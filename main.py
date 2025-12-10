import math
import time
import argparse, os, json, csv, logging
import random
import numpy as np
from typing import Dict, List, Tuple

try:
    import gtsam
except Exception:  # pragma: no cover - CLI will fail later if bindings missing
    gtsam = None
from cosmo_slam_centralised.loader import (load_jrl, iter_init_entries, iter_measurements,
                               LoaderConfig, summarize_schema, build_key_robot_map,
                               groundtruth_by_robot_key)
from cosmo_slam_centralised.models import InitEntry, PriorFactorPose3, BetweenFactorPose3
from cosmo_slam_centralised.graph import GraphBuilder, default_robot_infer
from cosmo_slam_centralised.isam import ISAM2Manager, incremental_optimize, optimize_all_batch
from cosmo_slam_common.kpi_logging import KPILogger
from cosmo_slam_common.bandwidth import BandwidthTracker, factor_bytes
from cosmo_slam_common.latency import LatencyTracker
from cosmo_slam_common.resource_monitor import ResourceMonitor
from cosmo_slam_common.viz import (
    plot_trajectories_2d,
    plot_trajectories_3d,
    plot_est_vs_gt_xy,
    plot_est_vs_gt_3d,
    plot_est_vs_gt_xy_aligned,
    plot_est_vs_gt_3d_aligned,
)
from cosmo_slam_common.metrics import align_and_ate_per_robot, compute_rpe_per_robot
from cosmo_slam_centralised.source import JRLFactorSource, ROS2FactorSource, FactorSourceError
from cosmo_slam_ros2.qos import parse_qos_options



def parse_args():
    ap = argparse.ArgumentParser(description="SLAM backends for COSMO-bench JRL datasets (centralised + decentralised).")
    ap.add_argument("--jrl", required=True, help="Path to .jrl JSON")
    ap.add_argument("--export-path", required=True, help="Directory to write outputs")
    ap.add_argument("--include-potential-outliers", action="store_true", help="(Reserved) Include potential outliers")
    ap.add_argument("--robust", choices=["none","huber","cauchy"], default="huber", help="Robust kernel")
    ap.add_argument("--robust-k", type=float, default=None, help="Robust tuning parameter")
    ap.add_argument("--batch-size", type=int, default=200, help="iSAM2 update batch size")
    ap.add_argument("--relin-th", type=float, default=0.1, help="iSAM2 relinearize threshold")
    ap.add_argument("--relin-skip", type=int, default=10, help="iSAM2 relinearize skip")
    ap.add_argument("--quat-order", choices=["wxyz","xyzw"], default="wxyz", help="Quaternion order in file")
    ap.add_argument("--log", default="INFO", help="Logging level")
    ap.add_argument("--solver", choices=["isam2","batch"], default="isam2", help="Choose iSAM2 (incremental) or LM (batch) solver")
    ap.add_argument("--eval-gt", action="store_true", help="Align to ground truth and report ATE; also export GT overlays")
    ap.add_argument("--backend", choices=["centralised", "centralized", "decentralised", "decentralized"], default="centralised",
                    help="Select centralised (global graph) or decentralised (per-robot DDF-SAM) backend")
    ap.add_argument("--resource-interval", type=float, default=0.5,
                    help="Resource monitor sampling interval in seconds (default 0.5)")
    ap.add_argument("--ddf-rounds", type=int, default=5,
                    help="Max decentralised optimisation rounds (backend=decentralised)")
    ap.add_argument("--ddf-convergence", type=float, default=5e-3,
                    help="Convergence threshold on interface translation change (backend=decentralised)")
    ap.add_argument("--ddf-rot-convergence", type=float, default=5e-3,
                    help="Convergence threshold on interface rotation change in radians (backend=decentralised)")
    ap.add_argument("--ddf-local-iters", type=int, default=25,
                    help="Max LM iterations per robot during decentralised solves")
    ap.add_argument("--ddf-relaxation", type=float, default=1.0,
                    help="Relaxation factor for inter-robot updates (1.0 disables relaxation)")
    ap.add_argument("--ddf-ingest-time-scale", type=float, default=0.0,
                    help="Scale factor converting factor stamp deltas to seconds when pacing ingest (0 disables)")
    ap.add_argument("--ddf-ingest-max-sleep", type=float, default=0.0,
                    help="Clamp per-factor ingest sleep to this many seconds (0 = no clamp)")
    ap.add_argument("--ddf-interface-time-scale", type=float, default=0.0,
                    help="Scale factor converting interface stamp deltas to seconds when pacing broadcasts (0 disables)")
    ap.add_argument("--ddf-interface-max-sleep", type=float, default=0.0,
                    help="Clamp per-interface sleep to this many seconds (0 = no clamp)")
    ap.add_argument("--ddf-multiprocess", action="store_true",
                    help="Run each DDF agent in its own process (requires --ddf-transport=ros2)")
    ap.add_argument("--central-transport", choices=["inproc", "ros2"], default="inproc",
                    help="Transport used for factor ingest in the centralised backend")
    ap.add_argument("--central-ros2-topic", dest="central_ros2_topic_prefix", default="/cosmo/factor_batch",
                    help="ROS 2 topic prefix for factor batches when --central-transport=ros2 (per robot topic = <prefix>/<robot>)")
    ap.add_argument("--central-ros2-reliability", choices=["reliable", "best_effort"], default="reliable",
                    help="Reliability QoS policy for the centralised ROS 2 subscriber")
    ap.add_argument("--central-ros2-durability", choices=["volatile", "transient_local"], default="volatile",
                    help="Durability QoS policy for the centralised ROS 2 subscriber")
    ap.add_argument("--central-ros2-depth", type=int, default=10,
                    help="Queue depth for the centralised ROS 2 subscriber")
    ap.add_argument("--central-ros2-max-batches", type=int, default=0,
                    help="Stop after receiving this many factor batches (0 = wait for idle timeout)")
    ap.add_argument("--central-ros2-queue-size", type=int, default=0,
                    help="Internal factor queue size when ingesting over ROS 2 (0 = unbounded)")
    ap.add_argument("--central-ros2-idle-timeout", type=float, default=1.0,
                    help="Seconds without messages before finishing ROS 2 ingest (requires at least one batch)")
    ap.add_argument("--ddf-transport", choices=["inproc", "ros2"], default="inproc",
                    help="Transport used for decentralised interface exchange")
    ap.add_argument("--ddf-ros2-topic-prefix", default="/cosmo/iface",
                    help="ROS 2 topic prefix for interface exchange when --ddf-transport=ros2")
    ap.add_argument("--ddf-ros2-reliability", choices=["reliable", "best_effort"], default="reliable",
                    help="Reliability QoS policy for decentralised ROS 2 publications")
    ap.add_argument("--ddf-ros2-durability", choices=["volatile", "transient_local"], default="volatile",
                    help="Durability QoS policy for decentralised ROS 2 publications")
    ap.add_argument("--ddf-ros2-depth", type=int, default=10,
                    help="Queue depth for decentralised ROS 2 publishers/subscribers")
    # Decentralised per-agent factor streaming (optional)
    ap.add_argument("--ddf-agent-factor-ros2", action="store_true",
                    help="Enable per-agent factor ingest over ROS 2 (/cosmo/factor_batch/<rid>) for identical streaming")
    ap.add_argument("--ddf-agent-factor-prefix", default="/cosmo/factor_batch",
                    help="ROS 2 topic prefix for per-agent factor ingest when --ddf-agent-factor-ros2 is set")
    ap.add_argument("--ddf-agent-factor-reliability", choices=["reliable", "best_effort"], default="reliable",
                    help="Reliability QoS for per-agent factor subscribers")
    ap.add_argument("--ddf-agent-factor-durability", choices=["volatile", "transient_local"], default="volatile",
                    help="Durability QoS for per-agent factor subscribers")
    ap.add_argument("--ddf-agent-factor-depth", type=int, default=10,
                    help="Queue depth for per-agent factor subscribers")
    ap.add_argument("--ddf-agent-factor-idle-timeout", type=float, default=1.0,
                    help="Seconds without factor messages before an agent closes its factor source")
    # Local solve scheduling during streaming
    ap.add_argument("--ddf-agent-solve-every-n", type=int, default=0,
                    help="Trigger a local solve after every N ingested factors (0 disables)")
    ap.add_argument("--ddf-agent-solve-period-s", type=float, default=0.0,
                    help="Trigger a local solve every T seconds during streaming (0 disables)")
    ap.add_argument("--use-sim-time", action="store_true",
                    help="Enable ROS 2 simulated time for ROS nodes (sets /use_sim_time=true)")
    return ap.parse_args()

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def robot_infer_from_prefixes(prefixes_csv: str):
    prefixes = [p.strip() for p in prefixes_csv.split(",") if p.strip()]
    def infer(key):
        s = str(key)
        for p in prefixes:
            if s.startswith(p):
                return p
        return default_robot_infer(key)
    return infer

def _rot3_to_quat_wxyz(R):
    # Prefer instance method if available
    if hasattr(R, "quaternion"):
        q = R.quaternion()  # expected [w, x, y, z]
        return float(q[0]), float(q[1]), float(q[2]), float(q[3])

    # Fallback: derive from rotation matrix (always available)
    M = R.matrix()
    m00, m01, m02 = float(M[0,0]), float(M[0,1]), float(M[0,2])
    m10, m11, m12 = float(M[1,0]), float(M[1,1]), float(M[1,2])
    m20, m21, m22 = float(M[2,0]), float(M[2,1]), float(M[2,2])

    tr = m00 + m11 + m22
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S

    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz) or 1.0
    return qw/n, qx/n, qy/n, qz/n


def _point3_to_xyz(t):
    """
    Support both GTSAM APIs:
    - Point3 with .x()/.y()/.z()
    - numpy-like array [x, y, z]
    - objects exposing .vector() -> array
    """
    # Method-style Point3
    if hasattr(t, "x"):
        try:
            return float(t.x()), float(t.y()), float(t.z())
        except Exception:
            pass
    # Vector-like
    vec = None
    if hasattr(t, "vector"):
        try:
            vec = t.vector()
        except Exception:
            vec = None
    if vec is None:
        vec = t
    try:
        return float(vec[0]), float(vec[1]), float(vec[2])
    except Exception as e:
        raise TypeError(f"Unsupported translation type {type(t)}") from e


def export_csv_per_robot(estimate, by_robot_keys: Dict[str, set], out_dir: str, graph_builder=None):
    ensure_dir(out_dir)
    for rid, keys in by_robot_keys.items():
        rows = []
        for k in sorted(keys, key=lambda x: str(x)):
            if estimate.exists(k):
                p = estimate.atPose3(k)
                tx, ty, tz = _point3_to_xyz(p.translation())
                qw, qx, qy, qz = _rot3_to_quat_wxyz(p.rotation())
                key_label = graph_builder.denormalize_key(k) if graph_builder else k
                rows.append({
                    "key": key_label,
                    "x": tx, "y": ty, "z": tz,
                    "qw": qw, "qx": qx, "qy": qy, "qz": qz,
                })
        csv_path = os.path.join(out_dir, f"trajectory_{rid}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            headers = ["key","x","y","z","qw","qx","qy","qz"]
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for r in rows:
                w.writerow(r)

def export_stats_json(counts: Dict[str,int], graph_error: float, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"factors": counts, "final_error": graph_error}, f, indent=2)


def export_rpe_csv(rpe_stats: Dict[str, Dict[str, Dict[str, float]]], out_path: str) -> None:
    """Write Relative Pose Error metrics to CSV for quick inspection."""
    ensure_dir(os.path.dirname(out_path))
    headers = ["robot", "window", "count", "rmse", "mean", "pstd", "mean_dt"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for rid, windows in sorted(rpe_stats.items()):
            for window, stats in sorted(windows.items(), key=lambda item: float(item[0])):
                writer.writerow([
                    rid,
                    window,
                    int(stats.get("count", 0)),
                    float(stats.get("rmse", 0.0)),
                    float(stats.get("mean", 0.0)),
                    float(stats.get("pstd", 0.0)),
                    float(stats.get("mean_dt", 0.0)),
                ])


def run_centralised(args,
                    doc,
                    cfg,
                    robot_map: Dict[str, str],
                    init_lookup: Dict[str, InitEntry],
                    factor_iter,
                    out_dir: str):
    """Execute the original centralised pipeline."""
    robust_kind = None if args.robust == "none" else args.robust
    gb = GraphBuilder(robot_map=robot_map, robust_kind=robust_kind, robust_k=args.robust_k)
    isam = ISAM2Manager(relinearize_threshold=args.relin_th, relinearize_skip=args.relin_skip)

    kpi_dir = os.path.join(out_dir, "kpi_metrics")
    ensure_dir(kpi_dir)
    kpi = KPILogger(extra_fields={"solver": args.solver},
                    log_path=os.path.join(kpi_dir, "kpi_events.jsonl"),
                    emit_to_logger=False)
    bandwidth = BandwidthTracker()
    latency = LatencyTracker()
    resource_monitor = ResourceMonitor()
    resource_meta = {
        "jrl": os.path.abspath(args.jrl),
        "solver": args.solver,
        "batch_size": args.batch_size,
        "relinearize_skip": args.relin_skip,
        "agent_count": len(robot_map) or len(gb.by_robot_keys) or None,
        "central_transport": args.central_transport,
    }
    if args.central_transport == "ros2":
        resource_meta.update({
            "central_ros2_topic_prefix": args.central_ros2_topic_prefix,
            "central_ros2_reliability": args.central_ros2_reliability,
            "central_ros2_durability": args.central_ros2_durability,
            "central_ros2_depth": args.central_ros2_depth,
            "transport": "ros2",
            "ros_domain_id": int(os.environ.get("ROS_DOMAIN_ID", "0")) if os.environ.get("ROS_DOMAIN_ID") else None,
        })
    resource_monitor.start(resource_meta)
    # Optional per-process aggregation via environment variable
    try:
        pids_env = os.environ.get("COSMO_RESOURCE_PIDS")
        if pids_env:
            pids = [int(x) for x in pids_env.split(",") if x.strip()]
            resource_monitor.set_pids(pids)
    except Exception:
        pass

    estimate = None
    ack_pub = None

    try:
        if args.solver == "isam2":
            # Optional ACK publisher for ROS2 factor batches (E2E send→use)
            if args.central_transport == "ros2":
                try:
                    from cosmo_slam_ros2.ack import Ros2AckPublisher  # type: ignore
                    ack_pub = Ros2AckPublisher(
                        topic_prefix=args.central_ros2_topic_prefix,
                        qos_profile={
                            "reliability": args.central_ros2_reliability,
                            "durability": args.central_ros2_durability,
                            "depth": args.central_ros2_depth,
                        },
                    )
                except Exception:
                    ack_pub = None

            estimate = incremental_optimize(
                factor_iter,
                init_lookup,
                gb,
                isam,
                batch_size=args.batch_size,
                kpi=kpi,
                bandwidth=bandwidth,
                latency=latency,
                ack_publisher=ack_pub,
            )
        else:
            batch_events = []
            for f in iter_measurements(doc, cfg):
                if isinstance(f, PriorFactorPose3):
                    gb.add_prior(f, init_lookup)
                    topic = f"prior/{gb.robot_of(f.key)}"
                    size = factor_bytes(f)
                    bandwidth.add_uplink(topic, size)
                    if latency:
                        batch_events.append(
                            latency.record_ingest(
                                "PriorFactorPose3",
                                float(getattr(f, "stamp", 0.0)),
                                gb.robot_of(f.key),
                                ingest_wall=time.time(),
                                ingest_mono=time.perf_counter(),
                                metadata={
                                    "key": str(f.key),
                                    **({"send_ts_mono": float(getattr(f, "send_ts_mono", None))} if getattr(f, "send_ts_mono", None) is not None else {}),
                                    **({"send_ts_wall": float(getattr(f, "send_ts_wall", None))} if getattr(f, "send_ts_wall", None) is not None else {}),
                                },
                            )
                        )
                    kpi.sensor_ingest("PriorFactorPose3", f.stamp, key=str(f.key), topic=topic, bytes=size)
                elif isinstance(f, BetweenFactorPose3):
                    gb.add_between(f, init_lookup)
                    rid1 = gb.robot_of(f.key1)
                    rid2 = gb.robot_of(f.key2)
                    topic = f"between/{rid1}" if rid1 == rid2 else f"between/{rid1}-{rid2}"
                    size = factor_bytes(f)
                    bandwidth.add_uplink(topic, size)
                    if latency:
                        batch_events.append(
                            latency.record_ingest(
                                "BetweenFactorPose3",
                                float(getattr(f, "stamp", 0.0)),
                                rid1,
                                ingest_wall=time.time(),
                                ingest_mono=time.perf_counter(),
                                metadata={
                                    "key1": str(f.key1),
                                    "key2": str(f.key2),
                                    **({"send_ts_mono": float(getattr(f, "send_ts_mono", None))} if getattr(f, "send_ts_mono", None) is not None else {}),
                                    **({"send_ts_wall": float(getattr(f, "send_ts_wall", None))} if getattr(f, "send_ts_wall", None) is not None else {}),
                                },
                            )
                        )
                    kpi.sensor_ingest("BetweenFactorPose3", f.stamp, key1=str(f.key1), key2=str(f.key2), topic=topic, bytes=size)
            if latency:
                latency.assign_batch(1, batch_events)
            estimate = optimize_all_batch(
                gb.graph,
                gb.initial,
                max_iters=200,
                kpi=kpi,
                bandwidth=bandwidth,
                latency=latency,
                batch_id=1,
                translation_cache={},
            )

        try:
            final_err = isam.error(gb.graph, estimate)
        except Exception:
            final_err = gb.graph.error(estimate)

        print("=== Centralised optimisation summary ===")
        for rid, keys in gb.by_robot_keys.items():
            print(f"Robot {rid}: #poses = {len(keys)}")
        print(f"Factors: {gb.counts}")
        print(f"Final graph error: {final_err:.6f}")

        if gb.loop_counts.get("intra"):
            print("Total intra-robot loop closures:")
            for rid, cnt in sorted(gb.loop_counts["intra"].items()):
                print(f"  Robot {rid}: {cnt}")
        if gb.loop_counts.get("inter_pairs"):
            print("Total inter-robot loop closures:")
            for (rid1, rid2), cnt in sorted(gb.loop_counts["inter_pairs"].items()):
                print(f"  Robots {rid1}-{rid2}: {cnt}")
        if gb.loop_counts.get("inter_per_robot"):
            print("Inter-robot closures per robot:")
            for rid, cnt in sorted(gb.loop_counts["inter_per_robot"].items()):
                print(f"  Robot {rid}: {cnt}")

        if args.eval_gt:
            gt_map = groundtruth_by_robot_key(doc, cfg)
            metrics = align_and_ate_per_robot(estimate, gb.by_robot_keys, gt_map, key_label=gb.denormalize_key)
            print("=== ATE (aligned) ===")
            for rid, m in metrics.items():
                print(f"Robot {rid}: matches={m['matches']}, rmse={m['rmse']}")
            rpe_stats = compute_rpe_per_robot(
                estimate,
                gb.by_robot_keys,
                gt_map,
                metrics,
                key_label=gb.denormalize_key,
            )
            if rpe_stats:
                for rid, stats in rpe_stats.items():
                    metrics.setdefault(rid, {})["rpe"] = stats
                rpe_csv = os.path.join(out_dir, "rpe_metrics.csv")
                export_rpe_csv(rpe_stats, rpe_csv)
            gt_xy = os.path.join(out_dir, "trajectories_xy_est_vs_gt.png")
            gt_3d = os.path.join(out_dir, "trajectories_3d_est_vs_gt.png")
            plot_est_vs_gt_xy(estimate, gb.by_robot_keys, gt_map, gt_xy)
            plot_est_vs_gt_3d(estimate, gb.by_robot_keys, gt_map, gt_3d)
            try:
                gt_xy_al = os.path.join(out_dir, "trajectories_xy_est_vs_gt_aligned.png")
                gt_3d_al = os.path.join(out_dir, "trajectories_3d_est_vs_gt_aligned.png")
                plot_est_vs_gt_xy_aligned(estimate, gb.by_robot_keys, gt_map, metrics, gt_xy_al)
                plot_est_vs_gt_3d_aligned(estimate, gb.by_robot_keys, gt_map, metrics, gt_3d_al)
            except Exception:
                pass
            with open(os.path.join(out_dir, "gt_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

        export_csv_per_robot(estimate, gb.by_robot_keys, os.path.join(out_dir, "trajectories"), gb)
        export_stats_json(gb.counts, final_err, os.path.join(out_dir, "graph_stats.json"))
        plot_trajectories_2d(estimate, gb.by_robot_keys, os.path.join(out_dir, "trajectories_xy.png"))
        plot_trajectories_3d(estimate, gb.by_robot_keys, os.path.join(out_dir, "trajectories_3d.png"))

        print(f"Artifacts written to: {out_dir}")

        bandwidth_path = os.path.join(kpi_dir, "bandwidth_stats.json")
        bandwidth.export_json(bandwidth_path)
        bw_logger = logging.getLogger("cosmo_slam.bandwidth")
        bw_logger.info("Bandwidth stats written to %s", bandwidth_path)
        bandwidth.log_summary(bw_logger)

        latency_path = os.path.join(kpi_dir, "latency_metrics.json")
        latency.export_json(latency_path)
        lat_logger = logging.getLogger("cosmo_slam.latency")
        lat_logger.info("Latency metrics written to %s", latency_path)
        latency.log_summary(lat_logger)
    finally:
        try:
            if ack_pub is not None:
                ack_pub.close()
        except Exception:
            pass
        resource_monitor.stop()

    resource_monitor.update_metadata(
        robots=len(gb.by_robot_keys),
        central_transport=args.central_transport,
        transport=("ros2" if args.central_transport == "ros2" else None),
        ros_domain_id=(int(os.environ.get("ROS_DOMAIN_ID", "0")) if args.central_transport == "ros2" and os.environ.get("ROS_DOMAIN_ID") else None),
    )
    resource_path = os.path.join(kpi_dir, "resource_profile.json")
    resource_monitor.export_json(resource_path)
    res_logger = logging.getLogger("cosmo_slam.resource")
    res_logger.info("Resource profile written to %s", resource_path)
    resource_monitor.log_summary(res_logger)
    kpi.close()

    return estimate, gb


def _select_ddf_bus(args):
    if getattr(args, "ddf_transport", "inproc") != "ros2":
        return None

    qos_profile = parse_qos_options(
        reliability=args.ddf_ros2_reliability,
        durability=args.ddf_ros2_durability,
        depth=args.ddf_ros2_depth,
    )
    topic_prefix = args.ddf_ros2_topic_prefix

    def _factory(robot_ids):
        from cosmo_slam_decentralised.communication import Ros2PeerBus

        try:
            return Ros2PeerBus(
                robot_ids,
                topic_prefix=topic_prefix,
                qos_profile=dict(qos_profile),
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialise ROS 2 peer bus: {exc}") from exc

    return _factory


def run_decentralised(args,
                      doc,
                      cfg,
                      robot_map: Dict[str, str],
                      init_lookup: Dict[str, InitEntry],
                      factors: List,
                      out_dir: str):
    """Execute the decentralised DDF-SAM backend and export lightweight artefacts."""
    if gtsam is None:
        raise RuntimeError("gtsam bindings are required for the decentralised backend")

    from cosmo_slam_decentralised.ddf_sam import DDFSAMBackend, PacingConfig
    from cosmo_slam_decentralised.mp_runner import run_multiprocess

    robust_kind = None if args.robust == "none" else args.robust
    # KPI and resource tracking for decentralised backend (parity with centralised)
    kpi_dir = os.path.join(out_dir, "kpi_metrics")
    ensure_dir(kpi_dir)
    kpi_events_path = os.path.join(kpi_dir, "kpi_events.jsonl")
    kpi: KPILogger | None = None
    bandwidth = BandwidthTracker()
    latency = LatencyTracker()
    resource_monitor = ResourceMonitor()
    resource_monitor.start({
        "jrl": os.path.abspath(args.jrl),
        "solver": "ddf_sam",
        "ddf_transport": args.ddf_transport,
        "ddf_rounds": args.ddf_rounds,
        "ddf_convergence": args.ddf_convergence,
        "ddf_rot_convergence": args.ddf_rot_convergence,
        "ddf_local_iters": args.ddf_local_iters,
    })
    # Optional per-process aggregation via environment variable
    try:
        pids_env = os.environ.get("COSMO_RESOURCE_PIDS")
        if pids_env:
            pids = [int(x) for x in pids_env.split(",") if x.strip()]
            resource_monitor.set_pids(pids)
    except Exception:
        pass

    pacing_cfg = PacingConfig(
        ingest_time_scale=args.ddf_ingest_time_scale,
        ingest_max_sleep=args.ddf_ingest_max_sleep,
        interface_time_scale=args.ddf_interface_time_scale,
        interface_max_sleep=args.ddf_interface_max_sleep,
    )

    # Multi-process mode: one process per agent communicating over ROS2
    if args.ddf_multiprocess:
        if getattr(args, "ddf_transport", "inproc") != "ros2":
            raise RuntimeError("--ddf-multiprocess requires --ddf-transport=ros2")
        qos_profile = dict(
            parse_qos_options(
                reliability=getattr(args, "ddf_ros2_reliability", "reliable"),
                durability=getattr(args, "ddf_ros2_durability", "volatile"),
                depth=getattr(args, "ddf_ros2_depth", 10),
            )
        )
        # Launch agents
        pids, bw_tracker, lat_tracker, ns_prefix = run_multiprocess(
            robot_map=robot_map,
            init_lookup=init_lookup,
            factors=factors,
            robust_kind=robust_kind,
            robust_k=args.robust_k,
            batch_max_iters=args.ddf_local_iters,
            ddf_rounds=args.ddf_rounds,
            convergence_tol=args.ddf_convergence,
            rotation_tol=args.ddf_rot_convergence,
            relaxation_alpha=args.ddf_relaxation,
            topic_prefix=getattr(args, "ddf_ros2_topic_prefix", "/cosmo/iface"),
            qos_profile=qos_profile,
            out_dir=out_dir,
            ingest_time_scale=args.ddf_ingest_time_scale,
            ingest_max_sleep=args.ddf_ingest_max_sleep,
            interface_time_scale=args.ddf_interface_time_scale,
            interface_max_sleep=args.ddf_interface_max_sleep,
            agent_factor_ros2=bool(getattr(args, "ddf_agent_factor_ros2", False)),
            agent_factor_prefix=getattr(args, "ddf_agent_factor_prefix", "/cosmo/factor_batch"),
            agent_factor_reliability=getattr(args, "ddf_agent_factor_reliability", "reliable"),
            agent_factor_durability=getattr(args, "ddf_agent_factor_durability", "volatile"),
            agent_factor_depth=int(getattr(args, "ddf_agent_factor_depth", 10)),
            agent_factor_idle_timeout=float(getattr(args, "ddf_agent_factor_idle_timeout", 1.0)),
            agent_solve_every_n=int(getattr(args, "ddf_agent_solve_every_n", 0)),
            agent_solve_period_s=float(getattr(args, "ddf_agent_solve_period_s", 0.0)),
        )
        # Aggregate child PIDs in resource monitor for ongoing sampling
        try:
            resource_monitor.set_pids([pid for pid in pids if pid])
        except Exception:
            pass
        # KPI exports from monitor (byte‑accurate bandwidth, E2E send→ingest)
        print("=== Decentralised optimisation summary (multiprocess) ===")
        if bw_tracker is not None:
            bandwidth_path = os.path.join(kpi_dir, "bandwidth_stats.json")
            bw_tracker.export_json(bandwidth_path)
            logging.getLogger("cosmo_slam.bandwidth").info("Bandwidth stats written to %s", bandwidth_path)
        if lat_tracker is not None:
            # Derive ingest->opt and ingest->broadcast latencies from merged per-agent KPI events
            try:
                import bisect
                kpi_path = os.path.join(kpi_dir, "kpi_events.jsonl")
                opt_map: Dict[str, list] = {}
                brd_map: Dict[str, list] = {}
                # Avoid PEP 585 syntax for broader Python compatibility
                ing_events = []  # list of (robot, ingest_ts)
                # 1) Build per-robot optimization/broadcast timelines from KPI events
                with open(kpi_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            ev = json.loads(line)
                        except Exception:
                            continue
                        ts = ev.get("ts")
                        if not isinstance(ts, (int, float)):
                            continue
                        name = ev.get("event")
                        rid = ev.get("robot")
                        if name == "optimization_start" and rid:
                            opt_map.setdefault(str(rid), []).append(float(ts))
                        elif name == "map_broadcast" and rid:
                            brd_map.setdefault(str(rid), []).append(float(ts))
                for lst in opt_map.values():
                    lst.sort()
                for lst in brd_map.values():
                    lst.sort()
                # 2) Extract ingest times from the monitor's latency events (InterfaceMessage arrival)
                for ev in getattr(lat_tracker, "events", []):
                    rid = ev.get("receiver")
                    ts_in = ev.get("ingest_wall")
                    if rid and isinstance(ts_in, (int, float)):
                        ing_events.append((str(rid), float(ts_in)))
                # For each ingest, pair with the next solve start / next broadcast for that robot
                for rid, ts_in in sorted(ing_events, key=lambda t: t[1]):
                    # Optimization delta
                    lst = opt_map.get(rid) or []
                    if lst:
                        idx = bisect.bisect_left(lst, ts_in)
                        if idx < len(lst):
                            d = float(lst[idx] - ts_in)
                            eid = lat_tracker.record_ingest("InterfaceIngest", 0.0, rid, ts_in, ts_in)
                            try:
                                lat_tracker._events[eid]["latency_ingest_to_opt"] = d  # type: ignore[attr-defined]
                            except Exception:
                                pass
                    # Broadcast delta
                    lstb = brd_map.get(rid) or []
                    if lstb:
                        idxb = bisect.bisect_left(lstb, ts_in)
                        if idxb < len(lstb):
                            d = float(lstb[idxb] - ts_in)
                            eid = lat_tracker.record_ingest("InterfaceIngest", 0.0, rid, ts_in, ts_in)
                            try:
                                lat_tracker._events[eid]["latency_ingest_to_broadcast"] = d  # type: ignore[attr-defined]
                            except Exception:
                                pass
            except Exception:
                logging.getLogger("cosmo_slam.latency").debug("Failed to derive ingest→opt/broadcast for decentralised MP", exc_info=False)

            latency_path = os.path.join(kpi_dir, "latency_metrics.json")
            lat_tracker.export_json(latency_path)
            logging.getLogger("cosmo_slam.latency").info("Latency metrics written to %s", latency_path)

        resource_monitor.stop()

        resource_monitor.update_metadata(
            robots=len(robot_map) or 0,
            ddf_transport=args.ddf_transport,
            transport=("ros2" if args.ddf_transport == "ros2" else None),
            ros_domain_id=(int(os.environ.get("ROS_DOMAIN_ID", "0")) if args.ddf_transport == "ros2" and os.environ.get("ROS_DOMAIN_ID") else None),
            multiprocess=True,
            pids=pids,
            ddf_ros2_topic_prefix=args.ddf_ros2_topic_prefix,
            ddf_ros2_namespaced_prefix=ns_prefix,
        )
        resource_path = os.path.join(kpi_dir, "resource_profile.json")
        resource_monitor.export_json(resource_path)
        res_logger = logging.getLogger("cosmo_slam.resource")
        res_logger.info("Resource profile written to %s", resource_path)
        resource_monitor.log_summary(res_logger)
        # kpi_events.jsonl is produced above from per-agent logs when available; ensure it exists
        try:
            if not os.path.exists(kpi_events_path):
                with open(kpi_events_path, "w", encoding="utf-8") as f:
                    f.write("")
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Reconstruct combined trajectories for plotting + GT alignment
        # ------------------------------------------------------------------
        trajectories_dir = os.path.join(out_dir, "trajectories")
        ensure_dir(trajectories_dir)

        key_cache: Dict[str, int] = {}
        reverse_cache: Dict[int, str] = {}

        def _normalize_key_for_merge(raw_key: str) -> int:
            s = str(raw_key)
            if s in key_cache:
                return key_cache[s]
            norm: int
            if s.isdigit():
                norm = int(s)
            elif s and s[0].isalpha():
                digits = ''.join(ch for ch in s[1:] if ch.isdigit())
                idx = int(digits or "0")
                try:
                    norm = gtsam.symbol(s[0], idx)
                except Exception:
                    norm = hash((s, len(key_cache)))
            else:
                try:
                    norm = int(s)
                except Exception:
                    norm = hash((s, len(key_cache)))
            key_cache[s] = norm
            reverse_cache[norm] = s
            return norm

        aggregated_estimate = gtsam.Values()
        aggregated_by_robot: Dict[str, set] = {}
        label_lookup: Dict[int, str] = {}
        have_values = False

        robot_ids = set(robot_map.values())
        try:
            for fname in os.listdir(trajectories_dir):
                if fname.startswith("trajectory_") and fname.endswith(".csv"):
                    robot_ids.add(fname[len("trajectory_"):-4])
        except Exception:
            pass
        robot_ids = sorted(robot_ids)

        robot_stats: Dict[str, Dict[str, object]] = {}
        max_iterations = 0
        converged_flags: List[bool] = []

        for rid in robot_ids:
            traj_path = os.path.join(trajectories_dir, f"trajectory_{rid}.csv")
            pose_count = 0
            if os.path.exists(traj_path):
                with open(traj_path, "r", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        key_label = row.get("key")
                        if not key_label:
                            continue
                        try:
                            tx = float(row.get("x", 0.0))
                            ty = float(row.get("y", 0.0))
                            tz = float(row.get("z", 0.0))
                            qw = float(row.get("qw", 1.0))
                            qx = float(row.get("qx", 0.0))
                            qy = float(row.get("qy", 0.0))
                            qz = float(row.get("qz", 0.0))
                        except Exception:
                            continue
                        norm_q = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz) or 1.0
                        qw_n, qx_n, qy_n, qz_n = qw / norm_q, qx / norm_q, qy / norm_q, qz / norm_q
                        try:
                            pose = gtsam.Pose3(
                                gtsam.Rot3.Quaternion(qw_n, qx_n, qy_n, qz_n),
                                gtsam.Point3(tx, ty, tz),
                            )
                        except Exception:
                            continue
                        norm_key = _normalize_key_for_merge(key_label)
                        if aggregated_estimate.exists(norm_key):
                            aggregated_estimate.update(norm_key, pose)
                        else:
                            aggregated_estimate.insert(norm_key, pose)
                        label_lookup[norm_key] = str(key_label)
                        robot_id = robot_map.get(str(key_label), default_robot_infer(str(key_label)))
                        aggregated_by_robot.setdefault(robot_id, set()).add(norm_key)
                        have_values = True
                        pose_count += 1

            status_path = os.path.join(trajectories_dir, f"agent_status_{rid}.json")
            status_data: Dict[str, object] = {}
            if os.path.exists(status_path):
                try:
                    with open(status_path, "r", encoding="utf-8") as fh:
                        status_data = json.load(fh)
                except Exception:
                    status_data = {}
            iterations = int(status_data.get("iterations", 0) or 0)
            max_iterations = max(max_iterations, iterations)
            conv_flag = bool(status_data.get("converged", False))
            converged_flags.append(conv_flag)
            robot_stats[rid] = {
                "poses": pose_count,
                "iterations": iterations or None,
                "converged": conv_flag,
                "pid": status_data.get("pid"),
            }

        # Compute message counts from merged KPI events
        map_broadcast_count = 0
        total_events = 0
        if os.path.exists(kpi_events_path):
            try:
                with open(kpi_events_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        total_events += 1
                        try:
                            evt = json.loads(line)
                        except Exception:
                            continue
                        if evt.get("event") == "map_broadcast":
                            map_broadcast_count += 1
            except Exception:
                pass

        stats_payload = {
            "iterations": max_iterations,
            "converged": all(converged_flags) if converged_flags else False,
            "agents": robot_stats,
            "messages_delivered": map_broadcast_count,
            "kpi_event_count": total_events,
        }
        stats_path = os.path.join(out_dir, "decentralised_stats.json")
        try:
            with open(stats_path, "w", encoding="utf-8") as fh:
                json.dump(stats_payload, fh, indent=2)
        except Exception:
            pass

        if have_values:
            try:
                plot_trajectories_2d(aggregated_estimate, aggregated_by_robot, os.path.join(out_dir, "trajectories_xy.png"))
                plot_trajectories_3d(aggregated_estimate, aggregated_by_robot, os.path.join(out_dir, "trajectories_3d.png"))
            except Exception:
                pass

        align_pose = None

        if args.eval_gt and have_values:
            gt_map = groundtruth_by_robot_key(doc, cfg)

            def _label(key):
                return label_lookup.get(int(key), str(key))

            try:
                metrics = align_and_ate_per_robot(aggregated_estimate, aggregated_by_robot, gt_map, key_label=_label)
                base_robot = next(
                    (rid for rid, m in metrics.items() if m.get("rmse") is not None and m.get("R")),
                    None,
                )
                if base_robot is not None:
                    R_np = np.asarray(metrics[base_robot]["R"], dtype=float)
                    t_np = np.asarray(metrics[base_robot]["t"], dtype=float)
                    try:
                        align_rot = gtsam.Rot3(R_np)
                        align_trans = gtsam.Point3(float(t_np[0]), float(t_np[1]), float(t_np[2]))
                        align_pose = gtsam.Pose3(align_rot, align_trans)
                        for key in list(aggregated_estimate.keys()):
                            pose = aggregated_estimate.atPose3(key)
                            aggregated_estimate.update(key, align_pose.compose(pose))
                    except Exception:
                        align_pose = None

                metrics = align_and_ate_per_robot(aggregated_estimate, aggregated_by_robot, gt_map, key_label=_label)
                print("=== ATE (aligned, decentralised) ===")
                for rid, m in metrics.items():
                    print(f"Robot {rid}: matches={m['matches']}, rmse={m['rmse']}")
                rpe_stats = compute_rpe_per_robot(
                    aggregated_estimate,
                    aggregated_by_robot,
                    gt_map,
                    metrics,
                    key_label=_label,
                )
                if rpe_stats:
                    for rid, stats in rpe_stats.items():
                        metrics.setdefault(rid, {})["rpe"] = stats
                    export_rpe_csv(rpe_stats, os.path.join(out_dir, "rpe_metrics.csv"))
                gt_xy = os.path.join(out_dir, "trajectories_xy_est_vs_gt.png")
                gt_3d = os.path.join(out_dir, "trajectories_3d_est_vs_gt.png")
                plot_est_vs_gt_xy(aggregated_estimate, aggregated_by_robot, gt_map, gt_xy)
                plot_est_vs_gt_3d(aggregated_estimate, aggregated_by_robot, gt_map, gt_3d)
                with open(os.path.join(out_dir, "gt_metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
            except Exception:
                logging.getLogger("cosmo_slam.decentralised").warning(
                    "Failed to align decentralised estimates with ground truth",
                    exc_info=True,
                )
        elif args.eval_gt and not have_values:
            logging.getLogger("cosmo_slam.decentralised").warning(
                "Decentralised multiprocess run produced no estimates; skipping GT alignment",
            )

        return {"pids": pids, "topic_prefix": ns_prefix}

    kpi = KPILogger(extra_fields={"solver": "ddf_sam"},
                    log_path=kpi_events_path,
                    emit_to_logger=False)

    backend = DDFSAMBackend(
        robot_map=robot_map,
        init_lookup=init_lookup,
        robust_kind=robust_kind,
        robust_k=args.robust_k,
        batch_max_iters=args.ddf_local_iters,
        kpi=kpi,
        bandwidth=bandwidth,
        latency=latency,
        bus_factory=_select_ddf_bus(args),
        pacing=pacing_cfg,
    )
    backend.ingest_factors(factors)

    result = backend.run(
        max_rounds=args.ddf_rounds,
        convergence_tol=args.ddf_convergence,
        rotation_tol=args.ddf_rot_convergence,
        relaxation_alpha=args.ddf_relaxation,
    )

    print("=== Decentralised optimisation summary ===")
    print(f"Iterations: {result.iterations} (converged={result.converged})")

    trajectories_dir = os.path.join(out_dir, "trajectories")
    ensure_dir(trajectories_dir)

    global_estimate = gtsam.Values()
    global_by_robot: Dict[str, set] = {}
    label_lookup: Dict[int, str] = {}
    have_values = False
    robot_stats: Dict[str, Dict[str, object]] = {}

    for rid, agent in backend.agents.items():
        estimate = agent.estimate_snapshot()
        gb = getattr(agent, "_latest_graph", None)
        if estimate is None or gb is None:
            robot_stats[rid] = {"poses": 0, "factors": {}}
            continue

        keys_for_robot = gb.by_robot_keys.get(rid, set())
        robot_stats[rid] = {
            "poses": len(keys_for_robot),
            "factors": dict(gb.counts),
        }
        if keys_for_robot:
            export_csv_per_robot(estimate, {rid: keys_for_robot}, trajectories_dir, gb)
        for key in keys_for_robot:
            if not estimate.exists(key):
                continue
            if not global_estimate.exists(key):
                global_estimate.insert(key, estimate.atPose3(key))
            global_by_robot.setdefault(rid, set()).add(key)
            label_lookup[int(key)] = gb.denormalize_key(key)
            have_values = True

    stats_path = os.path.join(out_dir, "decentralised_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "iterations": result.iterations,
            "converged": result.converged,
            "agents": robot_stats,
            "messages_delivered": backend.bus.delivered,
        }, f, indent=2)

    print(f"Artifacts written to: {out_dir}")

    # Export KPI summaries and resource profile
    try:
        bandwidth_path = os.path.join(kpi_dir, "bandwidth_stats.json")
        bandwidth.export_json(bandwidth_path)
        bw_logger = logging.getLogger("cosmo_slam.bandwidth")
        bw_logger.info("Bandwidth stats written to %s", bandwidth_path)
        bandwidth.log_summary(bw_logger)

        latency_path = os.path.join(kpi_dir, "latency_metrics.json")
        latency.export_json(latency_path)
        lat_logger = logging.getLogger("cosmo_slam.latency")
        lat_logger.info("Latency metrics written to %s", latency_path)
        latency.log_summary(lat_logger)
    finally:
        resource_monitor.stop()

    resource_monitor.update_metadata(
        robots=len(backend.agents),
        ddf_transport=args.ddf_transport,
        transport=("ros2" if args.ddf_transport == "ros2" else None),
        ros_domain_id=(int(os.environ.get("ROS_DOMAIN_ID", "0")) if args.ddf_transport == "ros2" and os.environ.get("ROS_DOMAIN_ID") else None),
    )
    resource_path = os.path.join(kpi_dir, "resource_profile.json")
    resource_monitor.export_json(resource_path)
    res_logger = logging.getLogger("cosmo_slam.resource")
    res_logger.info("Resource profile written to %s", resource_path)
    resource_monitor.log_summary(res_logger)
    kpi.close()

    if args.eval_gt and have_values:
        gt_map = groundtruth_by_robot_key(doc, cfg)

        def _label(key):
            return label_lookup.get(int(key), str(key))

        metrics = align_and_ate_per_robot(global_estimate, global_by_robot, gt_map, key_label=_label)
        print("=== ATE (aligned, decentralised) ===")
        for rid, m in metrics.items():
            print(f"Robot {rid}: matches={m['matches']}, rmse={m['rmse']}")
        rpe_stats = compute_rpe_per_robot(
            global_estimate,
            global_by_robot,
            gt_map,
            metrics,
            key_label=_label,
        )
        if rpe_stats:
            for rid, stats in rpe_stats.items():
                metrics.setdefault(rid, {})["rpe"] = stats
            export_rpe_csv(rpe_stats, os.path.join(out_dir, "rpe_metrics.csv"))
        gt_xy = os.path.join(out_dir, "trajectories_xy_est_vs_gt.png")
        gt_3d = os.path.join(out_dir, "trajectories_3d_est_vs_gt.png")
        plot_est_vs_gt_xy(global_estimate, global_by_robot, gt_map, gt_xy)
        plot_est_vs_gt_3d(global_estimate, global_by_robot, gt_map, gt_3d)
        try:
            gt_xy_al = os.path.join(out_dir, "trajectories_xy_est_vs_gt_aligned.png")
            gt_3d_al = os.path.join(out_dir, "trajectories_3d_est_vs_gt_aligned.png")
            plot_est_vs_gt_xy_aligned(global_estimate, global_by_robot, gt_map, metrics, gt_xy_al)
            plot_est_vs_gt_3d_aligned(global_estimate, global_by_robot, gt_map, metrics, gt_3d_al)
        except Exception:
            pass
        with open(os.path.join(out_dir, "gt_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    elif args.eval_gt:
        logging.getLogger("cosmo_slam.decentralised").warning(
            "Ground-truth evaluation requested but no decentralised estimates were produced yet.")

    if have_values:
        plot_trajectories_2d(global_estimate, global_by_robot, os.path.join(out_dir, "trajectories_xy.png"))
        plot_trajectories_3d(global_estimate, global_by_robot, os.path.join(out_dir, "trajectories_3d.png"))

    return result


def main():
    args = parse_args()
    logging.basicConfig(level=args.log.upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if getattr(args, "use_sim_time", False):
        os.environ["COSMO_USE_SIM_TIME"] = "1"
    if getattr(args, "resource_interval", None) is not None:
        try:
            if float(args.resource_interval) > 0.0:
                os.environ["COSMO_RESOURCE_INTERVAL"] = str(float(args.resource_interval))
        except Exception:
            pass

    seed_env = os.environ.get("COSMO_RUN_SEED")
    if seed_env:
        try:
            seed_val = int(seed_env)
            random.seed(seed_val)
            try:
                np.random.seed(seed_val)
            except Exception:
                pass
        except Exception:
            pass

    out_dir = os.path.abspath(args.export_path)
    ensure_dir(out_dir)

    cfg = LoaderConfig(quaternion_order=args.quat_order,
                       validate_schema=True,
                       include_potential_outliers=args.include_potential_outliers)
    doc = load_jrl(args.jrl, cfg)

    print("Schema peek:", json.dumps(summarize_schema(doc)))

    robot_map: Dict[str, str] = build_key_robot_map(doc)
    init_lookup: Dict[str, InitEntry] = {}
    for it in iter_init_entries(doc, cfg):
        init_lookup[str(it.key)] = it

    backend_choice = args.backend.lower()
    if backend_choice in ("centralized", "centralised"):
        factor_source = None
        try:
            if args.central_transport == "ros2":
                if args.solver != "isam2":
                    raise ValueError("ROS 2 factor ingest currently supports only --solver isam2")
                robots = sorted({str(rid) for rid in robot_map.values() if rid is not None} | {"global"})
                qos_profile = {
                    "reliability": args.central_ros2_reliability,
                    "durability": args.central_ros2_durability,
                    "depth": args.central_ros2_depth,
                }
                factor_source = ROS2FactorSource(
                    topic_prefix=args.central_ros2_topic_prefix,
                    robot_ids=robots,
                    qos_profile=qos_profile,
                    queue_size=args.central_ros2_queue_size,
                    spin_timeout=0.1,
                    idle_timeout=args.central_ros2_idle_timeout,
                )
            else:
                factor_source = JRLFactorSource(doc, cfg)

            with factor_source as src:
                factor_iter = src.iter_factors()
                run_centralised(args, doc, cfg, robot_map, init_lookup, factor_iter, out_dir)
        except FactorSourceError as exc:
            logging.getLogger("cosmo_slam.centralised").error("Failed to construct factor source: %s", exc)
            raise
    else:
        factors = list(iter_measurements(doc, cfg))
        if not factors:
            raise ValueError("No factors found in dataset; cannot run decentralised backend")
        run_decentralised(args, doc, cfg, robot_map, init_lookup, factors, out_dir)


if __name__ == "__main__":
    main()
