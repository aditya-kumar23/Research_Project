from __future__ import annotations

import os
import time
import uuid
import json
import logging
import multiprocessing as mp
import random
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import gtsam  # type: ignore
except Exception:
    gtsam = None  # type: ignore

from cosmo_slam_centralised.models import InitEntry
from cosmo_slam_decentralised.partition import partition_measurements, RobotMeasurementBundle
from cosmo_slam_decentralised.agents import DecentralizedRobotAgent
from cosmo_slam_decentralised.communication import Ros2PeerBus, InterfaceMessage
from cosmo_slam_decentralised.bandwidth import interface_message_bytes
from cosmo_slam_ros2.qos import parse_qos_options
from cosmo_slam_ros2.interface_msg import decode_interface_message
from cosmo_slam_common.bandwidth import BandwidthTracker
from cosmo_slam_common.latency import LatencyTracker
from cosmo_slam_common.kpi_logging import KPILogger
from cosmo_slam_ros2.sim_time import configure_sim_time

logger = logging.getLogger("cosmo_slam.decentralised.mp")


_seed_env = os.environ.get("COSMO_RUN_SEED")
if _seed_env:
    try:
        _seed_val = int(_seed_env)
        random.seed(_seed_val)
        np.random.seed(_seed_val)
    except Exception:
        pass


def _export_csv_per_robot(estimate: "gtsam.Values", keys: Iterable, out_path: str, gb=None) -> None:
    rows = []
    for k in sorted(keys, key=lambda x: str(x)):
        try:
            if not estimate.exists(k):
                continue
            p = estimate.atPose3(k)
            t = p.translation()
            r = p.rotation()
            try:
                tx, ty, tz = float(t.x()), float(t.y()), float(t.z())
            except Exception:
                vec = None
                if hasattr(t, "vector"):
                    try:
                        vec = t.vector()
                    except Exception:
                        vec = None
                if vec is None:
                    try:
                        vec = np.asarray(t, dtype=float)
                    except Exception:
                        vec = [0.0, 0.0, 0.0]
                tx, ty, tz = float(vec[0]), float(vec[1]), float(vec[2])
            try:
                qw, qx, qy, qz = map(float, getattr(r, "quaternion", lambda: [1,0,0,0])())
            except Exception:
                qw, qx, qy, qz = float(r.w()), float(r.x()), float(r.y()), float(r.z())
            key_label = gb.denormalize_key(k) if gb else k
            rows.append({
                "key": key_label, "x": tx, "y": ty, "z": tz,
                "qw": qw, "qx": qx, "qy": qy, "qz": qz,
            })
        except Exception:
            continue
    import csv
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["key","x","y","z","qw","qx","qy","qz"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _agent_loop(
    rid: str,
    bundle: RobotMeasurementBundle,
    robot_ids: List[str],
    *,
    robot_map: Dict[str, str],
    init_lookup: Dict[str, InitEntry],
    robust_kind: Optional[str],
    robust_k: Optional[float],
    batch_max_iters: int,
    topic_prefix: str,
    qos_profile: Dict[str, object],
    out_dir: str,
    metrics_dir: str,
    max_rounds: int,
    convergence_tol: float,
    rotation_tol: float,
    relaxation_alpha: float,
    # Pacing controls (dataset and interface replay)
    ingest_time_scale: float = 0.0,
    ingest_max_sleep: float = 0.0,
    interface_time_scale: float = 0.0,
    interface_max_sleep: float = 0.0,
    # Optional per-agent factor streaming over ROS2
    agent_factor_ros2: bool = False,
    agent_factor_prefix: str = "/cosmo/factor_batch",
    agent_factor_reliability: str = "reliable",
    agent_factor_durability: str = "volatile",
    agent_factor_depth: int = 10,
    agent_factor_idle_timeout: float = 1.0,
    # Local solve schedule during factor streaming
    agent_solve_every_n: int = 0,
    agent_solve_period_s: float = 0.0,
) -> None:
    logging.basicConfig(level=os.environ.get("COSMO_LOG", "INFO"))
    try:
        agent = DecentralizedRobotAgent(
            rid,
            {str(k): str(v) for k, v in robot_map.items()},
            {str(k): v for k, v in init_lookup.items()},
            robust_kind=robust_kind,
            robust_k=robust_k,
            batch_max_iters=batch_max_iters,
        )
    except Exception as exc:
        logger.error("Agent %s failed to initialise: %s", rid, exc)
        return

    # Simple stamp pacer (mirrors _StampPacer in ddf_sam but kept local)
    class _StampPacer:
        def __init__(self, time_scale: float, max_sleep: float) -> None:
            self._time_scale = float(time_scale or 0.0)
            self._max_sleep = max(0.0, float(max_sleep or 0.0))
            self._last: Optional[float] = None

        @property
        def enabled(self) -> bool:
            return self._time_scale > 0.0

        def sleep_for(self, stamp: float) -> None:
            if not self.enabled:
                return
            try:
                cur = float(stamp)
            except Exception:
                return
            if 'math' in globals() and math is not None:
                try:
                    if not math.isfinite(cur):
                        return
                except Exception:
                    pass
            if self._last is None:
                self._last = cur
                return
            delta = cur - self._last
            self._last = cur
            if delta <= 0.0:
                return
            sleep_s = delta / self._time_scale
            if self._max_sleep > 0.0:
                sleep_s = min(sleep_s, self._max_sleep)
            if sleep_s > 0.0:
                time.sleep(sleep_s)

    try:
        import math  # local import for pacer
    except Exception:
        math = None  # type: ignore[assignment]

    ingest_pacer = _StampPacer(ingest_time_scale, ingest_max_sleep)
    iface_pacer = _StampPacer(interface_time_scale, interface_max_sleep)

    # We defer ROS 2 bus creation until after optional factor streaming to avoid
    # double rclpy.init() in some environments. The streaming path will still
    # run local solves; inter-robot broadcasts begin once the bus is up.
    bus = None

    # KPI logger (per-agent file, merged by parent post-run)
    kpi_logger: Optional[KPILogger] = None
    try:
        os.makedirs(metrics_dir, exist_ok=True)
        kpi_path = os.path.join(metrics_dir, f"kpi_events_{rid}.jsonl")
        kpi_logger = KPILogger(
            enabled=True,
            extra_fields={"solver": "ddf_sam", "robot": rid},
            log_path=kpi_path,
            emit_to_logger=False,
        )
    except Exception as exc:
        logger.debug("Agent %s: failed to initialise KPI logger: %s", rid, exc)
        kpi_logger = None

    # Populate local graph contents for this agent: either via ROS2 factor stream
    # (identical to centralised ingest per robot) or from the pre-partitioned bundle.
    try:
        use_ros2_factor_stream = bool(agent_factor_ros2)
        iter_idx_global = 0
        if use_ros2_factor_stream:
            factor_prefix = agent_factor_prefix
            factor_rel = agent_factor_reliability
            factor_dur = agent_factor_durability
            factor_depth = int(agent_factor_depth)
            factor_idle = float(agent_factor_idle_timeout)
            qos = {"reliability": factor_rel, "durability": factor_dur, "depth": factor_depth}
            try:
                from cosmo_slam_centralised.source import ROS2FactorSource, FactorSourceError  # type: ignore
            except Exception as exc:
                logger.error("Agent %s failed to import ROS2FactorSource: %s", rid, exc)
                return
            try:
                with ROS2FactorSource(
                    topic_prefix=factor_prefix,
                    robot_ids=[rid],
                    qos_profile=qos,
                    queue_size=0,
                    spin_timeout=0.1,
                    idle_timeout=factor_idle,
                ) as src:
                    # Streaming solve schedule
                    iter_idx = 0
                    last_solve = time.perf_counter()
                    since_last = 0
                    for f in src.iter_factors():
                        # Route factor into agent buffers
                        agent.ingest_factor(f)
                        since_last += 1
                        now = time.perf_counter()
                        do_solve = False
                        if agent_solve_every_n and since_last >= max(1, int(agent_solve_every_n)):
                            do_solve = True
                        if (not do_solve) and agent_solve_period_s and (now - last_solve) >= max(0.0, float(agent_solve_period_s)):
                            do_solve = True
                        if do_solve:
                            iter_idx += 1
                            agent.solve_round(iteration=iter_idx, kpi=kpi_logger)
                            outgoing = agent.interface_messages(iteration=iter_idx)
                            for msg in outgoing:
                                if iface_pacer.enabled:
                                    try:
                                        iface_pacer.sleep_for(getattr(msg, "stamp", 0.0))
                                    except Exception:
                                        pass
                                try:
                                    msg.sent_mono_time = time.perf_counter()
                                    msg.sent_wall_time = time.time()
                                except Exception:
                                    pass
                                if bus is not None:
                                    bus.post(msg)
                                if kpi_logger is not None:
                                    try:
                                        topic = f"{topic_prefix}/{msg.receiver}"
                                        kpi_logger.map_broadcast(
                                            batch_id=iter_idx,
                                            pose_count=None,
                                            topic=topic,
                                            receiver=msg.receiver,
                                            sender=msg.sender,
                                            bytes=interface_message_bytes(msg),
                                        )
                                    except Exception:
                                        pass
                            last_solve = now
                            since_last = 0
                    # Per-factor KPI (byte accounting consistent with centralised path)
                        if kpi_logger is not None:
                            try:
                                from cosmo_slam_centralised.models import PriorFactorPose3, BetweenFactorPose3  # local import to avoid overhead
                                from cosmo_slam_common.bandwidth import factor_bytes  # type: ignore
                            except Exception:
                                PriorFactorPose3 = None  # type: ignore
                                BetweenFactorPose3 = None  # type: ignore
                                factor_bytes = None  # type: ignore
                            try:
                                topic = getattr(f, "__ros2_topic", None)
                                stamp = float(getattr(f, "stamp", 0.0) or 0.0)
                                size = factor_bytes(f) if callable(factor_bytes) else None
                                if PriorFactorPose3 is not None and isinstance(f, PriorFactorPose3):
                                    kpi_logger.sensor_ingest("PriorFactorPose3", stamp, key=str(getattr(f, "key", "")), topic=topic, bytes=size)
                                elif BetweenFactorPose3 is not None and isinstance(f, BetweenFactorPose3):
                                    kpi_logger.sensor_ingest(
                                        "BetweenFactorPose3",
                                        stamp,
                                        key1=str(getattr(f, "key1", "")),
                                        key2=str(getattr(f, "key2", "")),
                                        topic=topic,
                                        bytes=size,
                                    )
                            except Exception:
                                pass
                    iter_idx_global = iter_idx
            except Exception as exc:
                logger.error("Agent %s failed to stream ROS2 factors: %s", rid, exc)
                return
        else:
            # Fallback: pre-partitioned bundle ingestion (optionally paced by dataset stamps)
            # Priors
            for f in getattr(bundle, "priors", []) or []:
                if ingest_pacer.enabled:
                    ingest_pacer.sleep_for(float(getattr(f, "stamp", 0.0) or 0.0))
                agent.add_local_priors([f])
            # Local between factors
            for f in getattr(bundle, "local_between", []) or []:
                if ingest_pacer.enabled:
                    ingest_pacer.sleep_for(float(getattr(f, "stamp", 0.0) or 0.0))
                agent.add_local_between([f])
            # Inter-robot factors (owned by this agent)
            for f in getattr(bundle, "inter_between", []) or []:
                if ingest_pacer.enabled:
                    ingest_pacer.sleep_for(float(getattr(f, "stamp", 0.0) or 0.0))
                agent.add_inter_factor(f)
    except Exception as exc:
        logger.error("Agent %s failed to ingest factors: %s", rid, exc)
        return

    # Bring up ROS 2 bus now (after streaming) to avoid rclpy.init() collisions
    try:
        bus = Ros2PeerBus(robot_ids, topic_prefix=topic_prefix, qos_profile=qos_profile)
    except Exception as exc:
        logger.error("Agent %s failed to create ROS2 bus: %s", rid, exc)
        return

    # ACK publisher for 'use' events
    try:
        import rclpy  # type: ignore
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy  # type: ignore
        from std_msgs.msg import UInt8MultiArray  # type: ignore
        qos = QoSProfile(
            depth=int(qos_profile.get("depth", 10)),
            reliability=(
                ReliabilityPolicy.RELIABLE
                if str(qos_profile.get("reliability", "reliable")).lower() == "reliable"
                else ReliabilityPolicy.BEST_EFFORT
            ),
            durability=(
                DurabilityPolicy.TRANSIENT_LOCAL
                if str(qos_profile.get("durability", "volatile")).lower() == "transient_local"
                else DurabilityPolicy.VOLATILE
            ),
        )
        ack_node = rclpy.create_node(f"cosmo_slam_ack_{rid}")
        configure_sim_time(ack_node)
        ack_topic = f"{topic_prefix}/ack/{rid}"
        ack_pub = ack_node.create_publisher(UInt8MultiArray, ack_topic, qos)
    except Exception as exc:
        ack_node = None
        ack_pub = None
        logger.debug("Agent %s: failed to create ACK publisher: %s", rid, exc)

    try:
        # Initial solve and broadcast if no streaming solves occurred
        if iter_idx_global == 0:
            agent.solve_round(iteration=0, kpi=kpi_logger)
            outgoing = agent.interface_messages(iteration=0)
        else:
            outgoing = []
        for msg in outgoing:
            if iface_pacer.enabled:
                try:
                    iface_pacer.sleep_for(getattr(msg, "stamp", 0.0))
                except Exception:
                    pass
            # Refresh send timestamps right before publish for accurate E2E
            try:
                msg.sent_mono_time = time.perf_counter()
                msg.sent_wall_time = time.time()
            except Exception:
                pass
            bus.post(msg)
            if kpi_logger is not None:
                try:
                    topic = f"{topic_prefix}/{msg.receiver}"
                    kpi_logger.map_broadcast(
                        batch_id=0,
                        pose_count=None,
                        topic=topic,
                        receiver=msg.receiver,
                        sender=msg.sender,
                        bytes=interface_message_bytes(msg),
                    )
                except Exception:
                    pass

        def _msg_state(msg: InterfaceMessage):
            t = np.array([msg.translation.x, msg.translation.y, msg.translation.z], dtype=float)
            q = np.array([msg.rotation.w, msg.rotation.x, msg.rotation.y, msg.rotation.z], dtype=float)
            q = q / (np.linalg.norm(q) or 1.0)
            return t, q

        last_state = {(m.sender, m.key): _msg_state(m) for m in outgoing}

        converged = False
        start_iter = int(iter_idx_global + 1)
        for iteration in range(start_iter, start_iter + max_rounds):
            incoming = bus.drain(rid)
            if incoming:
                agent.receive_interface_messages(incoming, relaxation=relaxation_alpha)
                if kpi_logger is not None:
                    for m in incoming:
                        try:
                            kpi_logger.sensor_ingest(
                                "InterfaceMessage",
                                float(getattr(m, "stamp", 0.0)),
                                sender=str(getattr(m, "sender", "")),
                                receiver=str(rid),
                                key=str(getattr(m, "key", "")),
                                iteration=int(getattr(m, "iteration", 0)),
                            )
                        except Exception:
                            pass
                # Publish ACKs to signal 'use' for E2E send->use latency
                if ack_pub is not None and ack_node is not None:
                    try:
                        import json as _json
                        now_mono = time.perf_counter()
                        now_wall = time.time()
                        for m in incoming:
                            payload = {
                                "sender": str(getattr(m, "sender", "")),
                                "receiver": str(rid),
                                "key": str(getattr(m, "key", "")),
                                "iteration": int(getattr(m, "iteration", 0)),
                                "send_ts_mono": float(getattr(m, "sent_mono_time", 0.0) or 0.0),
                                "send_ts_wall": float(getattr(m, "sent_wall_time", 0.0) or 0.0),
                                "use_ts_mono": float(now_mono),
                                "use_ts_wall": float(now_wall),
                            }
                            data = _json.dumps(payload, separators=(",", ":")).encode("utf-8")
                            msg = UInt8MultiArray()
                            msg.data = list(data)
                            try:
                                ack_pub.publish(msg)
                            except Exception:
                                pass
                    except Exception:
                        pass
            agent.solve_round(iteration=iteration, kpi=kpi_logger)
            outgoing = agent.interface_messages(iteration=iteration)
            if not outgoing:
                converged = True
                break
            for msg in outgoing:
                if iface_pacer.enabled:
                    try:
                        iface_pacer.sleep_for(getattr(msg, "stamp", 0.0))
                    except Exception:
                        pass
                try:
                    msg.sent_mono_time = time.perf_counter()
                    msg.sent_wall_time = time.time()
                except Exception:
                    pass
                bus.post(msg)
                if kpi_logger is not None:
                    try:
                        topic = f"{topic_prefix}/{msg.receiver}"
                        kpi_logger.map_broadcast(
                            batch_id=iteration,
                            pose_count=None,
                            topic=topic,
                            receiver=msg.receiver,
                            sender=msg.sender,
                            bytes=interface_message_bytes(msg),
                        )
                    except Exception:
                        pass

            # Local convergence heuristic (max delta below thresholds)
            state = {(m.sender, m.key): _msg_state(m) for m in outgoing}
            if last_state:
                td, rd = [], []
                for key, (t, q) in state.items():
                    prev = last_state.get(key)
                    if prev is None:
                        continue
                    pt, pq = prev
                    td.append(float(np.linalg.norm(t - pt)))
                    dot = float(abs(np.dot(q, pq)))
                    dot = max(-1.0, min(1.0, dot))
                    rd.append(float(2.0 * np.arccos(dot)))
                if kpi_logger is not None and td and rd:
                    try:
                        kpi_logger._emit(
                            "ddf_round_delta",
                            iteration=iteration,
                            max_translation_delta=float(max(td)),
                            max_rotation_delta=float(max(rd)),
                        )
                    except Exception:
                        pass
                if td and rd and max(td) < convergence_tol and max(rd) < rotation_tol:
                    converged = True
                    break
            last_state = state

        # Export per-robot CSV trajectory
        try:
            est = agent.estimate_snapshot()
            gb = getattr(agent, "_latest_graph", None)
            if est is not None and gb is not None:
                keys = gb.by_robot_keys.get(rid, set())
                if keys:
                    os.makedirs(out_dir, exist_ok=True)
                    _export_csv_per_robot(est, keys, os.path.join(out_dir, f"trajectory_{rid}.csv"), gb)
        except Exception:
            pass

        # Emit a small status file
        try:
            status = {"pid": os.getpid(), "rid": rid, "converged": converged, "iterations": iteration}
            with open(os.path.join(out_dir, f"agent_status_{rid}.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=2)
        except Exception:
            pass
    finally:
        try:
            bus.close()
        except Exception:
            pass
        if ack_node is not None:
            try:
                ack_node.destroy_node()
            except Exception:
                pass
        if kpi_logger is not None:
            try:
                kpi_logger.close()
            except Exception:
                pass


def run_multiprocess(
    robot_map: Dict[str, str],
    init_lookup: Dict[str, InitEntry],
    factors: Iterable,
    *,
    robust_kind: Optional[str],
    robust_k: Optional[float],
    batch_max_iters: int,
    ddf_rounds: int,
    convergence_tol: float,
    rotation_tol: float,
    relaxation_alpha: float,
    topic_prefix: str,
    qos_profile: Dict[str, object],
    out_dir: str,
    # Pacing controls (dataset and interface replay)
    ingest_time_scale: float = 0.0,
    ingest_max_sleep: float = 0.0,
    interface_time_scale: float = 0.0,
    interface_max_sleep: float = 0.0,
    # Agent factor streaming (ROS2)
    agent_factor_ros2: bool = False,
    agent_factor_prefix: str = "/cosmo/factor_batch",
    agent_factor_reliability: str = "reliable",
    agent_factor_durability: str = "volatile",
    agent_factor_depth: int = 10,
    agent_factor_idle_timeout: float = 1.0,
    # Agent local solve schedule
    agent_solve_every_n: int = 0,
    agent_solve_period_s: float = 0.0,
) -> Tuple[List[int], Optional[BandwidthTracker], Optional[LatencyTracker], str]:
    """Spawn one process per robot and execute decentralised DDF asynchronously.

    Returns the list of child PIDs for resource aggregation.
    """
    bundles = partition_measurements(factors, robot_map)
    robot_ids = sorted(set(robot_map.values()) | set(bundles.keys()))
    if not robot_ids:
        raise ValueError("No robots detected for multi-process DDF")

    trajectories_dir = os.path.join(out_dir, "trajectories")
    metrics_dir = os.path.join(out_dir, "kpi_metrics")
    os.makedirs(trajectories_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Namespace ROS 2 topics by run id to avoid collisions
    run_id = uuid.uuid4().hex[:8]
    # ROS 2 topic tokens must start with an alphabetic character; prefix with 'run_'.
    namespaced_prefix = f"{topic_prefix.rstrip('/')}/run_{run_id}"

    # Prepare monitoring subscribers in the parent to capture byte-accurate
    # bandwidth and E2E (send->ingest) latency during the run.
    bw_tracker: Optional[BandwidthTracker] = None
    lat_tracker: Optional[LatencyTracker] = None
    try:
        import rclpy  # type: ignore
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy  # type: ignore
        from std_msgs.msg import UInt8MultiArray  # type: ignore

        rclpy.init(args=None)
        node = rclpy.create_node("cosmo_slam_mp_monitor")
        configure_sim_time(node)
        # Build QoS from profile
        qos = QoSProfile(
            depth=int(qos_profile.get("depth", 10)),
            reliability=(
                ReliabilityPolicy.RELIABLE
                if str(qos_profile.get("reliability", "reliable")).lower() == "reliable"
                else ReliabilityPolicy.BEST_EFFORT
            ),
            durability=(
                DurabilityPolicy.TRANSIENT_LOCAL
                if str(qos_profile.get("durability", "volatile")).lower() == "transient_local"
                else DurabilityPolicy.VOLATILE
            ),
        )
        bw_tracker = BandwidthTracker()
        lat_tracker = LatencyTracker()

        def _make_cb(expected_rid: str):
            def _cb(msg):
                try:
                    data = bytes(getattr(msg, "data", []) or [])
                    iface = decode_interface_message(data)
                except Exception as exc:
                    logger.debug("Monitor: failed to decode iface msg for %s: %s", expected_rid, exc)
                    return
                size = len(data)
                if bw_tracker is not None:
                    topic = f"iface/{iface.sender}-{iface.receiver}"
                    bw_tracker.add_uplink(topic, size)
                if lat_tracker is not None:
                    now_mono = time.perf_counter()
                    now_wall = time.time()
                    snd_mono = getattr(iface, "sent_mono_time", None)
                    meta = {
                        "sender": iface.sender,
                        "receiver": iface.receiver,
                        "key": str(getattr(iface, "key", "")),
                        "iteration": int(getattr(iface, "iteration", 0)),
                    }
                    if getattr(iface, "sent_wall_time", None) is not None:
                        meta["send_ts_wall"] = float(iface.sent_wall_time)
                    if snd_mono is not None:
                        try:
                            snd_mono_f = float(snd_mono)
                            meta["send_ts_mono"] = snd_mono_f
                            meta["e2e_send_to_ingest"] = now_mono - snd_mono_f
                        except Exception:
                            pass
                    try:
                        lat_tracker.record_ingest(
                            "InterfaceMessage",
                            float(getattr(iface, "stamp", 0.0)),
                            iface.receiver,
                            ingest_wall=now_wall,
                            ingest_mono=now_mono,
                            metadata=meta,
                        )
                    except Exception:
                        pass
            return _cb

        subscriptions = []
        ack_subscriptions = []
        for rid in robot_ids:
            topic = f"{namespaced_prefix}/{rid}"
            try:
                sub = node.create_subscription(UInt8MultiArray, topic, _make_cb(rid), qos)
                subscriptions.append(sub)
            except Exception as exc:
                logger.debug("Monitor: failed to subscribe to %s: %s", topic, exc)
        # ACK subscribers for 'use' events
        def _ack_cb_factory(expected_rid: str):
            def _ack_cb(msg):
                try:
                    data = bytes(getattr(msg, "data", []) or [])
                    import json as _json
                    doc = _json.loads(data.decode("utf-8"))
                except Exception:
                    return
                send_mono = float(doc.get("send_ts_mono", 0.0) or 0.0)
                use_mono = float(doc.get("use_ts_mono", 0.0) or 0.0)
                e2e_use = None
                if send_mono > 0.0 and use_mono > 0.0:
                    e2e_use = use_mono - send_mono
                if lat_tracker is not None:
                    try:
                        meta = {
                            "sender": str(doc.get("sender", "")),
                            "receiver": str(doc.get("receiver", "")),
                            "key": str(doc.get("key", "")),
                            "iteration": int(doc.get("iteration", 0)),
                            "send_ts_mono": send_mono if send_mono > 0.0 else None,
                        }
                        if e2e_use is not None:
                            meta["e2e_send_to_use"] = e2e_use
                        lat_tracker.record_ingest(
                            "InterfaceACK",
                            0.0,
                            expected_rid,
                            ingest_wall=float(doc.get("use_ts_wall", 0.0) or time.time()),
                            ingest_mono=use_mono or time.perf_counter(),
                            metadata=meta,
                        )
                    except Exception:
                        pass
            return _ack_cb

        for rid in robot_ids:
            topic = f"{namespaced_prefix}/ack/{rid}"
            try:
                sub = node.create_subscription(UInt8MultiArray, topic, _ack_cb_factory(rid), qos)
                ack_subscriptions.append(sub)
            except Exception as exc:
                logger.debug("Monitor: failed to subscribe to ACK %s: %s", topic, exc)
    except Exception as exc:
        logger.debug("Monitor initialisation failed; KPIs will be empty: %s", exc)
        node = None
        rclpy = None  # type: ignore
        subscriptions = []

    ctx = mp.get_context("spawn") if hasattr(mp, "get_context") else mp
    procs: List[mp.Process] = []
    pids: List[int] = []

    for rid in robot_ids:
        bundle = bundles.get(rid)
        if bundle is None:
            bundle = RobotMeasurementBundle()
        p = ctx.Process(
            target=_agent_loop,
            args=(rid, bundle, robot_ids),
            kwargs=dict(
                robot_map=robot_map,
                init_lookup=init_lookup,
                robust_kind=robust_kind,
                robust_k=robust_k,
                batch_max_iters=batch_max_iters,
                topic_prefix=namespaced_prefix,
                qos_profile=qos_profile,
                out_dir=trajectories_dir,
                metrics_dir=metrics_dir,
                max_rounds=ddf_rounds,
                convergence_tol=convergence_tol,
                rotation_tol=rotation_tol,
                relaxation_alpha=relaxation_alpha,
                ingest_time_scale=ingest_time_scale,
                ingest_max_sleep=ingest_max_sleep,
                interface_time_scale=interface_time_scale,
                interface_max_sleep=interface_max_sleep,
                agent_factor_ros2=agent_factor_ros2,
                agent_factor_prefix=agent_factor_prefix,
                agent_factor_reliability=agent_factor_reliability,
                agent_factor_durability=agent_factor_durability,
                agent_factor_depth=agent_factor_depth,
                agent_factor_idle_timeout=agent_factor_idle_timeout,
                agent_solve_every_n=agent_solve_every_n,
                agent_solve_period_s=agent_solve_period_s,
            ),
            daemon=True,
        )
        p.start()
        procs.append(p)
        pids.append(p.pid)

    # Pump monitor until all agents complete
    try:
        while any(p.is_alive() for p in procs):
            if node is not None and rclpy is not None:
                try:
                    # Lower spin timeout for tighter e2e send->ingest timing
                    rclpy.spin_once(node, timeout_sec=0.01)
                except Exception:
                    time.sleep(0.01)
            else:
                time.sleep(0.05)
        # Drain any final callbacks
        if node is not None and rclpy is not None:
            try:
                rclpy.spin_once(node, timeout_sec=0.0)
            except Exception:
                pass
    finally:
        # Ensure children joined
        for p in procs:
            try:
                p.join(timeout=0.1)
            except Exception:
                pass
        # Tear down monitor
        if node is not None:
            try:
                for sub in subscriptions:
                    try:
                        node.destroy_subscription(sub)
                    except Exception:
                        pass
                for sub in ack_subscriptions:
                    try:
                        node.destroy_subscription(sub)
                    except Exception:
                        pass
                node.destroy_node()
            except Exception:
                pass
        try:
            if 'rclpy' in globals() and rclpy is not None:
                rclpy.shutdown()
        except Exception:
            pass

    # Merge per-agent KPI logs into a single timeline for parity with centralised backend
    merged_events = []
    merged_path = os.path.join(metrics_dir, "kpi_events.jsonl")
    try:
        import json as _json

        for rid in robot_ids:
            agent_log = os.path.join(metrics_dir, f"kpi_events_{rid}.jsonl")
            if not os.path.exists(agent_log):
                continue
            try:
                with open(agent_log, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            doc = _json.loads(line)
                            merged_events.append(doc)
                        except Exception:
                            continue
            except Exception:
                continue
        if merged_events:
            merged_events.sort(key=lambda ev: ev.get("ts", 0.0))
            with open(merged_path, "w", encoding="utf-8") as fh:
                for ev in merged_events:
                    fh.write(_json.dumps(ev, sort_keys=True) + "\n")
        else:
            with open(merged_path, "w", encoding="utf-8") as fh:
                fh.write("")
    except Exception:
        try:
            with open(merged_path, "w", encoding="utf-8") as fh:
                fh.write("")
        except Exception:
            pass

    return pids, bw_tracker, lat_tracker, namespaced_prefix
