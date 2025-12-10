#!/usr/bin/env python3
"""Orchestrate ROS 2 KPI runs for centralised and decentralised backends.

This script launches:
- Centralised ROS 2 ingest: factor publisher + central solver (ACK/E2E metrics)
- Decentralised ROS 2 multi-process with paced ingest/interface (realistic DDF KPIs)

Defaults mirror the recommended settings in README and prior guidance:
- iSAM2, robust=cauchy,k=1.0, relin-th=0.05, relin-skip=5, wxyz, include outliers
- QoS: reliable + volatile with depth=10
- Publisher pacing: time-scale=3e9, max-sleep=0.3, batch-size=1
- DDF pacing: ingest/interface time-scale=1e9, max-sleep=0.1, rounds=8

Usage examples:
  python tools/orchestrate_kpi.py \
      --jrl datasets/sample.jrl \
      --out-base output_kpi_runs \
      --eval-gt

Using a config file (JSON or TOML) with all flags:
  python tools/orchestrate_kpi.py --config tools/orchestrate_kpi.example.json

CLI flags always override values from the config file.

You must have a ROS 2 environment sourced and rclpy available.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shlex
import signal
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from subprocess import Popen
from typing import Any, Dict, List, Tuple

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - optional TOML support
    tomllib = None  # type: ignore[assignment]

from cosmo_slam_centralised.kpi_derive import derive_kpis_for_run, write_json as write_kpi_json, _read_json


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _cmd_str(cmd: list[str]) -> str:
    return shlex.join(cmd)


def _spawn(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> Popen:
    print(f"[orchestrate] spawn: {_cmd_str(cmd)}")
    return Popen(cmd, cwd=str(cwd) if cwd else None, env=env)


def _wait(proc: Popen, name: str) -> int:
    code = proc.wait()
    print(f"[orchestrate] {name} exited with code {code}")
    return int(code or 0)


def _terminate(proc: Popen, name: str, timeout: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    try:
        print(f"[orchestrate] terminating {name} (pid={proc.pid})")
        proc.terminate()
        t0 = time.time()
        while proc.poll() is None and time.time() - t0 < timeout:
            time.sleep(0.1)
        if proc.poll() is None:
            print(f"[orchestrate] killing {name} (pid={proc.pid})")
            proc.kill()
    except Exception:
        pass


def _safe_float(val: Any) -> float | None:
    try:
        if val is None:
            return None
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None


_T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _t_critical_95(df: float) -> float:
    if df is None or df <= 0:
        return 1.96
    df_int = int(round(df))
    if df_int in _T_CRIT_95:
        return _T_CRIT_95[df_int]
    if df_int < 30:
        lower = max((k for k in _T_CRIT_95 if k < df_int), default=1)
        upper = min((k for k in _T_CRIT_95 if k > df_int), default=30)
        if lower == upper:
            return _T_CRIT_95.get(lower, 1.96)
        low_v = _T_CRIT_95.get(lower, 1.96)
        up_v = _T_CRIT_95.get(upper, 1.96)
        ratio = (df_int - lower) / (upper - lower)
        return low_v + ratio * (up_v - low_v)
    return 1.96


def _summary_stats(values: List[float]) -> Dict[str, float]:
    cleaned = [float(v) for v in values if _safe_float(v) is not None]
    if not cleaned:
        return {}
    n = len(cleaned)
    mean_val = mean(cleaned)
    stdev_val = stdev(cleaned) if n > 1 else 0.0
    halfwidth = 0.0
    if n > 1 and stdev_val > 0.0:
        halfwidth = _t_critical_95(n - 1) * stdev_val / math.sqrt(n)
    return {
        "count": n,
        "mean": mean_val,
        "stdev": stdev_val,
        "ci95_halfwidth": halfwidth,
    }


def _welch_t_test(sample_a: List[float], sample_b: List[float]) -> Dict[str, float | None]:
    a = [float(v) for v in sample_a if _safe_float(v) is not None]
    b = [float(v) for v in sample_b if _safe_float(v) is not None]
    result: Dict[str, float | None] = {"t_stat": None, "df": None, "p_value": None}
    if not a or not b:
        return result
    n1, n2 = len(a), len(b)
    m1, m2 = mean(a), mean(b)
    s1 = stdev(a) if n1 > 1 else 0.0
    s2 = stdev(b) if n2 > 1 else 0.0
    var1 = s1 ** 2
    var2 = s2 ** 2
    denom = math.sqrt((var1 / n1) + (var2 / n2))
    if denom == 0.0:
        return result
    t_stat = (m1 - m2) / denom
    result["t_stat"] = t_stat
    df = None
    if n1 > 1 and n2 > 1 and (var1 > 0.0 or var2 > 0.0):
        numerator = (var1 / n1 + var2 / n2) ** 2
        denominator = 0.0
        if var1 > 0.0:
            denominator += ((var1 / n1) ** 2) / (n1 - 1)
        if var2 > 0.0:
            denominator += ((var2 / n2) ** 2) / (n2 - 1)
        if denominator > 0.0:
            df = numerator / denominator
    result["df"] = df
    if df is not None:
        try:
            from scipy.stats import t as t_dist  # type: ignore

            result["p_value"] = float(2.0 * t_dist.sf(abs(t_stat), df))
        except Exception:
            result["p_value"] = None
    return result


def _collect_run_metrics(run_dir: Path) -> Dict[str, float | None]:
    metrics: Dict[str, float | None] = {}
    lat_path = run_dir / "kpi_metrics" / "latency_metrics.json"
    lat_json = _read_json(str(lat_path)) if lat_path.exists() else {}
    summary = lat_json.get("summary", {}) if isinstance(lat_json, dict) else {}
    for bucket_name, metric_key in (
        ("ingest_to_broadcast", "latency_ingest_to_broadcast_mean"),
        ("e2e_send_to_ingest", "latency_e2e_send_to_ingest_mean"),
    ):
        bucket = summary.get(bucket_name, {}) if isinstance(summary, dict) else {}
        metrics[metric_key] = _safe_float(bucket.get("mean"))

    bw_path = run_dir / "kpi_metrics" / "bandwidth_stats.json"
    bw_json = _read_json(str(bw_path)) if bw_path.exists() else {}
    if isinstance(bw_json, dict):
        for direction, metric_key in (("uplink", "bandwidth_uplink_bytes"), ("downlink", "bandwidth_downlink_bytes")):
            bucket = bw_json.get(direction, {})
            total = 0.0
            if isinstance(bucket, dict):
                for stats in bucket.values():
                    if isinstance(stats, dict):
                        total += float(stats.get("bytes", 0.0) or 0.0)
            metrics[metric_key] = total

    # Derived KPIs (loop-closure correction, convergence time, etc.)
    try:
        derived = derive_kpis_for_run(str(run_dir))
        write_kpi_json(str(run_dir / "kpi_metrics" / "derived_kpis.json"), derived)
    except Exception:
        derived = {}
    if isinstance(derived, dict):
        metrics["time_to_global_convergence_s"] = _safe_float(derived.get("time_to_global_convergence_s"))
        lcc = derived.get("loop_closure_correction")
        if isinstance(lcc, dict):
            metrics["loop_closure_correction_mean"] = _safe_float(lcc.get("mean"))
        delivery = derived.get("delivery_rate")
        if isinstance(delivery, dict):
            metrics["delivery_rate_mean"] = _safe_float(delivery.get("mean"))
    return metrics


def _write_replication_stats(out_base: Path, records: List[Dict[str, Any]]) -> None:
    central_series: Dict[str, List[float]] = {}
    decentral_series: Dict[str, List[float]] = {}
    rows: List[Tuple[int, str, str, float]] = []
    for rec in records:
        rep_idx = rec.get("rep")
        for system_key, series, prefix in (("central", central_series, "centralised"), ("decentral", decentral_series, "decentralised")):
            metrics = rec.get(system_key, {}) or {}
            for metric, value in metrics.items():
                if _safe_float(value) is None:
                    continue
                series.setdefault(metric, []).append(float(value))
                rows.append((rep_idx, prefix, metric, float(value)))

    stats_payload = {"centralised": {}, "decentralised": {}, "comparisons": {}}
    for metric, values in central_series.items():
        stats_payload["centralised"][metric] = _summary_stats(values)
    for metric, values in decentral_series.items():
        stats_payload["decentralised"][metric] = _summary_stats(values)

    for metric in sorted(set(central_series.keys()) & set(decentral_series.keys())):
        stats_payload["comparisons"][metric] = _welch_t_test(central_series[metric], decentral_series[metric])

    stats_path = out_base / "replication_stats.json"
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats_payload, fh, indent=2)
    csv_path = out_base / "replicate_metrics.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("rep,system,metric,value\n")
        for rep, system, metric, value in rows:
            fh.write(f"{rep},{system},{metric},{value}\n")


def _strip_json_comments(text: str) -> str:
    """Remove // line and /* ... */ block comments from JSON text.
    Preserves content within strings and escaped characters.
    """
    out: list[str] = []
    i, n = 0, len(text)
    in_str = False
    in_sl_comment = False
    in_ml_comment = False
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ''
        if in_sl_comment:
            if ch == '\n':
                in_sl_comment = False
                out.append(ch)
            i += 1
            continue
        if in_ml_comment:
            if ch == '*' and nxt == '/':
                in_ml_comment = False
                i += 2
            else:
                i += 1
            continue
        if in_str:
            if ch == '\\':
                # keep escape and next char
                out.append(ch)
                if i + 1 < n:
                    out.append(text[i + 1])
                    i += 2
                else:
                    i += 1
                continue
            if ch == '"':
                in_str = False
            out.append(ch)
            i += 1
            continue
        # not in string or comment
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == '/' and nxt == '/':
            in_sl_comment = True
            i += 2
            continue
        if ch == '/' and nxt == '*':
            in_ml_comment = True
            i += 2
            continue
        out.append(ch)
        i += 1
    return ''.join(out)


def _load_config_file(path: Path) -> Dict[str, Any]:
    """Load configuration from a JSON or TOML file.

    Supports flat keys matching CLI flag dest names, or nested groups like:
      - global, qos, solver, centralised, decentralised, impair
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            txt = f.read()
        try:
            raw = json.loads(txt)
        except json.JSONDecodeError:
            # Allow JSON with comments (JSONC)
            raw = json.loads(_strip_json_comments(txt))
    elif path.suffix.lower() == ".toml":
        if tomllib is None:
            raise RuntimeError("TOML config requires Python 3.11+ (tomllib). Use JSON or upgrade.")
        with path.open("rb") as f:  # tomllib expects bytes
            raw = tomllib.load(f)  # type: ignore[union-attr]
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}. Use .json or .toml")

    # Normalize possibly nested structure to argparse dest names
    if not isinstance(raw, dict):
        raise ValueError("Config root must be an object/dict")

    flat: Dict[str, Any] = {}
    # Accept flat passthrough keys (direct CLI dest names)
    for k, v in raw.items():
        if not isinstance(v, dict):
            flat[k.replace("-", "_")] = v

    # Accept common nested groups
    def put(src: Dict[str, Any], mapping: Dict[str, str]) -> None:
        for sk, dk in mapping.items():
            if sk in src:
                flat[dk] = src[sk]

    if isinstance(raw.get("global"), dict):
        put(raw["global"], {
            "jrl": "jrl",
            "out_base": "out_base",
            "eval_gt": "eval_gt",
            "ros2_domain_id": "ros2_domain_id",
            "dry_run": "dry_run",
            "skip_centralised": "skip_centralised",
            "skip_decentralised": "skip_decentralised",
            "replicas": "replicas",
            "seed_base": "seed_base",
            "use_sim_time": "use_sim_time",
            "resource_interval": "resource_interval",
        })
    if isinstance(raw.get("qos"), dict):
        put(raw["qos"], {
            "reliability": "qos_reliability",
            "durability": "qos_durability",
            "depth": "qos_depth",
        })
    if isinstance(raw.get("solver"), dict):
        put(raw["solver"], {
            "relin_th": "relin_th",
            "relin_skip": "relin_skip",
            "robust": "robust",
            "robust_k": "robust_k",
            "quat_order": "quat_order",
        })
    if isinstance(raw.get("centralised"), dict):
        put(raw["centralised"], {
            "factor_topic": "factor_topic",
            "publisher_batch_size": "publisher_batch_size",
            "publisher_time_scale": "publisher_time_scale",
            "publisher_max_sleep": "publisher_max_sleep",
            "publisher_idle_gap": "publisher_idle_gap",
            "publisher_wait_timeout": "publisher_wait_timeout",
            "skip": "skip_centralised",
        })
    if isinstance(raw.get("decentralised"), dict):
        put(raw["decentralised"], {
            "iface_topic_prefix": "iface_topic_prefix",
            "ddf_ingest_time_scale": "ddf_ingest_time_scale",
            "ddf_ingest_max_sleep": "ddf_ingest_max_sleep",
            "ddf_interface_time_scale": "ddf_interface_time_scale",
            "ddf_interface_max_sleep": "ddf_interface_max_sleep",
            "ddf_rounds": "ddf_rounds",
            "skip": "skip_decentralised",
        })
    if isinstance(raw.get("impair"), dict):
        put(raw["impair"], {
            "json": "impair_json",
            "file": "impair_file",
        })

    # Normalize impairment json object to a compact JSON string if provided as dict/list
    if "impair_json" in flat and isinstance(flat["impair_json"], (dict, list)):
        flat["impair_json"] = json.dumps(flat["impair_json"], separators=(",", ":"))

    return flat


def parse_args() -> argparse.Namespace:
    # Stage 1: parse only --config to load defaults from file
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--config", help="Path to JSON/TOML config for all flags")
    cfg_ns, _ = base.parse_known_args()

    ap = argparse.ArgumentParser(
        description="Orchestrate centralised and decentralised ROS 2 KPI runs.",
        parents=[base],
    )
    # Not required if provided via --config
    ap.add_argument("--jrl", required=False, help="Path to COSMO-bench .jrl dataset")
    ap.add_argument("--out-base", default="output_kpi_runs", help="Base directory for outputs")
    ap.add_argument("--skip-centralised", action="store_true", help="Skip centralised ROS 2 run")
    ap.add_argument("--skip-decentralised", action="store_true", help="Skip decentralised ROS 2 run")
    ap.add_argument("--eval-gt", action="store_true", help="Enable ground truth alignment and GT overlays")
    # Impairments (JSON string or file path). If set, applied to both central publisher and DDF peer bus.
    ap.add_argument("--impair-json", default=None, help="JSON string describing reproducible network impairments")
    ap.add_argument("--impair-file", default=None, help="Path to JSON file describing impairments")
    ap.add_argument("--ros2-domain-id", type=int, default=None, help="Optional ROS_DOMAIN_ID to set for all children")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    ap.add_argument("--replicas", type=int, default=1, help="Number of replicated runs to execute")
    ap.add_argument("--seed-base", type=int, default=None, help="Base seed used for replicated runs (seed+i)")
    ap.add_argument("--use-sim-time", action="store_true", help="Enable ROS 2 simulated time for all nodes")
    ap.add_argument("--resource-interval", type=float, default=0.5, help="Resource monitor sampling interval (seconds)")

    # QoS
    ap.add_argument("--qos-reliability", choices=["reliable", "best_effort"], default="reliable")
    ap.add_argument("--qos-durability", choices=["volatile", "transient_local"], default="volatile")
    ap.add_argument("--qos-depth", type=int, default=10)

    # Centralised topics + pacing (publisher)
    ap.add_argument("--factor-topic", default="/cosmo/factor_batch", help="Topic prefix for factor batches")
    ap.add_argument("--publisher-batch-size", type=int, default=1)
    ap.add_argument("--publisher-time-scale", type=float, default=3e9)
    ap.add_argument("--publisher-max-sleep", type=float, default=0.3)
    ap.add_argument("--publisher-idle-gap", type=float, default=2.0)
    ap.add_argument("--publisher-wait-timeout", type=float, default=60.0)

    # Decentralised topics + pacing
    ap.add_argument("--iface-topic-prefix", default="/cosmo/iface")
    ap.add_argument("--ddf-ingest-time-scale", type=float, default=1e9)
    ap.add_argument("--ddf-ingest-max-sleep", type=float, default=0.1)
    ap.add_argument("--ddf-interface-time-scale", type=float, default=1e9)
    ap.add_argument("--ddf-interface-max-sleep", type=float, default=0.1)
    ap.add_argument("--ddf-rounds", type=int, default=8)

    # Solver knobs (kept consistent across runs)
    ap.add_argument("--relin-th", type=float, default=0.05)
    ap.add_argument("--relin-skip", type=int, default=5)
    ap.add_argument("--robust", choices=["none", "huber", "cauchy"], default="cauchy")
    ap.add_argument("--robust-k", type=float, default=1.0)
    ap.add_argument("--quat-order", choices=["wxyz", "xyzw"], default="wxyz")
    # If config file is provided, load and set as defaults so CLI overrides
    if getattr(cfg_ns, "config", None):
        try:
            cfg_path = Path(cfg_ns.config).expanduser().resolve()
            cfg_defaults = _load_config_file(cfg_path)
            # set_defaults accepts any subset of known args
            ap.set_defaults(**cfg_defaults)
            print(f"[orchestrate] loaded config defaults from {cfg_path}")
        except Exception as e:  # pragma: no cover - defensive
            print(f"[orchestrate] Failed to load config: {e}", file=sys.stderr)
            sys.exit(2)

    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if not args.jrl and not args.dry_run:
        print("[orchestrate] ERROR: --jrl is required (via CLI or config)")
        return 2
    jrl = Path(args.jrl or "MISSING.jrl").resolve()
    if not args.dry_run and not jrl.exists():
        print(f"[orchestrate] ERROR: JRL file not found: {jrl}")
        return 2

    out_base = Path(args.out_base).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    if args.ros2_domain_id is not None:
        base_env["ROS_DOMAIN_ID"] = str(int(args.ros2_domain_id))
        print(f"[orchestrate] ROS_DOMAIN_ID={base_env['ROS_DOMAIN_ID']}")
    base_env.setdefault("PYTHONUNBUFFERED", "1")

    impairment_file_path: Path | None = None
    base_impair_spec: Dict[str, Any] | None = None
    if args.impair_file:
        impairment_file_path = Path(args.impair_file).resolve()
        if not args.dry_run and not impairment_file_path.exists():
            print(f"[orchestrate] ERROR: impairment file not found: {impairment_file_path}", file=sys.stderr)
            return 2
    elif args.impair_json:
        try:
            base_impair_spec = json.loads(args.impair_json)
        except json.JSONDecodeError:
            try:
                base_impair_spec = json.loads(_strip_json_comments(args.impair_json))
            except Exception as exc:
                print(f"[orchestrate] ERROR: invalid --impair-json: {exc}", file=sys.stderr)
                return 2

    replicas = max(1, int(args.replicas or 1))
    seed_base = args.seed_base

    # Compose common solver opts
    common_solver = [
        "--solver", "isam2",
        "--include-potential-outliers",
        "--relin-th", str(args.relin_th),
        "--relin-skip", str(args.relin_skip),
        "--quat-order", args.quat_order,
        "--robust", args.robust,
    ]
    if args.robust != "none" and args.robust_k is not None:
        common_solver += ["--robust-k", str(args.robust_k)]
    if args.eval_gt:
        common_solver.append("--eval-gt")
    if args.use_sim_time:
        common_solver.append("--use-sim-time")

    qos_opts = [
        "--qos-reliability", args.qos_reliability,
        "--qos-durability", args.qos_durability,
        "--qos-depth", str(int(args.qos_depth)),
    ]
    central_qos_opts = [
        "--central-ros2-reliability", args.qos_reliability,
        "--central-ros2-durability", args.qos_durability,
        "--central-ros2-depth", str(int(args.qos_depth)),
    ]
    ddf_qos_opts = [
        "--ddf-ros2-reliability", args.qos_reliability,
        "--ddf-ros2-durability", args.qos_durability,
        "--ddf-ros2-depth", str(int(args.qos_depth)),
    ]

    py = sys.executable
    replicate_records: List[Dict[str, Any]] = []
    overall_rc = 0
    children: list[tuple[str, Popen]] = []

    def _on_sigint(signum, frame):  # noqa: ARG001
        print("[orchestrate] SIGINT received; terminating children...")
        for name, proc in children:
            _terminate(proc, name)
        sys.exit(130)

    signal.signal(signal.SIGINT, _on_sigint)

    for rep_idx in range(replicas):
        rep_seed = seed_base + rep_idx if seed_base is not None else random.randint(0, 2**31 - 1)
        rep_label = f"[rep {rep_idx + 1}/{replicas}] " if replicas > 1 else ""
        rep_base = out_base if replicas == 1 else (out_base / f"rep_{rep_idx:03d}")
        out_central = rep_base / "central_ros2"
        out_ddf = rep_base / "ddf_ros2_mp_paced"
        _ensure_dir(out_central)
        _ensure_dir(out_ddf)

        env = base_env.copy()
        env["PYTHONHASHSEED"] = str(rep_seed)
        env["COSMO_RUN_SEED"] = str(rep_seed)
        env.setdefault("COSMO_RESOURCE_SEED", str(rep_seed))
        env.setdefault("COSMO_RANDOM_SEED", str(rep_seed))
        env.setdefault("NUMPY_SEED", str(rep_seed))
        if args.use_sim_time:
            env["COSMO_USE_SIM_TIME"] = "1"
        if args.resource_interval:
            try:
                env["COSMO_RESOURCE_INTERVAL"] = str(float(args.resource_interval))
            except Exception:
                pass

        def _env_for(out_dir: Path) -> Dict[str, str]:
            proc_env = env.copy()
            proc_env["COSMO_IMPAIR_OUT_DIR"] = str(out_dir / "kpi_metrics")
            if impairment_file_path is not None:
                proc_env["COSMO_IMPAIR_FILE"] = str(impairment_file_path)
                proc_env.pop("COSMO_IMPAIR", None)
            elif base_impair_spec is not None:
                spec = dict(base_impair_spec)
                spec["seed"] = int(rep_seed)
                proc_env["COSMO_IMPAIR"] = json.dumps(spec, separators=(",", ":"))
                proc_env.pop("COSMO_IMPAIR_FILE", None)
            elif args.impair_json:
                proc_env["COSMO_IMPAIR"] = args.impair_json
            return proc_env

        central_solver_cmd = [
            py, str(root / "main.py"),
            "--jrl", str(jrl),
            "--export-path", str(out_central),
            "--backend", "centralised",
            "--central-transport", "ros2",
            "--central-ros2-topic", args.factor_topic,
            "--central-ros2-idle-timeout", "60.0",
        ] + common_solver + central_qos_opts

        publisher_cmd = [
            py, str(root / "tools" / "ros2_factor_publisher.py"),
            "--jrl", str(jrl),
            "--topic", args.factor_topic,
            "--batch-size", str(int(args.publisher_batch_size)),
            "--time-scale", str(float(args.publisher_time_scale)),
            "--max-sleep", str(float(args.publisher_max_sleep)),
            "--idle-gap", str(float(args.publisher_idle_gap)),
            "--loop", "1",
            "--include-potential-outliers",
            "--wait-subscriber-timeout", str(float(args.publisher_wait_timeout)),
            "--quat-order", args.quat_order,
        ] + qos_opts

        ddf_cmd = [
            py, str(root / "main.py"),
            "--jrl", str(jrl),
            "--export-path", str(out_ddf),
            "--backend", "decentralised",
            "--ddf-transport", "ros2",
            "--ddf-multiprocess",
            "--ddf-ros2-topic-prefix", args.iface_topic_prefix,
            "--ddf-ingest-time-scale", str(float(args.ddf_ingest_time_scale)),
            "--ddf-ingest-max-sleep", str(float(args.ddf_ingest_max_sleep)),
            "--ddf-interface-time-scale", str(float(args.ddf_interface_time_scale)),
            "--ddf-interface-max-sleep", str(float(args.ddf_interface_max_sleep)),
            "--ddf-rounds", str(int(args.ddf_rounds)),
        ] + common_solver + ddf_qos_opts

        if args.dry_run:
            if not args.skip_centralised:
                print(f"[orchestrate]{rep_label}CENTRAL SOLVER:")
                print("  ", _cmd_str(central_solver_cmd))
                print(f"[orchestrate]{rep_label}FACTOR PUBLISHER:")
                print("  ", _cmd_str(publisher_cmd))
            if not args.skip_decentralised:
                print(f"[orchestrate]{rep_label}DECENTRALISED DDF:")
                print("  ", _cmd_str(ddf_cmd))
            replicate_records.append({"rep": rep_idx, "seed": rep_seed, "central": {}, "decentral": {}})
            continue

        central_metrics: Dict[str, float | None] = {}
        decentral_metrics: Dict[str, float | None] = {}

        if not args.skip_centralised:
            print(f"[orchestrate]{rep_label}=== Centralised ROS 2 run ===")
            env_solver = _env_for(out_central)
            solver = _spawn(central_solver_cmd, cwd=root, env=env_solver)
            children.append(("central_solver", solver))
            time.sleep(2.0)
            env_pub = _env_for(out_central)
            pub = _spawn(publisher_cmd, cwd=root, env=env_pub)
            children.append(("factor_publisher", pub))

            rc_pub = _wait(pub, "factor_publisher")
            rc_solver = _wait(solver, "central_solver")
            children.clear()
            overall_rc |= (rc_pub != 0) or (rc_solver != 0)
            if rc_pub == 0 and rc_solver == 0:
                central_metrics = _collect_run_metrics(out_central)
                print(f"[orchestrate]{rep_label}Centralised outputs: {out_central}")
                print(f"  bandwidth: {out_central}/kpi_metrics/bandwidth_stats.json")
                print(f"  latency:   {out_central}/kpi_metrics/latency_metrics.json")
                print(f"  resources: {out_central}/kpi_metrics/resource_profile.json")
            else:
                print(f"[orchestrate]{rep_label}Centralised run failed (solver={rc_solver}, publisher={rc_pub})")

        if not args.skip_decentralised:
            print(f"[orchestrate]{rep_label}=== Decentralised ROS 2 multi-process (paced) ===")
            env_ddf = _env_for(out_ddf)
            ddf = _spawn(ddf_cmd, cwd=root, env=env_ddf)
            children.append(("ddf_mp", ddf))
            rc_ddf = _wait(ddf, "ddf_mp")
            children.clear()
            overall_rc |= (rc_ddf != 0)
            if rc_ddf == 0:
                decentral_metrics = _collect_run_metrics(out_ddf)
                print(f"[orchestrate]{rep_label}Decentralised outputs: {out_ddf}")
                print(f"  bandwidth: {out_ddf}/kpi_metrics/bandwidth_stats.json")
                print(f"  latency:   {out_ddf}/kpi_metrics/latency_metrics.json")
                print(f"  resources: {out_ddf}/kpi_metrics/resource_profile.json")
            else:
                print(f"[orchestrate]{rep_label}Decentralised run failed (rc={rc_ddf})")

        replicate_records.append({
            "rep": rep_idx,
            "seed": rep_seed,
            "central": central_metrics,
            "decentral": decentral_metrics,
        })

    if replicas > 1 and not args.dry_run:
        _write_replication_stats(out_base, replicate_records)
        print(f"[orchestrate] Replication stats written to {out_base}/replication_stats.json")

    if overall_rc:
        print("[orchestrate] One or more runs failed")
        return 1
    print("[orchestrate] All requested runs completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
