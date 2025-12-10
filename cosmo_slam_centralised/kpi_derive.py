"""Derived KPI computations for Cosmo SLAM outputs.

This module post-processes existing KPI exports to compute:
- Loop-closure correction time (inter-robot between factor ingest → broadcast)
- Time-to-global convergence (approximation via timeline end markers)

It intentionally depends only on JSON files exported by the CLI runs, so it can
be used offline without changing the solver pipelines.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Lazy imports guarded in helpers to avoid hard dependency where not available


# -------------------------------
# Helpers
# -------------------------------

def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _stats(values: Iterable[float]) -> Dict[str, Optional[float]]:
    vals = sorted(float(v) for v in values if isinstance(v, (int, float)))
    if not vals:
        return {"count": 0}
    import statistics

    def pct(p: float) -> Optional[float]:
        if not vals:
            return None
        if p <= 0:
            return vals[0]
        if p >= 100:
            return vals[-1]
        rank = (p / 100.0) * (len(vals) - 1)
        lower = int(rank)
        upper = min(lower + 1, len(vals) - 1)
        weight = rank - lower
        return vals[lower] * (1 - weight) + vals[upper] * weight

    out: Dict[str, Optional[float]] = {
        "count": len(vals),
        "min": vals[0],
        "max": vals[-1],
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "p90": pct(90.0),
        "p95": pct(95.0),
        "p99": pct(99.0),
    }
    if len(vals) > 1:
        out["stdev"] = statistics.pstdev(vals)
    return out


def _infer_robot(key: str) -> str:
    """Infer robot ID from a key label (prefix letters before the first digit).

    Mirrors logic in cosmo_slam_centralised.graph.default_robot_infer.
    """
    if key is None:
        return "global"
    s = str(key)
    prefix = []
    for ch in s:
        if ch.isalpha():
            prefix.append(ch)
        else:
            break
    return "".join(prefix) or "global"


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# -------------------------------
# Loop-closure correction time
# -------------------------------

def extract_loop_closure_correction_times(latency_json: Dict[str, Any], robot_map: Optional[Dict[str, str]] = None) -> List[float]:
    """Return ingest→broadcast latencies for inter-robot between factors.

    For centralised iSAM2 runs, LatencyTracker records per-event fields:
    - factor_type: 'BetweenFactorPose3'
    - metadata: key1, key2
    - latency_ingest_to_broadcast (seconds)

    We infer robots from key labels and keep only inter-robot pairs.
    """
    events = latency_json.get("events") if isinstance(latency_json, dict) else None
    if not isinstance(events, list):
        return []
    out: List[float] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("factor_type") != "BetweenFactorPose3":
            continue
        k1 = ev.get("key1")
        k2 = ev.get("key2")
        if not (isinstance(k1, str) or isinstance(k1, int)):
            continue
        if not (isinstance(k2, str) or isinstance(k2, int)):
            continue
        r1 = str(robot_map.get(str(k1))) if robot_map and str(k1) in robot_map else _infer_robot(str(k1))
        r2 = str(robot_map.get(str(k2))) if robot_map and str(k2) in robot_map else _infer_robot(str(k2))
        if r1 == r2:
            # Intra-robot between edges are often odometry; skip.
            continue
        d = _safe_float(ev.get("latency_ingest_to_broadcast"))
        if d is not None and d >= 0.0:
            out.append(d)
    return out


def summarise_loop_closure_correction(latency_json: Dict[str, Any], robot_map: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    vals = extract_loop_closure_correction_times(latency_json, robot_map=robot_map)
    return {"loop_closure_correction": _stats(vals)}


def _stabilisation_time_after(
    events: List[Dict[str, Any]],
    *,
    start_ts: float,
    event_name: str,
    delta_field: str = "max_translation_delta",
    solver_allow: Optional[Iterable[str]] = None,
    epsilon: float = 1e-3,
    required: int = 5,
) -> Optional[float]:
    """Return the first timestamp where a run of ``required`` events stays <= ``epsilon``.

    Searches KPI events with name ``event_name`` occurring at or after ``start_ts`` and
    returns the timestamp of the last event within the first qualifying consecutive window
    whose ``delta_field`` does not exceed ``epsilon``. Returns ``None`` when no such
    window is found.
    """
    if not events:
        return None
    filtered: List[Tuple[float, float]] = []
    grouped: Dict[int, Tuple[float, float]] = {}
    has_iteration = False
    allow = set(solver_allow) if solver_allow else None
    for ev in events:
        if ev.get("event") != event_name:
            continue
        if allow is not None and ev.get("solver") not in allow:
            continue
        ts = _safe_float(ev.get("ts"))
        delta = _safe_float(ev.get(delta_field))
        if ts is None or delta is None:
            continue
        if ts < start_ts:
            continue
        if delta < 0:
            continue
        it = ev.get("iteration")
        if it is not None:
            try:
                idx = int(it)
                has_iteration = True
                prev = grouped.get(idx)
                # Keep the worst-case delta per iteration to be conservative
                if prev is None or delta > prev[1]:
                    grouped[idx] = (ts, delta)
            except Exception:
                filtered.append((ts, delta))
        else:
            filtered.append((ts, delta))
    if has_iteration:
        filtered.extend(grouped.values())
    if len(filtered) < required:
        return None
    filtered.sort(key=lambda t: t[0])
    for i in range(len(filtered) - required + 1):
        window = [filtered[i + j][1] for j in range(required)]
        if all(v <= epsilon for v in window):
            # End timestamp = ts of last event in the stable window
            return filtered[i + required - 1][0]
    return None


def extract_loop_closure_correction_stabilised(
    latency_json: Dict[str, Any],
    kpi_events: List[Dict[str, Any]],
    *,
    robot_map: Optional[Dict[str, str]] = None,
    epsilon: float = 1e-3,
    required: int = 5,
) -> List[float]:
    """Return stabilised TLC values per inter-robot loop-closure ingest event.

    Detection time t_det is the wall-clock ingest time of an inter-robot
    ``BetweenFactorPose3``. Stabilisation time t_stable is the first timestamp
    where the global updates remain within ``epsilon`` for ``required`` consecutive
    KPI events, measured from either:
      - centralised: ``optimization_end`` events (solver in {isam2,batch})
      - decentralised: ``ddf_round_delta`` events
    The returned values are t_stable - t_det for all eligible closures.
    """
    if not isinstance(latency_json, dict):
        return []
    events = latency_json.get("events")
    if not isinstance(events, list):
        return []

    # Decide which KPI stream to use for stabilisation
    use_ddf = any(ev.get("event") == "ddf_round_delta" for ev in kpi_events or [])
    if use_ddf:
        stab_event = "ddf_round_delta"
        delta_field = "max_translation_delta"
        allow = None  # DDF events omit standard solver labels or use 'ddf_sam'
    else:
        stab_event = "optimization_end"
        delta_field = "max_translation_delta"
        allow = {"isam2", "batch"}

    out: List[float] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("factor_type") != "BetweenFactorPose3":
            continue
        k1 = ev.get("key1"); k2 = ev.get("key2")
        if not (isinstance(k1, (int, str)) and isinstance(k2, (int, str))):
            continue
        r1 = str(robot_map.get(str(k1))) if robot_map and str(k1) in robot_map else _infer_robot(str(k1))
        r2 = str(robot_map.get(str(k2))) if robot_map and str(k2) in robot_map else _infer_robot(str(k2))
        if r1 == r2:
            continue  # only inter-robot
        t_det = _safe_float(ev.get("ingest_wall"))
        if t_det is None:
            continue
        t_stable = _stabilisation_time_after(
            kpi_events,
            start_ts=float(t_det),
            event_name=stab_event,
            delta_field=delta_field,
            solver_allow=allow,
            epsilon=float(epsilon),
            required=int(required),
        )
        if t_stable is None:
            continue
        dt = float(t_stable) - float(t_det)
        if dt >= 0.0:
            out.append(dt)
    return out


def summarise_loop_closure_correction_stabilised(
    latency_json: Dict[str, Any],
    kpi_events: List[Dict[str, Any]],
    *,
    robot_map: Optional[Dict[str, str]] = None,
    epsilon: float = 1e-3,
    required: int = 5,
) -> Dict[str, Any]:
    vals = extract_loop_closure_correction_stabilised(
        latency_json,
        kpi_events,
        robot_map=robot_map,
        epsilon=epsilon,
        required=required,
    )
    return {
        "loop_closure_correction_stabilised": _stats(vals),
        "loop_closure_correction_stabilised_params": {
            "epsilon": float(epsilon),
            "required": int(required),
        },
    }


# -------------------------------
# Time-to-global convergence (approx)
# -------------------------------

@dataclass
class ConvergenceResult:
    seconds: Optional[float]
    start_ts: Optional[float]
    end_ts: Optional[float]
    method: str  # description of how it was computed


def _read_kpi_events(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
    except Exception:
        return []
    return out


def _convergence_from_delta_events(events: List[Dict[str, Any]],
                                   *,
                                   event_name: str,
                                   delta_field: str = "max_translation_delta",
                                   solver_allow: Optional[Iterable[str]] = None,
                                   epsilon: float = 1e-3,
                                   required: int = 5) -> Optional[Tuple[float, float]]:
    """Return (start_ts, end_ts) when delta stays below epsilon for required events."""
    if not events:
        return None
    filtered: List[Tuple[float, float]] = []
    grouped: Dict[int, Tuple[float, float]] = {}
    has_iteration = False
    allow = set(solver_allow) if solver_allow else None
    for ev in events:
        if ev.get("event") != event_name:
            continue
        if allow is not None and ev.get("solver") not in allow:
            continue
        ts = _safe_float(ev.get("ts"))
        delta = _safe_float(ev.get(delta_field))
        if ts is None or delta is None:
            continue
        if delta < 0:
            continue
        iteration_val = ev.get("iteration")
        if iteration_val is not None:
            try:
                iteration_idx = int(iteration_val)
                has_iteration = True
                prev = grouped.get(iteration_idx)
                if prev is None or delta > prev[1]:
                    grouped[iteration_idx] = (ts, delta)
            except Exception:
                filtered.append((ts, delta))
        else:
            filtered.append((ts, delta))
    if has_iteration:
        filtered.extend(grouped.values())
    if len(filtered) < required:
        return None
    filtered.sort(key=lambda item: item[0])
    start_ts = filtered[0][0]
    for idx in range(len(filtered) - required + 1):
        window = [filtered[idx + j][1] for j in range(required)]
        if all(v <= epsilon for v in window):
            end_ts = filtered[idx + required - 1][0]
            return start_ts, end_ts
    return None


def time_to_global_convergence(run_dir: str) -> ConvergenceResult:
    """Estimate time-to-global convergence from KPI event timelines."""
    kpi_path = os.path.join(run_dir, "kpi_metrics", "kpi_events.jsonl")
    events = _read_kpi_events(kpi_path)
    if not events:
        return ConvergenceResult(seconds=None, start_ts=None, end_ts=None, method="no_events")
    first_ts = None
    for ev in events:
        ts_val = _safe_float(ev.get("ts"))
        if ts_val is None:
            continue
        if first_ts is None or ts_val < first_ts:
            first_ts = ts_val
    # Prefer formal epsilon-based convergence when deltas are available
    epsilon_result = _convergence_from_delta_events(
        events,
        event_name="optimization_end",
        solver_allow={"isam2", "batch"},
        epsilon=1e-3,
        required=5,
    )
    if epsilon_result is not None:
        start_ts, end_ts = epsilon_result
        if first_ts is not None:
            start_ts = first_ts
        return ConvergenceResult(
            seconds=max(0.0, end_ts - start_ts),
            start_ts=start_ts,
            end_ts=end_ts,
            method="delta_threshold",
        )
    ddf_result = _convergence_from_delta_events(
        events,
        event_name="ddf_round_delta",
        delta_field="max_translation_delta",
        solver_allow=None,
        epsilon=1e-3,
        required=5,
    )
    if ddf_result is not None:
        start_ts, end_ts = ddf_result
        if first_ts is not None:
            start_ts = first_ts
        return ConvergenceResult(
            seconds=max(0.0, end_ts - start_ts),
            start_ts=start_ts,
            end_ts=end_ts,
            method="delta_threshold_ddf",
        )
    ts0 = None
    ts_end = None
    last_brd = None
    last_opt = None
    for ev in events:
        ts = _safe_float(ev.get("ts"))
        if ts is None:
            continue
        if ts0 is None:
            ts0 = ts
        name = ev.get("event")
        if name == "map_broadcast":
            # Track the latest broadcast across the run
            last_brd = ts
        elif name == "optimization_end":
            last_opt = ts
    if last_brd is not None:
        ts_end = last_brd
        method = "last_map_broadcast"
    elif last_opt is not None:
        ts_end = last_opt
        method = "last_optimization_end"
    else:
        ts_end = events[-1].get("ts")
        ts_end = _safe_float(ts_end)
        method = "last_event_fallback"
    if ts0 is None or ts_end is None:
        return ConvergenceResult(seconds=None, start_ts=ts0, end_ts=ts_end, method=method)
    return ConvergenceResult(seconds=max(0.0, ts_end - ts0), start_ts=ts0, end_ts=ts_end, method=method)


def derive_kpis_for_run(run_dir: str) -> Dict[str, Any]:
    """Compute derived KPIs for a single output run directory.

    Returns a dictionary suitable for writing to JSON.
    """
    out: Dict[str, Any] = {}
    latency_path = os.path.join(run_dir, "kpi_metrics", "latency_metrics.json")
    lat = _read_json(latency_path)
    kpi_path = os.path.join(run_dir, "kpi_metrics", "kpi_events.jsonl")
    kpi_events = _read_kpi_events(kpi_path)
    # Try to load robot map from the run's JRL to robustly classify inter vs intra
    robot_map = _load_robot_map_for_run(run_dir)
    if lat:
        out.update(summarise_loop_closure_correction(lat, robot_map=robot_map))
        out.update(summarise_interface_correction(lat))
        if kpi_events:
            out.update(
                summarise_loop_closure_correction_stabilised(
                    lat, kpi_events, robot_map=robot_map, epsilon=1e-3, required=5
                )
            )
            out.update(
                summarise_interface_correction_stabilised(
                    lat, kpi_events, epsilon=1e-3, required=5
                )
            )
    conv = time_to_global_convergence(run_dir)
    out["time_to_global_convergence_s"] = conv.seconds
    out["time_to_global_convergence_method"] = conv.method
    out["timeline_start_ts"] = conv.start_ts
    out["timeline_end_ts"] = conv.end_ts
    robustness = _load_robustness_metrics(run_dir)
    if robustness:
        out.update(summarise_delivery_rates(robustness))
    return out


def write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# -------------------------------
# Backend-aware helpers
# -------------------------------

def _load_robot_map_for_run(run_dir: str) -> Optional[Dict[str, str]]:
    """Attempt to build key->robot mapping from the run's JRL path.

    Returns None if unavailable.
    """
    # Discover JRL path from resource profile metadata
    res_path = os.path.join(run_dir, "kpi_metrics", "resource_profile.json")
    meta = _read_json(res_path).get("metadata", {})
    jrl = meta.get("jrl") if isinstance(meta, dict) else None
    if not jrl or not os.path.exists(jrl):
        return None
    try:
        # Import locally to avoid base import cost when not used
        from cosmo_slam_centralised.loader import load_jrl, LoaderConfig, build_key_robot_map  # type: ignore

        doc = load_jrl(jrl, LoaderConfig())
        m = build_key_robot_map(doc)
        return dict(m) if m else None
    except Exception:
        return None


def _load_robustness_metrics(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "kpi_metrics", "robustness_metrics.json")
    return _read_json(path)


def summarise_delivery_rates(robustness_json: Dict[str, Any]) -> Dict[str, Any]:
    stats = robustness_json.get("stats", {}) if isinstance(robustness_json, dict) else {}
    topics = stats.get("topics", {}) if isinstance(stats, dict) else {}
    rates: List[float] = []
    per_topic: Dict[str, Dict[str, float | None]] = {}
    for topic, data in topics.items():
        if not isinstance(data, dict):
            continue
        attempts = _safe_float(data.get("attempts")) or 0.0
        drops = _safe_float(data.get("drops")) or 0.0
        delivered = _safe_float(data.get("delivered")) or (attempts - drops)
        eta = _safe_float(data.get("delivery_rate"))
        if eta is None and attempts > 0:
            eta = max(0.0, min(1.0, delivered / attempts))
        if eta is not None:
            rates.append(eta)
        per_topic[str(topic)] = {
            "attempts": float(attempts),
            "drops": float(drops),
            "delivered": float(delivered),
            "delivery_rate": eta,
        }
    summary: Dict[str, Any] = {"delivery_topics": per_topic}
    if rates:
        summary["delivery_rate"] = _stats(rates)
    return summary


# -------------------------------
# Decentralised analogue (optional)
# -------------------------------

def extract_interface_correction_times(latency_json: Dict[str, Any]) -> List[float]:
    """Return ingest→broadcast latencies recorded for interface events.

    The decentralised pipeline derives ingest→opt/broadcast deltas for interface
    arrivals and injects them into LatencyTracker events when merging per-agent
    KPIs. We read any event that carries 'latency_ingest_to_broadcast' but is
    not a BetweenFactorPose3 (to avoid overlap with centralised).
    """
    events = latency_json.get("events") if isinstance(latency_json, dict) else None
    if not isinstance(events, list):
        return []
    out: List[float] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("factor_type") == "BetweenFactorPose3":
            continue
        d = _safe_float(ev.get("latency_ingest_to_broadcast"))
        if d is not None and d >= 0.0:
            out.append(d)
    return out


def summarise_interface_correction(latency_json: Dict[str, Any]) -> Dict[str, Any]:
    vals = extract_interface_correction_times(latency_json)
    return {"interface_correction": _stats(vals)}


def extract_interface_correction_stabilised(
    latency_json: Dict[str, Any],
    kpi_events: List[Dict[str, Any]],
    *,
    epsilon: float = 1e-3,
    required: int = 5,
) -> List[float]:
    """Return stabilised interface-correction deltas using KPI delta timelines.

    For each latency event that is NOT a BetweenFactorPose3 (e.g., InterfaceMessage
    arrivals in decentralised runs), treat its ingest_wall as t_det and compute
    t_stable via the same stability search used for loop-closure stabilisation.
    Uses ddf_round_delta when present, otherwise optimisation_end.
    """
    if not isinstance(latency_json, dict):
        return []
    events = latency_json.get("events")
    if not isinstance(events, list):
        return []
    use_ddf = any(ev.get("event") == "ddf_round_delta" for ev in kpi_events or [])
    if use_ddf:
        stab_event = "ddf_round_delta"
        delta_field = "max_translation_delta"
        allow = None
    else:
        stab_event = "optimization_end"
        delta_field = "max_translation_delta"
        allow = {"isam2", "batch"}
    out: List[float] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("factor_type") == "BetweenFactorPose3":
            continue
        t_det = _safe_float(ev.get("ingest_wall"))
        if t_det is None:
            continue
        t_stable = _stabilisation_time_after(
            kpi_events,
            start_ts=float(t_det),
            event_name=stab_event,
            delta_field=delta_field,
            solver_allow=allow,
            epsilon=float(epsilon),
            required=int(required),
        )
        if t_stable is None:
            continue
        dt = float(t_stable) - float(t_det)
        if dt >= 0.0:
            out.append(dt)
    return out


def summarise_interface_correction_stabilised(
    latency_json: Dict[str, Any],
    kpi_events: List[Dict[str, Any]],
    *,
    epsilon: float = 1e-3,
    required: int = 5,
) -> Dict[str, Any]:
    vals = extract_interface_correction_stabilised(latency_json, kpi_events, epsilon=epsilon, required=required)
    return {
        "interface_correction_stabilised": _stats(vals),
        "interface_correction_stabilised_params": {
            "epsilon": float(epsilon),
            "required": int(required),
        },
    }
