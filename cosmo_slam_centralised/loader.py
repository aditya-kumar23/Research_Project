import json
from typing import Dict, Any, Optional, List, Union, Tuple, Set
from dataclasses import dataclass
import logging
import numpy as np
from .models import (
    Quaternion, Translation, InitEntry, PriorFactorPose3, BetweenFactorPose3,
    JRLDocument, to_covariance
)

logger = logging.getLogger("cosmo_slam.loader")

@dataclass
class LoaderConfig:
    quaternion_order: str = "wxyz"   # dataset uses [w,x,y,z]
    validate_schema: bool = True
    include_potential_outliers: bool = False

def _q_from_list(q: List[float], order: str) -> Quaternion:
    if order == "wxyz":
        if len(q) != 4: raise ValueError("Quaternion must be [w,x,y,z]")
        return Quaternion(q[0], q[1], q[2], q[3])
    elif order == "xyzw":
        if len(q) != 4: raise ValueError("Quaternion must be [x,y,z,w]")
        return Quaternion(q[3], q[0], q[1], q[2])
    else:
        raise ValueError(f"Unsupported quaternion order: {order}")

def _t_from_list(t: List[float]) -> Translation:
    if len(t) != 3: raise ValueError("Translation must be [x,y,z]")
    return Translation(t[0], t[1], t[2])

def load_jrl(path: str, cfg: Optional[LoaderConfig] = None) -> JRLDocument:
    cfg = cfg or LoaderConfig()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    doc = JRLDocument(
        measurements=data.get("measurements", {}),
        outlier_factors=data.get("outlier factors", []) or data.get("outlier_factors", []),
        potential_outlier_factors=data.get("potential outlier factors", []) or data.get("potential_outlier_factors", []),
        ground_truth=data.get("ground truth", {}) or data.get("groundtruth", {}) or data.get("ground_truth", {}),
        initialisation=data.get("initialisation", []) or data.get("initialization", []),
    )
    if cfg.validate_schema:
        if not isinstance(doc.measurements, (dict, list)):
            logger.warning("measurements is not dict/list; got %s", type(doc.measurements))
        if not isinstance(doc.initialisation, (list, dict)):
            logger.warning("initialisation is not list/dict; got %s", type(doc.initialisation))
    return doc


def _as_index_pair(item: Any) -> Optional[Tuple[int, int]]:
    """Return (frame_idx, measurement_idx) if `item` encodes such coordinates."""
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        try:
            return int(item[0]), int(item[1])
        except (TypeError, ValueError):
            return None
    if isinstance(item, dict):
        frame = item.get("frame", item.get("frame_idx"))
        meas = item.get("measurement", item.get("measurement_idx"))
        if frame is None or meas is None:
            return None
        try:
            return int(frame), int(meas)
        except (TypeError, ValueError):
            return None
    return None


def _build_outlier_lookup(entries: Any) -> Dict[str, Set[Tuple[int, int]]]:
    """Map robot id -> {(frame_idx, measurement_idx), ...} for quick filtering."""
    lookup: Dict[str, Set[Tuple[int, int]]] = {}
    if isinstance(entries, dict):
        for rid, seq in entries.items():
            coords: Set[Tuple[int, int]] = set()
            if isinstance(seq, list):
                for item in seq or []:
                    pair = _as_index_pair(item)
                    if pair is not None:
                        coords.add(pair)
            elif isinstance(seq, dict):
                for sub in seq.values():
                    if not isinstance(sub, list):
                        continue
                    for item in sub or []:
                        pair = _as_index_pair(item)
                        if pair is not None:
                            coords.add(pair)
            if coords:
                lookup[str(rid)] = coords
    elif isinstance(entries, list):
        coords: Set[Tuple[int, int]] = set()
        for item in entries or []:
            pair = _as_index_pair(item)
            if pair is not None:
                coords.add(pair)
        if coords:
            lookup["global"] = coords
    return lookup


def _is_flagged(lookup: Dict[str, Set[Tuple[int, int]]], rid: str, frame_idx: int, meas_idx: int) -> bool:
    key = str(rid)
    pair = (frame_idx, meas_idx)
    pairs = lookup.get(key)
    if pairs and pair in pairs:
        return True
    global_pairs = lookup.get("global")
    if global_pairs and pair in global_pairs:
        return True
    return False

def iter_init_entries(doc: JRLDocument, cfg: Optional[LoaderConfig] = None):
    """Yield InitEntry. Supports list OR dict-per-robot."""
    cfg = cfg or LoaderConfig()
    init = doc.initialisation or []
    if isinstance(init, list):
        iterable = [("global", x) for x in init]
    elif isinstance(init, dict):
        iterable = [(rid, x) for rid, lst in init.items() for x in (lst or [])]
    else:
        iterable = []

    for idx, (rid, it) in enumerate(iterable):
        try:
            rot_list = it.get("rotation") or (it.get("prior", {}) or {}).get("rotation")
            trans_list = it.get("translation") or (it.get("prior", {}) or {}).get("translation")
            rot = _q_from_list(rot_list, cfg.quaternion_order)
            trans = _t_from_list(trans_list)
            yield InitEntry(key=it["key"], rotation=rot, translation=trans, type=it.get("type", "Pose3"))
        except Exception as e:
            logger.warning("Skipping malformed initialization[%d]: %s", idx, e)

def _parse_prior(meas: Dict[str, Any], stamp: float, cfg: LoaderConfig) -> Optional[PriorFactorPose3]:
    try:
        prior = meas.get("prior", meas)
        rot = _q_from_list(prior["rotation"], cfg.quaternion_order)
        trans = _t_from_list(prior["translation"])
        cov = to_covariance(meas["covariance"])
        key = meas["key"]
        return PriorFactorPose3(key=key, rotation=rot, translation=trans, covariance=cov, stamp=float(stamp))
    except Exception as e:
        logger.warning("Skipping PriorFactorPose3: %s", e)
        return None

def _parse_between(meas: Dict[str, Any], stamp: float, cfg: LoaderConfig) -> Optional[BetweenFactorPose3]:
    try:
        m = meas.get("measurement", meas)
        rot = _q_from_list(m["rotation"], cfg.quaternion_order)
        trans = _t_from_list(m["translation"])
        cov = to_covariance(meas["covariance"])
        key1 = meas.get("key1", meas.get("key_from"))
        key2 = meas.get("key2", meas.get("key_to"))
        if key1 is None or key2 is None:
            raise KeyError("Missing key1/key2 (or key_from/key_to)")
        return BetweenFactorPose3(key1=key1, key2=key2, rotation=rot, translation=trans, covariance=cov, stamp=float(stamp))
    except Exception as e:
        logger.warning("Skipping BetweenFactorPose3: %s", e)
        return None

def iter_measurements(doc: JRLDocument, cfg: Optional[LoaderConfig] = None):
    """Yield Prior/Between factors sorted by 'stamp'.

    Handles:
    - dict per robot: measurements['a'] is a LIST OF FRAMES with {'stamp','measurements':[...]}
    - dict with type arrays (rare)
    - flat list with 'type' per item (rare)
    """
    cfg = cfg or LoaderConfig()
    out: List[Union[PriorFactorPose3, BetweenFactorPose3]] = []

    outlier_lookup = _build_outlier_lookup(doc.outlier_factors)
    potential_lookup = _build_outlier_lookup(doc.potential_outlier_factors)
    skipped_outliers = 0
    skipped_potential = 0

    ms = doc.measurements
    if isinstance(ms, dict):
        # Case 1: per-robot → list of frames
        for rid, seq in (ms or {}).items():
            if isinstance(seq, list) and seq and isinstance(seq[0], dict) and "measurements" in seq[0]:
                for frame_idx, frame in enumerate(seq):
                    stamp = frame.get("stamp", 0.0)
                    measurements = frame.get("measurements", []) or []
                    for meas_idx, m in enumerate(measurements):
                        if _is_flagged(outlier_lookup, rid, frame_idx, meas_idx):
                            skipped_outliers += 1
                            continue
                        if (not cfg.include_potential_outliers
                                and _is_flagged(potential_lookup, rid, frame_idx, meas_idx)):
                            skipped_potential += 1
                            continue
                        t = (m.get("type") or "").strip()
                        if t == "PriorFactorPose3":
                            p = _parse_prior(m, stamp, cfg);  out.append(p) if p else None
                        elif t == "BetweenFactorPose3":
                            b = _parse_between(m, stamp, cfg); out.append(b) if b else None
                continue
            # Case 2: per-robot → dict with type arrays
            if isinstance(seq, dict):
                for m in (seq.get("PriorFactorPose3", []) or []):
                    p = _parse_prior(m, m.get("stamp", 0.0), cfg);  out.append(p) if p else None
                for m in (seq.get("BetweenFactorPose3", []) or []):
                    b = _parse_between(m, m.get("stamp", 0.0), cfg); out.append(b) if b else None
                continue

            # Case 3: dict keyed by factor type with list payloads
            if isinstance(seq, list) and rid in {"PriorFactorPose3", "BetweenFactorPose3"}:
                parser = _parse_prior if rid == "PriorFactorPose3" else _parse_between
                for m in seq:
                    parsed = parser(m, m.get("stamp", 0.0), cfg)
                    if parsed:
                        out.append(parsed)

    elif isinstance(ms, list):
        # Case 3: flat list with 'type'
        for m in ms:
            t = (m.get("type") or "").strip()
            if t == "PriorFactorPose3":
                p = _parse_prior(m, m.get("stamp", 0.0), cfg);  out.append(p) if p else None
            elif t == "BetweenFactorPose3":
                b = _parse_between(m, m.get("stamp", 0.0), cfg); out.append(b) if b else None

    if skipped_outliers or skipped_potential:
        logger.info(
            "Filtered %d labelled outliers and %d potential outliers (include=%s)",
            skipped_outliers,
            skipped_potential,
            cfg.include_potential_outliers,
        )

    # Sort by stamp for ingest order
    out = [x for x in out if x is not None]
    out.sort(key=lambda x: x.stamp)
    for it in out:
        yield it

def build_key_robot_map(doc: JRLDocument) -> Dict[str, str]:
    """Return mapping 'str(key)' -> robot_id ('a'|'b'|'c') from initialization (dict form)."""
    mapping: Dict[str, str] = {}
    init = doc.initialisation
    if isinstance(init, dict):
        for rid, lst in init.items():
            for it in (lst or []):
                k = str(it.get("key"))
                if k is not None:
                    mapping[k] = rid
    return mapping


def summarize_schema(doc: JRLDocument) -> Dict[str, Any]:
    """Small summary for debugging."""
    ms = doc.measurements
    if isinstance(ms, dict):
        return {
            "measurements_type": "dict",
            "keys": list(ms.keys()),
            "per_robot_kind": {k: type(v).__name__ for k, v in ms.items()}
        }
    elif isinstance(ms, list):
        return {"measurements_type": "list", "len": len(ms)}
    else:
        return {"measurements_type": str(type(ms))}

# ---- Ground truth parsing (robust to schema variants) ----
def _parse_pose_like(d, cfg: LoaderConfig):
    """Accept {"rotation":[...], "translation":[...]}, or {"pose":{...}}."""
    src = d.get("pose", d)
    rot = _q_from_list(src["rotation"], cfg.quaternion_order)
    trans = _t_from_list(src["translation"])
    return rot, trans

def iter_groundtruth(doc: JRLDocument, cfg: Optional[LoaderConfig] = None):
    """Yield tuples: (robot_id, key, rotation, translation, stamp or None).

    Supports:
      groundtruth = { "a":[{key, rotation, translation, stamp?}, ...], "b":[...], ... }
      groundtruth = { "a":[{key, pose:{rotation,translation}, ...}, ...], ... }
      If 'key' is missing, uses the index within the list.
    """
    cfg = cfg or LoaderConfig()
    gt = doc.ground_truth
    if not isinstance(gt, dict):
        return
    for rid, seq in (gt or {}).items():
        if not isinstance(seq, list):
            continue
        for idx, item in enumerate(seq):
            try:
                key = item.get("key", idx)
                rot, trans = _parse_pose_like(item, cfg)
                stamp = item.get("stamp")
                yield rid, key, rot, trans, stamp
            except Exception as e:
                logger.warning("Skipping groundtruth[%s][%d]: %s", rid, idx, e)

def groundtruth_by_robot_key(doc: JRLDocument, cfg: Optional[LoaderConfig] = None):
    """Return dict: rid -> { str(key): (rotation, translation, stamp) }"""
    cfg = cfg or LoaderConfig()
    out: Dict[str, Dict[str, tuple]] = {}
    for rid, key, rot, trans, stamp in iter_groundtruth(doc, cfg):
        out.setdefault(rid, {})[str(key)] = (rot, trans, stamp)
    return out
# ----------------------------------------------------------
