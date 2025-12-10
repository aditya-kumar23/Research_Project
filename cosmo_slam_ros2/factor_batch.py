"""JSON-based factor batch serialisation scaffolding for ROS 2 transport.

The helpers defined here deliberately avoid depending on ``rclpy`` so the
centralised backend can exercise encode/decode logic in unit tests without
having ROS 2 available.  Future iterations can swap this module for a more
compact wire format without changing the public API.
"""

from __future__ import annotations

import json
import logging
from typing import Iterable, List, Union, Dict, Any
import uuid

import numpy as np

from cosmo_slam_centralised.models import (
    PriorFactorPose3,
    BetweenFactorPose3,
    Quaternion,
    Translation,
)

logger = logging.getLogger("cosmo_slam_ros2.factor_batch")

FactorLike = Union[PriorFactorPose3, BetweenFactorPose3]


def _serialise_key(key: Union[str, int]) -> Dict[str, Any]:
    if isinstance(key, int):
        return {"type": "int", "value": key}
    if isinstance(key, str):
        return {"type": "str", "value": key}
    # Fallback: representable string
    return {"type": "repr", "value": repr(key)}


def _deserialise_key(payload: Dict[str, Any]) -> Union[str, int]:
    kind = payload.get("type")
    value = payload.get("value")
    if kind == "int":
        try:
            return int(value)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Expected int-compatible value, got {value!r}") from exc
    if kind == "str":
        return str(value)
    if kind == "repr":
        # Best effort: leave as string
        return str(value)
    # Unknown tag, return raw
    return value


def _serialise_quaternion(q: Quaternion) -> List[float]:
    return [float(q.w), float(q.x), float(q.y), float(q.z)]


def _serialise_translation(t: Translation) -> List[float]:
    return [float(t.x), float(t.y), float(t.z)]


def _serialise_covariance(cov) -> List[List[float]]:
    arr = np.asarray(cov, dtype=float)
    if arr.shape == (6, 6):
        return arr.tolist()
    if arr.size == 36:
        return arr.reshape(6, 6).tolist()
    raise ValueError(f"Covariance must have 36 elements; got shape {arr.shape}")


def _deserialise_covariance(payload: Any) -> np.ndarray:
    arr = np.asarray(payload, dtype=float)
    if arr.shape == (6, 6):
        return arr
    if arr.size == 36:
        return arr.reshape(6, 6)
    raise ValueError(f"Decoded covariance has unexpected shape {arr.shape}")


def _serialise_factor(factor: FactorLike) -> Dict[str, Any]:
    if isinstance(factor, PriorFactorPose3):
        return {
            "type": "PriorFactorPose3",
            "key": _serialise_key(factor.key),
            "rotation": _serialise_quaternion(factor.rotation),
            "translation": _serialise_translation(factor.translation),
            "covariance": _serialise_covariance(factor.covariance),
            "stamp": float(getattr(factor, "stamp", 0.0)),
        }
    if isinstance(factor, BetweenFactorPose3):
        return {
            "type": "BetweenFactorPose3",
            "key1": _serialise_key(factor.key1),
            "key2": _serialise_key(factor.key2),
            "rotation": _serialise_quaternion(factor.rotation),
            "translation": _serialise_translation(factor.translation),
            "covariance": _serialise_covariance(factor.covariance),
            "stamp": float(getattr(factor, "stamp", 0.0)),
        }
    raise TypeError(f"Unsupported factor type {type(factor)!r}")


def _deserialise_factor(payload: Dict[str, Any]) -> FactorLike:
    kind = payload.get("type")
    if kind == "PriorFactorPose3":
        rotation = payload.get("rotation")
        translation = payload.get("translation")
        if rotation is None or translation is None:
            raise ValueError("Missing rotation/translation in PriorFactorPose3 payload")
        quaternion = Quaternion(*map(float, rotation))
        t = Translation(*map(float, translation))
        cov = _deserialise_covariance(payload.get("covariance"))
        key = _deserialise_key(payload.get("key"))
        stamp = float(payload.get("stamp", 0.0))
        return PriorFactorPose3(key=key, rotation=quaternion, translation=t, covariance=cov, stamp=stamp)
    if kind == "BetweenFactorPose3":
        rotation = payload.get("rotation")
        translation = payload.get("translation")
        if rotation is None or translation is None:
            raise ValueError("Missing rotation/translation in BetweenFactorPose3 payload")
        quaternion = Quaternion(*map(float, rotation))
        t = Translation(*map(float, translation))
        cov = _deserialise_covariance(payload.get("covariance"))
        key1 = _deserialise_key(payload.get("key1"))
        key2 = _deserialise_key(payload.get("key2"))
        stamp = float(payload.get("stamp", 0.0))
        return BetweenFactorPose3(key1=key1, key2=key2, rotation=quaternion, translation=t, covariance=cov, stamp=stamp)
    raise ValueError(f"Unsupported factor payload type {kind!r}")


def encode_factor_batch(factors: Iterable[FactorLike], *, version: int = 1) -> bytes:
    """Serialise a batch of factors into a compact JSON payload.

    Parameters
    ----------
    factors:
        Iterable of ``PriorFactorPose3`` or ``BetweenFactorPose3`` instances.
    version:
        Payload version tag inserted into the serialized document (defaults to 1).

    Returns
    -------
    bytes
        UTF-8 encoded JSON payload.
    """

    entries = []
    for factor in factors:
        entries.append(_serialise_factor(factor))
    # Attach sender timestamps to support end-to-end latency accounting on the
    # subscriber side. We prefer monotonic clock for deltas but include wall
    # time for correlation.
    try:
        import time as _time  # local import to avoid overhead when unused
        send_ts_mono = float(_time.perf_counter())
        send_ts_wall = float(_time.time())
    except Exception:
        send_ts_mono = None
        send_ts_wall = None
    payload = {"version": version, "factors": entries}
    # Add a message identifier so the subscriber can account bandwidth once per
    # received batch even if it expands into multiple factors.
    try:
        payload["message_id"] = uuid.uuid4().hex
    except Exception:
        pass
    if send_ts_mono is not None:
        payload["send_ts_mono"] = send_ts_mono
    if send_ts_wall is not None:
        payload["send_ts_wall"] = send_ts_wall
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def decode_factor_batch(payload: Union[str, bytes, bytearray]) -> List[FactorLike]:
    """Reconstruct factor objects from a payload created by :func:`encode_factor_batch`."""

    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    doc = json.loads(payload)
    version = doc.get("version", 1)
    if version != 1:
        logger.warning("Unknown factor batch version %s; attempting fallback decode", version)
    factors = doc.get("factors", [])
    # Timestamps and message_id may be present for end-to-end latency and
    # bandwidth accounting.
    send_ts_mono = doc.get("send_ts_mono", None)
    send_ts_wall = doc.get("send_ts_wall", None)
    message_id = doc.get("message_id", None)
    out: List[FactorLike] = []
    for idx, item in enumerate(factors):
        try:
            f = _deserialise_factor(item)
            # Attach send timestamp metadata if available. The factor objects
            # are dataclasses without __slots__, so dynamic attributes are OK.
            if send_ts_mono is not None:
                try:
                    setattr(f, "send_ts_mono", float(send_ts_mono))
                except Exception:
                    pass
            if send_ts_wall is not None:
                try:
                    setattr(f, "send_ts_wall", float(send_ts_wall))
                except Exception:
                    pass
            if message_id is not None:
                try:
                    setattr(f, "__ros2_msg_id", str(message_id))
                except Exception:
                    pass
            out.append(f)
        except Exception as exc:
            logger.warning("Skipping factor[%d]: %s", idx, exc)
    return out
