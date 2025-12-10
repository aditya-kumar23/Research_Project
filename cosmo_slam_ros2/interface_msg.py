"""Encode/decode helpers for decentralised interface messages."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

import numpy as np

from cosmo_slam_centralised.models import Quaternion, Translation

try:  # pragma: no cover - import guard for optional decentralised package
    from cosmo_slam_decentralised.communication import InterfaceMessage as _InterfaceMessage
except Exception:  # pragma: no cover - defensive: allow import without decentralised backend
    _InterfaceMessage = None  # type: ignore[misc]

if _InterfaceMessage is None:  # pragma: no cover - fallback for decoding without full package
    from dataclasses import dataclass

    @dataclass
    class _InterfaceMessage:  # type: ignore[override]
        sender: str
        receiver: str
        key: str
        rotation: Quaternion
        translation: Translation
        covariance: np.ndarray
        iteration: int
        stamp: float = 0.0
        sent_wall_time: float = 0.0
        sent_mono_time: float = 0.0

InterfaceMessage = _InterfaceMessage  # type: ignore[misc]

logger = logging.getLogger("cosmo_slam_ros2.interface_msg")


def _serialise_quaternion(q: Quaternion):
    return [float(q.w), float(q.x), float(q.y), float(q.z)]


def _serialise_translation(t: Translation):
    return [float(t.x), float(t.y), float(t.z)]


def _serialise_covariance(covariance) -> Any:
    arr = np.asarray(covariance, dtype=float)
    if arr.shape == (6, 6):
        return arr.tolist()
    if arr.size == 36:
        return arr.reshape(6, 6).tolist()
    raise ValueError(f"Interface covariance must have 36 elements; got shape {arr.shape}")


def _deserialise_covariance(payload: Any) -> np.ndarray:
    arr = np.asarray(payload, dtype=float)
    if arr.shape == (6, 6):
        return arr
    if arr.size == 36:
        return arr.reshape(6, 6)
    raise ValueError(f"Decoded covariance has unexpected shape {arr.shape}")


def _coerce_quaternion(values) -> Quaternion:
    if values is None:
        raise ValueError("Quaternion payload missing")
    vals = list(map(float, values))
    if len(vals) != 4:
        raise ValueError(f"Quaternion payload must have 4 elements, got {len(vals)}")
    return Quaternion(*vals)


def _coerce_translation(values) -> Translation:
    if values is None:
        raise ValueError("Translation payload missing")
    vals = list(map(float, values))
    if len(vals) != 3:
        raise ValueError(f"Translation payload must have 3 elements, got {len(vals)}")
    return Translation(*vals)


def encode_interface_message(msg: "InterfaceMessage", *, version: int = 1) -> bytes:
    """Encode an ``InterfaceMessage`` into a JSON payload."""

    # Allow duck-typed messages coming from spawned processes where the
    # dataclass identity may differ but the attributes match.
    required_attrs = ("sender", "receiver", "key", "rotation", "translation", "covariance")
    if not all(hasattr(msg, attr) for attr in required_attrs):
        raise TypeError(f"Interface message missing required fields: {type(msg)!r}")

    payload: Dict[str, Any] = {
        "version": version,
        "message": {
            "sender": msg.sender,
            "receiver": msg.receiver,
            "key": str(msg.key),
            "stamp": float(getattr(msg, "stamp", 0.0)),
            "iteration": int(getattr(msg, "iteration", 0)),
            "rotation": _serialise_quaternion(msg.rotation),
            "translation": _serialise_translation(msg.translation),
            "covariance": _serialise_covariance(msg.covariance),
        },
    }
    # Include sender timestamps for E2E latency measurement on receivers.
    sent_wall = getattr(msg, "sent_wall_time", None)
    sent_mono = getattr(msg, "sent_mono_time", None)
    if sent_mono is not None:
        try:
            payload["message"]["send_ts_mono"] = float(sent_mono)
        except Exception:
            pass
    if sent_wall is not None:
        try:
            payload["message"]["send_ts_wall"] = float(sent_wall)
        except Exception:
            pass
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def decode_interface_message(data: Any) -> "InterfaceMessage":
    """Decode bytes produced by :func:`encode_interface_message`."""

    if InterfaceMessage is None:  # pragma: no cover - defensive guard
        raise RuntimeError("cosmo_slam_decentralised.communication.InterfaceMessage is unavailable")

    if isinstance(data, (bytes, bytearray)):
        data = data.decode("utf-8")
    doc = json.loads(data)
    version = doc.get("version", 1)
    if version != 1:
        logger.warning("Unknown interface message version %s; attempting fallback decode", version)
    payload = doc.get("message", {})
    try:
        rotation = _coerce_quaternion(payload.get("rotation"))
        translation = _coerce_translation(payload.get("translation"))
        covariance = _deserialise_covariance(payload.get("covariance"))
        sender = str(payload.get("sender"))
        receiver = str(payload.get("receiver"))
        key = payload.get("key")
        stamp = float(payload.get("stamp", 0.0))
        iteration = int(payload.get("iteration", 0))
        # Back/forward compatible timestamp fields
        sent_wall_time = payload.get("send_ts_wall")
        if sent_wall_time is None:
            sent_wall_time = payload.get("sent_wall_time")
        if sent_wall_time is not None:
            sent_wall_time = float(sent_wall_time)
        sent_mono_time = payload.get("send_ts_mono")
        if sent_mono_time is not None:
            sent_mono_time = float(sent_mono_time)
    except Exception as exc:
        raise ValueError(f"Malformed interface message payload: {exc}") from exc

    message = InterfaceMessage(
        sender=sender,
        receiver=receiver,
        key=key,
        rotation=rotation,
        translation=translation,
        covariance=covariance,
        iteration=iteration,
        stamp=stamp,
        sent_wall_time=sent_wall_time,
        sent_mono_time=sent_mono_time,
    )
    return message
