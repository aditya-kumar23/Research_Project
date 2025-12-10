"""Bandwidth helpers for decentralised interface messages.

Mirrors the approximate byte accounting approach used by the centralised
backend by serialising a compact JSON payload and measuring its size.
"""
from __future__ import annotations

import json
import numpy as np

from .communication import InterfaceMessage
from cosmo_slam_ros2.interface_msg import encode_interface_message


def _flatten_cov(cov: np.ndarray):
    return np.asarray(cov, dtype=float).reshape(-1).tolist()


def interface_message_bytes(msg: InterfaceMessage) -> int:
    """Serialized payload size for a single interface message.

    Uses the same encoder as the ROS 2 transport to achieve byte-accurate
    accounting relative to `ros2 topic bw` for the interface topics.
    """
    try:
        payload = encode_interface_message(msg)
        return len(payload)
    except Exception:
        # Fallback to approximate JSON if encoding fails for any reason
        rot = [float(msg.rotation.w), float(msg.rotation.x), float(msg.rotation.y), float(msg.rotation.z)]
        trans = [float(msg.translation.x), float(msg.translation.y), float(msg.translation.z)]
        payload = {
            "sender": msg.sender,
            "receiver": msg.receiver,
            "key": str(msg.key),
            "stamp": float(getattr(msg, "stamp", 0.0)),
            "iteration": int(getattr(msg, "iteration", 0)),
            "rotation": rot,
            "translation": trans,
            "covariance": _flatten_cov(msg.covariance),
        }
        serialized = json.dumps(payload, separators=(",", ":"))
        return len(serialized.encode("utf-8"))
