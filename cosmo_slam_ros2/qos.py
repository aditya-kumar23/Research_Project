"""Helper utilities for configuring ROS 2 QoS profiles.

These helpers keep the CLI parsing layer decoupled from ``rclpy`` so we can
parse/validate QoS-related flags even when ROS 2 is not installed.  The returned
profiles are plain dictionaries; ROS-specific code can translate them into
``rclpy.qos.QoSProfile`` instances when integrating transport hooks.
"""

from __future__ import annotations

from typing import Dict, Optional

DEFAULT_RELIABILITY = "reliable"
DEFAULT_DURABILITY = "volatile"
DEFAULT_DEPTH = 10


def default_qos_profile() -> Dict[str, object]:
    """Return a copy of the default QoS profile."""

    return {
        "reliability": DEFAULT_RELIABILITY,
        "durability": DEFAULT_DURABILITY,
        "depth": DEFAULT_DEPTH,
    }


_VALID_RELIABILITY = {"reliable", "best_effort"}
_VALID_DURABILITY = {"volatile", "transient_local"}


def parse_qos_options(
    reliability: Optional[str] = None,
    durability: Optional[str] = None,
    depth: Optional[int] = None,
) -> Dict[str, object]:
    """Validate QoS CLI flags and return a profile dictionary."""

    profile = default_qos_profile()
    if reliability:
        norm = reliability.lower()
        if norm not in _VALID_RELIABILITY:
            raise ValueError(f"Unsupported reliability policy {reliability!r}")
        profile["reliability"] = norm
    if durability:
        norm = durability.lower()
        if norm not in _VALID_DURABILITY:
            raise ValueError(f"Unsupported durability policy {durability!r}")
        profile["durability"] = norm
    if depth is not None:
        if depth <= 0:
            raise ValueError("QoS depth must be positive")
        profile["depth"] = int(depth)
    return profile

