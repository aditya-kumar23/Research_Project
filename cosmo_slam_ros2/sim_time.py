"""Helpers for enabling ROS 2 simulated time on nodes."""
from __future__ import annotations

import os
from typing import Any


def use_sim_time() -> bool:
    flag = os.environ.get("COSMO_USE_SIM_TIME")
    if not flag:
        return False
    return flag.strip().lower() not in {"0", "false", "no"}


def configure_sim_time(node: Any) -> None:
    if not use_sim_time() or node is None:
        return
    try:
        from rclpy.parameter import Parameter  # type: ignore
    except Exception:
        return
    try:
        node.set_parameters([Parameter(name="use_sim_time", value=True)])
    except Exception:
        pass


__all__ = ["use_sim_time", "configure_sim_time"]
