"""Decentralised SLAM backend for COSMO-bench JRL datasets.

This package contains a light-weight simulation of a distributed (multi-robot)
pose graph backend inspired by DDF-SAM.  Each robot owns an agent that maintains
its local factor graph, exchanges interface summaries over a peer-to-peer bus,
and runs iterative distributed optimisation rounds.

The modules exposed here intentionally mirror the structure of the centralised
backend so that higher-level tooling (CLIs, notebooks, experiments) can switch
between centralised and decentralised execution with minimal glue code.
"""

from .partition import partition_measurements, RobotMeasurementBundle
from .communication import PeerToPeerBus, InterfaceMessage
from .agents import DecentralizedRobotAgent
from .ddf_sam import DDFSAMBackend, PacingConfig

__all__ = [
    "partition_measurements",
    "RobotMeasurementBundle",
    "PeerToPeerBus",
    "InterfaceMessage",
    "DecentralizedRobotAgent",
    "DDFSAMBackend",
    "PacingConfig",
]
