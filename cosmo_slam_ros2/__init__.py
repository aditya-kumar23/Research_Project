"""Scaffolding for ROS 2 helpers used by cosmo_slam.

This package currently provides lightweight JSON-based encode/decode helpers
for factor batches (centralised backend) and interface messages (decentralised
backend).  The helpers are intentionally simple so they can be exercised without
ROS 2 installed; future iterations can swap in binary encoders or ROS-specific
message adapters without changing the public API.
"""

from .factor_batch import encode_factor_batch, decode_factor_batch
from .interface_msg import encode_interface_message, decode_interface_message
from .qos import default_qos_profile, parse_qos_options

__all__ = [
    "encode_factor_batch",
    "decode_factor_batch",
    "encode_interface_message",
    "decode_interface_message",
    "default_qos_profile",
    "parse_qos_options",
]
