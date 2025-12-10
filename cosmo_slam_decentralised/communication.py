from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from collections import defaultdict, deque
import threading
import time
import logging
import numpy as np
from cosmo_slam_ros2.impair import ImpairmentPolicy
from cosmo_slam_ros2.sim_time import configure_sim_time

from cosmo_slam_centralised.models import Quaternion, Translation


@dataclass
class InterfaceMessage:
    """Pose + covariance estimate that one robot shares with a peer.

    The payload mirrors the data DDF-SAM agents exchange: a linearisation
    point (pose) and its covariance for separator variables that are shared
    across robot sub-graphs.
    """

    sender: str
    receiver: str
    key: str
    rotation: Quaternion
    translation: Translation
    covariance: np.ndarray
    iteration: int
    stamp: float = 0.0
    sent_wall_time: float = None
    # Monotonic timestamp from sender process for E2E latency (optional)
    sent_mono_time: float = None

    def __post_init__(self):
        if self.sent_wall_time is None:
            self.sent_wall_time = time.time()
        if self.sent_mono_time is None:
            try:
                self.sent_mono_time = time.perf_counter()
            except Exception:
                self.sent_mono_time = None
        self.covariance = np.asarray(self.covariance, dtype=float)


class PeerToPeerBus:
    """In-memory peer-to-peer message bus used to emulate decentralised exchange."""

    def __init__(self):
        self._mailboxes: Dict[str, deque] = defaultdict(deque)
        self._delivered: int = 0

    def post(self, message: InterfaceMessage) -> None:
        """Queue a message for the receiver.  FIFO order is preserved per receiver."""
        self._mailboxes[message.receiver].append(message)

    def broadcast(self, sender: str, receivers: List[str], key: str,
                  rotation: Quaternion, translation: Translation,
                  covariance: np.ndarray, iteration: int, stamp: float = 0.0) -> None:
        """Convenience helper to push the same payload to multiple peers."""
        for recv in receivers:
            self.post(InterfaceMessage(
                sender=sender,
                receiver=recv,
                key=str(key),
                rotation=rotation,
                translation=translation,
                covariance=np.asarray(covariance, dtype=float),
                iteration=iteration,
                stamp=stamp,
            ))

    def drain(self, robot_id: str) -> List[InterfaceMessage]:
        """Return and clear all pending messages for `robot_id`."""
        mailbox = self._mailboxes.get(robot_id)
        if not mailbox:
            return []
        msgs = list(mailbox)
        self._delivered += len(msgs)
        mailbox.clear()
        return msgs

    @property
    def pending_counts(self) -> Dict[str, int]:
        return {rid: len(q) for rid, q in self._mailboxes.items() if q}

    @property
    def delivered(self) -> int:
        return self._delivered

    def close(self) -> None:  # pragma: no cover - default no-op for compatibility
        return None


class Ros2PeerBus(PeerToPeerBus):  # pragma: no cover - requires ROS 2 runtime
    """ROS 2-backed peer-to-peer bus mirroring :class:`PeerToPeerBus` semantics."""

    def __init__(
        self,
        robot_ids: Iterable[str],
        *,
        topic_prefix: str = "/cosmo/iface",
        qos_profile: Optional[Dict[str, object]] = None,
        spin_timeout: float = 0.1,
    ) -> None:
        super().__init__()
        self._topic_prefix = (topic_prefix or "/cosmo/iface").rstrip("/") or "/cosmo/iface"
        self._robot_ids = sorted({str(rid) for rid in robot_ids})
        self._qos_profile = dict(qos_profile or {})
        self._spin_timeout = max(spin_timeout, 0.01)

        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._rclpy = None
        self._node = None
        self._publishers: Dict[str, object] = {}
        self._subscriptions: Dict[str, object] = {}
        self._thread: Optional[threading.Thread] = None
        self._msg_type = None
        self._decode = None
        self._encode = None
        self._qos_profile.setdefault("reliability", "reliable")
        self._qos_profile.setdefault("durability", "volatile")
        self._qos_profile.setdefault("depth", 10)

        self._logger = logging.getLogger("cosmo_slam.decentralised.ros2_bus")
        self._impair = ImpairmentPolicy.from_env() or ImpairmentPolicy(None)

        self._bootstrap_ros()
        for rid in self._robot_ids:
            self._ensure_subscription(rid)

        self._thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # ROS 2 setup helpers
    # ------------------------------------------------------------------
    def _bootstrap_ros(self) -> None:
        try:
            import rclpy  # type: ignore
            from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy  # type: ignore
            from std_msgs.msg import UInt8MultiArray  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "ROS 2 decentralised transport requested but rclpy/std_msgs are unavailable"
            ) from exc

        from cosmo_slam_ros2.interface_msg import encode_interface_message, decode_interface_message

        qos_kwargs = {"depth": int(self._qos_profile.get("depth", 10))}
        reliability = str(self._qos_profile.get("reliability", "reliable")).lower()
        durability = str(self._qos_profile.get("durability", "volatile")).lower()
        qos_kwargs["reliability"] = (
            ReliabilityPolicy.RELIABLE if reliability == "reliable" else ReliabilityPolicy.BEST_EFFORT
        )
        qos_kwargs["durability"] = (
            DurabilityPolicy.TRANSIENT_LOCAL if durability == "transient_local" else DurabilityPolicy.VOLATILE
        )

        self._rclpy = rclpy
        self._msg_type = UInt8MultiArray
        self._encode = encode_interface_message
        self._decode = decode_interface_message
        self._qos_profile = qos_kwargs

        if not self._rclpy_is_initialized(rclpy):
            self._rclpy_init(rclpy)
            self._should_shutdown = True
        else:
            self._should_shutdown = False

        self._node = rclpy.create_node("cosmo_slam_ddf_peer_bus")
        configure_sim_time(self._node)
        self._QoSProfile = QoSProfile  # store type to avoid re-import per call

    def _topic_for_robot(self, rid: str) -> str:
        return f"{self._topic_prefix}/{rid}"

    def _spin_loop(self):
        if self._rclpy is None or self._node is None:
            return
        while not self._closed.is_set():
            try:
                self._rclpy.spin_once(self._node, timeout_sec=self._spin_timeout)
            except Exception as exc:
                self._logger.debug("ROS 2 spin_once failed: %s", exc)
                time.sleep(self._spin_timeout)
        try:
            self._rclpy.spin_once(self._node, timeout_sec=0.0)
        except Exception:
            pass

    def _ensure_subscription(self, rid: str) -> None:
        if self._node is None or self._msg_type is None:
            return
        with self._lock:
            if rid in self._subscriptions:
                return
            topic = self._topic_for_robot(rid)
            qos = self._QoSProfile(**self._qos_profile)

            def _callback(msg, *, robot_id=rid):
                self._on_message(robot_id, msg)

            subscription = self._node.create_subscription(self._msg_type, topic, _callback, qos)
            self._subscriptions[rid] = subscription
            self._mailboxes.setdefault(rid, deque())

    def _ensure_publisher(self, rid: str) -> object:
        if self._node is None or self._msg_type is None:
            raise RuntimeError("ROS 2 peer bus not initialised")
        with self._lock:
            publisher = self._publishers.get(rid)
        if publisher is not None:
            return publisher
        topic = self._topic_for_robot(rid)
        qos = self._QoSProfile(**self._qos_profile)
        publisher = self._node.create_publisher(self._msg_type, topic, qos)
        with self._lock:
            existing = self._publishers.get(rid)
            if existing is not None:
                try:
                    self._node.destroy_publisher(publisher)
                except Exception:
                    pass
                return existing
            self._publishers[rid] = publisher
        self._ensure_subscription(rid)
        return publisher

    # ------------------------------------------------------------------
    # PeerToPeerBus API implementation
    # ------------------------------------------------------------------
    def post(self, message: InterfaceMessage) -> None:
        if self._encode is None:
            raise RuntimeError("ROS 2 peer bus cannot encode interface messages")
        topic_name = self._topic_for_robot(str(message.receiver))
        publisher = self._ensure_publisher(str(message.receiver))
        payload = self._encode(message)
        ros_msg = self._msg_type()
        ros_msg.data = list(payload)
        # Impairments (drop/throttle) if configured
        if self._impair is not None:
            try:
                delay, reason = self._impair.on_send(
                    sender=str(message.sender),
                    receiver=str(message.receiver),
                    bytes_len=len(payload),
                    sent_wall_time=float(getattr(message, "sent_wall_time", 0.0) or time.time()),
                    topic=topic_name,
                )
                if delay > 0.0:
                    time.sleep(delay)
                if reason is not None:
                    return  # drop
            except Exception:
                pass
        try:
            publisher.publish(ros_msg)
        except Exception as exc:
            self._logger.warning("Failed to publish interface message to %s: %s", message.receiver, exc)

    def drain(self, robot_id: str) -> List[InterfaceMessage]:
        self._pump_events()
        with self._lock:
            mailbox = self._mailboxes.get(robot_id)
            if not mailbox:
                return []
            msgs = list(mailbox)
            self._delivered += len(msgs)
            mailbox.clear()
            return msgs

    @property
    def pending_counts(self) -> Dict[str, int]:
        with self._lock:
            return {rid: len(q) for rid, q in self._mailboxes.items() if q}

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        # Export impairment stats if any
        try:
            if getattr(self, "_impair", None) is not None:
                self._impair.export()
        except Exception:
            pass
        try:
            if self._node is not None:
                with self._lock:
                    for pub in self._publishers.values():
                        try:
                            self._node.destroy_publisher(pub)
                        except Exception:
                            pass
                    for sub in self._subscriptions.values():
                        try:
                            self._node.destroy_subscription(sub)
                        except Exception:
                            pass
                self._node.destroy_node()
        except Exception as exc:
            self._logger.debug("Failed to destroy ROS 2 peer bus node: %s", exc)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        try:
            if self._rclpy and getattr(self, "_should_shutdown", False) and self._rclpy_is_initialized(self._rclpy):
                self._rclpy_shutdown(self._rclpy)
        except Exception as exc:
            self._logger.debug("Failed to shutdown rclpy for peer bus: %s", exc)
        self._subscriptions.clear()
        self._publishers.clear()
        self._node = None
        self._rclpy = None

    # ------------------------------------------------------------------
    # ROS 2 utilities
    # ------------------------------------------------------------------
    def _pump_events(self) -> None:
        if self._rclpy is None or self._node is None:
            return
        try:
            self._rclpy.spin_once(self._node, timeout_sec=0.0)
        except Exception:
            pass

    def _on_message(self, robot_id: str, msg) -> None:
        if self._decode is None:
            return
        try:
            payload = bytes(getattr(msg, "data", []) or [])
            iface_msg = self._decode(payload)
        except Exception as exc:
            self._logger.warning("Failed to decode interface message for %s: %s", robot_id, exc)
            return
        with self._lock:
            self._mailboxes[robot_id].append(iface_msg)

    def _rclpy_is_initialized(self, module) -> bool:
        if hasattr(module, "is_initialized"):
            try:
                return bool(module.is_initialized())
            except TypeError:
                return bool(module.is_initialized(args=None))
        try:
            from rclpy.utilities import is_initialized as util_is_initialized  # type: ignore

            return bool(util_is_initialized())
        except Exception:
            return bool(getattr(module, "_initialized", False))

    def _rclpy_init(self, module) -> None:
        if hasattr(module, "init"):
            try:
                module.init()
                return
            except TypeError:
                module.init(args=None)
                return
            except Exception as exc:  # tolerate duplicate init in shared context
                try:
                    if "must only be called once" in str(exc).lower():
                        return
                except Exception:
                    pass
                try:
                    if getattr(module, "is_initialized", None) and module.is_initialized():
                        return
                except Exception:
                    pass
                raise
        from rclpy import init as rclpy_init  # type: ignore

        try:
            rclpy_init()
        except Exception as exc:
            if "must only be called once" not in str(exc).lower():
                raise

    def _rclpy_shutdown(self, module) -> None:
        if hasattr(module, "shutdown"):
            try:
                module.shutdown()
            except TypeError:
                module.shutdown(args=None)
            return
        from rclpy import shutdown as rclpy_shutdown  # type: ignore

        rclpy_shutdown()
