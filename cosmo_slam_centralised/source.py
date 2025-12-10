"""Factor source abstractions for centralised ingest pipelines."""

from __future__ import annotations

import logging
import queue
import threading
import time
from contextlib import AbstractContextManager
from functools import partial
from typing import Dict, Iterable, Iterator, List, Optional, Set, Union

from .models import PriorFactorPose3, BetweenFactorPose3
from .loader import iter_measurements, LoaderConfig, JRLDocument
from cosmo_slam_ros2.sim_time import configure_sim_time

Factor = Union[PriorFactorPose3, BetweenFactorPose3]

logger = logging.getLogger("cosmo_slam.source")


class FactorSourceError(RuntimeError):
    """Raised when a factor source cannot be constructed or started."""


class FactorSource(AbstractContextManager):
    """Base class for factor providers used by the centralised backend."""

    def iter_factors(self) -> Iterable[Factor]:  # pragma: no cover - interface method
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - default no-op
        return None

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close()
        finally:
            return False


class JRLFactorSource(FactorSource):
    """Wrap ``iter_measurements`` so it matches the factor source interface."""

    def __init__(self, doc: JRLDocument, cfg: LoaderConfig):
        self._doc = doc
        self._cfg = cfg

    def iter_factors(self) -> Iterable[Factor]:
        return iter_measurements(self._doc, self._cfg)


class ROS2FactorSource(FactorSource):
    """Subscribe to per-robot ROS 2 topics and decode factor batches on the fly."""

    def __init__(
        self,
        topic_prefix: str,
        robot_ids: Iterable[str],
        qos_profile: Dict[str, object],
        *,
        queue_size: int = 0,
        spin_timeout: float = 0.1,
        idle_timeout: float = 0.0,
    ) -> None:
        self._topic_prefix = (topic_prefix or "/cosmo/factor_batch").rstrip("/") or "/cosmo/factor_batch"
        self._robot_ids = sorted({str(rid) for rid in robot_ids}) or ["global"]
        self._active_publishers: Set[str] = set(self._robot_ids)
        self._done_publishers: Set[str] = set()
        self._qos_profile = dict(qos_profile)
        self._queue: "queue.Queue" = queue.Queue(maxsize=queue_size if queue_size > 0 else 0)
        self._queue_maxsize = queue_size if queue_size > 0 else 0
        self._spin_timeout = max(spin_timeout, 0.01)
        self._idle_timeout = max(idle_timeout, 0.0)
        self._closed = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._message_count = 0
        self._should_shutdown = False
        self._last_message_time: Optional[float] = None

        # Runtime handles
        self._node = None
        self._subscriptions: List[object] = []
        self._control_subscription = None
        self._decode = None
        self._rclpy = None

        self._control_topic = f"{self._topic_prefix}/control"

    def _topic_for_robot(self, rid: str) -> str:
        return f"{self._topic_prefix}/{rid}"

    # ------------------------------------------------------------------
    # ROS 2 bootstrap helpers
    # ------------------------------------------------------------------
    def _ensure_rclpy(self):
        try:
            import rclpy  # type: ignore
            from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy  # type: ignore
            from std_msgs.msg import UInt8MultiArray, String  # type: ignore
        except Exception as exc:  # pragma: no cover - requires ROS 2 runtime
            raise FactorSourceError(
                "ROS 2 transport requested but rclpy/std_msgs are not available"
            ) from exc

        from cosmo_slam_ros2.factor_batch import decode_factor_batch

        qos_kwargs = dict(depth=int(self._qos_profile.get("depth", 10)))
        reliability = self._qos_profile.get("reliability", "reliable").lower()
        durability = self._qos_profile.get("durability", "volatile").lower()
        qos_kwargs["reliability"] = (
            ReliabilityPolicy.RELIABLE if reliability == "reliable" else ReliabilityPolicy.BEST_EFFORT
        )
        qos_kwargs["durability"] = (
            DurabilityPolicy.TRANSIENT_LOCAL if durability == "transient_local" else DurabilityPolicy.VOLATILE
        )
        data_qos = QoSProfile(**qos_kwargs)
        control_qos = QoSProfile(
            depth=1,
            reliability=data_qos.reliability,
            durability=data_qos.durability,
        )
        return rclpy, UInt8MultiArray, String, data_qos, control_qos, decode_factor_batch

    def _spin_loop(self, rclpy_module):  # pragma: no cover - requires ROS 2 runtime
        while not self._closed.is_set():
            rclpy_module.spin_once(self._node, timeout_sec=self._spin_timeout)
        # Drain remaining events to flush callbacks
        rclpy_module.spin_once(self._node, timeout_sec=0.0)

    # ------------------------------------------------------------------
    # FactorSource API
    # ------------------------------------------------------------------
    def __enter__(self):
        (
            rclpy,
            UInt8MultiArray,
            StringMsg,
            data_qos,
            control_qos,
            decode_factor_batch,
        ) = self._ensure_rclpy()
        self._rclpy = rclpy

        if not self._rclpy_is_initialized(rclpy):
            self._rclpy_init(rclpy)
            self._should_shutdown = True

        self._decode = decode_factor_batch
        self._node = rclpy.create_node("cosmo_slam_factor_listener")
        configure_sim_time(self._node)

        for rid in self._robot_ids:
            topic = self._topic_for_robot(rid)
            subscription = self._node.create_subscription(
                UInt8MultiArray,
                topic,
                partial(self._on_message, rid),
                data_qos,
            )
            self._subscriptions.append(subscription)
            logger.debug("Subscribed to factor topic %s", topic)

        self._control_subscription = self._node.create_subscription(
            StringMsg,
            self._control_topic,
            self._on_control,
            control_qos,
        )
        logger.debug("Subscribed to control topic %s", self._control_topic)

        self._thread = threading.Thread(target=self._spin_loop, args=(rclpy,), daemon=True)
        self._thread.start()
        return self

    def iter_factors(self) -> Iterator[Factor]:
        if self._node is None:
            raise FactorSourceError("ROS2FactorSource must be entered before iteration")
        while True:
            if self._closed.is_set() and self._queue.empty():
                break
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._should_stop():
                    break
                continue
            yield item
            self._queue.task_done()
        logger.info(
            "ROS2 factor source drained: messages=%d, robots=%s",
            self._message_count,
            sorted(self._done_publishers),
        )

    def _should_stop(self) -> bool:
        if not self._active_publishers and self._queue.empty():
            logger.info("All publishers finished; closing factor source")
            self.close()
            return True
        if (
            self._idle_timeout > 0.0
            and self._message_count > 0
            and self._last_message_time is not None
            and (time.monotonic() - self._last_message_time) > self._idle_timeout
        ):
            logger.info("No factor batches received for %.2fs; closing source", self._idle_timeout)
            self.close()
            return True
        return False

    def _on_message(self, rid: str, msg):  # pragma: no cover - requires ROS 2 runtime
        if self._closed.is_set():
            return
        self._message_count += 1
        try:
            payload = bytes(msg.data)
            factors = self._decode(payload)
        except Exception as exc:
            logger.warning("Failed to decode factor batch %d from %s: %s", self._message_count, rid, exc)
            return

        self._last_message_time = time.monotonic()
        topic = self._topic_for_robot(rid)
        msg_bytes = len(payload)
        # Annotate each factor with ROS2 transport context for byte-accurate
        # bandwidth accounting downstream.
        for factor in factors:
            try:
                setattr(factor, "__ros2_topic", topic)
                setattr(factor, "__ros2_msg_bytes", msg_bytes)
            except Exception:
                pass
            while True:
                try:
                    self._queue.put(factor, timeout=0.1)
                    break
                except queue.Full:
                    logger.warning(
                        "Factor queue full (maxsize=%s); dropping remaining factors from %s",
                        self._queue_maxsize if self._queue_maxsize else "unbounded",
                        rid,
                    )
                    return

    def _on_control(self, msg):  # pragma: no cover - requires ROS 2 runtime
        if self._closed.is_set():
            return
        data = (getattr(msg, "data", "") or "").strip()
        if not data:
            return
        if data == "DONE_ALL":
            logger.info("Received DONE_ALL control signal")
            self._done_publishers.update(self._active_publishers)
            self._active_publishers.clear()
            return
        if data.startswith("DONE:"):
            rid = data.split(":", 1)[1]
            if rid in self._active_publishers:
                self._active_publishers.discard(rid)
                self._done_publishers.add(rid)
                logger.info(
                    "Received completion from %s (remaining=%s)",
                    rid,
                    sorted(self._active_publishers),
                )
            else:
                logger.debug("Received DONE for %s but it was not registered", rid)

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            if self._node is not None:
                for sub in self._subscriptions:
                    try:
                        self._node.destroy_subscription(sub)
                    except Exception:
                        pass
                if self._control_subscription is not None:
                    try:
                        self._node.destroy_subscription(self._control_subscription)
                    except Exception:
                        pass
                self._node.destroy_node()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to destroy ROS 2 node: %s", exc)
        try:
            if self._rclpy and self._should_shutdown and self._rclpy_is_initialized(self._rclpy):
                self._rclpy_shutdown(self._rclpy)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to shutdown rclpy: %s", exc)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        self._node = None
        self._subscriptions = []
        self._control_subscription = None
        self._decode = None
        self._rclpy = None

    # ------------------------------------------------------------------
    # rclpy compatibility helpers
    # ------------------------------------------------------------------
    def _rclpy_is_initialized(self, module) -> bool:
        if hasattr(module, "is_initialized"):
            try:
                return bool(module.is_initialized())
            except TypeError:  # pragma: no cover - API variant accepting args
                return bool(module.is_initialized(args=None))
        try:  # pragma: no cover - exercised if utilities mode available
            from rclpy.utilities import is_initialized as util_is_initialized  # type: ignore

            return bool(util_is_initialized())
        except Exception:
            return bool(getattr(module, "_initialized", False))

    def _rclpy_init(self, module) -> None:
        # Be defensive: some environments report not-initialized yet still throw
        # when init() is called twice. Treat such errors as benign.
        if hasattr(module, "init"):
            try:
                module.init()
                return
            except TypeError:  # pragma: no cover - API variant accepting args
                module.init(args=None)
                return
            except Exception as exc:  # pragma: no cover - robust to duplicate init
                try:
                    if "must only be called once" in str(exc).lower():
                        return
                except Exception:
                    pass
                # As a last resort, if rclpy now reports initialized, continue
                try:
                    if getattr(module, "is_initialized", None):
                        if bool(module.is_initialized()):
                            return
                except Exception:
                    pass
                raise FactorSourceError(f"Unable to initialise rclpy: {exc}")
        try:  # pragma: no cover
            from rclpy import init as rclpy_init  # type: ignore

            try:
                rclpy_init()
            except Exception as exc:  # tolerate duplicate init
                if "must only be called once" not in str(exc).lower():
                    raise
        except Exception as exc:
            raise FactorSourceError(f"Unable to initialise rclpy: {exc}")

    def _rclpy_shutdown(self, module) -> None:
        if hasattr(module, "shutdown"):
            module.shutdown()
            return
        try:  # pragma: no cover
            from rclpy import shutdown as rclpy_shutdown  # type: ignore

            rclpy_shutdown()
        except Exception as exc:
            logger.debug("rclpy shutdown fallback failed: %s", exc)
