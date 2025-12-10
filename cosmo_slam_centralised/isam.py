from typing import Dict, Optional, Iterable, TYPE_CHECKING, List
import logging
import time
import math
try:
    import gtsam
except Exception:
    gtsam = None

from .models import PriorFactorPose3, BetweenFactorPose3, InitEntry
from .graph import GraphBuilder
from .bandwidth import BandwidthTracker, factor_bytes, map_payload_bytes
from .latency import LatencyTracker
try:
    from cosmo_slam_ros2.ack import Ros2AckPublisher  # type: ignore
except Exception:
    Ros2AckPublisher = None  # type: ignore

if TYPE_CHECKING:
    from .kpi_logging import KPILogger

logger = logging.getLogger("cosmo_slam.isam")


def _update_translation_cache(cache: Dict[int, tuple], estimate: "gtsam.Values") -> float:
    """Update translation cache and return max Euclidean delta between estimates."""
    if gtsam is None or estimate is None:
        return 0.0
    max_delta = 0.0
    try:
        keys = list(estimate.keys())
    except Exception:
        keys = []
    for key in keys:
        try:
            pose = estimate.atPose3(key)
        except Exception:
            continue
        trans = pose.translation()
        try:
            tx, ty, tz = float(trans.x()), float(trans.y()), float(trans.z())
        except Exception:
            tx = float(trans[0]); ty = float(trans[1]); tz = float(trans[2])
        prev = cache.get(int(key))
        if prev is not None:
            dx = tx - prev[0]
            dy = ty - prev[1]
            dz = tz - prev[2]
            delta = math.sqrt(dx * dx + dy * dy + dz * dz)
            if delta > max_delta:
                max_delta = delta
        cache[int(key)] = (tx, ty, tz)
    return max_delta


class ISAM2Manager:
    """Thin manager around GTSAM's iSAM2 (API-compatible across wheels)."""

    def __init__(self,
                 relinearize_threshold: float = 0.1,
                 relinearize_skip: int = 10,
                 cache_linearized: bool = True):
        if gtsam is None:
            raise RuntimeError("GTSAM not available; cannot run iSAM2")
        params = gtsam.ISAM2Params()

        # Compat helpers (some wheels use setters, others properties)
        def _set(obj, prop: str, value, setter: Optional[str] = None):
            if hasattr(obj, prop):
                try:
                    setattr(obj, prop, value); return
                except Exception:
                    pass
            if setter and hasattr(obj, setter):
                getattr(obj, setter)(value)

        _set(params, "relinearizeThreshold", relinearize_threshold, "setRelinearizeThreshold")
        _set(params, "relinearizeSkip",      relinearize_skip,      "setRelinearizeSkip")
        _set(params, "cacheLinearizedFactors", cache_linearized,    "setCacheLinearizedFactors")
        _set(params, "enableRelinearization", True,                 "setEnableRelinearization")

        self.isam = gtsam.ISAM2(params)
        self._estimate = gtsam.Values()

    def update(self, graph: "gtsam.NonlinearFactorGraph", initial: "gtsam.Values"):
        self.isam.update(graph, initial)
        self._estimate = self.isam.calculateEstimate()

    @property
    def estimate(self) -> "gtsam.Values":
        return self._estimate

    def error(self, graph: "gtsam.NonlinearFactorGraph", estimate: Optional["gtsam.Values"] = None) -> float:
        est = estimate or self._estimate
        return graph.error(est)


def incremental_optimize(
    factors: Iterable,
    init_lookup: Dict[str, InitEntry],
    graph_builder: GraphBuilder,
    isam: ISAM2Manager,
    batch_size: int = 100,
    kpi: Optional["KPILogger"] = None,
    bandwidth: Optional[BandwidthTracker] = None,
    latency: Optional[LatencyTracker] = None,
    ack_publisher: object = None,
):
    """Update iSAM2 with only-new factors/keys per batch."""
    added = 0
    batch_idx = 0
    batch_event_ids: List[int] = []
    # Track ROS 2 batch messages already accounted for to avoid double counting
    seen_ros_msgs = set()
    translation_cache: Dict[int, tuple] = {}
    for f in factors:
        if isinstance(f, PriorFactorPose3):
            graph_builder.add_prior(f, init_lookup)
            topic = None
            event_id = None
            if bandwidth:
                ros_msg_id = getattr(f, "__ros2_msg_id", None)
                ros_topic = getattr(f, "__ros2_topic", None)
                ros_bytes = getattr(f, "__ros2_msg_bytes", None)
                if ros_msg_id and ros_topic and isinstance(ros_bytes, int) and ros_msg_id not in seen_ros_msgs:
                    seen_ros_msgs.add(ros_msg_id)
                    topic = ros_topic
                    size = int(ros_bytes)
                    bandwidth.add_uplink(topic, size)
                else:
                    topic = f"prior/{graph_builder.robot_of(f.key)}"
                    size = factor_bytes(f)
                    bandwidth.add_uplink(topic, size)
            if latency:
                meta_extra = {}
                # Propagate ROS2 message metadata for ACKs and accounting
                if hasattr(f, "__ros2_msg_id"):
                    meta_extra["ros2_msg_id"] = getattr(f, "__ros2_msg_id")
                if hasattr(f, "__ros2_topic"):
                    meta_extra["ros2_topic"] = getattr(f, "__ros2_topic")
                if hasattr(f, "__ros2_msg_bytes"):
                    try:
                        meta_extra["ros2_msg_bytes"] = int(getattr(f, "__ros2_msg_bytes"))
                    except Exception:
                        pass
                event_id = latency.record_ingest(
                    "PriorFactorPose3",
                    float(getattr(f, "stamp", 0.0)),
                    graph_builder.robot_of(f.key),
                    ingest_wall=time.time(),
                    ingest_mono=time.perf_counter(),
                    metadata={
                        "key": str(f.key),
                        # Propagate ROS2 producer timestamps if present
                        **({"send_ts_mono": float(getattr(f, "send_ts_mono", None))} if getattr(f, "send_ts_mono", None) is not None else {}),
                        **({"send_ts_wall": float(getattr(f, "send_ts_wall", None))} if getattr(f, "send_ts_wall", None) is not None else {}),
                        **meta_extra,
                    },
                )
                batch_event_ids.append(event_id)
            if kpi:
                kpi.sensor_ingest(
                    "PriorFactorPose3",
                    f.stamp,
                    key=str(f.key),
                    topic=topic,
                    bytes=size if bandwidth else None,
                )
        elif isinstance(f, BetweenFactorPose3):
            graph_builder.add_between(f, init_lookup)
            topic = None
            size = None
            event_id = None
            if bandwidth:
                ros_msg_id = getattr(f, "__ros2_msg_id", None)
                ros_topic = getattr(f, "__ros2_topic", None)
                ros_bytes = getattr(f, "__ros2_msg_bytes", None)
                if ros_msg_id and ros_topic and isinstance(ros_bytes, int) and ros_msg_id not in seen_ros_msgs:
                    seen_ros_msgs.add(ros_msg_id)
                    topic = ros_topic
                    size = int(ros_bytes)
                    bandwidth.add_uplink(topic, size)
                else:
                    rid1 = graph_builder.robot_of(f.key1)
                    rid2 = graph_builder.robot_of(f.key2)
                    if rid1 == rid2:
                        topic = f"between/{rid1}"
                    else:
                        topic = f"between/{rid1}-{rid2}"
                    size = factor_bytes(f)
                    bandwidth.add_uplink(topic, size)
            if latency:
                rid1 = graph_builder.robot_of(f.key1)
                meta_extra = {}
                if hasattr(f, "__ros2_msg_id"):
                    meta_extra["ros2_msg_id"] = getattr(f, "__ros2_msg_id")
                if hasattr(f, "__ros2_topic"):
                    meta_extra["ros2_topic"] = getattr(f, "__ros2_topic")
                if hasattr(f, "__ros2_msg_bytes"):
                    try:
                        meta_extra["ros2_msg_bytes"] = int(getattr(f, "__ros2_msg_bytes"))
                    except Exception:
                        pass
                event_id = latency.record_ingest(
                    "BetweenFactorPose3",
                    float(getattr(f, "stamp", 0.0)),
                    rid1,
                    ingest_wall=time.time(),
                    ingest_mono=time.perf_counter(),
                    metadata={
                        "key1": str(f.key1),
                        "key2": str(f.key2),
                        **({"send_ts_mono": float(getattr(f, "send_ts_mono", None))} if getattr(f, "send_ts_mono", None) is not None else {}),
                        **({"send_ts_wall": float(getattr(f, "send_ts_wall", None))} if getattr(f, "send_ts_wall", None) is not None else {}),
                        **meta_extra,
                    },
                )
                batch_event_ids.append(event_id)
            if kpi:
                kpi.sensor_ingest(
                    "BetweenFactorPose3",
                    f.stamp,
                    key1=str(f.key1),
                    key2=str(f.key2),
                    topic=topic,
                    bytes=size if bandwidth else None,
                )
        else:
            continue

        added += 1
        if added % batch_size == 0:
            bg, bv = graph_builder.pop_batch()
            if bg.size() > 0 or bv.size() > 0:
                batch_idx += 1
                events_for_batch = batch_event_ids
                batch_event_ids = []
                if latency:
                    latency.assign_batch(batch_idx, events_for_batch)
                if kpi:
                    kpi.optimization_start(batch_idx, bg.size(), added)
                update_start_wall = time.time()
                update_start = time.perf_counter()
                # Mark 'use' for E2E send→use
                if latency:
                    latency.mark_use(batch_idx, update_start_wall, update_start)
                isam.update(bg, bv)
                opt_end_mono = time.perf_counter()
                opt_end_wall = time.time()
                duration = opt_end_mono - update_start
                estimate = isam.estimate
                max_delta = _update_translation_cache(translation_cache, estimate)
                updated = estimate.size() if hasattr(estimate, "size") else None
                if kpi:
                    kpi.optimization_end(
                        batch_idx,
                        duration,
                        updated_keys=updated,
                        max_translation_delta=max_delta,
                    )
                # Publish ACKs for ROS2 factor batches used in this optimization
                if ack_publisher is not None and latency is not None:
                    try:
                        # Gather unique ROS2 messages from this batch
                        ids = set()
                        for ev_id in events_for_batch:
                            ev = latency.events[ev_id]
                            msg_id = ev.get("ros2_msg_id")
                            rid = ev.get("robot")
                            if not msg_id or not rid:
                                continue
                            if msg_id in ids:
                                continue
                            ids.add(msg_id)
                            ack_publisher.publish_ack(
                                rid,
                                message_id=str(msg_id),
                                send_ts_mono=ev.get("send_ts_mono"),
                                send_ts_wall=ev.get("send_ts_wall"),
                                use_ts_mono=update_start,
                                use_ts_wall=update_start_wall,
                                bytes=ev.get("ros2_msg_bytes"),
                            )
                    except Exception:
                        pass
                payload_bytes = None
                if bandwidth and updated is not None:
                    down_topic = "map_broadcast"
                    payload_bytes = map_payload_bytes(updated)
                    bandwidth.add_downlink(down_topic, payload_bytes)
                broadcast_wall = time.time()
                broadcast_mono = time.perf_counter()
                if latency:
                    latency.complete_batch(batch_idx, opt_end_wall, opt_end_mono, broadcast_wall, broadcast_mono, payload_bytes)
                if bandwidth and updated is not None:
                    if kpi:
                        kpi.map_broadcast(batch_idx, pose_count=updated, topic=down_topic, bytes=payload_bytes)
                elif kpi:
                    kpi.map_broadcast(batch_idx, pose_count=updated)

    if added % batch_size != 0:
        bg, bv = graph_builder.pop_batch()
        if bg.size() > 0 or bv.size() > 0:
            batch_idx += 1
            events_for_batch = batch_event_ids
            batch_event_ids = []
            if latency:
                latency.assign_batch(batch_idx, events_for_batch)
            if kpi:
                kpi.optimization_start(batch_idx, bg.size(), added)
            update_start_wall = time.time()
            update_start = time.perf_counter()
            if latency:
                latency.mark_use(batch_idx, update_start_wall, update_start)
            isam.update(bg, bv)
            opt_end_mono = time.perf_counter()
            opt_end_wall = time.time()
            duration = opt_end_mono - update_start
            estimate = isam.estimate
            max_delta = _update_translation_cache(translation_cache, estimate)
            updated = estimate.size() if hasattr(estimate, "size") else None
            if kpi:
                kpi.optimization_end(
                    batch_idx,
                    duration,
                    updated_keys=updated,
                    max_translation_delta=max_delta,
                )
            if ack_publisher is not None and latency is not None:
                try:
                    ids = set()
                    for ev_id in events_for_batch:
                        ev = latency.events[ev_id]
                        msg_id = ev.get("ros2_msg_id")
                        rid = ev.get("robot")
                        if not msg_id or not rid:
                            continue
                        if msg_id in ids:
                            continue
                        ids.add(msg_id)
                        ack_publisher.publish_ack(
                            rid,
                            message_id=str(msg_id),
                            send_ts_mono=ev.get("send_ts_mono"),
                            send_ts_wall=ev.get("send_ts_wall"),
                            use_ts_mono=update_start,
                            use_ts_wall=update_start_wall,
                            bytes=ev.get("ros2_msg_bytes"),
                        )
                except Exception:
                    pass
            payload_bytes = None
            if bandwidth and updated is not None:
                down_topic = "map_broadcast"
                payload_bytes = map_payload_bytes(updated)
                bandwidth.add_downlink(down_topic, payload_bytes)
            broadcast_wall = time.time()
            broadcast_mono = time.perf_counter()
            if latency:
                latency.complete_batch(batch_idx, opt_end_wall, opt_end_mono, broadcast_wall, broadcast_mono, payload_bytes)
            if bandwidth and updated is not None:
                if kpi:
                    kpi.map_broadcast(batch_idx, pose_count=updated, topic=down_topic, bytes=payload_bytes)
            elif kpi:
                kpi.map_broadcast(batch_idx, pose_count=updated)

    return isam.estimate


def optimize_all_batch(graph: "gtsam.NonlinearFactorGraph",
                       initial: "gtsam.Values",
                       max_iters: int = 100,
                       kpi: Optional["KPILogger"] = None,
                       bandwidth: Optional[BandwidthTracker] = None,
                       latency: Optional[LatencyTracker] = None,
                       batch_id: int = 1,
                       translation_cache: Optional[Dict[int, tuple]] = None) -> "gtsam.Values":
    """Levenberg–Marquardt batch solve (fallback for unstable iSAM2 wheels)."""
    params = gtsam.LevenbergMarquardtParams()
    params.setlambdaInitial(1e-3)
    params.setMaxIterations(max_iters)
    if kpi:
        kpi.optimization_start(batch_id, graph.size(), graph.size())
    update_start = time.perf_counter()
    opt = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    estimate = opt.optimize()
    opt_end_mono = time.perf_counter()
    opt_end_wall = time.time()
    duration = opt_end_mono - update_start
    cache = translation_cache if translation_cache is not None else {}
    max_delta = _update_translation_cache(cache, estimate)
    updated = estimate.size() if hasattr(estimate, "size") else None
    if kpi:
        kpi.optimization_end(
            batch_id,
            duration,
            updated_keys=updated,
            max_translation_delta=max_delta,
        )
    payload_bytes = None
    if bandwidth and updated is not None:
        payload_bytes = map_payload_bytes(updated)
        bandwidth.add_downlink("map_broadcast", payload_bytes)
    broadcast_wall = time.time()
    broadcast_mono = time.perf_counter()
    if latency:
        latency.complete_batch(batch_id, opt_end_wall, opt_end_mono, broadcast_wall, broadcast_mono, payload_bytes)
    if bandwidth and updated is not None:
        if kpi:
            kpi.map_broadcast(batch_id, pose_count=updated, topic="map_broadcast", bytes=payload_bytes)
    elif kpi:
        kpi.map_broadcast(batch_id, pose_count=updated)
    return estimate
