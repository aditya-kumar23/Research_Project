#!/usr/bin/env python3
"""ROS 2 publishers that replay COSMO JRL factors per robot."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Set

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String, UInt8MultiArray

from cosmo_slam_centralised.graph import default_robot_infer
from cosmo_slam_centralised.loader import (
    LoaderConfig,
    build_key_robot_map,
    iter_measurements,
    load_jrl,
    summarize_schema,
)
from cosmo_slam_centralised.models import BetweenFactorPose3, PriorFactorPose3
from cosmo_slam_ros2.factor_batch import encode_factor_batch
from cosmo_slam_ros2.qos import parse_qos_options
from cosmo_slam_ros2.impair import ImpairmentPolicy
from cosmo_slam_ros2.sim_time import configure_sim_time

Factor = PriorFactorPose3 | BetweenFactorPose3
logger = logging.getLogger("cosmo_slam_ros2.publisher")


class FactorReplayNode(Node):
    """ROS 2 node that publishes factor batches on per-robot topics."""

    def __init__(
        self,
        topic_prefix: str,
        robot_ids: Sequence[str],
        data_qos: QoSProfile,
        control_qos: QoSProfile | None = None,
    ) -> None:
        super().__init__("cosmo_slam_factor_replay")
        configure_sim_time(self)
        self._topic_prefix = topic_prefix.rstrip("/") or "/cosmo/factor_batch"
        self._data_qos = data_qos
        self._control_qos = control_qos or data_qos
        self._robot_publishers: Dict[str, object] = {}
        self._sent_batches: Dict[str, int] = {}
        self._sent_factors: Dict[str, int] = {}
        self._total_batches = 0
        self._total_factors = 0
        self._all_robot_ids: Set[str] = set()
        self._control_topic = f"{self._topic_prefix}/control"
        self._control_pub = self.create_publisher(String, self._control_topic, self._control_qos)
        for rid in sorted(set(robot_ids)):
            self._create_publisher(rid)
        # Optional impairments
        self._impair = ImpairmentPolicy.from_env() or ImpairmentPolicy(None)

    def topic_for_robot(self, rid: str) -> str:
        rid = str(rid)
        return f"{self._topic_prefix}/{rid}"

    @property
    def robot_ids(self) -> List[str]:
        return sorted(self._robot_publishers.keys())

    @property
    def all_robot_ids(self) -> List[str]:
        return sorted(self._all_robot_ids)

    def subscriber_count(self, rid: str) -> int:
        topic = self.topic_for_robot(rid)
        try:
            return int(self.count_subscribers(topic))
        except AttributeError:  # pragma: no cover - compatibility
            return 0

    def _create_publisher(self, rid: str):
        rid = str(rid)
        topic = self.topic_for_robot(rid)
        pub = self.create_publisher(UInt8MultiArray, topic, self._data_qos)
        self._robot_publishers[rid] = pub
        self._sent_batches.setdefault(rid, 0)
        self._sent_factors.setdefault(rid, 0)
        self._all_robot_ids.add(rid)
        logger.debug("Created publisher for robot %s on %s", rid, topic)
        return pub

    def publish_batch(self, rid: str, factors: List[Factor]) -> None:
        if not factors:
            return
        rid = str(rid)
        topic_path = self.topic_for_robot(rid)
        publisher = self._robot_publishers.get(rid) or self._create_publisher(rid)
        payload = encode_factor_batch(factors)
        msg = UInt8MultiArray()
        msg.data = list(payload)
        # Apply impairments per robot stream if configured
        if self._impair is not None:
            try:
                delay, reason = self._impair.on_send(
                    sender=rid,
                    receiver="central",
                    bytes_len=len(payload),
                    sent_wall_time=time.time(),
                    topic=topic_path,
                )
                if delay > 0.0:
                    _sleep_for(delay)
                if reason is not None:
                    return  # drop
            except Exception:
                pass
        publisher.publish(msg)  # type: ignore[attr-defined]
        sent = len(factors)
        self._sent_batches[rid] = self._sent_batches.get(rid, 0) + 1
        self._sent_factors[rid] = self._sent_factors.get(rid, 0) + sent
        self._total_batches += 1
        self._total_factors += sent
        if self._total_batches % 50 == 0:
            logger.info(
                "Published %d batches (%d factors)",
                self._total_batches,
                self._total_factors,
            )

    def publish_done(self, rid: str) -> None:
        msg = String()
        msg.data = f"DONE:{rid}"
        self._control_pub.publish(msg)

    def publish_done_all(self) -> None:
        msg = String()
        msg.data = "DONE_ALL"
        self._control_pub.publish(msg)


def _sleep_for(delta: float) -> None:
    if delta <= 0.0:
        return
    time.sleep(delta)


def _wait_for_subscribers(node: FactorReplayNode, robot_ids: Sequence[str], timeout: float) -> None:
    if timeout <= 0.0:
        logger.info("Subscriber wait disabled; streaming immediately")
        return
    remaining = set(robot_ids)
    start = time.time()
    logger.info(
        "Waiting up to %.1fs for subscribers on topics %s",
        timeout,
        [node.topic_for_robot(rid) for rid in remaining],
    )
    while remaining:
        missing = {rid for rid in list(remaining) if node.subscriber_count(rid) == 0}
        if not missing:
            logger.info("All subscribers detected; starting replay")
            return
        if time.time() - start >= timeout:
            logger.warning("Timed out waiting for subscribers on %s", sorted(missing))
            return
        rclpy.spin_once(node, timeout_sec=0.1)
        _sleep_for(0.1)
        remaining = missing


def _resolve_robot(factor: Factor, robot_map: Dict[str, str]) -> str:
    if isinstance(factor, PriorFactorPose3):
        rid = robot_map.get(str(factor.key))
        key_hint = factor.key
    else:
        rid = robot_map.get(str(factor.key1)) or robot_map.get(str(factor.key2))
        key_hint = factor.key1
    if rid:
        return str(rid)
    try:
        return str(default_robot_infer(str(key_hint)))
    except Exception:
        return "global"


def publish_dataset(
    node: FactorReplayNode,
    factors: Iterable[Factor],
    *,
    robot_map: Dict[str, str],
    batch_size: int,
    time_scale: float,
    max_sleep: float,
) -> None:
    batch_size = max(1, int(batch_size))
    buffers: Dict[str, List[Factor]] = defaultdict(list)
    last_stamp: float | None = None

    for factor in factors:
        stamp = float(getattr(factor, "stamp", 0.0))
        if last_stamp is None:
            last_stamp = stamp
        else:
            if time_scale > 0.0:
                dt = (stamp - last_stamp) / time_scale
                if max_sleep > 0.0:
                    dt = min(dt, max_sleep)
                _sleep_for(max(0.0, dt))
            last_stamp = stamp

        rid = _resolve_robot(factor, robot_map)
        buffer = buffers[rid]
        buffer.append(factor)
        if len(buffer) >= batch_size:
            node.publish_batch(rid, buffer)
            buffers[rid] = []

    for rid, remaining in buffers.items():
        if remaining:
            node.publish_batch(rid, remaining)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Publish COSMO JRL factors to ROS 2 topics.")
    ap.add_argument("--jrl", required=True, help="Path to the .jrl JSON dataset")
    ap.add_argument(
        "--topic",
        default="/cosmo/factor_batch",
        help="Topic prefix for factor batches (per robot topic = <prefix>/<robot>)",
    )
    ap.add_argument("--include-potential-outliers", action="store_true", help="Include potential outlier factors")
    ap.add_argument("--quat-order", choices=["wxyz", "xyzw"], default="wxyz", help="Quaternion order in file")
    ap.add_argument("--batch-size", type=int, default=1, help="Number of factors to bundle per publication")
    ap.add_argument(
        "--time-scale",
        type=float,
        default=0.0,
        help="Scale factor for dataset timestamps (0 = publish immediately). Use 1e9 for nanosecond stamps.",
    )
    ap.add_argument(
        "--max-sleep",
        type=float,
        default=0.0,
        help="Clamp per-factor sleep to this maximum in seconds (0 = no cap)",
    )
    ap.add_argument(
        "--idle-gap",
        type=float,
        default=1.0,
        help="Seconds to wait after each dataset iteration to keep publishers alive",
    )
    ap.add_argument(
        "--loop",
        type=int,
        default=1,
        help="Repeat the dataset this many times (0 = infinite looping)",
    )
    ap.add_argument(
        "--wait-subscriber-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for subscribers on all robot topics before streaming (0 = disable)",
    )
    ap.add_argument("--log", default="INFO", help="Logging level (DEBUG, INFO, ...)")
    ap.add_argument("--qos-reliability", choices=["reliable", "best_effort"], default="reliable")
    ap.add_argument("--qos-durability", choices=["volatile", "transient_local"], default="volatile")
    ap.add_argument("--qos-depth", type=int, default=10)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = LoaderConfig(
        quaternion_order=args.quat_order,
        validate_schema=True,
        include_potential_outliers=args.include_potential_outliers,
    )

    logger.info("Loading JRL dataset: %s", args.jrl)
    doc = load_jrl(args.jrl, cfg)
    logger.info("Schema: %s", json.dumps(summarize_schema(doc)))

    robot_map = build_key_robot_map(doc)
    robots = sorted(set(robot_map.values())) or ["global"]
    logger.info("Robots detected: %s", robots)

    qos_options = parse_qos_options(
        reliability=args.qos_reliability,
        durability=args.qos_durability,
        depth=args.qos_depth,
    )
    data_qos = QoSProfile(
        depth=int(qos_options["depth"]),
        reliability=(
            ReliabilityPolicy.RELIABLE
            if qos_options["reliability"] == "reliable"
            else ReliabilityPolicy.BEST_EFFORT
        ),
        durability=(
            DurabilityPolicy.TRANSIENT_LOCAL
            if qos_options["durability"] == "transient_local"
            else DurabilityPolicy.VOLATILE
        ),
    )
    control_qos = QoSProfile(
        depth=1,
        reliability=data_qos.reliability,
        durability=data_qos.durability,
    )

    time_scale = float(args.time_scale)
    max_sleep = max(0.0, float(args.max_sleep))
    idle_gap = max(0.0, float(args.idle_gap))
    loop = int(args.loop)
    wait_timeout = max(0.0, float(args.wait_subscriber_timeout))

    node: FactorReplayNode | None = None
    try:
        rclpy.init()
        node = FactorReplayNode(args.topic, robots, data_qos, control_qos)
        _wait_for_subscribers(node, robots, wait_timeout)

        iteration = 0
        while True:
            iteration += 1
            logger.info("Publishing dataset iteration %d", iteration)
            factors = iter_measurements(doc, cfg)
            publish_dataset(
                node,
                factors,
                robot_map=robot_map,
                batch_size=args.batch_size,
                time_scale=time_scale,
                max_sleep=max_sleep,
            )
            if loop != 0 and iteration >= loop:
                break
            if idle_gap > 0.0:
                logger.info("Idle gap %.2fs before next iteration", idle_gap)
                _sleep_for(idle_gap)

        # Notify subscribers that we are finished
        for rid in node.all_robot_ids:
            logger.info("Publishing completion signal for robot %s", rid)
            node.publish_done(rid)
        node.publish_done_all()
        if idle_gap > 0.0:
            logger.info("Final idle gap %.2fs to allow subscribers to finish", idle_gap)
            _sleep_for(idle_gap)

    except KeyboardInterrupt:
        logger.info("Interrupted; shutting down publisher")
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()
        # Export impairment stats if configured
        try:
            if getattr(node, "_impair", None) is not None:
                node._impair.export()
        except Exception:
            pass


if __name__ == "__main__":
    main()
