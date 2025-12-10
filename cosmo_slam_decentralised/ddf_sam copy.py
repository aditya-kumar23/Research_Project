from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import logging
import math
import os
import random
import time
import numpy as np

try:
    import gtsam
except Exception as exc:  # pragma: no cover - triggered when gtsam missing
    gtsam = None

from cosmo_slam_centralised.models import InitEntry, PriorFactorPose3, BetweenFactorPose3
from cosmo_slam_centralised.loader import LoaderConfig, load_jrl, iter_init_entries, iter_measurements, build_key_robot_map

from .partition import partition_measurements, RobotMeasurementBundle
from .communication import PeerToPeerBus
from .agents import DecentralizedRobotAgent
from .bandwidth import interface_message_bytes

from cosmo_slam_common.kpi_logging import KPILogger
from cosmo_slam_common.bandwidth import BandwidthTracker, factor_bytes
from cosmo_slam_common.latency import LatencyTracker

logger = logging.getLogger("cosmo_slam.decentralised.ddf")


_seed_env = os.environ.get("COSMO_RUN_SEED")
if _seed_env:
    try:
        _seed_val = int(_seed_env)
        random.seed(_seed_val)
        np.random.seed(_seed_val)
    except Exception:
        pass


@dataclass
class BackendResult:
    """Aggregated output of a DDF-SAM run."""

    estimates: Dict[str, gtsam.Values]
    iterations: int
    converged: bool


@dataclass
class PacingConfig:
    """Configuration for emulating dataset pacing using factor/interface stamps.

    The values mirror the CLI options exposed by the centralised ROS 2 replay
    tool where ``time_scale`` converts dataset stamp deltas to seconds (use
    ``1e9`` for nanosecond stamps).  A zero or negative ``time_scale`` disables
    pacing.  ``max_sleep`` optionally caps individual sleep intervals to avoid
    very large gaps when datasets contain discontinuities.
    """

    ingest_time_scale: float = 0.0
    ingest_max_sleep: float = 0.0
    interface_time_scale: float = 0.0
    interface_max_sleep: float = 0.0


class _StampPacer:
    """Helper that sleeps according to monotonically increasing stamp values."""

    def __init__(self, time_scale: float, max_sleep: float) -> None:
        self._time_scale = float(time_scale or 0.0)
        self._max_sleep = max(0.0, float(max_sleep or 0.0))
        self._last_stamp: Optional[float] = None

    @property
    def enabled(self) -> bool:
        return self._time_scale > 0.0

    def reset(self) -> None:
        self._last_stamp = None

    def sleep_for(self, stamp: float) -> None:
        if not self.enabled:
            return
        try:
            current = float(stamp)
        except (TypeError, ValueError):
            return
        if not math.isfinite(current):
            return
        if self._last_stamp is None:
            self._last_stamp = current
            return
        delta = current - self._last_stamp
        self._last_stamp = current
        if delta <= 0.0:
            return
        sleep_s = delta / self._time_scale
        if self._max_sleep > 0.0:
            sleep_s = min(sleep_s, self._max_sleep)
        if sleep_s > 0.0:
            time.sleep(sleep_s)

class DDFSAMBackend:
    """High-level orchestrator for the decentralised backend.

    Usage pattern:

    ```python
    backend = DDFSAMBackend.from_jrl(path_to_jrl)
    result = backend.run(max_rounds=5)
    values_per_robot = result.estimates
    ```
    """

    def __init__(
        self,
        robot_map: Dict[str, str],
        init_lookup: Dict[str, InitEntry],
        *,
        robust_kind: Optional[str] = None,
        robust_k: Optional[float] = None,
        batch_max_iters: int = 25,
        bootstrap_sigma: float = 1e1,
        kpi: Optional[KPILogger] = None,
        bandwidth: Optional[BandwidthTracker] = None,
        latency: Optional[LatencyTracker] = None,
        bus: Optional[PeerToPeerBus] = None,
        bus_factory: Optional[Callable[[Iterable[str]], PeerToPeerBus]] = None,
        pacing: Optional[PacingConfig] = None,
    ) -> None:
        if gtsam is None:
            raise RuntimeError("DDFSAMBackend requires the gtsam Python bindings")
        self.robot_map = dict(robot_map)
        self.init_lookup = {str(k): v for k, v in init_lookup.items()}
        self.robust_kind = robust_kind
        self.robust_k = robust_k
        self.batch_max_iters = batch_max_iters
        self.bootstrap_sigma = float(bootstrap_sigma)

        if bus is not None and bus_factory is not None:
            raise ValueError("Specify either bus or bus_factory, not both")
        self.bus: Optional[PeerToPeerBus] = bus
        self._bus_factory = bus_factory
        self.agents: Dict[str, DecentralizedRobotAgent] = {}
        self._bundles: Dict[str, RobotMeasurementBundle] = {}
        # KPI / instrumentation
        self.kpi = kpi
        self.bandwidth = bandwidth
        self.latency = latency
        self._ingest_event_ids: List[int] = []

        self.pacing = pacing or PacingConfig()
        self._ingest_pacer = _StampPacer(
            self.pacing.ingest_time_scale, self.pacing.ingest_max_sleep
        )
        self._interface_pacer = _StampPacer(
            self.pacing.interface_time_scale, self.pacing.interface_max_sleep
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_jrl(
        cls,
        path: str,
        cfg: Optional[LoaderConfig] = None,
        *,
        robust_kind: Optional[str] = None,
        robust_k: Optional[float] = None,
        batch_max_iters: int = 25,
        bootstrap_sigma: float = 1e1,
    ) -> "DDFSAMBackend":
        cfg = cfg or LoaderConfig()
        doc = load_jrl(path, cfg)
        robot_map = build_key_robot_map(doc)
        init_lookup: Dict[str, InitEntry] = {}
        for entry in iter_init_entries(doc, cfg):
            init_lookup[str(entry.key)] = entry
        factors = list(iter_measurements(doc, cfg))
        backend = cls(
            robot_map=robot_map,
            init_lookup=init_lookup,
            robust_kind=robust_kind,
            robust_k=robust_k,
            batch_max_iters=batch_max_iters,
            bootstrap_sigma=bootstrap_sigma,
        )
        backend.ingest_factors(factors)
        return backend

    def ingest_factors(self, factors: Iterable[BetweenFactorPose3 | PriorFactorPose3]) -> None:
        if self._ingest_pacer.enabled:
            self._ingest_pacer.reset()

            def _pace_in_order(stamp: float) -> None:
                # Pace using the original factor order so timing matches the dataset stream.
                try:
                    self._ingest_pacer.sleep_for(stamp)
                except Exception:
                    pass

            self._bundles = partition_measurements(factors, self.robot_map, on_factor=_pace_in_order)
        else:
            self._bundles = partition_measurements(factors, self.robot_map)
        all_robot_ids = set(self.robot_map.values()) | set(self._bundles.keys())
        for rid in sorted(all_robot_ids):
            agent = DecentralizedRobotAgent(
                rid,
                self.robot_map,
                self.init_lookup,
                robust_kind=self.robust_kind,
                robust_k=self.robust_k,
                batch_max_iters=self.batch_max_iters,
                bootstrap_sigma=self.bootstrap_sigma,
            )
            bundle = self._bundles.get(rid, RobotMeasurementBundle())
            agent.add_local_priors(bundle.priors)
            agent.add_local_between(bundle.local_between)
            for factor in bundle.inter_between:
                agent.add_inter_factor(factor)
            self.agents[rid] = agent
        if not self.agents:
            raise ValueError("No robots detected in dataset; cannot run decentralised backend")

        self._ensure_bus()

        # KPI + bandwidth + latency for sensor ingest
        def _factor_stamp(factor) -> float:
            try:
                return float(getattr(factor, "stamp", 0.0))
            except Exception:
                return 0.0

        if self.kpi or self.bandwidth or self.latency or self._ingest_pacer.enabled:
            for rid, bundle in self._bundles.items():
                # Priors
                for f in bundle.priors:
                    stamp = _factor_stamp(f)
                    topic = f"prior/{self.robot_map.get(str(f.key), rid)}"
                    size = factor_bytes(f)
                    if self.bandwidth:
                        self.bandwidth.add_uplink(topic, size)
                    if self.latency:
                        try:
                            ev_id = self.latency.record_ingest(
                                "PriorFactorPose3",
                                stamp,
                                self.robot_map.get(str(f.key), rid),
                                ingest_wall=time.time(),
                                ingest_mono=time.perf_counter(),
                                metadata={"key": str(f.key)},
                            )
                            self._ingest_event_ids.append(ev_id)
                        except Exception:
                            pass
                    if self.kpi:
                        try:
                            self.kpi.sensor_ingest(
                                "PriorFactorPose3",
                                stamp,
                                key=str(f.key),
                                topic=topic,
                                bytes=size,
                            )
                        except Exception:
                            pass
                # Local between
                for f in bundle.local_between:
                    stamp = _factor_stamp(f)
                    rid1 = self.robot_map.get(str(f.key1), rid)
                    rid2 = self.robot_map.get(str(f.key2), rid)
                    topic = f"between/{rid1}" if rid1 == rid2 else f"between/{rid1}-{rid2}"
                    size = factor_bytes(f)
                    if self.bandwidth:
                        self.bandwidth.add_uplink(topic, size)
                    if self.latency:
                        try:
                            ev_id = self.latency.record_ingest(
                                "BetweenFactorPose3",
                                stamp,
                                rid1,
                                ingest_wall=time.time(),
                                ingest_mono=time.perf_counter(),
                                metadata={"key1": str(f.key1), "key2": str(f.key2)},
                            )
                            self._ingest_event_ids.append(ev_id)
                        except Exception:
                            pass
                    if self.kpi:
                        try:
                            self.kpi.sensor_ingest(
                                "BetweenFactorPose3",
                                stamp,
                                key1=str(f.key1),
                                key2=str(f.key2),
                                topic=topic,
                                bytes=size,
                            )
                        except Exception:
                            pass
                # Inter between (owned by this robot)
                for f in bundle.inter_between:
                    stamp = _factor_stamp(f)
                    rid1 = self.robot_map.get(str(f.key1), rid)
                    rid2 = self.robot_map.get(str(f.key2), rid)
                    topic = f"between/{rid1}-{rid2}"
                    size = factor_bytes(f)
                    if self.bandwidth:
                        self.bandwidth.add_uplink(topic, size)
                    if self.latency:
                        try:
                            ev_id = self.latency.record_ingest(
                                "BetweenFactorPose3",
                                stamp,
                                rid1,
                                ingest_wall=time.time(),
                                ingest_mono=time.perf_counter(),
                                metadata={"key1": str(f.key1), "key2": str(f.key2)},
                            )
                            self._ingest_event_ids.append(ev_id)
                        except Exception:
                            pass
                    if self.kpi:
                        try:
                            self.kpi.sensor_ingest(
                                "BetweenFactorPose3",
                                stamp,
                                key1=str(f.key1),
                                key2=str(f.key2),
                                topic=topic,
                                bytes=size,
                            )
                        except Exception:
                            pass

    def _ensure_bus(self) -> None:
        if self.bus is not None:
            return
        if self._bus_factory is not None:
            self.bus = self._bus_factory(self.agents.keys())
        else:
            self.bus = PeerToPeerBus()

    # ------------------------------------------------------------------
    # Solver loop
    # ------------------------------------------------------------------
    def run(
        self,
        max_rounds: int = 5,
        convergence_tol: float = 5e-3,
        rotation_tol: float = 5e-3,
        relaxation_alpha: float = 1.0,
    ) -> BackendResult:
        if not self.agents:
            raise RuntimeError("No agents configured; call ingest_factors() first")

        if self.bus is None:
            self._ensure_bus()
        bus = self.bus
        if bus is None:
            raise RuntimeError("Message bus is not configured")

        # Initial local solve (round 0) to seed interface states
        initial_messages: List = []
        for agent in self.agents.values():
            agent.solve_round(iteration=0, kpi=self.kpi)
        for agent in self.agents.values():
            initial_messages.extend(agent.interface_messages(iteration=0))
        # Post initial messages and account for bandwidth
        initial_bytes = 0
        opt_end_wall = time.time()
        opt_end_mono = time.perf_counter()
        for msg in initial_messages:
            if self._interface_pacer.enabled:
                self._interface_pacer.sleep_for(getattr(msg, "stamp", 0.0))
            bus.post(msg)
            try:
                size = interface_message_bytes(msg)
                initial_bytes += size
                if self.bandwidth:
                    topic = f"iface/{msg.sender}-{msg.receiver}"
                    self.bandwidth.add_uplink(topic, size)
                if self.kpi:
                    self.kpi.map_broadcast(0, pose_count=None, topic=f"iface/{msg.sender}-{msg.receiver}", bytes=size)
            except Exception:
                pass

        broadcast_wall = time.time()
        broadcast_mono = time.perf_counter()

        if self.latency and self._ingest_event_ids:
            try:
                self.latency.assign_batch(0, self._ingest_event_ids)
                self.latency.complete_batch(0, opt_end_wall, opt_end_mono, broadcast_wall, broadcast_mono, initial_bytes)
            except Exception:
                pass

        last_state: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
        def _msg_translation(msg) -> np.ndarray:
            if hasattr(msg.translation, "to_numpy"):
                return msg.translation.to_numpy()
            return np.array([msg.translation.x, msg.translation.y, msg.translation.z], dtype=float)

        def _msg_quaternion(msg) -> np.ndarray:
            if hasattr(msg.rotation, "to_numpy"):
                quat = msg.rotation.to_numpy()
            else:
                quat = np.array([msg.rotation.w, msg.rotation.x, msg.rotation.y, msg.rotation.z], dtype=float)
            norm = np.linalg.norm(quat)
            if norm == 0.0:
                return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            return quat / norm

        def _msg_state(msg):
            return _msg_translation(msg), _msg_quaternion(msg)

        def _quat_angle(q1: np.ndarray, q2: np.ndarray) -> float:
            dot = float(np.dot(q1, q2))
            dot = max(-1.0, min(1.0, abs(dot)))
            return 2.0 * np.arccos(dot)

        if initial_messages:
            last_state = {
                (msg.sender, msg.key): _msg_state(msg)
                for msg in initial_messages
            }

        iterations_completed = 0
        converged = False

        relaxation_alpha = float(max(0.0, min(1.0, relaxation_alpha)))

        agent_items = list(self.agents.items())

        for iteration in range(1, max_rounds + 1):
            iterations_completed = iteration
            random.shuffle(agent_items)

            # Step 1: deliver messages from previous iteration (randomised order)
            for rid, agent in agent_items:
                incoming = bus.drain(rid)
                if incoming:
                    agent.receive_interface_messages(incoming, relaxation=relaxation_alpha)

            # Step 2: local optimisation with latest peer priors
            for _, agent in agent_items:
                agent.solve_round(iteration=iteration, kpi=self.kpi)

            # Step 3: collect outgoing interface summaries for the next round
            outgoing: List = []
            for _, agent in agent_items:
                outgoing.extend(agent.interface_messages(iteration=iteration))
            if not outgoing:
                converged = True
                break
            round_bytes = 0
            for msg in outgoing:
                if self._interface_pacer.enabled:
                    self._interface_pacer.sleep_for(getattr(msg, "stamp", 0.0))
                bus.post(msg)
                try:
                    size = interface_message_bytes(msg)
                    round_bytes += size
                    if self.bandwidth:
                        topic = f"iface/{msg.sender}-{msg.receiver}"
                        self.bandwidth.add_uplink(topic, size)
                    if self.kpi:
                        self.kpi.map_broadcast(iteration, pose_count=None, topic=f"iface/{msg.sender}-{msg.receiver}", bytes=size)
                except Exception:
                    pass

            state = {
                (msg.sender, msg.key): _msg_state(msg)
                for msg in outgoing
            }
            if last_state:
                trans_deltas = []
                rot_deltas = []
                for key, (vec, quat) in state.items():
                    prev = last_state.get(key)
                    if prev is None:
                        continue
                    prev_vec, prev_quat = prev
                    trans_deltas.append(np.linalg.norm(vec - prev_vec))
                    rot_deltas.append(_quat_angle(quat, prev_quat))
                if self.kpi and trans_deltas and rot_deltas:
                    try:
                        self.kpi._emit(
                            "ddf_round_delta",
                            iteration=iteration,
                            max_translation_delta=float(max(trans_deltas)),
                            max_rotation_delta=float(max(rot_deltas)),
                        )
                    except Exception:
                        pass
                if trans_deltas and rot_deltas:
                    if max(trans_deltas) < convergence_tol and max(rot_deltas) < rotation_tol:
                        converged = True
                        break
            last_state = state

        estimates = {rid: agent.estimate_snapshot() for rid, agent in self.agents.items()}
        return BackendResult(estimates=estimates, iterations=iterations_completed, converged=converged)
