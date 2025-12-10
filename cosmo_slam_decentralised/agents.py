from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple, Union
from collections import defaultdict, deque
import logging
import math
import numpy as np

try:
    import gtsam
except Exception as exc:  # pragma: no cover - exercised only when gtsam missing
    gtsam = None

from cosmo_slam_centralised.models import (
    PriorFactorPose3,
    BetweenFactorPose3,
    InitEntry,
    Quaternion,
    Translation,
)
from cosmo_slam_centralised.graph import GraphBuilder, default_robot_infer
from cosmo_slam_centralised.isam import optimize_all_batch

from .communication import InterfaceMessage
from cosmo_slam_common.kpi_logging import KPILogger

logger = logging.getLogger("cosmo_slam.decentralised.agent")


@dataclass
class InterfaceEdge:
    """Book-keeping wrapper describing a single inter-robot loop closure."""

    factor: BetweenFactorPose3
    local_key: str
    remote_key: str
    remote_robot: str
    owns_factor: bool


def _rot3_to_quat_wxyz(rot: "gtsam.Rot3") -> Tuple[float, float, float, float]:
    if hasattr(rot, "quaternion"):
        q = rot.quaternion()
        return float(q[0]), float(q[1]), float(q[2]), float(q[3])
    if hasattr(rot, "toQuaternion"):
        try:
            q = rot.toQuaternion()
            return float(q[0]), float(q[1]), float(q[2]), float(q[3])
        except Exception:
            pass
    M = rot.matrix()
    m00, m01, m02 = float(M[0, 0]), float(M[0, 1]), float(M[0, 2])
    m10, m11, m12 = float(M[1, 0]), float(M[1, 1]), float(M[1, 2])
    m20, m21, m22 = float(M[2, 0]), float(M[2, 1]), float(M[2, 2])

    tr = m00 + m11 + m22
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S

    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) or 1.0
    return qw / n, qx / n, qy / n, qz / n


def _pose3_to_components(pose: gtsam.Pose3) -> Tuple[Quaternion, Translation]:
    R = pose.rotation()
    qw, qx, qy, qz = _rot3_to_quat_wxyz(R)
    t = pose.translation()
    try:
        tx, ty, tz = float(t.x()), float(t.y()), float(t.z())
    except AttributeError:
        tx, ty, tz = map(float, t)
    return Quaternion(qw, qx, qy, qz), Translation(tx, ty, tz)


class DecentralizedRobotAgent:
    """Maintains a per-robot graph and emits interface summaries (DDF-SAM style).

    The agent owns the factors whose endpoints include at least one pose tied to
    `robot_id`.  Intra-robot factors are solved locally.  For inter-robot loop
    closures the agent inserts a strong prior on the peer's interface pose using
    the most recent message received over the peer-to-peer bus.  This mirrors the
    separator summarisation step in DDF-SAM where robots exchange linearisation
    points and marginal covariances for shared variables.
    """

    def __init__(
        self,
        robot_id: str,
        robot_map: Dict[str, str],
        init_lookup: Dict[str, InitEntry],
        *,
        robust_kind: Optional[str] = None,
        robust_k: Optional[float] = None,
        batch_max_iters: int = 25,
        bootstrap_sigma: float = 1e1,
    ) -> None:
        if gtsam is None:
            raise RuntimeError("DecentralizedRobotAgent requires the gtsam bindings")
        self.robot_id = robot_id
        self._robot_map = dict(robot_map)
        self._base_init_lookup: Dict[str, InitEntry] = dict(init_lookup)
        self._robust_kind = robust_kind
        self._robust_k = robust_k
        self._batch_max_iters = batch_max_iters
        self._bootstrap_sigma = float(bootstrap_sigma)

        self._dataset_priors: List[PriorFactorPose3] = []
        self._local_between: List[BetweenFactorPose3] = []
        self._interface_edges: List[InterfaceEdge] = []
        self._active_edge_ids: Set[Tuple[str, str, float]] = set()
        self._pending_edges: Dict[str, List[InterfaceEdge]] = defaultdict(list)
        self._pending_edge_ids: Set[Tuple[str, str, float]] = set()

        # Remote estimates keyed by pose key string
        self._remote_estimates: Dict[str, InterfaceMessage] = {}

        self._latest_estimate: Optional[gtsam.Values] = None
        self._latest_graph: Optional[GraphBuilder] = None
        self._latest_marginals: Optional[gtsam.Marginals] = None
        self._last_iteration: Optional[int] = None

    # ------------------------------------------------------------------
    # Factor ingestion helpers
    # ------------------------------------------------------------------
    def add_local_priors(self, priors: Iterable[PriorFactorPose3]) -> None:
        count = 0
        for prior in priors:
            self._dataset_priors.append(prior)
            count += 1
        if count == 0:
            logger.debug("[%s] No dataset priors provided during ingestion", self.robot_id)

    def add_local_between(self, factors: Iterable[BetweenFactorPose3]) -> None:
        self._local_between.extend(factors)

    def _edge_identifier(self, edge: InterfaceEdge) -> Tuple[str, str, float]:
        stamp = float(getattr(edge.factor, "stamp", 0.0))
        return (edge.local_key, edge.remote_key, stamp)

    def _activate_pending_for_key(self, remote_key: str) -> None:
        if remote_key not in self._pending_edges:
            return
        ready = self._pending_edges.pop(remote_key)
        for edge in ready:
            edge_id = self._edge_identifier(edge)
            if edge_id in self._active_edge_ids:
                continue
            self._interface_edges.append(edge)
            self._active_edge_ids.add(edge_id)
            self._pending_edge_ids.discard(edge_id)

    def add_inter_factor(self, factor: BetweenFactorPose3) -> None:
        rid1 = self._robot_of(factor.key1)
        rid2 = self._robot_of(factor.key2)
        if rid1 == rid2:
            logger.debug("Factor %s-%s is intra-robot; routing to local list", factor.key1, factor.key2)
            self._local_between.append(factor)
            return
        if rid1 == self.robot_id:
            local_key = str(factor.key1)
            remote_key = str(factor.key2)
            remote_robot = rid2
        elif rid2 == self.robot_id:
            local_key = str(factor.key2)
            remote_key = str(factor.key1)
            remote_robot = rid1
        else:
            logger.debug("Skipping factor %s-%s: neither endpoint belongs to %s", factor.key1, factor.key2, self.robot_id)
            return
        owner_robot = min(rid1, rid2)
        edge = InterfaceEdge(
            factor=factor,
            local_key=local_key,
            remote_key=remote_key,
            remote_robot=remote_robot,
            owns_factor=(self.robot_id == owner_robot),
        )
        edge_id = self._edge_identifier(edge)
        if edge_id in self._active_edge_ids or edge_id in self._pending_edge_ids:
            return
        if str(remote_key) in self._remote_estimates:
            self._interface_edges.append(edge)
            self._active_edge_ids.add(edge_id)
        else:
            self._pending_edges[str(remote_key)].append(edge)
            self._pending_edge_ids.add(edge_id)

    def ingest_factor(self, factor: Union[PriorFactorPose3, BetweenFactorPose3]) -> None:
        """Ingest a single factor into local buffers based on ownership.

        - PriorFactorPose3 on this agent's keys → local priors
        - BetweenFactorPose3 with both endpoints belonging to this agent → local between
        - BetweenFactorPose3 with exactly one endpoint belonging to this agent → inter-edge
        Factors unrelated to this agent are ignored.
        """
        if isinstance(factor, PriorFactorPose3):
            if self._robot_of(factor.key) == self.robot_id:
                self._dataset_priors.append(factor)
            return
        if isinstance(factor, BetweenFactorPose3):
            rid1 = self._robot_of(factor.key1)
            rid2 = self._robot_of(factor.key2)
            if rid1 == rid2 == self.robot_id:
                self._local_between.append(factor)
                return
            if rid1 == self.robot_id or rid2 == self.robot_id:
                self.add_inter_factor(factor)
            return

    def _robot_of(self, key: str) -> str:
        skey = str(key)
        if skey in self._robot_map:
            return self._robot_map[skey]
        return default_robot_infer(skey)

    def _bootstrap_prior(self, init_lookup: Dict[str, InitEntry]) -> Optional[PriorFactorPose3]:
        candidate_key: Optional[Union[str, int]] = None
        candidate_entry: Optional[InitEntry] = None

        for key, entry in init_lookup.items():
            if self._robot_of(key) == self.robot_id:
                candidate_key = key
                candidate_entry = entry
                break

        if candidate_key is None:
            for factor in self._local_between:
                if self._robot_of(factor.key1) == self.robot_id:
                    candidate_key = factor.key1
                    candidate_entry = init_lookup.get(str(factor.key1)) or self._base_init_lookup.get(str(factor.key1))
                    break
                if self._robot_of(factor.key2) == self.robot_id:
                    candidate_key = factor.key2
                    candidate_entry = init_lookup.get(str(factor.key2)) or self._base_init_lookup.get(str(factor.key2))
                    break

        if candidate_key is None:
            for key, rid in self._robot_map.items():
                if rid == self.robot_id:
                    candidate_key = key
                    candidate_entry = init_lookup.get(str(key)) or self._base_init_lookup.get(str(key))
                    break

        if candidate_key is None:
            logger.warning(
                "[%s] Unable to infer key for bootstrap prior; graph may remain gauge-free",
                self.robot_id,
            )
            return None

        key_str = str(candidate_key)
        if candidate_entry is None:
            candidate_entry = self._base_init_lookup.get(key_str)
        if candidate_entry is None:
            candidate_entry = InitEntry(
                key=key_str,
                rotation=Quaternion(1.0, 0.0, 0.0, 0.0),
                translation=Translation(0.0, 0.0, 0.0),
            )
        init_lookup[key_str] = candidate_entry

        covariance = np.eye(6) * (self._bootstrap_sigma ** 2)
        return PriorFactorPose3(
            key=key_str,
            rotation=candidate_entry.rotation,
            translation=candidate_entry.translation,
            covariance=covariance,
            stamp=0.0,
        )

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------
    def _blend_messages(
        self,
        prev: InterfaceMessage,
        new: InterfaceMessage,
        alpha: float,
    ) -> InterfaceMessage:
        alpha = float(max(0.0, min(1.0, alpha)))
        if alpha <= 0.0:
            return prev
        if alpha >= 1.0 or prev is None:
            return new

        def _quat_array(msg: InterfaceMessage) -> np.ndarray:
            rot = msg.rotation
            if hasattr(rot, "to_numpy"):
                arr = rot.to_numpy()
            else:
                arr = np.array([rot.w, rot.x, rot.y, rot.z], dtype=float)
            norm = np.linalg.norm(arr)
            if norm == 0.0:
                return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            return arr / norm

        def _trans_array(msg: InterfaceMessage) -> np.ndarray:
            trans = msg.translation
            if hasattr(trans, "to_numpy"):
                return trans.to_numpy()
            return np.array([trans.x, trans.y, trans.z], dtype=float)

        prev_q = _quat_array(prev)
        new_q = _quat_array(new)
        if float(np.dot(prev_q, new_q)) < 0.0:
            new_q = -new_q
        blended_q = (1.0 - alpha) * prev_q + alpha * new_q
        norm_q = np.linalg.norm(blended_q)
        if norm_q == 0.0:
            blended_q = new_q
            norm_q = np.linalg.norm(blended_q) or 1.0
        blended_q /= norm_q

        prev_t = _trans_array(prev)
        new_t = _trans_array(new)
        blended_t = (1.0 - alpha) * prev_t + alpha * new_t

        relaxed_msg = InterfaceMessage(
            sender=new.sender,
            receiver=new.receiver,
            key=new.key,
            rotation=Quaternion(*blended_q.tolist()),
            translation=Translation(*blended_t.tolist()),
            covariance=new.covariance,
            iteration=new.iteration,
            stamp=new.stamp,
            sent_wall_time=new.sent_wall_time,
        )
        return relaxed_msg

    def receive_interface_messages(self, messages: Iterable[InterfaceMessage], *, relaxation: float = 1.0) -> None:
        activated: Set[str] = set()
        for msg in messages:
            key = str(msg.key)
            prev = self._remote_estimates.get(key)
            if prev is not None and getattr(msg, "iteration", 0) <= getattr(prev, "iteration", -1):
                continue
            if prev is not None and relaxation < 1.0:
                msg = self._blend_messages(prev, msg, relaxation)
            self._remote_estimates[key] = msg
            activated.add(key)
        for key in activated:
            self._activate_pending_for_key(key)

    def _build_remote_prior(self, key: str, init_lookup: Dict[str, InitEntry]) -> Optional[PriorFactorPose3]:
        key_str = str(key)
        if key_str in self._remote_estimates:
            msg = self._remote_estimates[key_str]
            init_lookup[key_str] = InitEntry(key=key_str, rotation=msg.rotation, translation=msg.translation)
            covariance = np.asarray(msg.covariance, dtype=float)
            if covariance.shape != (6, 6):
                covariance = np.eye(6) * (self._bootstrap_sigma ** 2)
            return PriorFactorPose3(
                key=key_str,
                rotation=msg.rotation,
                translation=msg.translation,
                covariance=covariance,
                stamp=float(msg.stamp or 0.0),
            )
        # Fallback: use initialisation with a very loose covariance so the
        # optimiser keeps the variable numerically stable until a peer update arrives.
        if key_str in init_lookup:
            entry = init_lookup[key_str]
            covariance = np.eye(6) * (self._bootstrap_sigma ** 2)
            return PriorFactorPose3(
                key=key_str,
                rotation=entry.rotation,
                translation=entry.translation,
                covariance=covariance,
                stamp=0.0,
            )
        return None

    # ------------------------------------------------------------------
    # Optimisation rounds
    # ------------------------------------------------------------------
    def solve_round(self, iteration: int, kpi: Optional[KPILogger] = None) -> Optional[gtsam.Values]:
        if gtsam is None:
            raise RuntimeError("gtsam bindings unavailable")
        init_lookup: Dict[str, InitEntry] = dict(self._base_init_lookup)
        gb = GraphBuilder(
            robot_map=self._robot_map,
            robust_kind=self._robust_kind,
            robust_k=self._robust_k,
        )

        priors = list(self._dataset_priors)
        if not priors:
            logger.warning(
                "[%s] Dataset provided no PriorFactorPose3; injecting weak bootstrap prior",
                self.robot_id,
            )
            bootstrap_prior = self._bootstrap_prior(init_lookup)
            if bootstrap_prior is not None:
                priors.append(bootstrap_prior)
        for prior in priors:
            gb.add_prior(prior, init_lookup)
        for factor in self._local_between:
            gb.add_between(factor, init_lookup)

        for key in list(self._pending_edges.keys()):
            if key in self._remote_estimates:
                self._activate_pending_for_key(key)

        added_remote: Set[str] = set()
        for edge in self._interface_edges:
            if edge.remote_key not in added_remote:
                prior = self._build_remote_prior(edge.remote_key, init_lookup)
                if prior is not None:
                    gb.add_prior(prior, init_lookup)
                added_remote.add(edge.remote_key)
            # Only add the inter factor if both endpoints are initialised –
            # otherwise we will revisit it once the peer shares a pose.
            if edge.owns_factor and str(edge.remote_key) in init_lookup:
                gb.add_between(edge.factor, init_lookup)
            else:
                logger.debug(
                    "[%s] Delaying inter-robot factor (%s,%s) until peer estimate arrives",
                    self.robot_id,
                    edge.local_key,
                    edge.remote_key,
                )

        if gb.graph.size() == 0:
            logger.warning("[%s] No factors in graph; skipping iteration", self.robot_id)
            return None

        if kpi:
            try:
                kpi.optimization_start(iteration, gb.graph.size(), gb.graph.size())
            except Exception:
                pass
        update_start = None
        try:
            import time as _time
            update_start = _time.perf_counter()
        except Exception:
            update_start = None
        estimate = optimize_all_batch(gb.graph, gb.initial, max_iters=self._batch_max_iters)
        if kpi:
            try:
                import time as _time
                duration = (_time.perf_counter() - update_start) if update_start is not None else 0.0
                updated = estimate.size() if hasattr(estimate, "size") else None
                kpi.optimization_end(iteration, duration, updated_keys=updated)
            except Exception:
                pass
        self._latest_estimate = estimate
        self._latest_graph = gb
        has_inter_edges = bool(self._interface_edges)
        if not has_inter_edges and self._pending_edges:
            has_inter_edges = any(self._pending_edges.values())
        ready_for_marginals = (not has_inter_edges) or bool(self._remote_estimates)
        if ready_for_marginals:
            try:
                self._latest_marginals = gtsam.Marginals(gb.graph, estimate)
            except Exception as exc:
                logger.warning("[%s] Failed to compute marginals: %s", self.robot_id, exc)
                self._latest_marginals = None
        else:
            # Wait until peer information anchors the separator variables before requesting covariances.
            self._latest_marginals = None
        self._last_iteration = iteration
        return estimate

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------
    def interface_messages(self, iteration: int) -> List[InterfaceMessage]:
        if self._latest_estimate is None or self._latest_graph is None:
            return []
        msgs_by_recipient: Dict[Tuple[str, str], InterfaceMessage] = {}
        marginals = self._latest_marginals
        all_edges: List[InterfaceEdge] = list(self._interface_edges)
        for pendings in self._pending_edges.values():
            all_edges.extend(pendings)
        for edge in all_edges:
            gb = self._latest_graph
            norm_key = gb.normalize_key(edge.local_key)
            if not self._latest_estimate.exists(norm_key):
                continue
            pose = self._latest_estimate.atPose3(norm_key)
            rot, trans = _pose3_to_components(pose)
            covariance = np.eye(6) * (self._bootstrap_sigma ** 2)
            if marginals is not None:
                try:
                    covariance = np.asarray(marginals.marginalCovariance(norm_key))
                except Exception:
                    covariance = np.eye(6) * (self._bootstrap_sigma ** 2)
            key = (edge.remote_robot, edge.local_key)
            if key in msgs_by_recipient:
                continue
            msgs_by_recipient[key] = InterfaceMessage(
                sender=self.robot_id,
                receiver=edge.remote_robot,
                key=edge.local_key,
                rotation=rot,
                translation=trans,
                covariance=covariance,
                iteration=iteration,
                stamp=edge.factor.stamp,
            )
        return list(msgs_by_recipient.values())

    def local_estimates(self) -> Dict[str, gtsam.Pose3]:
        if self._latest_estimate is None or self._latest_graph is None:
            return {}
        result: Dict[str, gtsam.Pose3] = {}
        gb = self._latest_graph
        for key in gb.keys_for_robot(self.robot_id):
            if self._latest_estimate.exists(key):
                result[gb.denormalize_key(key)] = self._latest_estimate.atPose3(key)
        return result

    def estimate_snapshot(self) -> Optional[gtsam.Values]:
        return self._latest_estimate
