from typing import Dict, Iterable, Optional, Union, Set
import logging
from collections import Counter
import string

try:
    import gtsam
except Exception:
    gtsam = None

from .models import PriorFactorPose3, BetweenFactorPose3, InitEntry, Quaternion, Translation
from .robust import gaussian_from_covariance, robustify

logger = logging.getLogger("cosmo_slam.graph")

_GLOBAL_KEY_MAP: Dict[str, int] = {}

def resolve_key(key: Union[str, int]) -> Union[str, int]:
    if isinstance(key, str) and key in _GLOBAL_KEY_MAP:
        return _GLOBAL_KEY_MAP[key]
    return key

if gtsam is not None and not getattr(gtsam.Values.exists, "_cosmo_wrapped", False):
    _orig_exists = gtsam.Values.exists

    def _exists_with_strings(self, key):
        if isinstance(key, str) and key in _GLOBAL_KEY_MAP:
            return _orig_exists(self, _GLOBAL_KEY_MAP[key])
        return _orig_exists(self, key)

    gtsam.Values.exists = _exists_with_strings
    gtsam.Values.exists._cosmo_wrapped = True


def default_robot_infer(key: Union[str, int]) -> str:
    if isinstance(key, int):
        return "global"
    s = str(key)
    prefix = []
    for ch in s:
        if ch.isalpha():
            prefix.append(ch)
        else:
            break
    return ''.join(prefix) or "global"


def pose_from(rot: Quaternion, trans: Translation):
    if gtsam is None:
        raise RuntimeError("GTSAM not available; cannot build Pose3")
    R = gtsam.Rot3.Quaternion(rot.w, rot.x, rot.y, rot.z)
    t = gtsam.Point3(trans.x, trans.y, trans.z)
    return gtsam.Pose3(R, t)


class GraphBuilder:
    """Maintains a global graph/values + per-batch deltas for iSAM2 updates.

    IMPORTANT: We DO NOT reuse the exact same factor object across two graphs.
    Some GTSAM Python wheels mis-handle shared_ptr lifetime across multiple
    NonlinearFactorGraphs, leading to use-after-free or double-delete.
    """

    def __init__(self, robot_map: Optional[Dict[str, str]] = None,
                 robot_infer=default_robot_infer,
                 robust_kind: Optional[str] = None,
                 robust_k: Optional[float] = None):
        if gtsam is None:
            raise RuntimeError("GTSAM not available; cannot build graph")
        # Global accumulators (for error reporting / exports)
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()

        # Per-batch deltas (fed into iSAM2.update)
        self._batch_graph = gtsam.NonlinearFactorGraph()
        self._batch_values = gtsam.Values()

        self.robot_map = robot_map or {}
        self.robot_infer = robot_infer
        self.robust_kind = robust_kind
        self.robust_k = robust_k
        self.poses_seen: Set[str] = set()
        self._values_seen: Set[int] = set()
        self.by_robot_keys: Dict[str, set] = {}
        self.counts = {"prior": 0, "between": 0, "skipped": 0}
        self.loop_counts = {
            "intra": Counter(),
            "inter_pairs": Counter(),
            "inter_per_robot": Counter(),
        }
        self._key_map: Dict[str, int] = {}
        self._reverse_key_map: Dict[int, str] = {}
        self._fallback_symbol_idx = 0

    def _robot_of(self, key: Union[str, int]) -> str:
        kid = str(key)
        if kid in self.robot_map:
            return self.robot_map[kid]
        return self.robot_infer(key)

    def robot_of(self, key: Union[str, int]) -> str:
        """Public accessor for robot inference (used by instrumentation)."""
        return self._robot_of(key)

    def _normalize_key(self, key: Union[str, int]) -> int:
        if isinstance(key, (int, float)) and not isinstance(key, bool):
            return int(key)
        kid = str(key)
        if kid in self._key_map:
            return self._key_map[kid]
        if kid.isdigit():
            norm = int(kid)
        elif gtsam is not None and kid and kid[0] in string.ascii_letters:
            prefix = kid[0]
            digits = ''.join(ch for ch in kid[1:] if ch.isdigit()) or "0"
            try:
                norm = gtsam.symbol(prefix, int(digits))
            except Exception:
                norm = gtsam.symbol(prefix, self._fallback_symbol_idx)
                self._fallback_symbol_idx += 1
        else:
            norm = hash(kid)
        self._key_map[kid] = norm
        self._reverse_key_map[norm] = kid
        _GLOBAL_KEY_MAP[kid] = norm
        return norm

    def normalize_key(self, key: Union[str, int]) -> int:
        """Public accessor for internal key normalisation."""
        return self._normalize_key(key)

    def denormalize_key(self, norm_key: Union[int, str]) -> str:
        if isinstance(norm_key, str):
            return norm_key
        return self._reverse_key_map.get(int(norm_key), str(norm_key))

    def original_key(self, norm_key: Union[int, str]) -> str:
        """Backward-compatible alias for denormalize_key."""
        return self.denormalize_key(norm_key)

    def keys_for_robot(self, robot_id: str) -> Iterable[int]:
        """Return a snapshot of the keys associated with `robot_id`."""
        keys = self.by_robot_keys.get(robot_id)
        if not keys:
            return ()
        return tuple(keys)

    def ensure_init(self, key: Union[str, int], init_lookup: Dict[str, InitEntry]) -> None:
        """Ensure 'key' exists; insert into both global Values and the current batch if new."""
        norm_key = self.normalize_key(key)
        if norm_key in self._values_seen:
            return
        self._values_seen.add(norm_key)
        kid = str(key)
        self.poses_seen.add(kid)
        if str(key) in init_lookup:
            entry = init_lookup[str(key)]
            pose = pose_from(entry.rotation, entry.translation)
        else:
            logger.warning("Missing initialization for key %s; using identity.", key)
            pose = gtsam.Pose3()
        # Insert in both global values and current batch values
        self.initial.insert(norm_key, pose)
        self._batch_values.insert(norm_key, pose)

        rid = self._robot_of(key)
        self.by_robot_keys.setdefault(rid, set()).add(norm_key)

    def _noise(self, cov):
        n = gaussian_from_covariance(cov)
        return robustify(n, self.robust_kind, self.robust_k)

    def add_prior(self, f: PriorFactorPose3, init_lookup: Dict[str, InitEntry]) -> None:
        try:
            self.ensure_init(f.key, init_lookup)
            pose = pose_from(f.rotation, f.translation)
            noise = self._noise(f.covariance)
            norm_key = self.normalize_key(f.key)
            # Create TWO separate factor objects (no sharing across graphs)
            pf_batch = gtsam.PriorFactorPose3(norm_key, pose, noise)
            pf_global = gtsam.PriorFactorPose3(norm_key, pose, noise)
            self._batch_graph.add(pf_batch)
            self.graph.add(pf_global)
            self.counts["prior"] += 1
        except Exception as e:
            logger.warning("Skipping prior on %s due to %s", f.key, e)
            self.counts["skipped"] += 1

    def add_between(self, f: BetweenFactorPose3, init_lookup: Dict[str, InitEntry]) -> None:
        try:
            self.ensure_init(f.key1, init_lookup)
            self.ensure_init(f.key2, init_lookup)
            rel = pose_from(f.rotation, f.translation)
            noise = self._noise(f.covariance)
            nk1 = self.normalize_key(f.key1)
            nk2 = self.normalize_key(f.key2)
            # Create TWO separate factor objects
            bf_batch = gtsam.BetweenFactorPose3(nk1, nk2, rel, noise)
            bf_global = gtsam.BetweenFactorPose3(nk1, nk2, rel, noise)
            self._batch_graph.add(bf_batch)
            self.graph.add(bf_global)
            self.counts["between"] += 1

            rid1 = self._robot_of(f.key1)
            rid2 = self._robot_of(f.key2)
            if rid1 == rid2:
                self.loop_counts["intra"][rid1] += 1
                logger.debug(
                    "Detected intra-robot loop closure: %s(%s) <-> %s(%s) @ stamp %.3f",
                    rid1,
                    f.key1,
                    rid2,
                    f.key2,
                    float(getattr(f, "stamp", 0.0)),
                )
            else:
                pair = tuple(sorted((rid1, rid2)))
                self.loop_counts["inter_pairs"][pair] += 1
                self.loop_counts["inter_per_robot"][rid1] += 1
                self.loop_counts["inter_per_robot"][rid2] += 1
                logger.debug(
                    "Detected inter-robot loop closure: %s(%s) <-> %s(%s) @ stamp %.3f",
                    rid1,
                    f.key1,
                    rid2,
                    f.key2,
                    float(getattr(f, "stamp", 0.0)),
                )
        except Exception as e:
            logger.warning("Skipping between %s-%s due to %s", f.key1, f.key2, e)
            self.counts["skipped"] += 1

    def pop_batch(self):
        """Return (batch_graph, batch_values) and reset deltas for the next mini-batch."""
        bg = self._batch_graph
        bv = self._batch_values
        # reset
        self._batch_graph = gtsam.NonlinearFactorGraph()
        self._batch_values = gtsam.Values()
        return bg, bv
