from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Union

from cosmo_slam_centralised.models import PriorFactorPose3, BetweenFactorPose3
from cosmo_slam_centralised.graph import default_robot_infer

Factor = Union[PriorFactorPose3, BetweenFactorPose3]


@dataclass
class RobotMeasurementBundle:
    """Container grouping factors relevant to a single robot.

    Attributes
    ----------
    priors:
        Prior factors acting on the robot's own pose variables.
    local_between:
        Between factors where both endpoints belong to the same robot
        (odometry edges and intra-robot loop closures).
    inter_between:
        Between factors where exactly one endpoint belongs to this robot
        and the other belongs to a different robot.  These are the edges
        that require peer-to-peer coordination when optimising.
    """

    priors: List[PriorFactorPose3] = field(default_factory=list)
    local_between: List[BetweenFactorPose3] = field(default_factory=list)
    inter_between: List[BetweenFactorPose3] = field(default_factory=list)

    def all_local_factors(self) -> List[Factor]:
        """Return priors + intra factors for convenience when seeding agents."""
        return [*self.priors, *self.local_between]


def _robot_of(key: Union[str, int], robot_map: Dict[str, str]) -> str:
    if isinstance(key, str) and key in robot_map:
        return robot_map[key]
    kid = str(key)
    if kid in robot_map:
        return robot_map[kid]
    return default_robot_infer(key)


def partition_measurements(
    factors: Iterable[Factor],
    robot_map: Dict[str, str],
    *,
    on_factor: Optional[Callable[[float], None]] = None,
) -> Dict[str, RobotMeasurementBundle]:
    """Split a factor stream into per-robot bundles and shared loop closures.

    Parameters
    ----------
    factors:
        Iterable of `PriorFactorPose3` or `BetweenFactorPose3` objects.
    robot_map:
        Mapping from pose key (stringified) to robot identifier.  The
        function falls back to `default_robot_infer` if the key is not
        present in the mapping, which keeps the helper robust even if the
        JRL file omits explicit per-robot initialisation blocks.

    Returns
    -------
    Dict[str, RobotMeasurementBundle]
        Bundles keyed by robot id.  Robots that only participate via
        inter-robot loop closures are still represented with an empty
        bundle so the caller can instantiate agents for them.
    """

    bundles: Dict[str, RobotMeasurementBundle] = {}

    def ensure(robot: str) -> RobotMeasurementBundle:
        if robot not in bundles:
            bundles[robot] = RobotMeasurementBundle()
        return bundles[robot]

    for factor in factors:
        if on_factor is not None:
            try:
                stamp = float(getattr(factor, "stamp", 0.0))
            except Exception:
                stamp = 0.0
            on_factor(stamp)
        if isinstance(factor, PriorFactorPose3):
            rid = _robot_of(factor.key, robot_map)
            ensure(rid).priors.append(factor)
        elif isinstance(factor, BetweenFactorPose3):
            rid1 = _robot_of(factor.key1, robot_map)
            rid2 = _robot_of(factor.key2, robot_map)
            b1 = ensure(rid1)
            b2 = ensure(rid2)
            if rid1 == rid2:
                b1.local_between.append(factor)
            else:
                b1.inter_between.append(factor)
                b2.inter_between.append(factor)
        else:
            # Should never happen because Loader only yields the two dataclasses,
            # but fall back to the first robot to keep the helper defensive.
            rid = default_robot_infer(getattr(factor, "key", "global"))
            ensure(rid)

    return bundles
