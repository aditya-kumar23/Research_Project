from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple, Iterable, Union
import numpy as np

@dataclass
class Quaternion:
    """Quaternion in [w, x, y, z] order.

    Why: We explicitly model the ordering to avoid confusion. COSMO-bench
    specifies wxyz here; if your dataset differs, convert at the loader.
    """
    w: float
    x: float
    y: float
    z: float

    def to_numpy(self) -> np.ndarray:
        return np.array([self.w, self.x, self.y, self.z], dtype=float)

@dataclass
class Translation:
    x: float
    y: float
    z: float

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

@dataclass
class InitEntry:
    key: Union[str, int]
    rotation: Quaternion
    translation: Translation
    type: str = "Pose3"

@dataclass
class PriorFactorPose3:
    key: Union[str, int]
    rotation: Quaternion  # prior rotation
    translation: Translation  # prior translation
    covariance: np.ndarray  # 6x6
    stamp: float

@dataclass
class BetweenFactorPose3:
    key1: Union[str, int]
    key2: Union[str, int]
    rotation: Quaternion  # measurement rotation
    translation: Translation  # measurement translation
    covariance: np.ndarray  # 6x6
    stamp: float

@dataclass
class JRLDocument:
    measurements: Dict[str, List[Dict[str, Any]]]
    outlier_factors: List[Dict[str, Any]]
    potential_outlier_factors: List[Dict[str, Any]]
    ground_truth: Dict[str, Any]
    initialisation: List[Dict[str, Any]]

def is_6x6_cov(mat: np.ndarray) -> bool:
    return isinstance(mat, np.ndarray) and mat.shape == (6, 6)

def to_covariance(cov_list: List[float]) -> np.ndarray:
    """Convert a flat list (36) or nested list (6x6) to a 6x6 ndarray.

    Why: JRL may store covariance either flattened row-major or as a 2-D list.
    We normalize into a 6x6 ndarray and validate SPD elsewhere.
    """
    arr = np.asarray(cov_list, dtype=float)
    if arr.size == 36 and arr.ndim == 1:
        return arr.reshape(6, 6)
    if arr.size == 36 and arr.ndim == 2 and arr.shape == (6, 6):
        return arr
    raise ValueError(f"Expected 36 elements for a 6x6 covariance, got shape {arr.shape} size {arr.size}")
