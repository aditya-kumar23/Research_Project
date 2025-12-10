from typing import Optional
import numpy as np

try:
    import gtsam
except Exception:
    gtsam = None

def make_spd(cov: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Jitter a covariance to be SPD if needed.

    Why: Real datasets sometimes include nearly singular covariances.
    We add diagonal jitter to ensure positive definiteness for GTSAM.
    Trade-off: Adds slight artificial confidence; eps is tiny.
    """
    cov = np.array(cov, dtype=float)
    # Symmetrize
    cov = 0.5 * (cov + cov.T)
    # Add jitter until PD
    jitter = eps
    for _ in range(8):
        try:
            np.linalg.cholesky(cov + np.eye(6) * jitter)
            return cov + np.eye(6) * jitter
        except np.linalg.LinAlgError:
            jitter *= 10.0
    # Last resort
    return cov + np.eye(6) * jitter

def gaussian_from_covariance(cov: np.ndarray):
    """Create a GTSAM Gaussian noise model from a 6x6 covariance.

    Ensures:
      - symmetric positive-definite (via jitter)
      - float64 dtype
      - contiguous row-major memory
    """
    if gtsam is None:
        raise RuntimeError("GTSAM not available; cannot build noise model")
    cov = make_spd(cov)
    cov = np.array(cov, dtype=np.float64, order="C")
    return gtsam.noiseModel.Gaussian.Covariance(cov)


def robustify(base, kind: Optional[str] = None, k: Optional[float] = None):
    """Wrap a base noise model with a robust kernel.

    kind: 'huber' | 'cauchy' | None
    k: tuning constant (default: Huber 1.345, Cauchy 1.0)
    """
    if gtsam is None:
        raise RuntimeError("GTSAM not available; cannot build robust model")
    if not kind:
        return base
    kind = kind.lower()
    if kind == "huber":
        k = 1.345 if k is None else k
        loss = gtsam.noiseModel.mEstimator.Huber(k)
    elif kind == "cauchy":
        k = 1.0 if k is None else k
        loss = gtsam.noiseModel.mEstimator.Cauchy(k)
    else:
        raise ValueError(f"Unsupported robust kernel: {kind}")
    return gtsam.noiseModel.Robust.Create(loss, base)
