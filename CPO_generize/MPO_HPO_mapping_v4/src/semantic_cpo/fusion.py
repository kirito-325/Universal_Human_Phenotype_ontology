from __future__ import annotations

import math

import numpy as np


def fuse_scalar(
    method: str,
    s: float,
    m: float | None,
    q: float,
    epsilon: float = 0.05,
    r_s: float | None = None,
    r_m: float | None = None,
) -> float:
    s = min(1.0, max(0.0, float(s)))
    q = min(1.0, max(0.0, float(q)))
    if m is None or not math.isfinite(m) or q == 0.0:
        return s
    m = min(1.0, max(0.0, float(m)))
    if method == "positive":
        value = s + epsilon * q * m * (1.0 - s)
    elif method == "linear":
        value = s + epsilon * q * (m - s)
    elif method == "bounded_geometric":
        value = s + epsilon * q * (math.sqrt(s * m) - s)
    elif method == "amplitude_gated_rank":
        if r_s is None or r_m is None:
            return s
        value = s + epsilon * q * m * (float(r_m) - float(r_s))
    else:
        raise ValueError(f"unknown fusion method: {method}")
    return min(1.0, max(0.0, value))


def fuse_array(
    method: str,
    s: np.ndarray,
    m: np.ndarray,
    q: np.ndarray,
    epsilon: float = 0.05,
    r_s: np.ndarray | None = None,
    r_m: np.ndarray | None = None,
) -> np.ndarray:
    s_used = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)
    m_used = np.clip(np.asarray(m, dtype=np.float64), 0.0, 1.0)
    q_used = np.clip(np.asarray(q, dtype=np.float64), 0.0, 1.0)
    computable = np.isfinite(m_used) & (q_used > 0)
    output = s_used.copy()
    if method == "positive":
        candidate = s_used + epsilon * q_used * m_used * (1.0 - s_used)
    elif method == "linear":
        candidate = s_used + epsilon * q_used * (m_used - s_used)
    elif method == "bounded_geometric":
        candidate = s_used + epsilon * q_used * (np.sqrt(s_used * m_used) - s_used)
    elif method == "amplitude_gated_rank":
        if r_s is None or r_m is None:
            return output.astype(np.float32)
        candidate = s_used + epsilon * q_used * m_used * (r_m - r_s)
        computable &= np.isfinite(r_s) & np.isfinite(r_m)
    else:
        raise ValueError(f"unknown fusion method: {method}")
    output[computable] = candidate[computable]
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def midcdf(value: float, background: np.ndarray) -> float | None:
    finite = np.sort(np.asarray(background, dtype=np.float64)[np.isfinite(background)])
    if not len(finite):
        return None
    left = int(np.searchsorted(finite, value, side="left"))
    right = int(np.searchsorted(finite, value, side="right"))
    return (left + 0.5 * (right - left)) / len(finite)

