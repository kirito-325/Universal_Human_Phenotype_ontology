from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScoreResult:
    shape: float
    support: float
    conflict: float
    shape_forward: float
    shape_reverse: float
    support_forward: float
    support_reverse: float
    conflict_forward: float
    conflict_reverse: float
    comparable: bool


def directional_bma(matrix: np.ndarray) -> tuple[float, float, float]:
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("BMA requires a non-empty two-dimensional matrix")
    forward = float(np.max(matrix, axis=1).mean())
    reverse = float(np.max(matrix, axis=0).mean())
    return forward, reverse, (forward + reverse) / 2


def score_pair(
    shape_matrix: np.ndarray,
    direction_matrix: np.ndarray | None = None,
) -> ScoreResult:
    shape_forward, shape_reverse, shape = directional_bma(shape_matrix)
    if direction_matrix is None:
        nan = float("nan")
        return ScoreResult(
            shape,
            nan,
            nan,
            shape_forward,
            shape_reverse,
            nan,
            nan,
            nan,
            nan,
            False,
        )
    if direction_matrix.shape != shape_matrix.shape:
        raise ValueError("shape and direction matrices must have the same shape")
    comparable = bool(
        np.any((shape_matrix > 0) & np.isin(direction_matrix, (1, -1)))
    )
    if not comparable:
        nan = float("nan")
        return ScoreResult(
            shape,
            nan,
            nan,
            shape_forward,
            shape_reverse,
            nan,
            nan,
            nan,
            nan,
            False,
        )
    support_matrix = shape_matrix * (direction_matrix == 1)
    conflict_matrix = shape_matrix * (direction_matrix == -1)
    support_forward, support_reverse, support = directional_bma(support_matrix)
    conflict_forward, conflict_reverse, conflict = directional_bma(conflict_matrix)
    return ScoreResult(
        shape,
        support,
        conflict,
        shape_forward,
        shape_reverse,
        support_forward,
        support_reverse,
        conflict_forward,
        conflict_reverse,
        True,
    )
