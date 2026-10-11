from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Atom:
    atom_id: str
    cl_id: str
    go_id: str
    direction_state: str = "unknown"
    direction_target: str = ""
    direction_property: str = ""
    relation_role: str = ""


def direction_multiplier(left: Atom, right: Atom) -> tuple[float, str]:
    known = {"increased", "decreased", "present", "absent"}
    opposite = {
        ("increased", "decreased"),
        ("decreased", "increased"),
        ("present", "absent"),
        ("absent", "present"),
    }
    comparable = (
        left.cl_id == right.cl_id
        and left.go_id == right.go_id
        and left.direction_target
        and (left.direction_target, left.direction_property)
        == (right.direction_target, right.direction_property)
        and left.direction_state in known
        and right.direction_state in known
    )
    if not comparable:
        return 1.0, "unknown"
    if (left.direction_state, right.direction_state) in opposite:
        return 0.0, "opposed"
    return 1.0, "same"


def bma(kernel: np.ndarray, left_weights: np.ndarray | None = None, right_weights: np.ndarray | None = None) -> float:
    values = np.asarray(kernel, dtype=np.float64)
    if not values.size or not values.shape[0] or not values.shape[1]:
        return math.nan
    left = np.full(values.shape[0], 1.0 / values.shape[0]) if left_weights is None else np.asarray(left_weights, dtype=np.float64)
    right = np.full(values.shape[1], 1.0 / values.shape[1]) if right_weights is None else np.asarray(right_weights, dtype=np.float64)
    if left.sum() <= 0 or right.sum() <= 0:
        return math.nan
    left = left / left.sum()
    right = right / right.sum()
    return 0.5 * (float(left @ values.max(axis=1)) + float(right @ values.max(axis=0)))


def mutual_best(kernel: np.ndarray, left_ids: Sequence[str], right_ids: Sequence[str]) -> float:
    values = np.asarray(kernel, dtype=np.float64)
    n, m = values.shape
    if not n or not m:
        return math.nan
    row_best = [min(range(m), key=lambda j: (-values[i, j], right_ids[j])) for i in range(n)]
    col_best = [min(range(n), key=lambda i: (-values[i, j], left_ids[i])) for j in range(m)]
    total = 0.0
    for i, j in enumerate(row_best):
        if col_best[j] == i and values[i, j] > 0:
            total += 0.5 * (1.0 / n + 1.0 / m) * values[i, j]
    return float(total)


def one_to_one(kernel: np.ndarray) -> float:
    values = np.asarray(kernel, dtype=np.float64)
    n, m = values.shape
    if not n or not m:
        return math.nan
    rows, columns = linear_sum_assignment(values, maximize=True)
    return float(0.5 * (1.0 / n + 1.0 / m) * values[rows, columns].sum())


def balanced_ot(kernel: np.ndarray) -> float:
    values = np.asarray(kernel, dtype=np.float64)
    n, m = values.shape
    if not n or not m:
        return math.nan
    mass = math.lcm(n, m)
    expanded = np.repeat(np.repeat(values, mass // n, axis=0), mass // m, axis=1)
    rows, columns = linear_sum_assignment(expanded, maximize=True)
    return float(expanded[rows, columns].sum() / mass)


def aggregate(
    method: str,
    kernel: np.ndarray,
    left_ids: Sequence[str],
    right_ids: Sequence[str],
    left_weights: np.ndarray | None = None,
    right_weights: np.ndarray | None = None,
) -> float:
    if method == "bma":
        return bma(kernel)
    if method == "ic_bma":
        return bma(kernel, left_weights, right_weights)
    if method == "mutual_best":
        return mutual_best(kernel, left_ids, right_ids)
    if method == "one_to_one":
        return one_to_one(kernel)
    if method == "balanced_ot":
        return balanced_ot(kernel)
    raise ValueError(f"unknown profile method: {method}")

