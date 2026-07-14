"""Finite-set metrics for multi-object tracking experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GospaResult:
    """GOSPA distance and its alpha=2 decomposition."""

    distance: float
    powered_distance: float
    localization: float
    missed: float
    false: float
    matched_count: int
    missed_count: int
    false_count: int


def _linear_sum_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    """Solve a rectangular linear assignment with the Hungarian algorithm."""

    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2:
        raise ValueError("cost must be a two-dimensional matrix")
    original_rows, original_columns = cost.shape
    if original_rows == 0 or original_columns == 0:
        return []

    transposed = original_rows > original_columns
    if transposed:
        cost = cost.T
    row_count, column_count = cost.shape

    row_potential = np.zeros(row_count + 1)
    column_potential = np.zeros(column_count + 1)
    column_match = np.zeros(column_count + 1, dtype=int)
    predecessor = np.zeros(column_count + 1, dtype=int)

    for row in range(1, row_count + 1):
        column_match[0] = row
        minimum_slack = np.full(column_count + 1, np.inf)
        used = np.zeros(column_count + 1, dtype=bool)
        column = 0
        while True:
            used[column] = True
            matched_row = column_match[column]
            delta = np.inf
            next_column = 0
            for candidate_column in range(1, column_count + 1):
                if used[candidate_column]:
                    continue
                reduced_cost = (
                    cost[matched_row - 1, candidate_column - 1]
                    - row_potential[matched_row]
                    - column_potential[candidate_column]
                )
                if reduced_cost < minimum_slack[candidate_column]:
                    minimum_slack[candidate_column] = reduced_cost
                    predecessor[candidate_column] = column
                if minimum_slack[candidate_column] < delta:
                    delta = minimum_slack[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(column_count + 1):
                if used[candidate_column]:
                    row_potential[column_match[candidate_column]] += delta
                    column_potential[candidate_column] -= delta
                else:
                    minimum_slack[candidate_column] -= delta
            column = next_column
            if column_match[column] == 0:
                break

        while True:
            previous_column = predecessor[column]
            column_match[column] = column_match[previous_column]
            column = previous_column
            if column == 0:
                break

    assignments: list[tuple[int, int]] = []
    for column in range(1, column_count + 1):
        row = column_match[column]
        if row == 0:
            continue
        pair = (row - 1, column - 1)
        assignments.append((pair[1], pair[0]) if transposed else pair)
    return assignments


def gospa(
    truth: np.ndarray,
    estimates: np.ndarray,
    cutoff: float = 10.0,
    order: float = 2.0,
    alpha: float = 2.0,
) -> GospaResult:
    """Compute GOSPA and its missed/false/localization decomposition.

    The decomposition is returned for ``alpha=2``. Assignment pairs at or
    beyond the cutoff are represented as one missed and one false target,
    matching the reference GOSPA implementation.
    """

    truth = np.asarray(truth, dtype=float)
    estimates = np.asarray(estimates, dtype=float)
    if truth.ndim != 2 or estimates.ndim != 2:
        raise ValueError("truth and estimates must be two-dimensional arrays")
    if truth.shape[1] != estimates.shape[1]:
        raise ValueError("truth and estimates must have the same point dimension")
    if cutoff <= 0.0:
        raise ValueError("cutoff must be positive")
    if order < 1.0:
        raise ValueError("order must be at least one")
    if alpha != 2.0:
        raise ValueError("the decomposition currently requires alpha=2")

    truth_count = len(truth)
    estimate_count = len(estimates)
    cardinality_cost = cutoff**order / alpha
    if truth_count == 0 or estimate_count == 0:
        missed_count = truth_count
        false_count = estimate_count
        missed = cardinality_cost * missed_count
        false = cardinality_cost * false_count
        total = missed + false
        return GospaResult(
            distance=float(total ** (1.0 / order)),
            powered_distance=float(total),
            localization=0.0,
            missed=float(missed),
            false=float(false),
            matched_count=0,
            missed_count=missed_count,
            false_count=false_count,
        )

    distances = np.linalg.norm(truth[:, None, :] - estimates[None, :, :], axis=2)
    assignments = _linear_sum_assignment(np.minimum(distances, cutoff) ** order)
    localization = 0.0
    matched_count = 0
    for truth_index, estimate_index in assignments:
        distance = distances[truth_index, estimate_index]
        if distance < cutoff:
            localization += distance**order
            matched_count += 1

    missed_count = truth_count - matched_count
    false_count = estimate_count - matched_count
    missed = cardinality_cost * missed_count
    false = cardinality_cost * false_count
    total = localization + missed + false
    return GospaResult(
        distance=float(total ** (1.0 / order)),
        powered_distance=float(total),
        localization=float(localization),
        missed=float(missed),
        false=float(false),
        matched_count=matched_count,
        missed_count=missed_count,
        false_count=false_count,
    )
