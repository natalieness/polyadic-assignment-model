"""Exact-margin null sampling and conditional Gibbs target-label swaps."""

from __future__ import annotations

import numpy as np


def validate_assignment(
    assignment: np.ndarray,
    site_degrees: np.ndarray,
    target_counts: np.ndarray,
) -> None:
    assignment = np.asarray(assignment)
    if not np.issubdtype(assignment.dtype, np.integer) or np.any(assignment < 0):
        raise RuntimeError("Assignment must be a nonnegative integer matrix")
    if not np.array_equal(assignment.sum(axis=1), np.asarray(site_degrees, dtype=int)):
        raise RuntimeError("Assignment rows do not match the fixed site degrees")
    if not np.array_equal(assignment.sum(axis=0), np.asarray(target_counts, dtype=int)):
        raise RuntimeError("Assignment columns do not match the fixed target totals")


def sample_random_null(
    site_degrees: np.ndarray,
    target_counts: np.ndarray,
    rng: np.random.Generator,
    *,
    max_same_target_per_site: int | None = None,
    max_restarts: int = 100,
) -> np.ndarray:
    """Shuffle labelled target-contact tokens into fixed site slots."""
    degrees = np.asarray(site_degrees, dtype=int)
    counts = np.asarray(target_counts, dtype=int)
    if np.any(degrees < 0) or np.any(counts < 0):
        raise ValueError("Site degrees and target counts cannot be negative")
    if int(degrees.sum()) != int(counts.sum()):
        raise ValueError("Fixed degrees and target totals have different contact totals")
    if max_same_target_per_site is not None:
        capacity = int(max_same_target_per_site)
        if np.any(degrees > capacity * len(counts)):
            raise ValueError("A site degree exceeds the total configured target capacity")
        if np.any(counts > capacity * len(degrees)):
            raise ValueError("A target total exceeds the total configured site capacity")
    site_slots = np.repeat(np.arange(len(degrees), dtype=int), degrees)
    target_tokens = np.repeat(np.arange(len(counts), dtype=int), counts)
    for _ in range(max_restarts):
        assignment = np.zeros((len(degrees), len(counts)), dtype=int)
        shuffled_sites = rng.permutation(site_slots)
        shuffled_targets = rng.permutation(target_tokens)
        np.add.at(assignment, (shuffled_sites, shuffled_targets), 1)
        if (
            max_same_target_per_site is None
            or int(assignment.max(initial=0)) <= max_same_target_per_site
        ):
            validate_assignment(assignment, degrees, counts)
            return assignment
    raise RuntimeError(
        f"Could not satisfy max_same_target_per_site after {max_restarts} null draws"
    )


def _multiplicity_profile(row: np.ndarray) -> tuple[int, ...]:
    return tuple(sorted(int(value) for value in row[row > 0]))


def _row_objective(
    row: np.ndarray,
    geometry_row: np.ndarray,
    cotarget_scores: np.ndarray,
    *,
    geometry_weight: float,
    cotargeting_weight: float,
    geometry_repeat_discount: float,
    cotarget_aggregation: str,
    cotarget_site_clip: float | None,
    self_repeat_penalty: float,
) -> float:
    effective_geometry_contacts = np.where(
        row > 0,
        1.0 + geometry_repeat_discount * (row - 1),
        0.0,
    )
    objective = geometry_weight * float(effective_geometry_contacts @ geometry_row)
    present = np.flatnonzero(row > 0)
    if cotargeting_weight and len(present) >= 2:
        values = np.asarray([
            cotarget_scores[left, right]
            for left_number, left in enumerate(present)
            for right in present[left_number + 1:]
        ], dtype=float)
        if len(values):
            cotarget_value = (
                float(np.mean(values))
                if cotarget_aggregation == "mean"
                else float(np.sum(values))
            )
            if cotarget_site_clip is not None:
                cotarget_value = float(np.clip(
                    cotarget_value, -cotarget_site_clip, cotarget_site_clip
                ))
            objective += cotargeting_weight * cotarget_value
    if self_repeat_penalty:
        objective -= self_repeat_penalty * float(np.sum(row * (row - 1) / 2.0))
    return objective


def gibbs_target_swaps(
    initial_assignment: np.ndarray,
    site_degrees: np.ndarray,
    target_counts: np.ndarray,
    geometry_scores: np.ndarray,
    cotarget_scores: np.ndarray,
    rng: np.random.Generator,
    *,
    geometry_weight: float,
    cotargeting_weight: float,
    temperature: float,
    geometry_repeat_discount: float,
    cotarget_aggregation: str,
    cotarget_site_clip: float | None,
    self_repeat_penalty: float,
    preserve_multiplicity_profile: bool,
    max_same_target_per_site: int | None,
    steps: int,
) -> np.ndarray:
    """Resample two-site target allocations without changing either margin.

    Each update selects two sites and two targets represented across those sites.
    It enumerates the complete feasible 2x2 table with the same row and column
    margins, then samples one table from its exact conditional distribution under
    the configured geometry/co-targeting objective.
    """
    result = np.asarray(initial_assignment, dtype=int).copy()
    degrees = np.asarray(site_degrees, dtype=int)
    counts = np.asarray(target_counts, dtype=int)
    geometry_scores = np.asarray(geometry_scores, dtype=float)
    cotarget_scores = np.asarray(cotarget_scores, dtype=float)
    validate_assignment(result, degrees, counts)
    if geometry_scores.shape != result.shape:
        raise ValueError("Geometry score matrix has the wrong shape")
    if cotarget_scores.shape != (result.shape[1], result.shape[1]):
        raise ValueError("Co-target score matrix has the wrong shape")
    active_sites = np.flatnonzero(degrees > 0)
    if len(active_sites) < 2 or steps <= 0:
        return result
    profiles = [_multiplicity_profile(row) for row in result]
    for _ in range(int(steps)):
        left_site, right_site = rng.choice(active_sites, size=2, replace=False)
        pooled_support = np.flatnonzero(
            result[left_site] + result[right_site] > 0
        )
        if len(pooled_support) < 2:
            continue
        left_target, right_target = rng.choice(
            pooled_support, size=2, replace=False
        )

        # A feasible 2x2 table is determined by its upper-left value. The other
        # three cells follow from the fixed two row and two column subtotals.
        left_subtotal = int(
            result[left_site, left_target] + result[left_site, right_target]
        )
        right_subtotal = int(
            result[right_site, left_target] + result[right_site, right_target]
        )
        left_target_subtotal = int(
            result[left_site, left_target] + result[right_site, left_target]
        )
        lower = max(0, left_target_subtotal - right_subtotal)
        upper = min(left_subtotal, left_target_subtotal)

        candidates: list[tuple[np.ndarray, np.ndarray, float]] = []
        for new_left_left in range(lower, upper + 1):
            proposed_left = result[left_site].copy()
            proposed_right = result[right_site].copy()
            proposed_left[left_target] = new_left_left
            proposed_left[right_target] = left_subtotal - new_left_left
            proposed_right[left_target] = left_target_subtotal - new_left_left
            proposed_right[right_target] = (
                right_subtotal - proposed_right[left_target]
            )
            if max_same_target_per_site is not None and (
                proposed_left.max(initial=0) > max_same_target_per_site
                or proposed_right.max(initial=0) > max_same_target_per_site
            ):
                continue
            if preserve_multiplicity_profile and (
                _multiplicity_profile(proposed_left) != profiles[left_site]
                or _multiplicity_profile(proposed_right) != profiles[right_site]
            ):
                continue
            proposed_left_objective = _row_objective(
                proposed_left, geometry_scores[left_site], cotarget_scores,
                geometry_weight=geometry_weight,
                cotargeting_weight=cotargeting_weight,
                geometry_repeat_discount=geometry_repeat_discount,
                cotarget_aggregation=cotarget_aggregation,
                cotarget_site_clip=cotarget_site_clip,
                self_repeat_penalty=self_repeat_penalty,
            )
            proposed_right_objective = _row_objective(
                proposed_right, geometry_scores[right_site], cotarget_scores,
                geometry_weight=geometry_weight,
                cotargeting_weight=cotargeting_weight,
                geometry_repeat_discount=geometry_repeat_discount,
                cotarget_aggregation=cotarget_aggregation,
                cotarget_site_clip=cotarget_site_clip,
                self_repeat_penalty=self_repeat_penalty,
            )
            candidates.append((
                proposed_left,
                proposed_right,
                proposed_left_objective + proposed_right_objective,
            ))

        if len(candidates) < 2:
            continue
        log_probabilities = np.asarray([
            objective / temperature for _, _, objective in candidates
        ])
        log_probabilities -= np.max(log_probabilities)
        probabilities = np.exp(log_probabilities)
        probabilities /= probabilities.sum()
        chosen = int(rng.choice(len(candidates), p=probabilities))
        result[left_site] = candidates[chosen][0]
        result[right_site] = candidates[chosen][1]
    validate_assignment(result, degrees, counts)
    return result


# Retain the old public name for existing notebooks and scripts.
metropolis_target_swaps = gibbs_target_swaps
