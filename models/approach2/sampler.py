"""Exact-margin null sampling and conditional Gibbs target-label swaps."""

from __future__ import annotations

from math import exp

import numpy as np

from .sampler_numba import (
    NO_CAP,
    NO_CLIP,
    NUMBA_AVAILABLE,
    draw_uniform_blocks,
    gibbs_kernel,
)


class _UniformStream:
    """Hand out uniforms one at a time, generating them in blocks.

    A scalar ``rng.random()`` costs about 130 ns against 45 ns for reading a buffered
    Python float, and the Gibbs loop needs three draws per step. Blocks come from the
    supplied generator, so a given seed still fixes the whole sequence.
    """

    __slots__ = ("_rng", "_block", "_values", "_position")

    def __init__(self, rng: np.random.Generator, block: int = 8192):
        self._rng = rng
        self._block = block
        self._values = rng.random(block).tolist()
        self._position = 0

    def next(self) -> float:
        if self._position == self._block:
            self._values = self._rng.random(self._block).tolist()
            self._position = 0
        value = self._values[self._position]
        self._position += 1
        return value


def _distinct_pair(values: list[int], first_uniform: float, second_uniform: float):
    """Draw two distinct entries of ``values``, uniformly over ordered pairs.

    The second index is drawn from the remaining ``n - 1`` slots and shifted past the
    first, which is what ``rng.choice(values, size=2, replace=False)`` does but without
    its argument validation and permutation machinery.
    """
    count = len(values)
    first = int(first_uniform * count)
    second = int(second_uniform * (count - 1))
    if second >= first:
        second += 1
    return values[first], values[second]


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
    cotarget_include_self: bool = False,
) -> float:
    """Score a whole site row. Reference definition of the sampler's objective.

    The Gibbs loop does not call this — it rebuilds only the part a swap can change,
    via ``_static_row_part`` and ``_moving_row_objective``. The tests pin those two
    against this function so the fast path cannot drift from the definition.
    """
    effective_geometry_contacts = np.where(
        row > 0,
        1.0 + geometry_repeat_discount * (row - 1),
        0.0,
    )
    objective = geometry_weight * float(effective_geometry_contacts @ geometry_row)
    present = np.flatnonzero(row > 0)
    if cotargeting_weight:
        values = [
            float(cotarget_scores[left, right])
            for left_number, left in enumerate(present)
            for right in present[left_number + 1:]
        ]
        if cotarget_include_self:
            # One term per repeated target, mirroring the site-unit diagonal that
            # ``pair_count_matrix`` builds for the propensity estimate. A site holding
            # a single target three times contributes one term, not three.
            values.extend(
                float(cotarget_scores[target, target])
                for target in present
                if row[target] >= 2
            )
        if values:
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


def _pair_score_matrix(cotarget_scores: np.ndarray) -> np.ndarray:
    """Mirror the pair scores into a symmetric matrix.

    ``_row_objective`` only ever reads ``cotarget_scores[a, b]`` for ``a <= b``, so the
    upper triangle is copied into both halves, the diagonal is carried through for the
    self-repeat term, and every other entry is dropped.
    """
    scores = np.asarray(cotarget_scores, dtype=float)
    upper = np.triu(scores, 1)
    return upper + upper.T + np.diag(np.diagonal(scores))


def _pair_score_lookup(cotarget_scores: np.ndarray) -> list[list[float]]:
    """The same mirrored scores as nested lists, for the interpreted sampler.

    Sites hold a handful of targets each, where Python list indexing costs roughly
    half of NumPy's scalar-access overhead. The compiled kernel wants the opposite
    and takes ``_pair_score_matrix`` directly.
    """
    return _pair_score_matrix(cotarget_scores).tolist()


def _static_row_part(
    row: list[int],
    pooled_support: list[int],
    left_target: int,
    right_target: int,
    pair_scores: list[list[float]] | None,
    include_self: bool = False,
) -> tuple[int, int, float, float, float, float, int]:
    """Summarise the part of one row that the two moving columns cannot change.

    A swap rewrites a row only at ``left_target`` and ``right_target``, so everything
    the objective needs about the remaining occupied targets is fixed for the whole
    candidate enumeration: how many there are, their largest multiplicity, their
    internal pair-score sum, and each moving column's pair-score sum against them.

    With ``include_self`` it also returns the non-moving repeats' own score sum and
    how many of them there are, since both feed the site total and its mean divisor.
    """
    others = [
        target for target in pooled_support
        if row[target] and target != left_target and target != right_target
    ]
    static_max = max((row[target] for target in others), default=0)
    if pair_scores is None:
        return len(others), static_max, 0.0, 0.0, 0.0, 0.0, 0
    static_pairs = 0.0
    for position, target in enumerate(others):
        scores = pair_scores[target]
        for other in others[position + 1:]:
            static_pairs += scores[other]
    left_scores = pair_scores[left_target]
    right_scores = pair_scores[right_target]
    left_cross = 0.0
    right_cross = 0.0
    for target in others:
        left_cross += left_scores[target]
        right_cross += right_scores[target]
    static_self = 0.0
    static_repeats = 0
    if include_self:
        for target in others:
            if row[target] >= 2:
                static_self += pair_scores[target][target]
                static_repeats += 1
    return (
        len(others), static_max, static_pairs, left_cross, right_cross,
        static_self, static_repeats,
    )


def _moving_row_objective(
    left_value: int,
    right_value: int,
    left_geometry: float,
    right_geometry: float,
    static_count: int,
    static_pairs: float,
    left_cross: float,
    right_cross: float,
    moving_pair_score: float,
    static_self: float = 0.0,
    static_repeats: int = 0,
    left_self_score: float = 0.0,
    right_self_score: float = 0.0,
    *,
    geometry_weight: float,
    cotargeting_weight: float,
    geometry_repeat_discount: float,
    cotarget_aggregation: str,
    cotarget_site_clip: float | None,
    self_repeat_penalty: float,
    cotarget_include_self: bool = False,
) -> float:
    """Score one row's two moving columns, given ``_static_row_part`` for the rest.

    This is ``_row_objective`` minus the terms that are identical for every candidate
    in a step. Those cancel in the conditional distribution, so only the moving part
    is rebuilt per candidate. The co-targeting term is not separable — its mean and
    clip act on the whole row — so the full pair sum is reassembled from the static
    part rather than differenced.
    """
    objective = 0.0
    if geometry_weight:
        geometry = 0.0
        if left_value:
            geometry += (1.0 + geometry_repeat_discount * (left_value - 1)) * left_geometry
        if right_value:
            geometry += (1.0 + geometry_repeat_discount * (right_value - 1)) * right_geometry
        objective = geometry_weight * geometry
    if cotargeting_weight:
        occupied = static_count + (left_value > 0) + (right_value > 0)
        # Every distinct-pair component is already zero when fewer than two targets
        # are occupied, so the running term count is the only guard needed.
        pairs = static_pairs
        if left_value:
            pairs += left_cross
        if right_value:
            pairs += right_cross
        if left_value and right_value:
            pairs += moving_pair_score
        terms = occupied * (occupied - 1) // 2
        if cotarget_include_self:
            pairs += static_self
            terms += static_repeats
            if left_value >= 2:
                pairs += left_self_score
                terms += 1
            if right_value >= 2:
                pairs += right_self_score
                terms += 1
        if terms:
            if cotarget_aggregation == "mean":
                pairs /= terms
            if cotarget_site_clip is not None:
                if pairs < -cotarget_site_clip:
                    pairs = -cotarget_site_clip
                elif pairs > cotarget_site_clip:
                    pairs = cotarget_site_clip
            objective += cotargeting_weight * pairs
    if self_repeat_penalty:
        objective -= self_repeat_penalty * (
            left_value * (left_value - 1) / 2.0 + right_value * (right_value - 1) / 2.0
        )
    return objective


def gibbs_target_swaps_nonumba(
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
    geometry_repeat_discount: float, # 0.0 means repeats add  nothing, 0.5 repeats add half score, 1 repeat adds full score
    cotarget_aggregation: str,
    cotarget_site_clip: float | None,
    self_repeat_penalty: float,
    preserve_multiplicity_profile: bool,
    max_same_target_per_site: int | None,
    steps: int,
    cotarget_include_self: bool = False,
) -> np.ndarray:
    """Resample two-site target allocations without changing either margin.

    Each update selects two sites and two targets represented across those sites.
    It enumerates the complete feasible 2x2 table with the same row and column
    margins, then samples one table from its exact conditional distribution under
    the configured geometry/co-targeting objective.

    A candidate differs from the current state in four integer cells, so the loop
    works on those cells rather than on copied rows: see ``_static_row_part`` for the
    per-row constants and ``_moving_row_objective`` for the per-candidate score.

    This is the interpreted reference. ``gibbs_target_swaps`` runs the identical
    chain through a compiled kernel and is roughly forty times faster; the two are
    interchangeable, down to producing the same assignment for the same seed.
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

    uniforms = _UniformStream(rng)
    sites = active_sites.tolist()
    geometry_rows = geometry_scores.tolist()
    pair_scores = _pair_score_lookup(cotarget_scores) if cotargeting_weight else None
    row_scores = dict(
        geometry_weight=geometry_weight,
        cotargeting_weight=cotargeting_weight,
        geometry_repeat_discount=geometry_repeat_discount,
        cotarget_aggregation=cotarget_aggregation,
        cotarget_site_clip=cotarget_site_clip,
        self_repeat_penalty=self_repeat_penalty,
        cotarget_include_self=cotarget_include_self,
    )

    for _ in range(int(steps)):
        left_site, right_site = _distinct_pair(
            sites, uniforms.next(), uniforms.next()
        )
        left_row = result[left_site]
        right_row = result[right_site]
        pooled = np.flatnonzero(left_row + right_row).tolist()
        if len(pooled) < 2:
            continue
        left_target, right_target = _distinct_pair(
            pooled, uniforms.next(), uniforms.next()
        )
        left_values = left_row.tolist()
        right_values = right_row.tolist()

        # A feasible 2x2 table is determined by its upper-left value. The other
        # three cells follow from the fixed two row and two column subtotals.
        left_subtotal = left_values[left_target] + left_values[right_target]
        right_subtotal = right_values[left_target] + right_values[right_target]
        left_target_subtotal = left_values[left_target] + right_values[left_target]
        lower = max(0, left_target_subtotal - right_subtotal)
        upper = min(left_subtotal, left_target_subtotal)
        if upper < lower:
            continue

        (
            left_static_count, left_static_max,
            left_static_pairs, left_left_cross, left_right_cross,
            left_static_self, left_static_repeats,
        ) = _static_row_part(
            left_values, pooled, left_target, right_target, pair_scores,
            cotarget_include_self,
        )
        (
            right_static_count, right_static_max,
            right_static_pairs, right_left_cross, right_right_cross,
            right_static_self, right_static_repeats,
        ) = _static_row_part(
            right_values, pooled, left_target, right_target, pair_scores,
            cotarget_include_self,
        )
        left_geometry = geometry_rows[left_site]
        right_geometry = geometry_rows[right_site]
        moving_pair_score = (
            pair_scores[left_target][right_target] if pair_scores is not None else 0.0
        )
        if pair_scores is not None and cotarget_include_self:
            left_self_score = pair_scores[left_target][left_target]
            right_self_score = pair_scores[right_target][right_target]
        else:
            left_self_score = 0.0
            right_self_score = 0.0

        # Only the moving cells can alter a row's multiplicity profile, so comparing
        # their non-zero values against the current ones settles the whole row.
        left_profile = sorted(
            value for value in (left_values[left_target], left_values[right_target]) if value
        )
        right_profile = sorted(
            value for value in (right_values[left_target], right_values[right_target]) if value
        )

        candidates: list[tuple[int, int, int, int]] = []
        objectives: list[float] = []
        for new_left_left in range(lower, upper + 1):
            new_left_right = left_subtotal - new_left_left
            new_right_left = left_target_subtotal - new_left_left
            new_right_right = right_subtotal - new_right_left
            if max_same_target_per_site is not None and (
                max(left_static_max, new_left_left, new_left_right)
                > max_same_target_per_site
                or max(right_static_max, new_right_left, new_right_right)
                > max_same_target_per_site
            ):
                continue
            if preserve_multiplicity_profile and (
                sorted(value for value in (new_left_left, new_left_right) if value)
                != left_profile
                or sorted(value for value in (new_right_left, new_right_right) if value)
                != right_profile
            ):
                continue
            objective = _moving_row_objective(
                new_left_left, new_left_right,
                left_geometry[left_target], left_geometry[right_target],
                left_static_count, left_static_pairs,
                left_left_cross, left_right_cross, moving_pair_score,
                left_static_self, left_static_repeats,
                left_self_score, right_self_score,
                **row_scores,
            ) + _moving_row_objective(
                new_right_left, new_right_right,
                right_geometry[left_target], right_geometry[right_target],
                right_static_count, right_static_pairs,
                right_left_cross, right_right_cross, moving_pair_score,
                right_static_self, right_static_repeats,
                left_self_score, right_self_score,
                **row_scores,
            )
            candidates.append(
                (new_left_left, new_left_right, new_right_left, new_right_right)
            )
            objectives.append(objective)

        if len(candidates) < 2:
            continue
        # Softmax over a handful of candidates, then an inverse-CDF draw. Both stay in
        # Python: a two-element NumPy exp/normalise costs more than the loop it saves.
        largest = max(objectives)
        # subtracting largest is a stability measure, prevent unnecessarily large exponentials 
        # without changing relative probabilities
        weights = [exp((objective - largest) / temperature) for objective in objectives]
        threshold = uniforms.next() * sum(weights)
        cumulative = 0.0
        chosen = len(weights) - 1
        for index, weight in enumerate(weights):
            cumulative += weight
            if threshold < cumulative:
                chosen = index
                break
        new_left_left, new_left_right, new_right_left, new_right_right = candidates[chosen]
        result[left_site, left_target] = new_left_left
        result[left_site, right_target] = new_left_right
        result[right_site, left_target] = new_right_left
        result[right_site, right_target] = new_right_right
    validate_assignment(result, degrees, counts)
    return result


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
    cotarget_include_self: bool = False,
) -> np.ndarray:
    """Run ``gibbs_target_swaps_nonumba``'s chain through the compiled kernel.

    Same arguments, same validation, same chain: for a given seed this returns the
    assignment the interpreted reference returns, cell for cell, because the
    uniforms are pre-drawn in the blocks ``_UniformStream`` would have used. The
    two are therefore substitutable in either direction, and
    ``test_numba_matches_the_python_reference`` checks that across the option grid.

    Without numba installed this delegates to the reference rather than running the
    kernel uncompiled, which would be far slower than the interpreted loop the
    reference was tuned into.
    """
    if not NUMBA_AVAILABLE:
        return gibbs_target_swaps_nonumba(
            initial_assignment, site_degrees, target_counts,
            geometry_scores, cotarget_scores, rng,
            geometry_weight=geometry_weight,
            cotargeting_weight=cotargeting_weight,
            temperature=temperature,
            geometry_repeat_discount=geometry_repeat_discount,
            cotarget_aggregation=cotarget_aggregation,
            cotarget_site_clip=cotarget_site_clip,
            self_repeat_penalty=self_repeat_penalty,
            preserve_multiplicity_profile=preserve_multiplicity_profile,
            max_same_target_per_site=max_same_target_per_site,
            steps=steps,
            cotarget_include_self=cotarget_include_self,
        )

    result = np.asarray(initial_assignment, dtype=np.int64).copy()
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

    steps = int(steps)
    gibbs_kernel(
        result,
        np.ascontiguousarray(active_sites, dtype=np.int64),
        np.ascontiguousarray(geometry_scores),
        np.ascontiguousarray(_pair_score_matrix(cotarget_scores)),
        draw_uniform_blocks(rng, steps),
        float(geometry_weight),
        float(cotargeting_weight),
        float(temperature),
        float(geometry_repeat_discount),
        cotarget_aggregation == "mean",
        NO_CLIP if cotarget_site_clip is None else float(cotarget_site_clip),
        float(self_repeat_penalty),
        bool(preserve_multiplicity_profile),
        NO_CAP if max_same_target_per_site is None else int(max_same_target_per_site),
        bool(cotarget_include_self),
        bool(cotargeting_weight),
        steps,
        # A feasible table's upper bound never exceeds a site degree, so this many
        # candidates always fit.
        int(degrees.max(initial=0)) + 2,
    )
    validate_assignment(result, degrees, counts)
    return result


# Retain the old public name for existing notebooks and scripts.
metropolis_target_swaps = gibbs_target_swaps
