"""Compiled inner loop for the conditional Gibbs sampler.

``gibbs_kernel`` is a line-for-line port of the loop in
``gibbs_target_swaps_nonumba``. It exists only to run that same chain faster, so
the two must agree exactly rather than merely in distribution: given the same
uniform sequence they produce the identical assignment, which
``test_numba_matches_the_python_reference`` pins down.

Three things have to change shape to survive ``njit``:

* ``None`` cannot be typed, so the optional clip and multiplicity cap arrive as
  the ``NO_CLIP``/``NO_CAP`` sentinels. Both sit outside the range
  ``Approach2Config.validate`` permits, so no real setting can collide with them.
* String options cannot be compared cheaply, so ``cotarget_aggregation`` arrives
  as the boolean ``use_mean``.
* ``np.random.Generator`` cannot be passed in, so the caller pre-draws the
  uniforms in ``UNIFORM_BLOCK``-sized blocks, exactly as ``_UniformStream``
  would have drawn them lazily. That is what keeps existing seeds reproducible.

The list-of-lists pair lookup that helps CPython is a pessimisation here, so the
kernel takes the symmetric score matrix directly.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without numba installed
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        """Leave the function uncompiled; callers dispatch on NUMBA_AVAILABLE."""
        if args and callable(args[0]):
            return args[0]
        return lambda function: function


# Sentinels for arguments the reference implementation types as optional.
NO_CLIP = np.inf
NO_CAP = -1
# _UniformStream hands out uniforms from blocks of this size, and a step consumes
# at most five: two to pick the sites, two to pick the targets, one to choose the
# candidate. Matching the block size keeps the drawn sequence identical.
UNIFORM_BLOCK = 8192
MAX_UNIFORMS_PER_STEP = 5


def draw_uniform_blocks(rng: np.random.Generator, steps: int) -> np.ndarray:
    """Pre-draw the uniforms ``_UniformStream(rng)`` would hand out over ``steps``.

    Blocks come from the supplied generator in the same order and the same size,
    so the concatenation is the sequence the reference implementation consumes.
    Rounding up to whole blocks can advance the generator past what a step budget
    strictly needs, which is harmless because every caller seeds a fresh
    generator per chain.
    """
    blocks = max(1, math.ceil(MAX_UNIFORMS_PER_STEP * steps / UNIFORM_BLOCK))
    return np.concatenate([rng.random(UNIFORM_BLOCK) for _ in range(blocks)])


@njit(cache=True, nogil=True)
def _profile(first, second):
    """The row's two moving values, as (count of non-zeros, smaller, larger).

    Comparing this triple is the typed stand-in for comparing the sorted non-zero
    lists that ``preserve_multiplicity_profile`` builds in the reference.
    """
    if first and second:
        if first <= second:
            return 2, first, second
        return 2, second, first
    if first:
        return 1, first, 0
    if second:
        return 1, second, 0
    return 0, 0, 0


@njit(cache=True, nogil=True)
def _moving_objective(
    left_value, right_value, left_geometry, right_geometry,
    static_count, static_pairs, left_cross, right_cross, moving_pair_score,
    static_self, static_repeats, left_self_score, right_self_score,
    geometry_weight, cotargeting_weight, geometry_repeat_discount,
    use_mean, site_clip, self_repeat_penalty, include_self,
):
    """Typed port of ``_moving_row_objective``; see that function for the reasoning."""
    objective = 0.0
    if geometry_weight != 0.0:
        geometry = 0.0
        if left_value:
            geometry += (1.0 + geometry_repeat_discount * (left_value - 1)) * left_geometry
        if right_value:
            geometry += (1.0 + geometry_repeat_discount * (right_value - 1)) * right_geometry
        objective = geometry_weight * geometry
    if cotargeting_weight != 0.0:
        occupied = static_count
        if left_value > 0:
            occupied += 1
        if right_value > 0:
            occupied += 1
        pairs = static_pairs
        if left_value:
            pairs += left_cross
        if right_value:
            pairs += right_cross
        if left_value and right_value:
            pairs += moving_pair_score
        terms = occupied * (occupied - 1) // 2
        if include_self:
            pairs += static_self
            terms += static_repeats
            if left_value >= 2:
                pairs += left_self_score
                terms += 1
            if right_value >= 2:
                pairs += right_self_score
                terms += 1
        if terms:
            if use_mean:
                pairs /= terms
            if site_clip != NO_CLIP:
                if pairs < -site_clip:
                    pairs = -site_clip
                elif pairs > site_clip:
                    pairs = site_clip
            objective += cotargeting_weight * pairs
    if self_repeat_penalty != 0.0:
        objective -= self_repeat_penalty * (
            left_value * (left_value - 1) / 2.0 + right_value * (right_value - 1) / 2.0
        )
    return objective


@njit(cache=True, nogil=True)
def gibbs_kernel(
    result, sites, geometry, pair_scores, uniforms,
    geometry_weight, cotargeting_weight, temperature, geometry_repeat_discount,
    use_mean, site_clip, self_repeat_penalty, preserve_profile, max_same,
    include_self, use_pairs, steps, buffer_size,
):
    """Run the two-site conditional Gibbs chain in place on ``result``.

    Returns how many uniforms were consumed, which the tests use to confirm the
    budget in ``draw_uniform_blocks`` is never exceeded.
    """
    n_targets = result.shape[1]
    n_sites = len(sites)
    # Scratch reused across steps: the kernel allocates nothing in the loop, which
    # is most of what it buys over the interpreted version.
    pooled = np.empty(n_targets, dtype=np.int64)
    others = np.empty(n_targets, dtype=np.int64)
    candidates = np.empty((buffer_size, 4), dtype=np.int64)
    objectives = np.empty(buffer_size, dtype=np.float64)
    weights = np.empty(buffer_size, dtype=np.float64)
    cursor = 0

    for _ in range(steps):
        # _distinct_pair, inlined: the second index skips past the first.
        first = int(uniforms[cursor] * n_sites)
        cursor += 1
        second = int(uniforms[cursor] * (n_sites - 1))
        cursor += 1
        if second >= first:
            second += 1
        left_site = sites[first]
        right_site = sites[second]

        n_pooled = 0
        for target in range(n_targets):
            if result[left_site, target] + result[right_site, target] > 0:
                pooled[n_pooled] = target
                n_pooled += 1
        if n_pooled < 2:
            continue

        first = int(uniforms[cursor] * n_pooled)
        cursor += 1
        second = int(uniforms[cursor] * (n_pooled - 1))
        cursor += 1
        if second >= first:
            second += 1
        left_target = pooled[first]
        right_target = pooled[second]

        left_ll = result[left_site, left_target]
        left_rr = result[left_site, right_target]
        right_ll = result[right_site, left_target]
        right_rr = result[right_site, right_target]
        left_subtotal = left_ll + left_rr
        right_subtotal = right_ll + right_rr
        left_target_subtotal = left_ll + right_ll
        lower = left_target_subtotal - right_subtotal
        if lower < 0:
            lower = 0
        upper = left_subtotal
        if left_target_subtotal < upper:
            upper = left_target_subtotal
        if upper < lower:
            continue

        # _static_row_part for both rows: everything the moving columns cannot change.
        left_static_count = 0
        left_static_max = 0
        left_static_pairs = 0.0
        left_left_cross = 0.0
        left_right_cross = 0.0
        left_static_self = 0.0
        left_static_repeats = 0
        right_static_count = 0
        right_static_max = 0
        right_static_pairs = 0.0
        right_left_cross = 0.0
        right_right_cross = 0.0
        right_static_self = 0.0
        right_static_repeats = 0

        for row_index in range(2):
            site = left_site if row_index == 0 else right_site
            n_others = 0
            static_max = 0
            for index in range(n_pooled):
                target = pooled[index]
                if target == left_target or target == right_target:
                    continue
                value = result[site, target]
                if value:
                    others[n_others] = target
                    n_others += 1
                    if value > static_max:
                        static_max = value
            static_pairs = 0.0
            left_cross = 0.0
            right_cross = 0.0
            static_self = 0.0
            static_repeats = 0
            if use_pairs:
                for position in range(n_others):
                    target = others[position]
                    for other_position in range(position + 1, n_others):
                        static_pairs += pair_scores[target, others[other_position]]
                for position in range(n_others):
                    target = others[position]
                    left_cross += pair_scores[left_target, target]
                    right_cross += pair_scores[right_target, target]
                if include_self:
                    for position in range(n_others):
                        target = others[position]
                        if result[site, target] >= 2:
                            static_self += pair_scores[target, target]
                            static_repeats += 1
            if row_index == 0:
                left_static_count = n_others
                left_static_max = static_max
                left_static_pairs = static_pairs
                left_left_cross = left_cross
                left_right_cross = right_cross
                left_static_self = static_self
                left_static_repeats = static_repeats
            else:
                right_static_count = n_others
                right_static_max = static_max
                right_static_pairs = static_pairs
                right_left_cross = left_cross
                right_right_cross = right_cross
                right_static_self = static_self
                right_static_repeats = static_repeats

        moving_pair_score = pair_scores[left_target, right_target] if use_pairs else 0.0
        if use_pairs and include_self:
            left_self_score = pair_scores[left_target, left_target]
            right_self_score = pair_scores[right_target, right_target]
        else:
            left_self_score = 0.0
            right_self_score = 0.0

        left_count, left_small, left_large = _profile(left_ll, left_rr)
        right_count, right_small, right_large = _profile(right_ll, right_rr)

        n_candidates = 0
        for new_left_left in range(lower, upper + 1):
            new_left_right = left_subtotal - new_left_left
            new_right_left = left_target_subtotal - new_left_left
            new_right_right = right_subtotal - new_right_left
            if max_same != NO_CAP:
                biggest = left_static_max
                if new_left_left > biggest:
                    biggest = new_left_left
                if new_left_right > biggest:
                    biggest = new_left_right
                if biggest > max_same:
                    continue
                biggest = right_static_max
                if new_right_left > biggest:
                    biggest = new_right_left
                if new_right_right > biggest:
                    biggest = new_right_right
                if biggest > max_same:
                    continue
            if preserve_profile:
                count, small, large = _profile(new_left_left, new_left_right)
                if count != left_count or small != left_small or large != left_large:
                    continue
                count, small, large = _profile(new_right_left, new_right_right)
                if count != right_count or small != right_small or large != right_large:
                    continue
            objective = _moving_objective(
                new_left_left, new_left_right,
                geometry[left_site, left_target], geometry[left_site, right_target],
                left_static_count, left_static_pairs,
                left_left_cross, left_right_cross, moving_pair_score,
                left_static_self, left_static_repeats,
                left_self_score, right_self_score,
                geometry_weight, cotargeting_weight, geometry_repeat_discount,
                use_mean, site_clip, self_repeat_penalty, include_self,
            ) + _moving_objective(
                new_right_left, new_right_right,
                geometry[right_site, left_target], geometry[right_site, right_target],
                right_static_count, right_static_pairs,
                right_left_cross, right_right_cross, moving_pair_score,
                right_static_self, right_static_repeats,
                left_self_score, right_self_score,
                geometry_weight, cotargeting_weight, geometry_repeat_discount,
                use_mean, site_clip, self_repeat_penalty, include_self,
            )
            candidates[n_candidates, 0] = new_left_left
            candidates[n_candidates, 1] = new_left_right
            candidates[n_candidates, 2] = new_right_left
            candidates[n_candidates, 3] = new_right_right
            objectives[n_candidates] = objective
            n_candidates += 1

        if n_candidates < 2:
            continue
        # Softmax then inverse-CDF draw, accumulated in the reference's order so the
        # floating-point result is identical rather than merely close.
        largest = objectives[0]
        for index in range(1, n_candidates):
            if objectives[index] > largest:
                largest = objectives[index]
        total = 0.0
        for index in range(n_candidates):
            weights[index] = math.exp((objectives[index] - largest) / temperature)
            total += weights[index]
        threshold = uniforms[cursor] * total
        cursor += 1
        cumulative = 0.0
        chosen = n_candidates - 1
        for index in range(n_candidates):
            cumulative += weights[index]
            if threshold < cumulative:
                chosen = index
                break
        result[left_site, left_target] = candidates[chosen, 0]
        result[left_site, right_target] = candidates[chosen, 1]
        result[right_site, left_target] = candidates[chosen, 2]
        result[right_site, right_target] = candidates[chosen, 3]
    return cursor
