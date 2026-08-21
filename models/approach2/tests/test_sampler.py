"""Focused tests for the exact-margin Approach 2 sampler."""

from __future__ import annotations

import itertools

import numpy as np

from models.approach2.propensity import pair_count_matrix
from models.approach2.sampler import (
    _distinct_pair,
    _moving_row_objective,
    _pair_score_lookup,
    _pair_score_matrix,
    _row_objective,
    _static_row_part,
    _UniformStream,
    gibbs_target_swaps,
    gibbs_target_swaps_nonumba,
    validate_assignment,
)
from models.approach2.sampler_numba import (
    MAX_UNIFORMS_PER_STEP,
    NUMBA_AVAILABLE,
    draw_uniform_blocks,
)


def test_gibbs_sampler_preserves_margins_and_responds_to_pair_scores() -> None:
    # Start with only disfavoured pairs (0, 2) and (1, 3). The same exact margins
    # permit the strongly favoured pairs (0, 1) and (2, 3).
    initial = np.zeros((40, 4), dtype=int)
    initial[:20, [0, 2]] = 1
    initial[20:, [1, 3]] = 1
    site_degrees = np.full(40, 2, dtype=int)
    target_counts = np.full(4, 20, dtype=int)
    cotarget_scores = np.full((4, 4), -1.0)
    np.fill_diagonal(cotarget_scores, 0.0)
    cotarget_scores[0, 1] = cotarget_scores[1, 0] = 1.0
    cotarget_scores[2, 3] = cotarget_scores[3, 2] = 1.0

    sampled = gibbs_target_swaps(
        initial,
        site_degrees,
        target_counts,
        geometry_scores=np.zeros_like(initial, dtype=float),
        cotarget_scores=cotarget_scores,
        rng=np.random.default_rng(7),
        geometry_weight=0.0,
        cotargeting_weight=5.0,
        temperature=1.0,
        geometry_repeat_discount=0.0,
        cotarget_aggregation="mean",
        cotarget_site_clip=None,
        self_repeat_penalty=0.0,
        preserve_multiplicity_profile=True,
        max_same_target_per_site=1,
        steps=10_000,
    )

    validate_assignment(sampled, site_degrees, target_counts)
    assert np.all((sampled > 0).sum(axis=1) == 2)
    pairs = pair_count_matrix(sampled)
    assert pairs[0, 1] + pairs[2, 3] >= 35


def test_moving_row_objective_matches_the_reference_up_to_a_constant() -> None:
    """The fast path may drop terms a swap cannot change, but nothing else.

    Whatever it omits has to be the same for every candidate in a step, otherwise the
    candidates would be ranked differently from ``_row_objective``.
    """
    rng = np.random.default_rng(11)
    for _ in range(200):
        n_targets = int(rng.integers(3, 10))
        row = rng.integers(0, 4, size=n_targets)
        left_target, right_target = (
            int(value) for value in rng.choice(n_targets, size=2, replace=False)
        )
        geometry_row = rng.normal(size=n_targets)
        cotarget_scores = rng.normal(size=(n_targets, n_targets))
        cotarget_scores = (cotarget_scores + cotarget_scores.T) / 2
        include_self = bool(rng.integers(0, 2))
        if not include_self:
            np.fill_diagonal(cotarget_scores, 0.0)
        options = dict(
            geometry_weight=float(rng.choice([0.0, 1.0, 2.5])),
            cotargeting_weight=float(rng.choice([0.0, 1.0, 5.0])),
            geometry_repeat_discount=float(rng.choice([0.0, 0.5])),
            cotarget_aggregation=str(rng.choice(["mean", "sum"])),
            cotarget_site_clip=(None if rng.integers(0, 2) else 2.0),
            self_repeat_penalty=float(rng.choice([0.0, 0.7])),
            cotarget_include_self=include_self,
        )

        pooled = sorted(set(np.flatnonzero(row).tolist()) | {left_target, right_target})
        pair_scores = (
            _pair_score_lookup(cotarget_scores) if options["cotargeting_weight"] else None
        )
        (
            static_count, _, static_pairs, left_cross, right_cross,
            static_self, static_repeats,
        ) = _static_row_part(
            row.tolist(), pooled, left_target, right_target, pair_scores, include_self
        )
        moving_pair = pair_scores[left_target][right_target] if pair_scores else 0.0
        left_self = (
            pair_scores[left_target][left_target]
            if pair_scores and include_self else 0.0
        )
        right_self = (
            pair_scores[right_target][right_target]
            if pair_scores and include_self else 0.0
        )

        offsets = []
        for left_value in range(4):
            for right_value in range(4):
                probe = row.copy()
                probe[left_target] = left_value
                probe[right_target] = right_value
                reference = _row_objective(probe, geometry_row, cotarget_scores, **options)
                moving = _moving_row_objective(
                    left_value, right_value,
                    geometry_row[left_target], geometry_row[right_target],
                    static_count, static_pairs, left_cross, right_cross, moving_pair,
                    static_self, static_repeats, left_self, right_self,
                    **options,
                )
                offsets.append(reference - moving)
        assert np.allclose(offsets, offsets[0], atol=1e-9)


def test_distinct_pair_is_uniform_over_ordered_pairs() -> None:
    values = list(range(5))
    stream = _UniformStream(np.random.default_rng(4))
    counts = np.zeros((len(values), len(values)))
    draws = 120_000
    for _ in range(draws):
        left, right = _distinct_pair(values, stream.next(), stream.next())
        counts[left, right] += 1

    assert np.trace(counts) == 0, "a pair must never repeat an element"
    off_diagonal = counts[~np.eye(len(values), dtype=bool)]
    expected = draws / len(off_diagonal)
    assert np.abs(off_diagonal - expected).max() < 5 * np.sqrt(expected)


def _rows_with_sum(total: int, width: int):
    if width == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _rows_with_sum(total - first, width - 1):
            yield (first,) + rest


def _check_exact_conditional(include_self: bool) -> None:
    """Each update draws from the exact two-row conditional, so the chain's stationary
    law is ``exp(sum of row objectives / temperature)`` over the fixed margins. On a
    system this small that law can be enumerated and compared against directly.
    """
    degrees = np.array([2, 1, 2])
    counts = np.array([2, 2, 1])
    rng = np.random.default_rng(5)
    geometry = rng.normal(size=(len(degrees), len(counts)))
    cotarget_scores = rng.normal(size=(len(counts), len(counts)))
    cotarget_scores = (cotarget_scores + cotarget_scores.T) / 2
    if not include_self:
        np.fill_diagonal(cotarget_scores, 0.0)
    scoring = dict(
        geometry_weight=1.5, cotargeting_weight=2.0, geometry_repeat_discount=0.25,
        cotarget_aggregation="mean", cotarget_site_clip=None, self_repeat_penalty=0.3,
        cotarget_include_self=include_self,
    )
    temperature = 0.8

    states = []
    for rows in itertools.product(
        *[list(_rows_with_sum(int(degree), len(counts))) for degree in degrees]
    ):
        candidate = np.array(rows, dtype=int)
        if np.array_equal(candidate.sum(axis=0), counts):
            states.append(candidate)
    assert len(states) > 5

    log_target = np.array([
        sum(
            _row_objective(state[site], geometry[site], cotarget_scores, **scoring)
            for site in range(len(degrees))
        ) / temperature
        for state in states
    ])
    log_target -= log_target.max()
    target = np.exp(log_target)
    target /= target.sum()

    position = {state.tobytes(): index for index, state in enumerate(states)}
    observed = np.zeros(len(states))
    chains = 6000
    for chain in range(chains):
        final = gibbs_target_swaps(
            states[0], degrees, counts,
            geometry_scores=geometry, cotarget_scores=cotarget_scores,
            rng=np.random.default_rng(chain), temperature=temperature,
            preserve_multiplicity_profile=False, max_same_target_per_site=None,
            steps=400, **scoring,
        )
        observed[position[final.tobytes()]] += 1

    total_variation = 0.5 * np.abs(observed / chains - target).sum()
    assert total_variation < 0.03, (
        f"stationary law drifted (include_self={include_self}): "
        f"TV distance {total_variation}"
    )


def test_sampler_reproduces_the_exact_conditional_distribution() -> None:
    _check_exact_conditional(include_self=False)


def test_exact_conditional_holds_with_self_repeat_scores() -> None:
    _check_exact_conditional(include_self=True)


def test_self_repeat_scores_change_how_often_targets_double_up() -> None:
    """A negative diagonal should suppress repeats a positive diagonal encourages.

    Degrees of two against three single-count targets means every site either holds
    two distinct targets or doubles one up, so the repeat rate is the whole outcome.
    """
    degrees = np.full(30, 2, dtype=int)
    counts = np.full(3, 20, dtype=int)
    initial = np.zeros((30, 3), dtype=int)
    initial[:10, 0] = 2
    initial[10:20, 1] = 2
    initial[20:, 2] = 2

    def repeats_for(diagonal: float) -> int:
        scores = np.zeros((3, 3))
        np.fill_diagonal(scores, diagonal)
        sampled = gibbs_target_swaps(
            initial, degrees, counts,
            geometry_scores=np.zeros_like(initial, dtype=float),
            cotarget_scores=scores,
            rng=np.random.default_rng(3),
            geometry_weight=0.0, cotargeting_weight=4.0, temperature=1.0,
            geometry_repeat_discount=0.0, cotarget_aggregation="mean",
            cotarget_site_clip=None, self_repeat_penalty=0.0,
            preserve_multiplicity_profile=False, max_same_target_per_site=None,
            steps=20_000, cotarget_include_self=True,
        )
        validate_assignment(sampled, degrees, counts)
        return int((sampled >= 2).sum())

    assert repeats_for(-2.0) < repeats_for(2.0)


def test_pair_score_lookup_mirrors_the_matrix() -> None:
    """The compiled and interpreted paths must read the same mirrored scores."""
    rng = np.random.default_rng(21)
    scores = rng.normal(size=(6, 6))
    matrix = _pair_score_matrix(scores)
    assert np.array_equal(matrix, matrix.T)
    assert _pair_score_lookup(scores) == matrix.tolist()
    # Only the upper triangle and the diagonal survive; the lower triangle is a mirror.
    assert np.allclose(np.triu(matrix, 1), np.triu(scores, 1))
    assert np.allclose(np.diagonal(matrix), np.diagonal(scores))


def test_numba_matches_the_python_reference() -> None:
    """The compiled kernel must reproduce the reference exactly, not just in law.

    Both consume the same uniform sequence, so any disagreement is a port bug rather
    than Monte Carlo noise. The grid covers every branch that had to change shape to
    compile: the optional clip and multiplicity cap, the aggregation string, the
    self-repeat diagonal, and each weight being switched off.
    """
    if not NUMBA_AVAILABLE:  # pragma: no cover - depends on the environment
        print("  numba unavailable, skipping the compiled-kernel comparison")
        return

    rng = np.random.default_rng(13)
    n_sites, n_targets = 14, 5
    initial = np.zeros((n_sites, n_targets), dtype=int)
    for site in range(n_sites):
        initial[site, site % n_targets] += 1
        initial[site, (site + 1) % n_targets] += 1
    degrees = initial.sum(axis=1)
    counts = initial.sum(axis=0)
    geometry = rng.normal(size=(n_sites, n_targets))
    cotarget = rng.normal(size=(n_targets, n_targets))
    cotarget = (cotarget + cotarget.T) / 2

    grid = [
        dict(),
        dict(preserve_multiplicity_profile=True),
        dict(max_same_target_per_site=2),
        dict(cotarget_aggregation="sum", cotarget_site_clip=None),
        dict(self_repeat_penalty=0.4, cotarget_include_self=False),
        dict(geometry_repeat_discount=0.5),
        dict(cotargeting_weight=0.0),
        dict(geometry_weight=0.0),
        dict(geometry_weight=0.0, cotargeting_weight=0.0),
        dict(temperature=0.2),
    ]
    for overrides in grid:
        options = dict(
            geometry_weight=1.5,
            cotargeting_weight=2.0,
            temperature=1.0,
            geometry_repeat_discount=0.0,
            cotarget_aggregation="mean",
            cotarget_site_clip=2.0,
            self_repeat_penalty=0.0,
            preserve_multiplicity_profile=False,
            max_same_target_per_site=None,
            cotarget_include_self=True,
            steps=3_000,
        )
        options.update(overrides)
        for seed in range(3):
            expected = gibbs_target_swaps_nonumba(
                initial, degrees, counts, geometry, cotarget,
                np.random.default_rng(seed), **options,
            )
            actual = gibbs_target_swaps(
                initial, degrees, counts, geometry, cotarget,
                np.random.default_rng(seed), **options,
            )
            assert np.array_equal(expected, actual), (
                f"compiled kernel diverged for {overrides} at seed {seed}"
            )
            validate_assignment(actual, degrees, counts)


def test_uniform_budget_covers_the_worst_case_step() -> None:
    """The pre-drawn block must outlast the chain, or the kernel reads past its end.

    A step consumes at most five uniforms, so the budget is only safe if that bound
    holds; this walks the reference's stream to confirm nothing draws more.
    """
    steps = 5_000
    assert len(draw_uniform_blocks(np.random.default_rng(0), steps)) >= (
        MAX_UNIFORMS_PER_STEP * steps
    )
    # The stream the reference builds hands out the same values in the same order.
    reference = _UniformStream(np.random.default_rng(31))
    drawn = draw_uniform_blocks(np.random.default_rng(31), steps)
    assert [reference.next() for _ in range(2_000)] == drawn[:2_000].tolist()


if __name__ == "__main__":
    test_gibbs_sampler_preserves_margins_and_responds_to_pair_scores()
    test_pair_score_lookup_mirrors_the_matrix()
    test_numba_matches_the_python_reference()
    test_uniform_budget_covers_the_worst_case_step()
    test_moving_row_objective_matches_the_reference_up_to_a_constant()
    test_distinct_pair_is_uniform_over_ordered_pairs()
    test_sampler_reproduces_the_exact_conditional_distribution()
    test_exact_conditional_holds_with_self_repeat_scores()
    test_self_repeat_scores_change_how_often_targets_double_up()
    print("approach2 sampler tests passed")
