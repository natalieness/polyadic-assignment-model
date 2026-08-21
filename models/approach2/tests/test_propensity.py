"""Tests for co-occurrence counting and individual co-targeting propensities."""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from models.approach2.config import CotargetConfig
from models.approach2.propensity import (
    _standardize_symmetric_scores,
    estimate_individual_propensity,
    load_supplied_propensity,
    mean_pair_count_matrix,
    pair_count_matrix,
)


# Target 0 repeats at the first site, target 1 at the last, target 2 never repeats.
OBSERVED = np.array([
    [3, 0, 0],
    [1, 1, 0],
    [0, 2, 1],
])
# Same site degrees and target totals, different arrangement and different repeats.
EXPECTED_SAMPLE = np.array([
    [1, 1, 1],
    [1, 1, 0],
    [2, 1, 0],
])
OFF_DIAGONAL = ~np.eye(3, dtype=bool)


def _config(**overrides) -> CotargetConfig:
    return replace(CotargetConfig(), **overrides)


def test_pair_counts_stay_in_site_units() -> None:
    counts = pair_count_matrix(OBSERVED)
    assert np.array_equal(np.diagonal(counts), [0, 0, 0])
    # A site is counted once for a pair however many contacts it holds.
    assert counts[0, 1] == 1
    assert counts[1, 2] == 1
    assert counts[0, 2] == 0


def test_self_repeat_diagonal_counts_sites_not_contact_pairs() -> None:
    counts = pair_count_matrix(OBSERVED, include_self_repeats=True)
    # Site units: one site each for targets 0 and 1, never for target 2. A contact
    # pair count would have read 3 for target 0, which is the unit mistake to avoid.
    assert np.array_equal(np.diagonal(counts), [1, 1, 0])
    assert np.array_equal(
        counts[OFF_DIAGONAL], pair_count_matrix(OBSERVED)[OFF_DIAGONAL]
    )


def test_mean_pair_counts_forward_the_flag() -> None:
    samples = np.stack([OBSERVED, OBSERVED])
    assert np.array_equal(np.diagonal(mean_pair_count_matrix(samples)), [0, 0, 0])
    assert np.array_equal(
        np.diagonal(mean_pair_count_matrix(samples, include_self_repeats=True)),
        [1.0, 1.0, 0.0],
    )


def test_self_scores_are_scaled_by_the_pair_score_spread() -> None:
    scores = np.array([
        [4.0, 1.0, -1.0],
        [1.0, 4.0, 3.0],
        [-1.0, 3.0, 4.0],
    ])
    result = _standardize_symmetric_scores(scores, True, None, True)
    sd = float(np.std([1.0, -1.0, 3.0]))
    assert np.allclose(np.diagonal(result), 4.0 / sd)
    assert np.allclose(result[0, 1], 1.0 / sd)
    # Without the flag the diagonal is discarded, as before.
    assert np.array_equal(
        np.diagonal(_standardize_symmetric_scores(scores, True, None, False)),
        np.zeros(3),
    )


def test_propensity_diagonal_appears_only_when_requested() -> None:
    expectation = np.stack([EXPECTED_SAMPLE, EXPECTED_SAMPLE])
    without, _, _ = estimate_individual_propensity(
        OBSERVED, expectation, _config(include_self_repeats=False)
    )
    with_self, _, _ = estimate_individual_propensity(
        OBSERVED, expectation, _config(include_self_repeats=True)
    )
    assert np.array_equal(np.diagonal(without), np.zeros(3))
    assert np.any(np.diagonal(with_self) != 0.0)
    # Target 1 repeats more often than expected, target 0 exactly as expected.
    assert np.diagonal(with_self)[1] > 0.0
    assert np.isclose(np.diagonal(with_self)[0], 0.0)
    # Enabling self repeats must leave every pair score untouched, so the flag is
    # safe to switch on without re-interpreting the off-diagonal numbers.
    assert np.allclose(with_self[OFF_DIAGONAL], without[OFF_DIAGONAL])


def test_supplied_scores_read_self_rows_only_when_requested() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "scores.csv"
        path.write_text(
            "target_a,target_b,score\n"
            "10,20,2.0\n"
            "10,30,-2.0\n"
            "20,20,4.0\n"
        )
        target_ids = np.array([10, 20, 30])
        without = load_supplied_propensity(path, target_ids, _config())
        with_self = load_supplied_propensity(
            path, target_ids, _config(include_self_repeats=True)
        )
    assert np.array_equal(np.diagonal(without), np.zeros(3))
    assert with_self[1, 1] > 0.0
    assert np.allclose(with_self[OFF_DIAGONAL], without[OFF_DIAGONAL])


if __name__ == "__main__":
    test_pair_counts_stay_in_site_units()
    test_self_repeat_diagonal_counts_sites_not_contact_pairs()
    test_mean_pair_counts_forward_the_flag()
    test_self_scores_are_scaled_by_the_pair_score_spread()
    test_propensity_diagonal_appears_only_when_requested()
    test_supplied_scores_read_self_rows_only_when_requested()
    print("approach2 propensity tests passed")
