"""Focused tests for the exact-margin Approach 2 sampler."""

from __future__ import annotations

import numpy as np

from models.approach2.propensity import pair_count_matrix
from models.approach2.sampler import gibbs_target_swaps, validate_assignment


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


if __name__ == "__main__":
    test_gibbs_sampler_preserves_margins_and_responds_to_pair_scores()
    print("approach2 sampler test passed")
