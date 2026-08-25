"""Distinct-target co-occurrence matrices and individual-neuron propensities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CotargetConfig


def pair_count_matrix(
    assignment: np.ndarray, include_self_repeats: bool = False
) -> np.ndarray:
    """Number of sites containing each unordered pair of distinct targets.

    With ``include_self_repeats`` the diagonal holds the number of sites where a
    target appears more than once. That keeps the whole matrix in site units: the
    off diagonal counts sites, so the diagonal counts sites too rather than the
    ``m * (m - 1) / 2`` contact pairs a multiplicity-weighted matrix would use.
    """
    assignment = np.asarray(assignment)
    present = assignment > 0
    # Integer accumulation is exact and avoids spurious floating-point BLAS
    # overflow warnings seen for these small binary matrices on some NumPy builds.
    integer_presence = present.astype(np.int64)
    result = integer_presence.T @ integer_presence
    if include_self_repeats:
        np.fill_diagonal(result, (assignment >= 2).sum(axis=0))
    else:
        np.fill_diagonal(result, 0)
    return result


def mean_pair_count_matrix(
    samples: np.ndarray, include_self_repeats: bool = False
) -> np.ndarray:
    samples = np.asarray(samples)
    if samples.ndim != 3:
        raise ValueError("Samples must have shape samples x sites x targets")
    return np.mean(
        [pair_count_matrix(sample, include_self_repeats) for sample in samples],
        axis=0,
    )


def _standardize_symmetric_scores(
    scores: np.ndarray,
    standardize: bool,
    clip: float | None,
    include_self_repeats: bool = False,
) -> np.ndarray:
    result = np.asarray(scores, dtype=float).copy()
    triangle = np.triu_indices(len(result), 1)
    values = result[triangle]
    if standardize and len(values):
        sd = float(np.std(values))
        if sd > 1e-12:
            # Zero means neutral enrichment and must stay zero, especially for
            # unsupported or unspecified target pairs.
            result[triangle] = values / sd
            result[(triangle[1], triangle[0])] = result[triangle]
            if include_self_repeats:
                # Divide self scores by the same figure rather than their own spread,
                # so a diagonal entry and a pair entry stay comparable once the
                # sampler adds them into one weighted site objective.
                np.fill_diagonal(result, np.diagonal(result) / sd)
        else:
            result.fill(0.0)
    if clip is not None:
        result = np.clip(result, -clip, clip)
    if not include_self_repeats:
        np.fill_diagonal(result, 0.0)
    return result


def estimate_individual_propensity(
    observed_assignment: np.ndarray,
    expected_samples: np.ndarray,
    config: CotargetConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate shrunk log enrichment relative to a simulated expectation."""
    include_self = config.include_self_repeats
    observed = pair_count_matrix(observed_assignment, include_self)
    expected = mean_pair_count_matrix(expected_samples, include_self)
    raw = np.log((observed + config.smoothing) / (expected + config.smoothing))
    support = observed + expected
    weight = support / (support + config.shrinkage)
    shrunk = weight * raw
    if not include_self:
        np.fill_diagonal(shrunk, 0.0)
    scores = _standardize_symmetric_scores(
        shrunk, config.standardize_scores, config.score_clip, include_self
    )
    return scores, observed, expected


def load_supplied_propensity(
    path: str | Path,
    target_ids: np.ndarray,
    config: CotargetConfig,
    skeleton_id: int | None = None,
) -> np.ndarray:
    """Read target_a,target_b,score rows; unspecified requested pairs get zero.
    If 'pre' column, exists, filter to only the current skeleton_id if available.

    Rows with ``target_a == target_b`` carry a self-repeat score and are only read
    when ``config.include_self_repeats`` is set; otherwise they are skipped.
    """
    table = pd.read_csv(path)
    required = {"target_a", "target_b", "score"}
    if not required <= set(table.columns):
        raise ValueError(f"Supplied co-target file requires columns {sorted(required)}")
    if "pre" in table.columns and skeleton_id is not None:
        if skeleton_id in table["pre"].values:
            table = table[table["pre"] == skeleton_id]
        else:
            print(f"Warning: skeleton {skeleton_id} not found in supplied co-target file")
            print('Using all rows instead of filtering by "pre" column')
    include_self = config.include_self_repeats
    lookup = {int(target): index for index, target in enumerate(target_ids)}
    result = np.zeros((len(target_ids), len(target_ids)), dtype=float)
    for row in table.itertuples(index=False):
        left = lookup.get(int(row.target_a))
        right = lookup.get(int(row.target_b))
        if left is None or right is None:
            continue
        if left == right and not include_self:
            continue
        result[left, right] = float(row.score)
        result[right, left] = float(row.score)
    return _standardize_symmetric_scores(
        result, config.standardize_scores, config.score_clip, include_self
    )
