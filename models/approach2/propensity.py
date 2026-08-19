"""Distinct-target co-occurrence matrices and individual-neuron propensities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CotargetConfig


def pair_count_matrix(assignment: np.ndarray) -> np.ndarray:
    """Number of sites containing each unordered pair of distinct targets."""
    assignment = np.asarray(assignment)
    present = assignment > 0
    # Integer accumulation is exact and avoids spurious floating-point BLAS
    # overflow warnings seen for these small binary matrices on some NumPy builds.
    integer_presence = present.astype(np.int64)
    result = integer_presence.T @ integer_presence
    np.fill_diagonal(result, 0)
    return result


def mean_pair_count_matrix(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples)
    if samples.ndim != 3:
        raise ValueError("Samples must have shape samples x sites x targets")
    return np.mean([pair_count_matrix(sample) for sample in samples], axis=0)


def _standardize_symmetric_scores(
    scores: np.ndarray,
    standardize: bool,
    clip: float | None,
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
        else:
            result.fill(0.0)
    if clip is not None:
        result = np.clip(result, -clip, clip)
    np.fill_diagonal(result, 0.0)
    return result


def estimate_individual_propensity(
    observed_assignment: np.ndarray,
    expected_samples: np.ndarray,
    config: CotargetConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate shrunk log enrichment relative to a simulated expectation."""
    observed = pair_count_matrix(observed_assignment)
    expected = mean_pair_count_matrix(expected_samples)
    raw = np.log((observed + config.smoothing) / (expected + config.smoothing))
    support = observed + expected
    weight = support / (support + config.shrinkage)
    shrunk = weight * raw
    np.fill_diagonal(shrunk, 0.0)
    scores = _standardize_symmetric_scores(
        shrunk, config.standardize_scores, config.score_clip
    )
    return scores, observed, expected


def load_supplied_propensity(
    path: str | Path,
    target_ids: np.ndarray,
    config: CotargetConfig,
) -> np.ndarray:
    """Read target_a,target_b,score rows; unspecified requested pairs get zero."""
    table = pd.read_csv(path)
    required = {"target_a", "target_b", "score"}
    if not required <= set(table.columns):
        raise ValueError(f"Supplied co-target file requires columns {sorted(required)}")
    lookup = {int(target): index for index, target in enumerate(target_ids)}
    result = np.zeros((len(target_ids), len(target_ids)), dtype=float)
    for row in table.itertuples(index=False):
        left = lookup.get(int(row.target_a))
        right = lookup.get(int(row.target_b))
        if left is None or right is None or left == right:
            continue
        result[left, right] = float(row.score)
        result[right, left] = float(row.score)
    return _standardize_symmetric_scores(
        result, config.standardize_scores, config.score_clip
    )
