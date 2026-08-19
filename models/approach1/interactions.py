"""Training-only target priors and shrunk co-targeting propensities."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Mapping

import pandas as pd


@dataclass
class TargetIdentityPrior:
    smoothing: float = 1.0
    counts: dict[int, int] = field(default_factory=dict)
    total: int = 0

    def fit(self, connector_rows: pd.DataFrame) -> "TargetIdentityPrior":
        counter: Counter[int] = Counter()
        for targets in connector_rows["postsynaptic_to"]:
            counter.update(int(target) for target in targets)
        self.counts = dict(counter)
        self.total = int(sum(counter.values()))
        return self

    def log_probability(self, target_id: int) -> float:
        vocabulary = max(len(self.counts), 1)
        numerator = self.counts.get(int(target_id), 0) + self.smoothing
        denominator = self.total + self.smoothing * vocabulary
        return math.log(numerator / max(denominator, 1e-12))


@dataclass
class CoTargetingModel:
    smoothing: float = 1.0
    shrinkage: float = 10.0
    minimum_observations: int = 2
    identity_scores: dict[tuple[int, int], float] = field(default_factory=dict)
    celltype_scores: dict[tuple[str, str], float] = field(default_factory=dict)

    @staticmethod
    def _ordered_pair(left, right):
        return (left, right) if left <= right else (right, left)

    def _scores(
        self,
        marginal: Counter,
        joint: Counter,
        n_sites: int,
    ) -> dict:
        result = {}
        for pair, observations in joint.items():
            if observations < self.minimum_observations:
                continue
            left, right = pair
            raw = math.log(
                ((observations + self.smoothing) * max(n_sites, 1))
                / ((marginal[left] + self.smoothing) * (marginal[right] + self.smoothing))
            )
            weight = observations / (observations + self.shrinkage)
            result[pair] = weight * raw
        return result

    def fit(self, connector_rows: pd.DataFrame, celltypes: Mapping[int, str]) -> "CoTargetingModel":
        identity_marginal: Counter[int] = Counter()
        identity_joint: Counter[tuple[int, int]] = Counter()
        type_marginal: Counter[str] = Counter()
        type_joint: Counter[tuple[str, str]] = Counter()
        n_sites = len(connector_rows)
        for targets in connector_rows["postsynaptic_to"]:
            identities = sorted({int(target) for target in targets})
            identity_marginal.update(identities)
            identity_joint.update(self._ordered_pair(left, right) for left, right in combinations(identities, 2))
            types = sorted({celltypes.get(target, "unknown") for target in identities})
            type_marginal.update(types)
            type_joint.update(self._ordered_pair(left, right) for left, right in combinations(types, 2))
        self.identity_scores = self._scores(identity_marginal, identity_joint, n_sites)
        self.celltype_scores = self._scores(type_marginal, type_joint, n_sites)
        return self

    def score(self, left: int, right: int, celltypes: Mapping[int, str]) -> float:
        if int(left) == int(right):
            return 0.0
        identity_pair = self._ordered_pair(int(left), int(right))
        if identity_pair in self.identity_scores:
            return self.identity_scores[identity_pair]
        left_type = celltypes.get(int(left), "unknown")
        right_type = celltypes.get(int(right), "unknown")
        return self.celltype_scores.get(self._ordered_pair(left_type, right_type), 0.0)
