"""Small dependency-light feature encoder and regularized Poisson GLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass
class DictFeatureEncoder:
    """Encode numeric and low-cardinality categorical dictionary features."""

    numeric_names: list[str] = field(default_factory=list)
    category_values: dict[str, list[str]] = field(default_factory=dict)
    means: np.ndarray | None = None
    scales: np.ndarray | None = None

    def fit(self, records: Iterable[Mapping[str, Any]]) -> "DictFeatureEncoder":
        records = list(records)
        numeric: set[str] = set()
        categories: dict[str, set[str]] = {}
        for record in records:
            for key, value in record.items():
                if isinstance(value, (str, bool)):
                    categories.setdefault(key, set()).add(str(value))
                elif value is not None:
                    numeric.add(key)
        self.numeric_names = sorted(numeric)
        self.category_values = {key: sorted(values) for key, values in sorted(categories.items())}
        matrix = self._raw_transform(records)
        numeric_count = len(self.numeric_names)
        if numeric_count:
            self.means = matrix[:, :numeric_count].mean(axis=0)
            self.scales = matrix[:, :numeric_count].std(axis=0)
            self.scales[self.scales < 1e-12] = 1.0
        else:
            self.means = np.zeros(0)
            self.scales = np.ones(0)
        return self

    @property
    def feature_names(self) -> list[str]:
        result = list(self.numeric_names)
        for key, values in self.category_values.items():
            result.extend(f"{key}={value}" for value in values)
        return result

    def _raw_transform(self, records: Iterable[Mapping[str, Any]]) -> np.ndarray:
        records = list(records)
        columns = len(self.numeric_names) + sum(len(v) for v in self.category_values.values())
        matrix = np.zeros((len(records), columns), dtype=np.float64)
        for row_index, record in enumerate(records):
            for column_index, name in enumerate(self.numeric_names):
                value = record.get(name, 0.0)
                matrix[row_index, column_index] = 0.0 if value is None else float(value)
            offset = len(self.numeric_names)
            for key, values in self.category_values.items():
                value = str(record.get(key, "__missing__"))
                try:
                    category_index = values.index(value)
                except ValueError:
                    category_index = -1
                if category_index >= 0:
                    matrix[row_index, offset + category_index] = 1.0
                offset += len(values)
        return matrix

    def transform(self, records: Iterable[Mapping[str, Any]]) -> np.ndarray:
        matrix = self._raw_transform(records)
        numeric_count = len(self.numeric_names)
        if numeric_count:
            if self.means is None or self.scales is None:
                raise RuntimeError("Feature encoder is not fitted")
            matrix[:, :numeric_count] = (matrix[:, :numeric_count] - self.means) / self.scales
        return matrix

    def fit_transform(self, records: Iterable[Mapping[str, Any]]) -> np.ndarray:
        records = list(records)
        self.fit(records)
        return self.transform(records)


@dataclass
class PoissonGLM:
    """L2-regularized log-link Poisson regression fitted with damped Newton steps."""

    alpha: float = 1.0
    max_iter: int = 80
    tolerance: float = 1e-7
    coefficients: np.ndarray | None = None
    converged: bool = False
    iterations: int = 0

    @staticmethod
    def _augment(matrix: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(matrix)), matrix])

    @staticmethod
    def _objective(
        design: np.ndarray,
        target: np.ndarray,
        beta: np.ndarray,
        alpha: float,
        weights: np.ndarray | None = None,
    ) -> float:
        eta = np.clip(design @ beta, -20.0, 20.0)
        mu = np.exp(eta)
        penalty = 0.5 * alpha * float(beta[1:] @ beta[1:])
        weights = np.ones(len(target)) if weights is None else weights
        return float(np.mean(weights * (mu - target * eta)) + penalty)

    def fit(self, matrix: np.ndarray, target: np.ndarray, sample_weight: np.ndarray | None = None) -> "PoissonGLM":
        matrix = np.asarray(matrix, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if matrix.ndim != 2 or target.shape != (len(matrix),):
            raise ValueError("Invalid matrix/target shapes")
        if np.any(target < 0):
            raise ValueError("Poisson targets must be nonnegative")
        if not len(matrix):
            raise ValueError("Cannot fit an empty model")
        design = self._augment(matrix)
        weights = np.ones(len(target)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        weights = weights / max(float(weights.mean()), 1e-12)
        beta = np.zeros(design.shape[1], dtype=float)
        beta[0] = np.log(max(float(np.average(target, weights=weights)), 1e-4))
        regularizer = np.eye(design.shape[1]) * self.alpha
        regularizer[0, 0] = 0.0

        previous = self._objective(design, target, beta, self.alpha, weights)
        for iteration in range(1, self.max_iter + 1):
            eta = np.clip(design @ beta, -20.0, 20.0)
            mu = np.exp(eta)
            gradient = design.T @ (weights * (mu - target)) / len(target) + regularizer @ beta
            hessian = (design.T * (weights * mu)) @ design / len(target) + regularizer
            hessian.flat[:: hessian.shape[0] + 1] += 1e-8
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            scale = 1.0
            accepted = False
            while scale >= 1e-6:
                candidate = beta - scale * step
                objective = self._objective(design, target, candidate, self.alpha, weights)
                if np.isfinite(objective) and objective <= previous:
                    beta = candidate
                    accepted = True
                    break
                scale *= 0.5
            self.iterations = iteration
            if not accepted:
                break
            improvement = previous - objective
            previous = objective
            if improvement <= self.tolerance * (1.0 + abs(previous)):
                self.converged = True
                break
        self.coefficients = beta
        return self

    def predict_mean(self, matrix: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("Model has not been fitted")
        eta = np.clip(self._augment(np.asarray(matrix, dtype=float)) @ self.coefficients, -20.0, 20.0)
        return np.exp(eta)


@dataclass
class EncodedPoissonModel:
    alpha: float = 1.0
    max_iter: int = 80
    encoder: DictFeatureEncoder = field(default_factory=DictFeatureEncoder)
    regressor: PoissonGLM | None = None

    def fit(self, records: list[Mapping[str, Any]], target: np.ndarray) -> "EncodedPoissonModel":
        matrix = self.encoder.fit_transform(records)
        self.regressor = PoissonGLM(alpha=self.alpha, max_iter=self.max_iter).fit(matrix, target)
        return self

    def predict_mean(self, records: list[Mapping[str, Any]]) -> np.ndarray:
        if self.regressor is None:
            raise RuntimeError("Model has not been fitted")
        return self.regressor.predict_mean(self.encoder.transform(records))
