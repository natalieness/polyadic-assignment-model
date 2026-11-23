"""Matched factorial simulation and geometry/co-target attribution outputs."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.approach1.data import DataRepository

from .config import Approach2Config
from .data import FixedNeuronInputs, load_observed_inputs
from .geometry import GeometryResult, geometry_scores
from .propensity import (
    estimate_individual_propensity,
    load_supplied_propensity,
    mean_pair_count_matrix,
    pair_count_matrix,
)
from .sampler import gibbs_target_swaps, sample_random_null, validate_assignment


CONDITIONS = ("null", "geometry", "cotargeting", "combined")

# Stamped into every JSON result so a file that predates a field change is
# rejected on load rather than silently rebuilt with missing attributes.
RESULT_FORMAT = "approach2.AttributionResult"
RESULT_VERSION = 1


def _rng(config: Approach2Config, skeleton_id: int, sample: int, stream: int):
    return np.random.default_rng(np.random.SeedSequence([
        config.sampler.random_seed,
        int(skeleton_id),
        int(sample),
        int(stream),
    ]))


def permutation_expectation_samples(
    inputs: FixedNeuronInputs,
    config: Approach2Config,
) -> np.ndarray:
    """Draw independent margin-preserving shuffles for the co-targeting expectation.

    Each draw shuffles labelled target tokens into the fixed site slots, so the site
    degrees and target totals stay exact while the pairing is random. Nothing here
    reads the simulated conditions: the expectation is not the null or geometry
    chain reused, so it carries no MCMC autocorrelation and its draw count is set by
    ``cotarget.n_perm_expectation`` rather than by ``sampler.prediction_samples``.
    """
    return np.stack([
        sample_random_null(
            inputs.site_degrees,
            inputs.target_counts,
            _rng(config, inputs.skeleton_id, index, 50),
            max_same_target_per_site=config.sampler.max_same_target_per_site,
            max_restarts=config.sampler.max_null_restarts,
        )
        for index in range(int(config.cotarget.n_perm_expectation))
    ])


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float).ravel()
    right = np.asarray(right, dtype=float).ravel()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    numerator = float(np.sum(left * right))
    return numerator / denominator if denominator else 0.0


def _encode_array(values: np.ndarray) -> dict[str, Any]:
    """Tag an array with its dtype and shape so the load side restores both.

    ``tolist`` converts NumPy scalars to Python ones, which ``json`` can write;
    NaN and inf survive the trip because Python's encoder and decoder agree on
    the non-standard ``NaN``/``Infinity`` tokens.
    """
    array = np.asarray(values)
    return {
        "__ndarray__": True,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "data": array.tolist(),
    }


def _decode_array(payload: dict[str, Any]) -> np.ndarray:
    return np.asarray(payload["data"], dtype=np.dtype(payload["dtype"])).reshape(
        payload["shape"]
    )


def _encode_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Store a frame column-wise: per-column ``tolist`` keeps mixed dtypes intact."""
    return {
        "__dataframe__": True,
        "columns": [str(name) for name in frame.columns],
        "index": frame.index.tolist(),
        "dtypes": {str(name): str(dtype) for name, dtype in frame.dtypes.items()},
        "values": {str(name): frame[name].tolist() for name in frame.columns},
    }


def _decode_frame(payload: dict[str, Any]) -> pd.DataFrame:
    columns = payload["columns"]
    frame = pd.DataFrame(
        {name: payload["values"][name] for name in columns},
        columns=columns,
        index=payload["index"],
    )
    for name, dtype in payload["dtypes"].items():
        try:
            frame[name] = frame[name].astype(dtype)
        except (TypeError, ValueError):
            # A column that picked up NaN cannot go back to an integer dtype.
            # The values are already correct, so leave the widened dtype alone.
            pass
    return frame


def _assignment_rows(
    assignment: np.ndarray,
    inputs: FixedNeuronInputs,
) -> pd.DataFrame:
    rows = []
    for site, target in zip(*np.nonzero(assignment)):
        rows.append({
            "site_id": int(site),
            "connector_id": int(inputs.sites.iloc[site]["connector_id"]),
            "presynaptic_node_id": int(inputs.sites.iloc[site]["presynaptic_node_id"]),
            "degree": int(inputs.site_degrees[site]),
            "target_id": int(inputs.target_ids[target]),
            "multiplicity": int(assignment[site, target]),
            "target_total": int(inputs.target_counts[target]),
        })
    return pd.DataFrame.from_records(rows)


def _assignment_summary(samples: np.ndarray, inputs: FixedNeuronInputs) -> pd.DataFrame:
    presence = np.mean(samples > 0, axis=0)
    mean = np.mean(samples, axis=0)
    sd = np.std(samples, axis=0)
    rows = []
    for site, target in zip(*np.nonzero(presence > 0)):
        rows.append({
            "site_id": int(site),
            "connector_id": int(inputs.sites.iloc[site]["connector_id"]),
            "presynaptic_node_id": int(inputs.sites.iloc[site]["presynaptic_node_id"]),
            "degree": int(inputs.site_degrees[site]),
            "target_id": int(inputs.target_ids[target]),
            "target_total": int(inputs.target_counts[target]),
            "presence_probability": float(presence[site, target]),
            "mean_multiplicity": float(mean[site, target]),
            "multiplicity_sd": float(sd[site, target]),
        })
    return pd.DataFrame.from_records(rows)


def _condition_summary(
    samples_by_condition: dict[str, np.ndarray],
    inputs: FixedNeuronInputs,
    geometry: GeometryResult,
    include_self_repeats: bool = False,
    count_multiples: bool = False,
) -> pd.DataFrame:
    observed_pairs = pair_count_matrix(
        inputs.observed_assignment, include_self_repeats, count_multiples
    )
    total_contacts = int(inputs.site_degrees.sum())
    rows = []
    for condition, samples in samples_by_condition.items():
        metrics = []
        for assignment in samples:
            contact_l1 = float(np.abs(assignment - inputs.observed_assignment).sum())
            repeated = float(np.maximum(assignment - 1, 0).sum())
            same_pairs = float((assignment * (assignment - 1) / 2.0).sum())
            distinct = (assignment > 0).sum(axis=1)
            distinct_pairs = float(np.sum(distinct * (distinct - 1) / 2.0))
            geometry_score = float(np.sum(assignment * geometry.scores) / max(total_contacts, 1))
            finite = np.isfinite(geometry.distances_um)
            distance_weight = assignment * finite
            distance_denominator = float(distance_weight.sum())
            mean_distance = (
                float(np.nansum(assignment * geometry.distances_um) / distance_denominator)
                if distance_denominator else float("nan")
            )
            metrics.append({
                "contact_accuracy": 1.0 - contact_l1 / max(2 * total_contacts, 1),
                "assignment_cosine": _cosine(assignment, inputs.observed_assignment),
                "pair_cosine": _cosine(
                    pair_count_matrix(assignment, include_self_repeats, count_multiples),
                    observed_pairs,
                ),
                "repeated_contacts": repeated,
                "same_target_pairs": same_pairs,
                "distinct_target_pairs": distinct_pairs,
                "mean_geometry_score": geometry_score,
                "mean_assigned_distance_um": mean_distance,
            })
        row: dict[str, Any] = {"condition": condition, "samples": len(samples)}
        for metric in metrics[0]:
            values = np.asarray([item[metric] for item in metrics], dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_sd"] = float(np.nanstd(values))
        rows.append(row)
    return pd.DataFrame.from_records(rows)


@dataclass
class AttributionResult:
    inputs: FixedNeuronInputs
    geometry: GeometryResult
    cotarget_scores: np.ndarray
    observed_pair_counts: np.ndarray
    propensity_expectation: np.ndarray
    samples: dict[str, np.ndarray]
    condition_summary: pd.DataFrame
    diagnostics: dict[str, Any]

    def validate(self) -> None:
        expected_shape = (
            len(self.inputs.site_degrees),
            len(self.inputs.target_ids),
        )
        profiles = None
        for condition in CONDITIONS:
            values = self.samples[condition]
            if values.ndim != 3 or values.shape[1:] != expected_shape:
                raise RuntimeError(f"{condition} samples have the wrong shape")
            for assignment in values:
                validate_assignment(
                    assignment,
                    self.inputs.site_degrees,
                    self.inputs.target_counts,
                )
            condition_profiles = [
                [tuple(sorted(row[row > 0].tolist())) for row in assignment]
                for assignment in values
            ]
            if profiles is None:
                profiles = condition_profiles
            elif self.diagnostics["preserve_multiplicity_profile"] and condition_profiles != profiles:
                raise RuntimeError("Matched conditions changed a locked multiplicity profile")

    def save(self, output_dir: str | Path, save_raw_samples: bool = True) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.inputs.sites.to_csv(output / "fixed_sites.csv", index=False)
        targets = pd.DataFrame({
            "target_id": self.inputs.target_ids,
            "target_total": self.inputs.target_counts,
            "backbone_available": self.geometry.backbone_available,
        })
        targets.to_csv(output / "targets.csv", index=False)
        geometry_rows = []
        for site in range(len(self.inputs.site_degrees)):
            for target in range(len(self.inputs.target_ids)):
                geometry_rows.append({
                    "site_id": site,
                    "target_id": int(self.inputs.target_ids[target]),
                    "distance_um": float(self.geometry.distances_um[site, target]),
                    "geometry_score": float(self.geometry.scores[site, target]),
                })
        pd.DataFrame.from_records(geometry_rows).to_csv(
            output / "geometry_scores.csv", index=False
        )
        cotarget_rows = []
        # Self scores live on the diagonal, so the row range starts one earlier when
        # they are in play. target_a == target_b marks a self-repeat score.
        include_self = bool(self.diagnostics.get("cotargeting_include_self_repeats"))
        for left in range(len(self.inputs.target_ids)):
            for right in range(
                left if include_self else left + 1, len(self.inputs.target_ids)
            ):
                cotarget_rows.append({
                    "target_a": int(self.inputs.target_ids[left]),
                    "target_b": int(self.inputs.target_ids[right]),
                    "score": float(self.cotarget_scores[left, right]),
                })
        pd.DataFrame.from_records(cotarget_rows).to_csv(
            output / "cotarget_scores.csv", index=False
        )
        _assignment_rows(self.inputs.observed_assignment, self.inputs).to_csv(
            output / "observed_assignments.csv", index=False
        )
        for condition, samples in self.samples.items():
            _assignment_rows(samples[0], self.inputs).to_csv(
                output / f"assignments_{condition}.csv", index=False
            )
            _assignment_summary(samples, self.inputs).to_csv(
                output / f"assignment_summary_{condition}.csv", index=False
            )
        self.condition_summary.to_csv(output / "condition_summary.csv", index=False)
        (output / "diagnostics.json").write_text(
            json.dumps(self.diagnostics, indent=2) + "\n"
        )
        if save_raw_samples:
            np.savez_compressed(
                output / "assignment_samples.npz",
                **self.samples,
                target_ids=self.inputs.target_ids,
                site_degrees=self.inputs.site_degrees,
                target_counts=self.inputs.target_counts,
            )

    def to_dict(self, include_samples: bool = True) -> dict[str, Any]:
        """Represent the whole result as JSON-safe primitives, field for field."""
        return {
            "format": RESULT_FORMAT,
            "version": RESULT_VERSION,
            "inputs": {
                "skeleton_id": int(self.inputs.skeleton_id),
                "sites": _encode_frame(self.inputs.sites),
                "site_degrees": _encode_array(self.inputs.site_degrees),
                "target_ids": _encode_array(self.inputs.target_ids),
                "target_counts": _encode_array(self.inputs.target_counts),
                "observed_assignment": _encode_array(self.inputs.observed_assignment),
            },
            "geometry": {
                "distances_um": _encode_array(self.geometry.distances_um),
                "scores": _encode_array(self.geometry.scores),
                "backbone_available": _encode_array(self.geometry.backbone_available),
                "raw_score_sd": float(self.geometry.raw_score_sd),
                "distance_threshold_um": float(self.geometry.distance_threshold_um),
            },
            "cotarget_scores": _encode_array(self.cotarget_scores),
            "observed_pair_counts": _encode_array(self.observed_pair_counts),
            "propensity_expectation": _encode_array(self.propensity_expectation),
            "samples": {
                condition: (
                    _encode_array(values)
                    if include_samples and values is not None
                    else None
                )
                for condition, values in self.samples.items()
            },
            "condition_summary": _encode_frame(self.condition_summary),
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AttributionResult":
        """Rebuild the real dataclasses, so attribute access works as before."""
        if payload.get("format") != RESULT_FORMAT:
            raise ValueError(f"Not an {RESULT_FORMAT} payload: {payload.get('format')!r}")
        if payload.get("version") != RESULT_VERSION:
            raise ValueError(
                f"Unsupported result version {payload.get('version')!r}; "
                f"this build writes version {RESULT_VERSION}"
            )
        inputs = payload["inputs"]
        geometry = payload["geometry"]
        return cls(
            inputs=FixedNeuronInputs(
                skeleton_id=int(inputs["skeleton_id"]),
                sites=_decode_frame(inputs["sites"]),
                site_degrees=_decode_array(inputs["site_degrees"]),
                target_ids=_decode_array(inputs["target_ids"]),
                target_counts=_decode_array(inputs["target_counts"]),
                observed_assignment=_decode_array(inputs["observed_assignment"]),
            ),
            geometry=GeometryResult(
                distances_um=_decode_array(geometry["distances_um"]),
                scores=_decode_array(geometry["scores"]),
                backbone_available=_decode_array(geometry["backbone_available"]),
                raw_score_sd=float(geometry["raw_score_sd"]),
                distance_threshold_um=float(geometry["distance_threshold_um"]),
            ),
            cotarget_scores=_decode_array(payload["cotarget_scores"]),
            observed_pair_counts=_decode_array(payload["observed_pair_counts"]),
            propensity_expectation=_decode_array(payload["propensity_expectation"]),
            samples={
                condition: (_decode_array(values) if values is not None else None)
                for condition, values in payload["samples"].items()
            },
            condition_summary=_decode_frame(payload["condition_summary"]),
            diagnostics=payload["diagnostics"],
        )

    def to_json(self, path: str | Path, include_samples: bool = True) -> None:
        """Write the entire result to one JSON file.

        ``include_samples=False`` stores ``None`` in place of each condition's
        draws. The samples dominate the file size by a wide margin, so that is
        the switch to reach for when only the summaries are wanted. A ``.gz``
        suffix compresses on the way out and is picked up again by ``from_json``.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(include_samples=include_samples))
        if path.suffix == ".gz":
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(text)
        else:
            path.write_text(text)

    @classmethod
    def from_json(cls, path: str | Path) -> "AttributionResult":
        """Load a file written by ``to_json`` back into an ``AttributionResult``."""
        path = Path(path)
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                return cls.from_dict(json.load(handle))
        return cls.from_dict(json.loads(path.read_text()))


def load_result(path: str | Path) -> AttributionResult:
    """Convenience wrapper around ``AttributionResult.from_json``."""
    return AttributionResult.from_json(path)


def run_attribution_experiment(
    repository: DataRepository,
    skeleton_id: int,
    config: Approach2Config,
) -> AttributionResult:
    """Run matched null/geometry/co-targeting/combined simulations for one neuron."""
    config.validate()
    inputs = load_observed_inputs(repository, int(skeleton_id))
    geometry = geometry_scores(repository, inputs, config.geometry)
    n_samples = config.sampler.prediction_samples
    n_contacts = int(inputs.site_degrees.sum())
    zero_geometry = np.zeros_like(geometry.scores)
    zero_cotarget = np.zeros((len(inputs.target_ids), len(inputs.target_ids)), dtype=float)
    base_samples = []
    geometry_samples = []

    if config.sampler.max_same_target_per_site is not None and (
        inputs.observed_assignment.max(initial=0)
        > config.sampler.max_same_target_per_site
        and config.sampler.multiplicity_profile_source == "observed"
    ):
        raise ValueError("Observed multiplicity exceeds max_same_target_per_site")

    # The expectation comes from its own permutations, so the co-targeting scores are
    # settled before any condition is simulated rather than after the null run.
    expectation_samples = permutation_expectation_samples(inputs, config)
    include_self_repeats = bool(config.cotarget.include_self_repeats)
    observed_pairs = pair_count_matrix(inputs.observed_assignment, include_self_repeats)
    if not config.cotarget.enabled or config.cotarget.source == "none":
        cotarget_scores = zero_cotarget
        expected_for_propensity = mean_pair_count_matrix(
            expectation_samples, include_self_repeats
        )
    elif config.cotarget.source == "supplied":
        cotarget_scores = load_supplied_propensity(
            config.cotarget.supplied_scores or "",
            inputs.target_ids,
            config.cotarget,
            skeleton_id=inputs.skeleton_id,
        )
        expected_for_propensity = mean_pair_count_matrix(
            expectation_samples, include_self_repeats
        )
    else:
        cotarget_scores, observed_pairs, expected_for_propensity = (
            estimate_individual_propensity(
                inputs.observed_assignment,
                expectation_samples,
                config.cotarget,
            )
        )

    for sample_index in range(n_samples):
        if config.sampler.multiplicity_profile_source == "observed":
            base = gibbs_target_swaps(
                inputs.observed_assignment,
                inputs.site_degrees,
                inputs.target_counts,
                zero_geometry,
                zero_cotarget,
                _rng(config, inputs.skeleton_id, sample_index, 10),
                geometry_weight=0.0,
                cotargeting_weight=0.0,
                temperature=config.sampler.temperature,
                geometry_repeat_discount=0.0,
                cotarget_aggregation=config.cotarget.aggregation,
                cotarget_site_clip=config.cotarget.site_score_clip,
                self_repeat_penalty=0.0,
                preserve_multiplicity_profile=config.sampler.preserve_multiplicity_profile,
                max_same_target_per_site=config.sampler.max_same_target_per_site,
                steps=config.sampler.null_burnin_steps_per_contact * n_contacts,
            )
        else:
            base = sample_random_null(
                inputs.site_degrees,
                inputs.target_counts,
                _rng(config, inputs.skeleton_id, sample_index, 10),
                max_same_target_per_site=config.sampler.max_same_target_per_site,
                max_restarts=config.sampler.max_null_restarts,
            )
        base_samples.append(base)
        geometry_samples.append(gibbs_target_swaps(
            base,
            inputs.site_degrees,
            inputs.target_counts,
            geometry.scores,
            zero_cotarget,
            _rng(config, inputs.skeleton_id, sample_index, 20),
            geometry_weight=(
                config.sampler.geometry_weight if config.geometry.enabled else 0.0
            ),
            cotargeting_weight=0.0,
            temperature=config.sampler.temperature,
            geometry_repeat_discount=config.geometry.repeat_discount,
            cotarget_aggregation=config.cotarget.aggregation,
            cotarget_site_clip=config.cotarget.site_score_clip,
            self_repeat_penalty=config.sampler.self_repeat_penalty,
            preserve_multiplicity_profile=config.sampler.preserve_multiplicity_profile,
            max_same_target_per_site=config.sampler.max_same_target_per_site,
            steps=config.sampler.mcmc_steps_per_contact * n_contacts,
        ))
    null_samples = np.stack(base_samples)
    geometry_samples_array = np.stack(geometry_samples)

    cotarget_samples = []
    combined_samples = []
    for sample_index, base in enumerate(null_samples):
        common = dict(
            site_degrees=inputs.site_degrees,
            target_counts=inputs.target_counts,
            cotarget_scores=cotarget_scores,
            temperature=config.sampler.temperature,
            geometry_repeat_discount=config.geometry.repeat_discount,
            cotarget_aggregation=config.cotarget.aggregation,
            cotarget_site_clip=config.cotarget.site_score_clip,
            self_repeat_penalty=config.sampler.self_repeat_penalty,
            preserve_multiplicity_profile=config.sampler.preserve_multiplicity_profile,
            max_same_target_per_site=config.sampler.max_same_target_per_site,
            steps=config.sampler.mcmc_steps_per_contact * n_contacts,
            cotarget_include_self=include_self_repeats,
        )
        cotarget_samples.append(gibbs_target_swaps(
            base,
            geometry_scores=zero_geometry,
            rng=_rng(config, inputs.skeleton_id, sample_index, 30),
            geometry_weight=0.0,
            cotargeting_weight=(
                config.sampler.cotargeting_weight if config.cotarget.enabled else 0.0
            ),
            **common,
        ))
        combined_samples.append(gibbs_target_swaps(
            base,
            geometry_scores=geometry.scores,
            rng=_rng(config, inputs.skeleton_id, sample_index, 40),
            geometry_weight=(
                config.sampler.geometry_weight if config.geometry.enabled else 0.0
            ),
            cotargeting_weight=(
                config.sampler.cotargeting_weight if config.cotarget.enabled else 0.0
            ),
            **common,
        ))

    samples = {
        "null": null_samples,
        "geometry": geometry_samples_array,
        "cotargeting": np.stack(cotarget_samples),
        "combined": np.stack(combined_samples),
    }
    diagnostics = {
        "sampler": "two_site_conditional_gibbs",
        "skeleton_id": int(inputs.skeleton_id),
        "n_sites": len(inputs.site_degrees),
        "n_targets": len(inputs.target_ids),
        "n_contacts": n_contacts,
        "prediction_samples": n_samples,
        "geometry_enabled": bool(config.geometry.enabled),
        "target_backbones_available": int(geometry.backbone_available.sum()),
        "target_backbones_requested": len(inputs.target_ids),
        "geometry_raw_score_sd": geometry.raw_score_sd,
        "cotargeting_enabled": bool(config.cotarget.enabled),
        "cotargeting_source": config.cotarget.source,
        "cotargeting_n_perm_expectation": int(config.cotarget.n_perm_expectation),
        "cotargeting_include_self_repeats": include_self_repeats,
        "descriptive_same_neuron_propensity": bool(
            config.cotarget.enabled and config.cotarget.source == "observed_neuron"
        ),
        "multiplicity_profile_source": config.sampler.multiplicity_profile_source,
        "preserve_multiplicity_profile": config.sampler.preserve_multiplicity_profile,
        "geometry_weight": config.sampler.geometry_weight,
        "cotargeting_weight": config.sampler.cotargeting_weight,
        "temperature": config.sampler.temperature,
        "random_seed": config.sampler.random_seed,
    }
    result = AttributionResult(
        inputs=inputs,
        geometry=geometry,
        cotarget_scores=cotarget_scores,
        observed_pair_counts=observed_pairs,
        propensity_expectation=expected_for_propensity,
        samples=samples,
        condition_summary=_condition_summary(
            samples, inputs, geometry, include_self_repeats
        ),
        diagnostics=diagnostics,
    )
    result.validate()
    return result


def run_attribution_experiment_shapley(
    repository: DataRepository,
    skeleton_id: int,
    config: Approach2Config,
    count_multiples: bool | None = None,
) -> AttributionResult:
    """Matched simulations with the co-targeting propensity measured over geometry.

    Two things differ from ``run_attribution_experiment``:

    1. The propensity denominator is the **geometry** condition's draws, not the
       independent margin-preserving permutations. The enrichment therefore asks
       what co-targeting structure survives once geometry has had its turn, in the
       spirit of a Shapley-style ordering, rather than what survives the degree and
       total margins alone. Because the geometry chain has to finish before the
       scores exist, the conditions are simulated in two passes. The expectation is
       drawn ``sampler.prediction_samples`` times, so ``cotarget.n_perm_expectation``
       goes unused here; raise the prediction count if the denominator looks noisy.
       With ``geometry.enabled`` false the geometry draws are just the null chain
       carried further, and the result collapses towards the permutation version.
    2. Pair counts can run in contact-pair units via ``count_multiples``, where a
       site holding ``[A, A, B]`` gives the A/B pair ``+2`` rather than ``+1``.
       ``None`` takes the setting from ``config.cotarget.count_multiples``.

    Both conditions still start from the same null draws, so the matched-pairs
    comparison across conditions is unchanged.
    """
    config.validate()
    inputs = load_observed_inputs(repository, int(skeleton_id))
    geometry = geometry_scores(repository, inputs, config.geometry)
    n_samples = config.sampler.prediction_samples
    n_contacts = int(inputs.site_degrees.sum())
    zero_geometry = np.zeros_like(geometry.scores)
    zero_cotarget = np.zeros((len(inputs.target_ids), len(inputs.target_ids)), dtype=float)
    base_samples = []
    geometry_samples = []
    include_self_repeats = bool(config.cotarget.include_self_repeats)
    if count_multiples is None:
        count_multiples = bool(config.cotarget.count_multiples)

    if config.sampler.max_same_target_per_site is not None and (
        inputs.observed_assignment.max(initial=0)
        > config.sampler.max_same_target_per_site
        and config.sampler.multiplicity_profile_source == "observed"
    ):
        raise ValueError("Observed multiplicity exceeds max_same_target_per_site")

    for sample_index in range(n_samples):
        if config.sampler.multiplicity_profile_source == "observed":
            base = gibbs_target_swaps(
                inputs.observed_assignment,
                inputs.site_degrees,
                inputs.target_counts,
                zero_geometry,
                zero_cotarget,
                _rng(config, inputs.skeleton_id, sample_index, 10),
                geometry_weight=0.0,
                cotargeting_weight=0.0,
                temperature=config.sampler.temperature,
                geometry_repeat_discount=0.0,
                cotarget_aggregation=config.cotarget.aggregation,
                cotarget_site_clip=config.cotarget.site_score_clip,
                self_repeat_penalty=0.0,
                preserve_multiplicity_profile=config.sampler.preserve_multiplicity_profile,
                max_same_target_per_site=config.sampler.max_same_target_per_site,
                steps=config.sampler.null_burnin_steps_per_contact * n_contacts,
            )
        else:
            base = sample_random_null(
                inputs.site_degrees,
                inputs.target_counts,
                _rng(config, inputs.skeleton_id, sample_index, 10),
                max_same_target_per_site=config.sampler.max_same_target_per_site,
                max_restarts=config.sampler.max_null_restarts,
            )
        base_samples.append(base)
        geometry_samples.append(gibbs_target_swaps(
            base,
            inputs.site_degrees,
            inputs.target_counts,
            geometry.scores,
            zero_cotarget,
            _rng(config, inputs.skeleton_id, sample_index, 20),
            geometry_weight=(
                config.sampler.geometry_weight if config.geometry.enabled else 0.0
            ),
            cotargeting_weight=0.0,
            temperature=config.sampler.temperature,
            geometry_repeat_discount=config.geometry.repeat_discount,
            cotarget_aggregation=config.cotarget.aggregation,
            cotarget_site_clip=config.cotarget.site_score_clip,
            self_repeat_penalty=config.sampler.self_repeat_penalty,
            preserve_multiplicity_profile=config.sampler.preserve_multiplicity_profile,
            max_same_target_per_site=config.sampler.max_same_target_per_site,
            steps=config.sampler.mcmc_steps_per_contact * n_contacts,
        ))
    null_samples = np.stack(base_samples)
    geometry_samples_array = np.stack(geometry_samples)

    # The geometry draws stand in for the permutations here, so the scores can only
    # be settled now rather than ahead of the first condition.
    expectation_samples = geometry_samples_array
    observed_pairs = pair_count_matrix(
        inputs.observed_assignment, include_self_repeats, count_multiples
    )
    if not config.cotarget.enabled or config.cotarget.source == "none":
        cotarget_scores = zero_cotarget
        expected_for_propensity = mean_pair_count_matrix(
            expectation_samples, include_self_repeats, count_multiples
        )
    elif config.cotarget.source == "supplied":
        cotarget_scores = load_supplied_propensity(
            config.cotarget.supplied_scores or "",
            inputs.target_ids,
            config.cotarget,
            skeleton_id=inputs.skeleton_id,
        )
        expected_for_propensity = mean_pair_count_matrix(
            expectation_samples, include_self_repeats, count_multiples
        )
    else:
        cotarget_scores, observed_pairs, expected_for_propensity = (
            estimate_individual_propensity(
                inputs.observed_assignment,
                expectation_samples,
                config.cotarget,
                count_multiples=count_multiples,
            )
        )

    cotarget_samples = []
    combined_samples = []
    for sample_index, base in enumerate(null_samples):
        common = dict(
            site_degrees=inputs.site_degrees,
            target_counts=inputs.target_counts,
            cotarget_scores=cotarget_scores,
            temperature=config.sampler.temperature,
            geometry_repeat_discount=config.geometry.repeat_discount,
            cotarget_aggregation=config.cotarget.aggregation,
            cotarget_site_clip=config.cotarget.site_score_clip,
            self_repeat_penalty=config.sampler.self_repeat_penalty,
            preserve_multiplicity_profile=config.sampler.preserve_multiplicity_profile,
            max_same_target_per_site=config.sampler.max_same_target_per_site,
            steps=config.sampler.mcmc_steps_per_contact * n_contacts,
            cotarget_include_self=include_self_repeats,
        )
        cotarget_samples.append(gibbs_target_swaps(
            base,
            geometry_scores=zero_geometry,
            rng=_rng(config, inputs.skeleton_id, sample_index, 30),
            geometry_weight=0.0,
            cotargeting_weight=(
                config.sampler.cotargeting_weight if config.cotarget.enabled else 0.0
            ),
            **common,
        ))
        combined_samples.append(gibbs_target_swaps(
            base,
            geometry_scores=geometry.scores,
            rng=_rng(config, inputs.skeleton_id, sample_index, 40),
            geometry_weight=(
                config.sampler.geometry_weight if config.geometry.enabled else 0.0
            ),
            cotargeting_weight=(
                config.sampler.cotargeting_weight if config.cotarget.enabled else 0.0
            ),
            **common,
        ))

    samples = {
        "null": null_samples,
        "geometry": geometry_samples_array,
        "cotargeting": np.stack(cotarget_samples),
        "combined": np.stack(combined_samples),
    }
    diagnostics = {
        "sampler": "two_site_conditional_gibbs",
        "skeleton_id": int(inputs.skeleton_id),
        "n_sites": len(inputs.site_degrees),
        "n_targets": len(inputs.target_ids),
        "n_contacts": n_contacts,
        "prediction_samples": n_samples,
        "geometry_enabled": bool(config.geometry.enabled),
        "target_backbones_available": int(geometry.backbone_available.sum()),
        "target_backbones_requested": len(inputs.target_ids),
        "geometry_raw_score_sd": geometry.raw_score_sd,
        "cotargeting_enabled": bool(config.cotarget.enabled),
        "cotargeting_source": config.cotarget.source,
        # n_perm_expectation is deliberately absent: the geometry draws are the
        # expectation, so the denominator has prediction_samples draws instead.
        "cotargeting_expectation_source": "geometry_samples",
        "cotargeting_n_expectation_samples": n_samples,
        "cotargeting_include_self_repeats": include_self_repeats,
        "cotargeting_count_multiples": bool(count_multiples),
        "descriptive_same_neuron_propensity": bool(
            config.cotarget.enabled and config.cotarget.source == "observed_neuron"
        ),
        "multiplicity_profile_source": config.sampler.multiplicity_profile_source,
        "preserve_multiplicity_profile": config.sampler.preserve_multiplicity_profile,
        "geometry_weight": config.sampler.geometry_weight,
        "cotargeting_weight": config.sampler.cotargeting_weight,
        "temperature": config.sampler.temperature,
        "random_seed": config.sampler.random_seed,
    }
    result = AttributionResult(
        inputs=inputs,
        geometry=geometry,
        cotarget_scores=cotarget_scores,
        observed_pair_counts=observed_pairs,
        propensity_expectation=expected_for_propensity,
        samples=samples,
        condition_summary=_condition_summary(
            samples, inputs, geometry, include_self_repeats, count_multiples
        ),
        diagnostics=diagnostics,
    )
    result.validate()
    return result


def run_sampler_with_permutation(
    repository: DataRepository,
    skeleton_id: int,
    config: Approach2Config,
) -> AttributionResult:
    config.validate()
    inputs = load_observed_inputs(repository, int(skeleton_id))
    geometry = geometry_scores(repository, inputs, config.geometry)
    n_samples = config.sampler.prediction_samples
    n_contacts = int(inputs.site_degrees.sum())
    zero_geometry = np.zeros_like(geometry.scores)
    zero_cotarget = np.zeros((len(inputs.target_ids), len(inputs.target_ids)), dtype=float)
    base_samples = []
    geometry_samples = []

    if config.sampler.max_same_target_per_site is not None and (
        inputs.observed_assignment.max(initial=0)
        > config.sampler.max_same_target_per_site
        and config.sampler.multiplicity_profile_source == "observed"
    ):
        raise ValueError("Observed multiplicity exceeds max_same_target_per_site")

    # The expectation comes from its own permutations, so the co-targeting scores are
    # settled before any condition is simulated rather than after the null run.
    expectation_samples = permutation_expectation_samples(inputs, config)
    print(f"expectation_samples.shape: {expectation_samples.shape}")
    include_self_repeats = bool(config.cotarget.include_self_repeats)
    observed_pairs = pair_count_matrix(inputs.observed_assignment, include_self_repeats)
    if not config.cotarget.enabled or config.cotarget.source == "none":
        cotarget_scores = zero_cotarget
        expected_for_propensity = mean_pair_count_matrix(
            expectation_samples, include_self_repeats
        )
    elif config.cotarget.source == "supplied":
        cotarget_scores = load_supplied_propensity(
            config.cotarget.supplied_scores or "",
            inputs.target_ids,
            config.cotarget,
            skeleton_id=inputs.skeleton_id,
        )
        expected_for_propensity = mean_pair_count_matrix(
            expectation_samples, include_self_repeats
        )
    else:
        cotarget_scores, observed_pairs, expected_for_propensity = (
            estimate_individual_propensity(
                inputs.observed_assignment,
                expectation_samples,
                config.cotarget,
            )
        )

    for sample_index in range(n_samples):
        # rather than just leaving it for the null, always do the gibbs target swaps for the base
        base = gibbs_target_swaps(
            expectation_samples[sample_index, :, :],
            inputs.site_degrees,
            inputs.target_counts,
            zero_geometry,
            zero_cotarget,
            _rng(config, inputs.skeleton_id, sample_index, 10),
            geometry_weight=0.0,
            cotargeting_weight=0.0,
            temperature=config.sampler.temperature,
            geometry_repeat_discount=0.0,
            cotarget_aggregation=config.cotarget.aggregation,
            cotarget_site_clip=config.cotarget.site_score_clip,
            self_repeat_penalty=0.0,
            preserve_multiplicity_profile=config.sampler.preserve_multiplicity_profile,
            max_same_target_per_site=config.sampler.max_same_target_per_site,
            steps=config.sampler.null_burnin_steps_per_contact * n_contacts,
        )
        base_samples.append(base)
        geometry_samples.append(gibbs_target_swaps(
            base,
            inputs.site_degrees,
            inputs.target_counts,
            geometry.scores,
            zero_cotarget,
            _rng(config, inputs.skeleton_id, sample_index, 20),
            geometry_weight=(
                config.sampler.geometry_weight if config.geometry.enabled else 0.0
            ),
            cotargeting_weight=0.0,
            temperature=config.sampler.temperature,
            geometry_repeat_discount=config.geometry.repeat_discount,
            cotarget_aggregation=config.cotarget.aggregation,
            cotarget_site_clip=config.cotarget.site_score_clip,
            self_repeat_penalty=config.sampler.self_repeat_penalty,
            preserve_multiplicity_profile=config.sampler.preserve_multiplicity_profile,
            max_same_target_per_site=config.sampler.max_same_target_per_site,
            steps=config.sampler.mcmc_steps_per_contact * n_contacts,
        ))
    null_samples = np.stack(base_samples)
    geometry_samples_array = np.stack(geometry_samples)

    cotarget_samples = []
    combined_samples = []
    for sample_index, base in enumerate(null_samples):
        common = dict(
            site_degrees=inputs.site_degrees,
            target_counts=inputs.target_counts,
            cotarget_scores=cotarget_scores,
            temperature=config.sampler.temperature,
            geometry_repeat_discount=config.geometry.repeat_discount,
            cotarget_aggregation=config.cotarget.aggregation,
            cotarget_site_clip=config.cotarget.site_score_clip,
            self_repeat_penalty=config.sampler.self_repeat_penalty,
            preserve_multiplicity_profile=config.sampler.preserve_multiplicity_profile,
            max_same_target_per_site=config.sampler.max_same_target_per_site,
            steps=config.sampler.mcmc_steps_per_contact * n_contacts,
            cotarget_include_self=include_self_repeats,
        )
        cotarget_samples.append(gibbs_target_swaps(
            base,
            geometry_scores=zero_geometry,
            rng=_rng(config, inputs.skeleton_id, sample_index, 30),
            geometry_weight=0.0,
            cotargeting_weight=(
                config.sampler.cotargeting_weight if config.cotarget.enabled else 0.0
            ),
            **common,
        ))
        combined_samples.append(gibbs_target_swaps(
            base,
            geometry_scores=geometry.scores,
            rng=_rng(config, inputs.skeleton_id, sample_index, 40),
            geometry_weight=(
                config.sampler.geometry_weight if config.geometry.enabled else 0.0
            ),
            cotargeting_weight=(
                config.sampler.cotargeting_weight if config.cotarget.enabled else 0.0
            ),
            **common,
        ))

    samples = {
        "null": null_samples,
        "geometry": geometry_samples_array,
        "cotargeting": np.stack(cotarget_samples),
        "combined": np.stack(combined_samples),
    }
    diagnostics = {
        "sampler": "two_site_conditional_gibbs",
        "skeleton_id": int(inputs.skeleton_id),
        "n_sites": len(inputs.site_degrees),
        "n_targets": len(inputs.target_ids),
        "n_contacts": n_contacts,
        "prediction_samples": n_samples,
        "geometry_enabled": bool(config.geometry.enabled),
        "target_backbones_available": int(geometry.backbone_available.sum()),
        "target_backbones_requested": len(inputs.target_ids),
        "geometry_raw_score_sd": geometry.raw_score_sd,
        "cotargeting_enabled": bool(config.cotarget.enabled),
        "cotargeting_source": config.cotarget.source,
        "cotargeting_n_perm_expectation": int(config.cotarget.n_perm_expectation),
        "cotargeting_include_self_repeats": include_self_repeats,
        "descriptive_same_neuron_propensity": bool(
            config.cotarget.enabled and config.cotarget.source == "observed_neuron"
        ),
        "multiplicity_profile_source": config.sampler.multiplicity_profile_source,
        "preserve_multiplicity_profile": config.sampler.preserve_multiplicity_profile,
        "geometry_weight": config.sampler.geometry_weight,
        "cotargeting_weight": config.sampler.cotargeting_weight,
        "temperature": config.sampler.temperature,
        "random_seed": config.sampler.random_seed,
    }
    result = AttributionResult(
        inputs=inputs,
        geometry=geometry,
        cotarget_scores=cotarget_scores,
        observed_pair_counts=observed_pairs,
        propensity_expectation=expected_for_propensity,
        samples=samples,
        condition_summary=_condition_summary(
            samples, inputs, geometry, include_self_repeats
        ),
        diagnostics=diagnostics,
    )
    result.validate()
    return result