"""Configurable staged model with exact-count constrained decoding."""

from __future__ import annotations

import copy
import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .config import DegreeSummary, ModelConfig
from .constraints import (
    allocate_learned_degrees,
    assign_degree_multiset,
    degrees_from_histogram,
    degrees_from_summary,
    residualize_partner_scores,
    sample_partner_assignment,
    select_exact_sites,
    solve_partner_transport,
)
from .data import ConnectorNeuron, DataRepository
from .features import (
    aggregate_availability,
    generate_candidates,
    partner_feature_record,
    site_feature_records,
    target_distance_fields,
)
from .interactions import CoTargetingModel, TargetIdentityPrior
from .regression import EncodedPoissonModel


@dataclass
class PredictionResult:
    skeleton_id: int
    sites: pd.DataFrame
    assignments: pd.DataFrame
    assignment_matrix: np.ndarray
    target_ids: np.ndarray
    assignment_samples: np.ndarray | None = None
    assignment_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def validate(self, n_sites: int, target_counts: Mapping[int, int]) -> None:
        if len(self.sites) != n_sites:
            raise RuntimeError("Prediction returned the wrong number of sites")
        if not np.array_equal(
            self.assignment_matrix.sum(axis=1), self.sites["predicted_degree"].to_numpy(int)
        ):
            raise RuntimeError("Assignment row sums do not match predicted degrees")
        expected = np.asarray([target_counts[int(target)] for target in self.target_ids], dtype=int)
        if not np.array_equal(self.assignment_matrix.sum(axis=0), expected):
            raise RuntimeError("Assignment column sums do not match requested target counts")
        if self.assignment_samples is not None:
            if self.assignment_samples.ndim != 3:
                raise RuntimeError("assignment_samples must be samples x sites x targets")
            for sample in self.assignment_samples:
                if not np.array_equal(sample.sum(axis=1), self.sites["predicted_degree"].to_numpy(int)):
                    raise RuntimeError("A sampled assignment has incorrect row sums")
                if not np.array_equal(sample.sum(axis=0), expected):
                    raise RuntimeError("A sampled assignment has incorrect target totals")


@dataclass
class _NeuronContext:
    candidates: pd.DataFrame
    site_records: list[dict[str, Any]]
    distances: dict[int, np.ndarray]
    distance_available: dict[int, bool]
    target_counts: dict[int, int]


class PolyadicAssignmentModel:
    """Approach 1: site intensity -> degree allocation -> partner assignment."""

    VERSION = 2

    def __init__(self, config: ModelConfig | None = None):
        self.config = ModelConfig() if config is None else copy.deepcopy(config)
        self.config.validate()
        self.site_model = EncodedPoissonModel(
            alpha=self.config.site.alpha, max_iter=self.config.site.max_iter
        )
        self.degree_model = EncodedPoissonModel(
            alpha=self.config.degree.alpha, max_iter=self.config.degree.max_iter
        )
        self.partner_model = EncodedPoissonModel(
            alpha=self.config.partner.alpha, max_iter=self.config.partner.max_iter
        )
        self.target_prior = TargetIdentityPrior()
        self.co_targeting = CoTargetingModel(
            smoothing=self.config.partner.co_targeting_smoothing,
            shrinkage=self.config.partner.co_targeting_shrinkage,
            minimum_observations=self.config.partner.co_targeting_min_observations,
        )
        self.fitted = False
        self.training_summary: dict[str, Any] = {}

    def _load_backbones(self, repository: DataRepository, target_ids: Iterable[int]):
        if not self.config.features.use_target_backbones:
            return {}
        return repository.load_backbones(
            target_ids, missing_policy=self.config.geometry.missing_backbone_policy
        )

    def _context(
        self,
        repository: DataRepository,
        skeleton_id: int,
        target_counts: Mapping[int, int],
        force_node_ids: set[int] | None = None,
    ) -> _NeuronContext:
        morphology = repository.morphologies.load(skeleton_id)
        candidates = generate_candidates(morphology, self.config, force_node_ids=force_node_ids)
        normalized_counts = {int(target): int(count) for target, count in target_counts.items()}
        if not normalized_counts or any(count <= 0 for count in normalized_counts.values()):
            raise ValueError("target_counts must contain positive counts")
        backbones = self._load_backbones(repository, normalized_counts)
        distances, available = target_distance_fields(
            candidates, normalized_counts, backbones, self.config
        )
        availability = aggregate_availability(
            normalized_counts, distances, available, self.config
        )
        records = site_feature_records(
            candidates,
            availability,
            self.config,
            presynaptic_celltype=repository.celltypes.get(int(skeleton_id)),
            hemisphere=repository.hemispheres.get(int(skeleton_id)),
        )
        return _NeuronContext(candidates, records, distances, available, normalized_counts)

    @staticmethod
    def _sample_indices(rng: np.random.Generator, n_rows: int, maximum: int) -> np.ndarray:
        if n_rows <= maximum:
            return np.arange(n_rows)
        return np.sort(rng.choice(n_rows, size=maximum, replace=False))

    def _training_records_for_neuron(
        self,
        repository: DataRepository,
        neuron: ConnectorNeuron,
        rng: np.random.Generator,
    ) -> tuple[list[dict[str, Any]], list[float], list[dict[str, Any]], list[float], list[dict[str, Any]], list[float]]:
        observed_nodes = neuron.rows["presynaptic_to_node"].astype(int).tolist()
        context = self._context(
            repository,
            neuron.skeleton_id,
            neuron.target_counts,
            force_node_ids=set(observed_nodes),
        )
        node_to_candidate = {
            int(node_id): index for index, node_id in enumerate(context.candidates["node_id"].astype(int))
        }
        observed_node_counts = Counter(observed_nodes)
        positive_indices = np.asarray(
            [node_to_candidate[node] for node in observed_node_counts], dtype=int
        )
        negative_indices = np.asarray(
            [index for index, node in enumerate(context.candidates["node_id"].astype(int)) if node not in observed_node_counts],
            dtype=int,
        )
        requested_negatives = int(
            np.ceil(len(positive_indices) * self.config.site.negative_to_positive_ratio)
        )
        if len(negative_indices) > requested_negatives:
            negative_indices = rng.choice(negative_indices, size=requested_negatives, replace=False)
        site_indices = np.concatenate([positive_indices, negative_indices])
        site_indices = site_indices[
            self._sample_indices(
                rng, len(site_indices), self.config.training.max_site_rows_per_neuron
            )
        ]
        site_records = [context.site_records[index] for index in site_indices]
        site_targets = [
            float(observed_node_counts.get(int(context.candidates.iloc[index]["node_id"]), 0))
            for index in site_indices
        ]

        degree_records: list[dict[str, Any]] = []
        degree_targets: list[float] = []
        partner_records: list[dict[str, Any]] = []
        partner_targets: list[float] = []
        neuron_targets = np.asarray(sorted(context.target_counts), dtype=int)
        total_contacts = sum(context.target_counts.values())
        for row in neuron.rows.itertuples(index=False):
            candidate_index = node_to_candidate[int(row.presynaptic_to_node)]
            base = context.site_records[candidate_index]
            degree_records.append(base)
            degree_targets.append(float(len(row.postsynaptic_to)))
            counts_here = Counter(int(target) for target in row.postsynaptic_to)
            positive_targets = np.asarray(sorted(counts_here), dtype=int)
            negative_pool = np.setdiff1d(neuron_targets, positive_targets, assume_unique=True)
            requested = self.config.partner.negative_targets_per_positive * len(positive_targets)
            if len(negative_pool) > requested:
                negative_pool = rng.choice(negative_pool, size=requested, replace=False)
            selected_targets = np.concatenate([positive_targets, negative_pool])
            for target_id in selected_targets:
                target_id = int(target_id)
                partner_records.append(
                    partner_feature_record(
                        base,
                        candidate_index,
                        target_id,
                        context.target_counts[target_id],
                        total_contacts,
                        context.distances,
                        context.distance_available,
                        repository.celltypes.get(target_id),
                        self.target_prior.log_probability(target_id),
                        self.config,
                    )
                )
                partner_targets.append(float(counts_here.get(target_id, 0)))

        partner_keep = self._sample_indices(
            rng, len(partner_records), self.config.training.max_partner_rows_per_neuron
        )
        partner_records = [partner_records[index] for index in partner_keep]
        partner_targets = [partner_targets[index] for index in partner_keep]
        return (
            site_records,
            site_targets,
            degree_records,
            degree_targets,
            partner_records,
            partner_targets,
        )

    def fit(
        self,
        repository: DataRepository,
        train_ids: Iterable[int],
        validation_ids: Iterable[int] | None = None,
    ) -> dict[str, Any]:
        train_ids = [int(value) for value in train_ids]
        if self.config.training.max_train_neurons is not None:
            train_ids = train_ids[: self.config.training.max_train_neurons]
        if not train_ids:
            raise ValueError("No training neuron IDs were supplied")
        missing = set(train_ids) - set(repository.connectors.neuron_ids.tolist())
        if missing:
            raise ValueError(f"Training neurons lack connector rows: {sorted(missing)[:20]}")
        connector_rows = repository.connectors.subset(train_ids)
        self.target_prior.fit(connector_rows)
        if self.config.features.use_co_targeting:
            self.co_targeting.fit(connector_rows, repository.celltypes)

        rng = np.random.default_rng(self.config.training.random_seed)
        all_site_records: list[dict[str, Any]] = []
        all_site_targets: list[float] = []
        all_degree_records: list[dict[str, Any]] = []
        all_degree_targets: list[float] = []
        all_partner_records: list[dict[str, Any]] = []
        all_partner_targets: list[float] = []
        for number, skeleton_id in enumerate(train_ids, start=1):
            records = self._training_records_for_neuron(
                repository, repository.connectors.neuron(skeleton_id), rng
            )
            site_records, site_targets, degree_records, degree_targets, partner_records, partner_targets = records
            all_site_records.extend(site_records)
            all_site_targets.extend(site_targets)
            all_degree_records.extend(degree_records)
            all_degree_targets.extend(degree_targets)
            all_partner_records.extend(partner_records)
            all_partner_targets.extend(partner_targets)
            if self.config.training.verbose and (number == 1 or number % 50 == 0 or number == len(train_ids)):
                print(
                    f"Prepared {number}/{len(train_ids)} neurons: "
                    f"{len(all_site_records):,} site, {len(all_degree_records):,} degree, "
                    f"{len(all_partner_records):,} partner rows"
                )

        self.site_model.fit(all_site_records, np.asarray(all_site_targets, dtype=float))
        self.degree_model.fit(all_degree_records, np.asarray(all_degree_targets, dtype=float))
        self.partner_model.fit(all_partner_records, np.asarray(all_partner_targets, dtype=float))
        self.fitted = True
        self.training_summary = {
            "train_neurons": len(train_ids),
            "site_rows": len(all_site_records),
            "degree_rows": len(all_degree_records),
            "partner_rows": len(all_partner_records),
            "site_model_converged": bool(self.site_model.regressor and self.site_model.regressor.converged),
            "degree_model_converged": bool(self.degree_model.regressor and self.degree_model.regressor.converged),
            "partner_model_converged": bool(self.partner_model.regressor and self.partner_model.regressor.converged),
        }
        if validation_ids is not None:
            from .evaluation import evaluate_split
            self.training_summary["validation"] = evaluate_split(
                self, repository, [int(value) for value in validation_ids]
            )
        return self.training_summary

    def _degree_prediction(
        self,
        degree_means: np.ndarray,
        total_contacts: int,
        degree_mode: str,
        degree_histogram: Mapping[int, int] | None,
        degree_summary: DegreeSummary | None,
    ) -> np.ndarray:
        if degree_mode == "exact_histogram":
            if degree_histogram is None:
                raise ValueError("exact_histogram mode requires degree_histogram")
            multiset = degrees_from_histogram(
                degree_histogram, len(degree_means), total_contacts
            )
            return assign_degree_multiset(multiset, degree_means)
        if degree_mode == "supplied_summary":
            if degree_summary is None:
                raise ValueError("supplied_summary mode requires degree_summary")
            multiset = degrees_from_summary(
                degree_summary, len(degree_means), total_contacts, degree_means
            )
            return assign_degree_multiset(multiset, degree_means)
        return allocate_learned_degrees(
            degree_means,
            total_contacts,
            self.config.degree.minimum,
            self.config.degree.maximum,
        )

    def predict(
        self,
        repository: DataRepository,
        skeleton_id: int,
        n_sites: int | None,
        target_counts: Mapping[int, int],
        *,
        exact_site_node_ids: Iterable[int] | None = None,
        degree_mode: str | None = None,
        degree_histogram: Mapping[int, int] | None = None,
        degree_summary: DegreeSummary | None = None,
        prediction_samples: int | None = None,
    ) -> PredictionResult:
        if not self.fitted:
            raise RuntimeError("Fit or load the model before prediction")
        skeleton_id = int(skeleton_id)
        exact_sites = None if exact_site_node_ids is None else [int(value) for value in exact_site_node_ids]
        if exact_sites is not None:
            if n_sites is not None and int(n_sites) != len(exact_sites):
                raise ValueError("n_sites differs from exact_site_node_ids length")
            n_sites = len(exact_sites)
        if n_sites is None or int(n_sites) <= 0:
            raise ValueError("A positive n_sites or exact_site_node_ids is required")
        n_sites = int(n_sites)
        total_contacts = sum(int(value) for value in target_counts.values())
        if total_contacts < n_sites:
            raise ValueError("Target counts imply fewer than one contact per presynaptic site")
        context = self._context(
            repository,
            skeleton_id,
            target_counts,
            force_node_ids=set(exact_sites) if exact_sites is not None else None,
        )
        intensity = self.site_model.predict_mean(context.site_records)
        if exact_sites is None:
            selected_indices = select_exact_sites(intensity, n_sites, self.config)
            location_mode = "predicted"
        else:
            lookup = {
                int(node_id): index for index, node_id in enumerate(context.candidates["node_id"].astype(int))
            }
            selected_indices = np.asarray([lookup[node_id] for node_id in exact_sites], dtype=int)
            location_mode = "exact"

        selected_candidates = context.candidates.iloc[selected_indices].reset_index(drop=True).copy()
        selected_records = [context.site_records[index] for index in selected_indices]
        selected_intensity = intensity[selected_indices]
        degree_means = self.degree_model.predict_mean(selected_records)
        degree_mode = self.config.degree.mode if degree_mode is None else degree_mode
        if degree_mode not in {"learned", "exact_histogram", "supplied_summary"}:
            raise ValueError(f"Unknown degree mode: {degree_mode}")
        degrees = self._degree_prediction(
            degree_means, total_contacts, degree_mode, degree_histogram, degree_summary
        )

        target_ids = np.asarray(sorted(int(value) for value in target_counts), dtype=int)
        target_count_array = np.asarray([int(target_counts[int(target)]) for target in target_ids])
        selected_distances = {
            int(target): context.distances[int(target)][selected_indices] for target in target_ids
        }
        selected_available = {
            int(target): context.distance_available[int(target)] for target in target_ids
        }
        pair_records: list[dict[str, Any]] = []
        for site_index, site_record in enumerate(selected_records):
            for target_id in target_ids:
                target_id = int(target_id)
                pair_records.append(
                    partner_feature_record(
                        site_record,
                        site_index,
                        target_id,
                        int(target_counts[target_id]),
                        total_contacts,
                        selected_distances,
                        selected_available,
                        repository.celltypes.get(target_id),
                        self.target_prior.log_probability(target_id),
                        self.config,
                    )
                )
        pair_means = self.partner_model.predict_mean(pair_records)
        raw_scores = np.log(np.clip(pair_means, 1e-12, None)).reshape(n_sites, len(target_ids))
        partner_config = self.config.partner
        if partner_config.residualize_scores:
            information_scores, interaction_strength = residualize_partner_scores(
                raw_scores,
                standardize=partner_config.standardize_residual_scores,
                clip=partner_config.score_clip,
                uninformative_threshold=partner_config.uninformative_score_threshold,
            )
        else:
            information_scores = raw_scores.copy()
            interaction_strength = float(np.std(information_scores))
            if interaction_strength <= partner_config.uninformative_score_threshold:
                information_scores.fill(0.0)
            elif partner_config.standardize_residual_scores:
                information_scores = (information_scores - information_scores.mean()) / interaction_strength
            if partner_config.score_clip is not None:
                information_scores = np.clip(
                    information_scores, -partner_config.score_clip, partner_config.score_clip
                )

        rng = np.random.default_rng(self.config.training.random_seed + skeleton_id)
        assignment_mode = partner_config.assignment_mode
        if assignment_mode == "deterministic":
            scores = raw_scores + rng.normal(
                scale=partner_config.random_tie_break, size=raw_scores.shape
            )
            assignment = solve_partner_transport(
                degrees,
                target_count_array,
                scores,
                partner_config.max_same_target_per_site,
            )
            if self.config.features.use_co_targeting and partner_config.co_targeting_weight:
                base_scores = scores.copy()
                for _ in range(partner_config.co_targeting_iterations):
                    adjusted = base_scores.copy()
                    for site_index in range(n_sites):
                        present = target_ids[assignment[site_index] > 0]
                        for target_index, target_id in enumerate(target_ids):
                            bonuses = [
                                self.co_targeting.score(
                                    int(target_id), int(other), repository.celltypes
                                )
                                for other in present
                                if int(other) != int(target_id)
                            ]
                            if bonuses:
                                bonus = (
                                    float(np.mean(bonuses))
                                    if partner_config.co_targeting_aggregation == "mean"
                                    else float(np.sum(bonuses))
                                )
                                if partner_config.co_targeting_score_clip is not None:
                                    bonus = float(np.clip(
                                        bonus,
                                        -partner_config.co_targeting_score_clip,
                                        partner_config.co_targeting_score_clip,
                                    ))
                                adjusted[site_index, target_index] += (
                                    partner_config.co_targeting_weight * bonus
                                )
                    new_assignment = solve_partner_transport(
                        degrees,
                        target_count_array,
                        adjusted,
                        partner_config.max_same_target_per_site,
                    )
                    if np.array_equal(new_assignment, assignment):
                        break
                    assignment = new_assignment
                    scores = adjusted
            assignment_samples_array = assignment[None, :, :]
        else:
            co_target_scores = None
            if (
                assignment_mode != "random_null"
                and self.config.features.use_co_targeting
                and partner_config.co_targeting_weight
            ):
                co_target_scores = np.zeros((len(target_ids), len(target_ids)), dtype=float)
                for left_index, left in enumerate(target_ids):
                    for right_index in range(left_index + 1, len(target_ids)):
                        right = target_ids[right_index]
                        value = self.co_targeting.score(
                            int(left), int(right), repository.celltypes
                        )
                        co_target_scores[left_index, right_index] = value
                        co_target_scores[right_index, left_index] = value
            number_of_samples = (
                partner_config.prediction_samples
                if prediction_samples is None
                else int(prediction_samples)
            )
            if number_of_samples < 1:
                raise ValueError("prediction_samples must be at least one")
            information_mix = 0.0 if assignment_mode == "random_null" else partner_config.information_mix
            assignment_samples_array = np.stack([
                sample_partner_assignment(
                    degrees,
                    target_count_array,
                    information_scores,
                    np.random.default_rng(np.random.SeedSequence([
                        self.config.training.random_seed,
                        skeleton_id,
                        sample_index,
                    ])),
                    information_mix=information_mix,
                    temperature=partner_config.assignment_temperature,
                    repeat_information_discount=partner_config.repeat_information_discount,
                    self_repeat_penalty=partner_config.self_repeat_penalty,
                    preserve_null_repeat_mass=partner_config.preserve_null_repeat_mass,
                    max_same_target_per_site=partner_config.max_same_target_per_site,
                    co_target_scores=co_target_scores,
                    co_targeting_weight=(
                        0.0 if assignment_mode == "random_null"
                        else partner_config.co_targeting_weight
                    ),
                    co_targeting_apply_to=partner_config.co_targeting_apply_to,
                    co_targeting_aggregation=partner_config.co_targeting_aggregation,
                    co_targeting_score_clip=partner_config.co_targeting_score_clip,
                    max_restarts=partner_config.stochastic_max_restarts,
                    mcmc_steps_per_contact=partner_config.mcmc_steps_per_contact,
                )
                for sample_index in range(number_of_samples)
            ])
            assignment = assignment_samples_array[0]
            scores = information_scores

        sites = selected_candidates[
            ["node_id", "parent_node_id", "compartment", "x", "y", "z", "branch_id", "cum_dist"]
        ].copy()
        sites.insert(0, "predicted_site_id", np.arange(n_sites, dtype=int))
        sites["site_location_mode"] = location_mode
        sites["site_intensity"] = selected_intensity
        sites["predicted_degree_mean"] = degree_means
        sites["predicted_degree"] = degrees

        assignment_rows = []
        for site_index in range(n_sites):
            for target_index, target_id in enumerate(target_ids):
                multiplicity = int(assignment[site_index, target_index])
                if not multiplicity:
                    continue
                assignment_rows.append({
                    "predicted_site_id": site_index,
                    "presynaptic_node_id": int(sites.iloc[site_index]["node_id"]),
                    "target_id": int(target_id),
                    "multiplicity": multiplicity,
                    "target_requested_total": int(target_counts[int(target_id)]),
                    "target_celltype": repository.celltypes.get(int(target_id), "unknown"),
                    "target_backbone_available": bool(selected_available[int(target_id)]),
                    "distance_to_target_backbone_um": (
                        float(selected_distances[int(target_id)][site_index])
                        if selected_available[int(target_id)] else np.nan
                    ),
                    "assignment_score": float(scores[site_index, target_index]),
                })
        assignments = pd.DataFrame.from_records(assignment_rows)
        presence_probability = np.mean(assignment_samples_array > 0, axis=0)
        mean_multiplicity = np.mean(assignment_samples_array, axis=0)
        multiplicity_sd = np.std(assignment_samples_array, axis=0)
        summary_rows = []
        for site_index in range(n_sites):
            for target_index, target_id in enumerate(target_ids):
                if presence_probability[site_index, target_index] <= 0:
                    continue
                summary_rows.append({
                    "predicted_site_id": site_index,
                    "presynaptic_node_id": int(sites.iloc[site_index]["node_id"]),
                    "target_id": int(target_id),
                    "presence_probability": float(presence_probability[site_index, target_index]),
                    "mean_multiplicity": float(mean_multiplicity[site_index, target_index]),
                    "multiplicity_sd": float(multiplicity_sd[site_index, target_index]),
                    "target_requested_total": int(target_counts[int(target_id)]),
                })
        assignment_summary = pd.DataFrame.from_records(summary_rows)
        repeated_contacts_per_sample = np.maximum(
            assignment_samples_array - 1, 0
        ).sum(axis=(1, 2))
        same_target_pairs_per_sample = (
            assignment_samples_array * (assignment_samples_array - 1) // 2
        ).sum(axis=(1, 2))
        result = PredictionResult(
            skeleton_id=skeleton_id,
            sites=sites,
            assignments=assignments,
            assignment_matrix=assignment,
            target_ids=target_ids,
            assignment_samples=assignment_samples_array,
            assignment_summary=assignment_summary,
            diagnostics={
                "site_location_mode": location_mode,
                "degree_mode": degree_mode,
                "n_candidates": len(context.candidates),
                "n_sites": n_sites,
                "total_contacts": total_contacts,
                "target_backbones_available": int(sum(selected_available.values())),
                "target_backbones_requested": len(selected_available),
                "assignment_mode": assignment_mode,
                "prediction_samples": int(len(assignment_samples_array)),
                "information_mix": (
                    0.0 if assignment_mode == "random_null"
                    else float(partner_config.information_mix)
                ),
                "raw_site_target_interaction_sd": interaction_strength,
                "information_scores_zeroed_as_uninformative": bool(
                    interaction_strength <= partner_config.uninformative_score_threshold
                ),
                "representative_sample": 0,
                "mean_repeated_contacts_per_sample": float(np.mean(repeated_contacts_per_sample)),
                "mean_same_target_pairs_per_sample": float(np.mean(same_target_pairs_per_sample)),
            },
        )
        result.validate(n_sites, target_counts)
        return result

    def save(self, path: str | Path) -> None:
        if not self.fitted:
            raise RuntimeError("Cannot save an unfitted model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump({"version": self.VERSION, "model": self}, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "PolyadicAssignmentModel":
        with Path(path).open("rb") as stream:
            payload = pickle.load(stream)
        if payload.get("version") != cls.VERSION:
            raise ValueError(
                f"Unsupported saved model version {payload.get('version')}; expected {cls.VERSION}"
            )
        model = payload.get("model")
        if not isinstance(model, cls):
            raise TypeError("Saved object is not a PolyadicAssignmentModel")
        return model
