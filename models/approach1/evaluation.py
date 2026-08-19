"""Held-out neuron evaluation with site matching before degree/partner scoring."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np

from .data import DataRepository


def _match_sites(predicted_xyz: np.ndarray, observed_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = np.linalg.norm(predicted_xyz[:, None, :] - observed_xyz[None, :, :], axis=2)
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
        predicted_index, observed_index = linear_sum_assignment(distances)
    except ImportError:
        remaining_predicted = set(range(len(predicted_xyz)))
        remaining_observed = set(range(len(observed_xyz)))
        pairs = []
        while remaining_predicted:
            best = min(
                (
                    (distances[predicted, observed], predicted, observed)
                    for predicted in remaining_predicted
                    for observed in remaining_observed
                ),
                key=lambda value: value[0],
            )
            _, predicted, observed = best
            pairs.append((predicted, observed))
            remaining_predicted.remove(predicted)
            remaining_observed.remove(observed)
        predicted_index = np.asarray([pair[0] for pair in pairs], dtype=int)
        observed_index = np.asarray([pair[1] for pair in pairs], dtype=int)
    return predicted_index, observed_index, distances[predicted_index, observed_index]


def evaluate_neuron(model, repository: DataRepository, skeleton_id: int, oracle_sites: bool = False) -> dict:
    neuron = repository.connectors.neuron(skeleton_id)
    exact_nodes = neuron.rows["presynaptic_to_node"].astype(int).tolist()
    result = model.predict(
        repository,
        skeleton_id,
        neuron.n_sites,
        neuron.target_counts,
        exact_site_node_ids=exact_nodes if oracle_sites else None,
        degree_mode="learned",
        prediction_samples=1,
    )
    morphology = repository.morphologies.load(skeleton_id).nodes.set_index("node_id")
    observed_xyz = morphology.loc[exact_nodes, ["x", "y", "z"]].to_numpy(float)
    predicted_xyz = result.sites[["x", "y", "z"]].to_numpy(float)
    predicted_index, observed_index, matched_distance = _match_sites(predicted_xyz, observed_xyz)
    matched_distance_um = matched_distance * model.config.geometry.coordinate_scale_to_um

    observed_degree = np.asarray(
        [len(targets) for targets in neuron.rows["postsynaptic_to"]], dtype=int
    )
    predicted_degree = result.sites["predicted_degree"].to_numpy(int)
    degree_error = predicted_degree[predicted_index] - observed_degree[observed_index]

    target_to_column = {int(target): index for index, target in enumerate(result.target_ids)}
    observed_assignment = np.zeros_like(result.assignment_matrix)
    for site, targets in enumerate(neuron.rows["postsynaptic_to"]):
        counts = Counter(int(target) for target in targets)
        for target, count in counts.items():
            observed_assignment[site, target_to_column[target]] = count
    matched_predicted = result.assignment_matrix[predicted_index]
    matched_observed = observed_assignment[observed_index]
    contact_l1 = np.abs(matched_predicted - matched_observed).sum()
    denominator = max(int(matched_observed.sum()) * 2, 1)
    flattened_predicted = matched_predicted.ravel().astype(float)
    flattened_observed = matched_observed.ravel().astype(float)
    cosine_denominator = np.linalg.norm(flattened_predicted) * np.linalg.norm(flattened_observed)
    cosine = (
        float(flattened_predicted @ flattened_observed / cosine_denominator)
        if cosine_denominator else 0.0
    )
    return {
        "skeleton_id": int(skeleton_id),
        "n_sites": neuron.n_sites,
        "n_contacts": int(observed_assignment.sum()),
        "site_distance_mean_um": float(np.mean(matched_distance_um)),
        "site_distance_median_um": float(np.median(matched_distance_um)),
        "site_within_1um": float(np.mean(matched_distance_um <= 1.0)),
        "site_within_2um": float(np.mean(matched_distance_um <= 2.0)),
        "degree_mae": float(np.mean(np.abs(degree_error))),
        "degree_exact_fraction": float(np.mean(degree_error == 0)),
        "partner_contact_accuracy": float(1.0 - contact_l1 / denominator),
        "partner_assignment_cosine": cosine,
        "oracle_sites": bool(oracle_sites),
    }


def evaluate_split(model, repository: DataRepository, skeleton_ids: Iterable[int], oracle_sites: bool = False) -> dict:
    skeleton_ids = [int(value) for value in skeleton_ids]
    rows = []
    failures = []
    for skeleton_id in skeleton_ids:
        try:
            rows.append(evaluate_neuron(model, repository, int(skeleton_id), oracle_sites=oracle_sites))
        except Exception as error:  # Preserve the rest of a long held-out evaluation.
            failures.append({"skeleton_id": int(skeleton_id), "error": str(error)})
    metric_names = [
        "site_distance_mean_um", "site_distance_median_um", "site_within_1um",
        "site_within_2um", "degree_mae", "degree_exact_fraction",
        "partner_contact_accuracy", "partner_assignment_cosine",
    ]
    summary = {
        metric: float(np.mean([row[metric] for row in rows])) if rows else float("nan")
        for metric in metric_names
    }
    return {
        "neurons_requested": len(skeleton_ids),
        "neurons_evaluated": len(rows),
        "failures": failures,
        "mean_metrics": summary,
        "per_neuron": rows,
    }
