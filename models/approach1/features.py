"""Candidate generation and switchable biological feature construction."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import ModelConfig
from .data import Morphology


def add_topology_columns(nodes: pd.DataFrame, coordinate_scale: float) -> pd.DataFrame:
    result = nodes.copy()
    node_ids = result["node_id"].astype("int64").to_numpy()
    parent_ids = result["parent_node_id"].astype("int64").to_numpy()
    child_counts = Counter(parent_ids[parent_ids >= 0].tolist())
    result["child_count"] = [child_counts.get(int(node), 0) for node in node_ids]
    result["is_root"] = (result["parent_node_id"] < 0).astype(int)
    result["is_leaf"] = (result["child_count"] == 0).astype(int)
    result["is_branchpoint"] = (result["child_count"] > 1).astype(int)
    result["step_um"] = result["step_dist"].fillna(0.0).astype(float) * coordinate_scale
    result["cum_um"] = result["cum_dist"].astype(float) * coordinate_scale
    max_by_compartment = result.groupby("compartment")["cum_um"].transform("max").clip(lower=1e-9)
    result["relative_path"] = result["cum_um"] / max_by_compartment
    for axis in ["x", "y", "z"]:
        result[f"{axis}_um"] = result[axis].astype(float) * coordinate_scale
        span = max(float(result[f"{axis}_um"].max() - result[f"{axis}_um"].min()), 1e-9)
        result[f"{axis}_relative"] = (result[f"{axis}_um"] - result[f"{axis}_um"].min()) / span
    return result


def generate_candidates(
    morphology: Morphology,
    config: ModelConfig,
    force_node_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Approximately uniform candidate nodes, always retaining forced training sites."""
    nodes = add_topology_columns(morphology.nodes, config.geometry.coordinate_scale_to_um)
    spacing = config.site.candidate_spacing_um
    forced = set() if force_node_ids is None else {int(value) for value in force_node_ids}
    keep: set[int] = set(forced)
    for (_, _), group in nodes.groupby(["compartment", "branch_id"], sort=False):
        ordered = group.sort_values(["cum_um", "rank", "node_id"])
        last_distance = -np.inf
        for row in ordered.itertuples(index=False):
            distance = float(row.cum_um)
            structural = bool(row.is_root or row.is_leaf or row.is_branchpoint)
            if structural or distance - last_distance >= spacing:
                keep.add(int(row.node_id))
                last_distance = distance
    selected = nodes.loc[nodes["node_id"].isin(keep)].copy()
    if forced - set(selected["node_id"].astype(int)):
        missing = sorted(forced - set(selected["node_id"].astype(int)))
        raise ValueError(f"Forced presynaptic nodes are missing from morphology: {missing[:20]}")
    return selected.sort_values(["compartment", "cum_um", "node_id"]).reset_index(drop=True)


def _brute_nearest(points: np.ndarray, target: np.ndarray, chunk_size: int) -> np.ndarray:
    result = np.full(len(points), np.inf, dtype=float)
    for start in range(0, len(points), chunk_size):
        block = points[start : start + chunk_size]
        # Split the target too, avoiding a potentially huge block x target tensor.
        best = np.full(len(block), np.inf, dtype=float)
        for target_start in range(0, len(target), chunk_size * 4):
            target_block = target[target_start : target_start + chunk_size * 4]
            squared = np.sum((block[:, None, :] - target_block[None, :, :]) ** 2, axis=2)
            best = np.minimum(best, np.min(squared, axis=1))
        result[start : start + len(block)] = np.sqrt(best)
    return result


def nearest_distances_um(points_um: np.ndarray, target_um: np.ndarray, chunk_size: int) -> np.ndarray:
    if not len(target_um):
        return np.full(len(points_um), np.nan)
    try:
        from scipy.spatial import cKDTree  # type: ignore
        return cKDTree(target_um).query(points_um, workers=-1)[0]
    except ImportError:
        return _brute_nearest(points_um, target_um, chunk_size)


def target_distance_fields(
    candidates: pd.DataFrame,
    target_counts: Mapping[int, int],
    backbones: Mapping[int, Morphology],
    config: ModelConfig,
) -> tuple[dict[int, np.ndarray], dict[int, bool]]:
    """Distance to each target; missing/disabled backbones are uniform fields."""
    points = candidates[["x_um", "y_um", "z_um"]].to_numpy(float)
    distances: dict[int, np.ndarray] = {}
    available: dict[int, bool] = {}
    for target_id in target_counts:
        target_id = int(target_id)
        if config.features.use_target_backbones and target_id in backbones:
            target = backbones[target_id].nodes[["x", "y", "z"]].to_numpy(float)
            target *= config.geometry.coordinate_scale_to_um
            distances[target_id] = nearest_distances_um(points, target, config.geometry.distance_chunk_size)
            available[target_id] = True
        else:
            distances[target_id] = np.zeros(len(candidates), dtype=float)
            available[target_id] = False
    return distances, available


def aggregate_availability(
    target_counts: Mapping[int, int],
    distances: Mapping[int, np.ndarray],
    available: Mapping[int, bool],
    config: ModelConfig,
) -> dict[str, np.ndarray]:
    n_candidates = len(next(iter(distances.values()))) if distances else 0
    total_contacts = max(sum(int(value) for value in target_counts.values()), 1)
    bandwidth = config.geometry.availability_bandwidth_um
    kernel_sum = np.zeros(n_candidates, dtype=float)
    nearest = np.full(n_candidates, np.inf, dtype=float)
    available_weight = 0
    within = {radius: np.zeros(n_candidates, dtype=float) for radius in config.geometry.availability_radii_um}
    for target_id, count in target_counts.items():
        weight = int(count)
        distance = distances[int(target_id)]
        if available[int(target_id)]:
            available_weight += weight
            nearest = np.minimum(nearest, distance)
            kernel_sum += weight * np.exp(-0.5 * (distance / bandwidth) ** 2)
            for radius in within:
                within[radius] += weight * (distance <= radius)
        else:
            # Uniform fallback: a missing target contributes equally everywhere.
            kernel_sum += weight
            for radius in within:
                within[radius] += weight
    if not available_weight:
        nearest[:] = 0.0
    result = {
        "target_availability": kernel_sum / total_contacts,
        "nearest_target_um": nearest,
        "backbone_contact_fraction": np.full(n_candidates, available_weight / total_contacts),
    }
    for radius, values in within.items():
        result[f"target_fraction_within_{radius:g}um"] = values / total_contacts
    return result


def site_feature_records(
    candidates: pd.DataFrame,
    availability: Mapping[str, np.ndarray],
    config: ModelConfig,
    presynaptic_celltype: str | None = None,
    hemisphere: str | None = None,
) -> list[dict[str, Any]]:
    flags = config.features
    records: list[dict[str, Any]] = []
    for index, row in candidates.iterrows():
        record: dict[str, Any] = {"intercept_feature": 1.0}
        if flags.use_compartment:
            record["compartment"] = str(row["compartment"])
        if flags.use_path_distance:
            record["relative_path"] = float(row["relative_path"])
            record["relative_path_squared"] = float(row["relative_path"]) ** 2
        if flags.use_branch_features:
            record["child_count"] = float(row["child_count"])
            record["is_leaf"] = float(row["is_leaf"])
            record["is_branchpoint"] = float(row["is_branchpoint"])
        if flags.use_local_cable:
            record["step_um"] = float(row["step_um"])
        if flags.use_coordinates:
            for axis in ["x", "y", "z"]:
                record[f"{axis}_relative"] = float(row[f"{axis}_relative"])
        if flags.use_target_availability:
            for name, values in availability.items():
                record[name] = float(values[index])
        if flags.use_presynaptic_celltype:
            record["presynaptic_celltype"] = presynaptic_celltype or "unknown"
        if flags.use_hemisphere:
            record["hemisphere"] = hemisphere or "unknown"
        records.append(record)
    return records


def partner_feature_record(
    site_record: Mapping[str, Any],
    site_index: int,
    target_id: int,
    target_count: int,
    total_contacts: int,
    distances: Mapping[int, np.ndarray],
    distance_available: Mapping[int, bool],
    target_celltype: str | None,
    target_log_prior: float,
    config: ModelConfig,
) -> dict[str, Any]:
    flags = config.features
    record = dict(site_record)
    if flags.use_target_count_features:
        record["target_count"] = float(target_count)
        record["target_fraction"] = float(target_count) / max(total_contacts, 1)
        record["log_target_count"] = float(np.log1p(target_count))
    if flags.use_target_identity_prior:
        record["target_identity_log_prior"] = float(target_log_prior)
    if flags.use_target_celltype:
        target_type = target_celltype or "unknown"
        record["target_celltype"] = target_type

        # Main effects for site and target are already fixed by the requested
        # row/column totals and are removed before assignment. These explicit
        # interactions let cell type express *where* a target type tends to be
        # contacted, while the stochastic decoder still bounds its influence.
        if "compartment" in site_record:
            compartment = str(site_record["compartment"])
            record["compartment_x_target_celltype"] = f"{compartment}|{target_type}"
            if flags.use_presynaptic_celltype and "presynaptic_celltype" in site_record:
                presynaptic_type = str(site_record["presynaptic_celltype"])
                record["presynaptic_x_compartment_x_target_celltype"] = (
                    f"{presynaptic_type}|{compartment}|{target_type}"
                )
        for site_feature in [
            "relative_path",
            "relative_path_squared",
            "is_leaf",
            "is_branchpoint",
            "step_um",
        ]:
            if site_feature in site_record:
                record[f"{site_feature}_x_target_celltype={target_type}"] = (
                    float(site_record[site_feature])
                )
    if flags.use_target_distance_for_partner:
        record["target_backbone_available"] = float(distance_available[int(target_id)])
        record["distance_to_target_um"] = float(distances[int(target_id)][site_index])
        bandwidth = config.geometry.availability_bandwidth_um
        if distance_available[int(target_id)]:
            record["target_geometry_kernel"] = float(
                np.exp(-0.5 * (distances[int(target_id)][site_index] / bandwidth) ** 2)
            )
        else:
            record["target_geometry_kernel"] = 1.0
    return record
