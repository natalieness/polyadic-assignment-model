"""Target-backbone geometry for fixed presynaptic sites."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from models.approach1.config import DataPaths
from models.approach1.data import DataRepository

from .config import GeometryConfig
from .data import FixedNeuronInputs

# Backbone variants that the local maximum distance table provides a column for. A
# target-backbone file declares its variant through its name, as in
# ``data/neuron_backbone_top3S.csv``.
BACKBONE_VARIANTS = ("Sover1", "top2S", "top3S")
LOCAL_MAX_DIST_ID_COLUMN = "presynaptic_to"


@dataclass
class GeometryResult:
    distances_um: np.ndarray
    scores: np.ndarray
    backbone_available: np.ndarray
    raw_score_sd: float
    distance_threshold_um: float = float("nan")


def _nearest_distances(points: np.ndarray, target: np.ndarray, chunk_size: int) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree  # type: ignore

        return cKDTree(target).query(points, workers=-1)[0]
    except ImportError:
        result = np.full(len(points), np.inf, dtype=float)
        for start in range(0, len(points), chunk_size):
            block = points[start : start + chunk_size]
            best = np.full(len(block), np.inf, dtype=float)
            for target_start in range(0, len(target), chunk_size * 4):
                target_block = target[target_start : target_start + chunk_size * 4]
                squared = np.sum(
                    (block[:, None, :] - target_block[None, :, :]) ** 2,
                    axis=2,
                )
                best = np.minimum(best, np.min(squared, axis=1))
            result[start : start + len(block)] = np.sqrt(best)
        return result


def backbone_variant(backbone_path: str | None) -> str:
    """Name the backbone variant a target-backbone file holds, from its file name."""
    if backbone_path is None:
        raise ValueError(
            "paths.target_backbones must be set to resolve a local maximum target distance"
        )
    stem = Path(backbone_path).stem
    matched = [name for name in BACKBONE_VARIANTS if stem.endswith(name)]
    if not matched:
        raise ValueError(
            f"Cannot tell which backbone variant '{backbone_path}' holds. Its file name "
            f"must end with one of {list(BACKBONE_VARIANTS)}, as in "
            "data/neuron_backbone_top3S.csv"
        )
    return max(matched, key=len)


@lru_cache(maxsize=None)
def load_local_max_distances(table_path: str, variant: str) -> dict[int, float]:
    """Map each presynaptic skeleton to its maximum site-to-target-backbone distance.

    Distances stay in raw coordinate units here, matching the node coordinates they
    were measured from. Cached per file and variant, because every neuron in a run
    reads the same table.
    """
    path = Path(table_path)
    if not path.exists():
        raise ValueError(f"Local maximum distance table '{table_path}' does not exist")
    table = pd.read_csv(path)
    missing = {LOCAL_MAX_DIST_ID_COLUMN, variant} - set(table.columns)
    if missing:
        raise ValueError(
            f"'{table_path}' is missing the columns {sorted(missing)} needed for the "
            f"'{variant}' backbones"
        )
    return dict(
        zip(
            table[LOCAL_MAX_DIST_ID_COLUMN].astype(int),
            table[variant].astype(float),
        )
    )


def local_max_target_distance_um(
    paths: DataPaths, skeleton_id: int, config: GeometryConfig
) -> float:
    """Return one neuron's own maximum target-backbone distance, in micrometres."""
    if paths.local_max_dist is None:
        raise ValueError(
            "paths.local_max_dist must be set to use local maximum target distances"
        )
    variant = backbone_variant(paths.target_backbones)
    if variant not in BACKBONE_VARIANTS:
        raise ValueError(
            f"Cannot use local maximum target distances with backbone variant '{variant}'"
        )
    distances = load_local_max_distances(str(paths.local_max_dist), variant)
    # don't allow a local max that is lower than the median max distance across all neurons
    # in case a neuron is really weird or has few connections 
    median_max_dist = np.median(list(distances.values()))
    distance = distances.get(int(skeleton_id))
    if distance is None or not np.isfinite(distance):
        raise ValueError(
            f"No usable '{variant}' local maximum distance for skeleton {skeleton_id} "
            f"in '{paths.local_max_dist}'"
        )
    if distance < median_max_dist:
        distance = median_max_dist
    return float(distance) * config.coordinate_scale_to_um


def geometry_scores(
    repository: DataRepository,
    inputs: FixedNeuronInputs,
    config: GeometryConfig,
    use_binary_possibility: bool = True, # binary possibility of target in reach or not
    use_binary_and_soft: bool = True, # binary possibility of target in reach or not and soft gaussian kernel for distance to target backbone
    use_local_max_target_dist: bool = True # using local max target distance to target backbones instead of a fixed bandwidth
) -> GeometryResult:
    """Return target-specific, column-centred proximity scores.

    A missing target backbone receives a zero column, meaning equal likelihood
    at every site. Pure target and site main effects are irrelevant under fixed
    margins; column centring keeps each target's score explicitly relative to
    its own available locations.
    """
    n_sites = len(inputs.site_degrees)
    n_targets = len(inputs.target_ids)
    distances = np.full((n_sites, n_targets), np.nan, dtype=float)
    scores = np.zeros((n_sites, n_targets), dtype=float)
    available = np.zeros(n_targets, dtype=bool)
    if not config.enabled:
        return GeometryResult(distances, scores, available, 0.0)

    backbones = repository.load_backbones(
        inputs.target_ids,
        missing_policy=config.missing_backbone_policy,
    )
    site_points = inputs.sites[["x", "y", "z"]].to_numpy(float)
    site_points *= config.coordinate_scale_to_um

    # get local max target distance if requested
    if use_local_max_target_dist:
        dist_threshold = local_max_target_distance_um(
            repository.paths, inputs.skeleton_id, config
        )
    else:
        dist_threshold = config.bandwidth_um

    for target_index, target_id in enumerate(inputs.target_ids):
        backbone = backbones.get(int(target_id))
        if backbone is None:
            print(f'No backbone for target {target_id}, skipping')
            continue
        target_points = backbone.nodes[["x", "y", "z"]].to_numpy(float)
        target_points *= config.coordinate_scale_to_um
        target_distance = _nearest_distances(
            site_points, target_points, config.distance_chunk_size
        )
        distances[:, target_index] = target_distance
        if use_binary_and_soft:
            kernel = (target_distance <= (dist_threshold)).astype(float) * np.exp(-0.5 * (target_distance / dist_threshold) ** 2)
        elif use_binary_possibility:
            kernel = (target_distance <= (dist_threshold)).astype(float)
        else:
            kernel = np.exp(-0.5 * (target_distance / dist_threshold) ** 2)
        scores[:, target_index] = kernel - kernel.mean()  #NOTE: also tried median norm  np.median(kernel) 
        available[target_index] = True

    variable_values = scores[:, available].ravel()
    raw_sd = float(np.std(variable_values)) if len(variable_values) else 0.0
    if raw_sd <= config.uninformative_threshold:
        scores.fill(0.0)
    elif config.standardize_scores:
        scores[:, available] /= raw_sd
    if config.score_clip is not None:
        scores = np.clip(scores, -config.score_clip, config.score_clip)
    scores[:, ~available] = 0.0
    return GeometryResult(distances, scores, available, raw_sd, dist_threshold)
