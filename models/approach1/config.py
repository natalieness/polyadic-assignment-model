"""Configuration objects for the staged polyadic-assignment model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping


@dataclass
class DataPaths:
    """Input paths. Relative paths are resolved from the current directory."""

    connectors: str = "data/con_short_2026_correctnodes.csv"
    detailed_nodes: str = "data/nodes_manual_wdendrites_euclid.csv"
    target_backbones: str | None = "data/neuron_backbones_10um.csv"
    celltypes: str | None = "data/celltype_df.csv"
    hemispheres: str | None = "data/skid_hemisphere.csv"
    bilateral_pairs: str | None = "data/pairs-2022-02-14.csv"
    morphology_cache: str = "data/approach1_cache/detailed"
    backbone_cache: str = "data/approach1_cache/backbones"
    # Per-presynaptic-neuron maximum observed site-to-target-backbone distance, with
    # one column per backbone variant (Sover1, top2S, top3S).
    local_max_dist: str | None = "data/max_presyn_to_backbone_distances_by_skid.csv"


@dataclass
class FeatureFlags:
    """Ablation switches. Required inputs cannot be disabled."""

    # Skeleton structure is always used to define possible site locations.
    use_compartment: bool = True
    use_path_distance: bool = True
    use_branch_features: bool = True
    use_coordinates: bool = False
    use_local_cable: bool = True

    # Optional spatial information. False gives uniform target availability.
    use_target_backbones: bool = True
    use_target_availability: bool = True
    use_target_distance_for_partner: bool = True

    # Optional metadata and learned interactions.
    use_target_celltype: bool = True
    use_presynaptic_celltype: bool = True
    use_hemisphere: bool = False
    use_co_targeting: bool = True

    # Target identity/counts remain present as constraints even when their
    # learned prior features are ablated.
    use_target_identity_prior: bool = True
    use_target_count_features: bool = True


@dataclass
class SiteModelConfig:
    candidate_spacing_um: float = 0.75
    negative_to_positive_ratio: float = 5.0
    max_sites_per_candidate: int = 5
    duplicate_site_penalty: float = 1.0
    selection: str = "sample"  # sample | topk
    random_seed: int = 42
    alpha: float = 1.0
    max_iter: int = 80


@dataclass
class DegreeModelConfig:
    alpha: float = 1.0
    max_iter: int = 80
    minimum: int = 1
    maximum: int = 30
    mode: str = "learned"  # learned | exact_histogram | supplied_summary


@dataclass
class PartnerModelConfig:
    alpha: float = 1.0
    max_iter: int = 80
    negative_targets_per_positive: int = 3

    # Final exact-margin assignment. ``stochastic`` starts from the random
    # target-token null and lets learned information only tilt that null.
    # ``random_null`` ignores all learned pair/co-targeting scores, while
    # ``deterministic`` retains the original maximum-score transport decoder.
    assignment_mode: str = "stochastic"  # stochastic | random_null | deterministic
    information_mix: float = 0.25
    assignment_temperature: float = 2.0
    residualize_scores: bool = True
    standardize_residual_scores: bool = True
    score_clip: float | None = 2.0
    uninformative_score_threshold: float = 1e-6
    prediction_samples: int = 100
    stochastic_max_restarts: int = 100
    mcmc_steps_per_contact: int = 10

    # Repeated copies of one target at one site remain possible. Learned
    # first-contact information is discounted for later copies, preventing a
    # strong placement score from automatically becoming a repeat preference.
    self_repeat_penalty: float = 0.0
    repeat_information_discount: float = 0.0
    preserve_null_repeat_mass: bool = True
    max_same_target_per_site: int | None = None

    # Co-targeting is a bounded nudge among otherwise available targets.
    co_targeting_weight: float = 0.1
    co_targeting_apply_to: str = "first_copy_only"  # first_copy_only | all_copies
    co_targeting_aggregation: str = "mean"  # mean | sum
    co_targeting_score_clip: float | None = 1.0

    # Used only by the legacy deterministic decoder.
    co_targeting_iterations: int = 3
    co_targeting_smoothing: float = 1.0
    co_targeting_shrinkage: float = 10.0
    co_targeting_min_observations: int = 2
    random_tie_break: float = 1e-7


@dataclass
class GeometryConfig:
    coordinate_scale_to_um: float = 0.001
    availability_bandwidth_um: float = 5.0
    availability_radii_um: list[float] = field(default_factory=lambda: [1.0, 2.0, 5.0, 10.0])
    distance_chunk_size: int = 512
    missing_backbone_policy: str = "uniform"  # uniform | error


@dataclass
class TrainingConfig:
    max_site_rows_per_neuron: int = 2_000
    max_partner_rows_per_neuron: int = 5_000
    max_train_neurons: int | None = None
    random_seed: int = 42
    verbose: bool = True


@dataclass
class ModelConfig:
    paths: DataPaths = field(default_factory=DataPaths)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    site: SiteModelConfig = field(default_factory=SiteModelConfig)
    degree: DegreeModelConfig = field(default_factory=DegreeModelConfig)
    partner: PartnerModelConfig = field(default_factory=PartnerModelConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        if self.site.candidate_spacing_um <= 0:
            raise ValueError("candidate_spacing_um must be positive")
        if self.site.max_sites_per_candidate < 1:
            raise ValueError("max_sites_per_candidate must be at least one")
        if self.degree.minimum < 1 or self.degree.maximum < self.degree.minimum:
            raise ValueError("invalid degree limits")
        if self.site.selection not in {"sample", "topk"}:
            raise ValueError("site.selection must be 'sample' or 'topk'")
        if self.degree.mode not in {"learned", "exact_histogram", "supplied_summary"}:
            raise ValueError("invalid degree.mode")
        if self.geometry.missing_backbone_policy not in {"uniform", "error"}:
            raise ValueError("invalid missing_backbone_policy")
        if self.partner.assignment_mode not in {"stochastic", "random_null", "deterministic"}:
            raise ValueError("partner.assignment_mode must be stochastic, random_null, or deterministic")
        if not 0.0 <= self.partner.information_mix <= 1.0:
            raise ValueError("partner.information_mix must be between zero and one")
        if self.partner.assignment_temperature <= 0:
            raise ValueError("partner.assignment_temperature must be positive")
        if self.partner.score_clip is not None and self.partner.score_clip <= 0:
            raise ValueError("partner.score_clip must be positive or null")
        if self.partner.uninformative_score_threshold < 0:
            raise ValueError("partner.uninformative_score_threshold cannot be negative")
        if self.partner.prediction_samples < 1:
            raise ValueError("partner.prediction_samples must be at least one")
        if self.partner.stochastic_max_restarts < 1:
            raise ValueError("partner.stochastic_max_restarts must be at least one")
        if self.partner.mcmc_steps_per_contact < 0:
            raise ValueError("partner.mcmc_steps_per_contact cannot be negative")
        if self.partner.self_repeat_penalty < 0:
            raise ValueError("partner.self_repeat_penalty cannot be negative")
        if self.partner.preserve_null_repeat_mass and self.partner.self_repeat_penalty:
            raise ValueError(
                "self_repeat_penalty cannot alter a locked null multiplicity profile; "
                "set preserve_null_repeat_mass=false to use the penalty"
            )
        if not 0.0 <= self.partner.repeat_information_discount <= 1.0:
            raise ValueError("partner.repeat_information_discount must be between zero and one")
        if self.partner.max_same_target_per_site is not None and self.partner.max_same_target_per_site < 1:
            raise ValueError("partner.max_same_target_per_site must be positive or null")
        if self.partner.co_targeting_apply_to not in {"first_copy_only", "all_copies"}:
            raise ValueError("partner.co_targeting_apply_to must be first_copy_only or all_copies")
        if self.partner.co_targeting_aggregation not in {"mean", "sum"}:
            raise ValueError("partner.co_targeting_aggregation must be mean or sum")
        if self.partner.co_targeting_score_clip is not None and self.partner.co_targeting_score_clip <= 0:
            raise ValueError("partner.co_targeting_score_clip must be positive or null")
        if self.features.use_target_availability and not self.features.use_target_backbones:
            # This combination is meaningful: it is the explicit uniform fallback.
            pass

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ModelConfig":
        known = {item.name for item in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
        config = cls(
            paths=DataPaths(**values.get("paths", {})),
            features=FeatureFlags(**values.get("features", {})),
            site=SiteModelConfig(**values.get("site", {})),
            degree=DegreeModelConfig(**values.get("degree", {})),
            partner=PartnerModelConfig(**values.get("partner", {})),
            geometry=GeometryConfig(**values.get("geometry", {})),
            training=TrainingConfig(**values.get("training", {})),
        )
        config.validate()
        return config


@dataclass(frozen=True)
class DegreeSummary:
    minimum: int
    q1: float
    median: float
    q3: float
    maximum: int

    def validate(self) -> None:
        values = [self.minimum, self.q1, self.median, self.q3, self.maximum]
        if any(left > right for left, right in zip(values, values[1:])):
            raise ValueError("Degree summary must satisfy min <= q1 <= median <= q3 <= max")
        if self.minimum < 1:
            raise ValueError("Polyadic degree must be at least one")
