"""Configuration for fixed-site, fixed-degree assignment attribution."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

from models.approach1.config import DataPaths


@dataclass
class GeometryConfig:
    enabled: bool = True
    coordinate_scale_to_um: float = 0.001
    bandwidth_um: float = 5.0
    distance_chunk_size: int = 512
    missing_backbone_policy: str = "uniform"  # uniform | error
    standardize_scores: bool = True
    score_clip: float | None = 2.0
    uninformative_threshold: float = 1e-8
    repeat_discount: float = 0.0


@dataclass
class CotargetConfig:
    enabled: bool = True
    source: str = "observed_neuron"  # observed_neuron | supplied | none
    supplied_scores: str | None = None
    expectation: str = "geometry"  # geometry | null
    smoothing: float = 1.0
    shrinkage: float = 10.0
    standardize_scores: bool = True
    score_clip: float | None = 2.0
    aggregation: str = "mean"  # mean | sum
    site_score_clip: float | None = 2.0


@dataclass
class SamplerConfig:
    prediction_samples: int = 100
    random_seed: int = 42
    geometry_weight: float = 1.0
    cotargeting_weight: float = 1.0
    temperature: float = 1.0
    mcmc_steps_per_contact: int = 20
    null_burnin_steps_per_contact: int = 50
    multiplicity_profile_source: str = "observed"  # observed | random_null
    preserve_multiplicity_profile: bool = True
    self_repeat_penalty: float = 0.0
    max_same_target_per_site: int | None = None
    max_null_restarts: int = 100


@dataclass
class OutputConfig:
    save_raw_samples: bool = True


@dataclass
class Approach2Config:
    paths: DataPaths = field(default_factory=DataPaths)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    cotarget: CotargetConfig = field(default_factory=CotargetConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        if self.geometry.bandwidth_um <= 0:
            raise ValueError("geometry.bandwidth_um must be positive")
        if self.geometry.distance_chunk_size < 1:
            raise ValueError("geometry.distance_chunk_size must be positive")
        if self.geometry.missing_backbone_policy not in {"uniform", "error"}:
            raise ValueError("geometry.missing_backbone_policy must be uniform or error")
        if self.geometry.score_clip is not None and self.geometry.score_clip <= 0:
            raise ValueError("geometry.score_clip must be positive or null")
        if not 0.0 <= self.geometry.repeat_discount <= 1.0:
            raise ValueError("geometry.repeat_discount must be between zero and one")
        if self.cotarget.source not in {"observed_neuron", "supplied", "none"}:
            raise ValueError("cotarget.source must be observed_neuron, supplied, or none")
        if self.cotarget.source == "supplied" and not self.cotarget.supplied_scores:
            raise ValueError("cotarget.supplied_scores is required for supplied scores")
        if self.cotarget.expectation not in {"geometry", "null"}:
            raise ValueError("cotarget.expectation must be geometry or null")
        if self.cotarget.smoothing <= 0 or self.cotarget.shrinkage < 0:
            raise ValueError("invalid cotarget smoothing or shrinkage")
        if self.cotarget.aggregation not in {"mean", "sum"}:
            raise ValueError("cotarget.aggregation must be mean or sum")
        if self.cotarget.score_clip is not None and self.cotarget.score_clip <= 0:
            raise ValueError("cotarget.score_clip must be positive or null")
        if self.cotarget.site_score_clip is not None and self.cotarget.site_score_clip <= 0:
            raise ValueError("cotarget.site_score_clip must be positive or null")
        if self.sampler.prediction_samples < 1:
            raise ValueError("sampler.prediction_samples must be at least one")
        if self.sampler.temperature <= 0:
            raise ValueError("sampler.temperature must be positive")
        if self.sampler.geometry_weight < 0 or self.sampler.cotargeting_weight < 0:
            raise ValueError("sampler weights cannot be negative")
        if self.sampler.mcmc_steps_per_contact < 0:
            raise ValueError("sampler.mcmc_steps_per_contact cannot be negative")
        if self.sampler.null_burnin_steps_per_contact < 0:
            raise ValueError("sampler.null_burnin_steps_per_contact cannot be negative")
        if self.sampler.multiplicity_profile_source not in {"observed", "random_null"}:
            raise ValueError("multiplicity_profile_source must be observed or random_null")
        if self.sampler.self_repeat_penalty < 0:
            raise ValueError("sampler.self_repeat_penalty cannot be negative")
        if self.sampler.preserve_multiplicity_profile and self.sampler.self_repeat_penalty:
            raise ValueError(
                "self_repeat_penalty has no effect with a locked multiplicity profile"
            )
        if (
            self.sampler.max_same_target_per_site is not None
            and self.sampler.max_same_target_per_site < 1
        ):
            raise ValueError("max_same_target_per_site must be positive or null")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Approach2Config":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "Approach2Config":
        unknown = set(values) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
        config = cls(
            paths=DataPaths(**values.get("paths", {})),
            geometry=GeometryConfig(**values.get("geometry", {})),
            cotarget=CotargetConfig(**values.get("cotarget", {})),
            sampler=SamplerConfig(**values.get("sampler", {})),
            output=OutputConfig(**values.get("output", {})),
        )
        config.validate()
        return config
