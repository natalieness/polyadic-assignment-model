"""Dependency-light end-to-end smoke test; runnable without pytest."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from models.approach1.config import DataPaths, DegreeSummary, ModelConfig
from models.approach1.constraints import residualize_partner_scores, sample_partner_assignment
from models.approach1.features import partner_feature_record
from models.approach1.data import DataRepository
from models.approach1.model import PolyadicAssignmentModel


def _write_morphology(cache: Path, skeleton_id: int, y: float = 0.0) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    n = 24
    node_id = np.arange(skeleton_id * 1000, skeleton_id * 1000 + n, dtype=np.int64)
    parent = np.concatenate([[-1], node_id[:-1]])
    np.savez_compressed(
        cache / f"{skeleton_id}.npz",
        node_id=node_id,
        parent_node_id=parent,
        skeleton_id=np.full(n, skeleton_id, dtype=np.int64),
        compartment=np.asarray(["axon"] * n, dtype="U8"),
        x=np.arange(n, dtype=float) * 500.0,
        y=np.full(n, y, dtype=float),
        z=np.zeros(n, dtype=float),
        rank=np.arange(n, dtype=np.int64),
        branch_id=np.zeros(n, dtype=np.int64),
        step_dist=np.concatenate([[np.nan], np.full(n - 1, 500.0)]),
        cum_dist=np.arange(n, dtype=float) * 500.0,
    )


def run_smoke_test() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        detailed = root / "detailed"
        backbones = root / "backbones"
        for skeleton_id, y in [(1, 0.0), (2, 1000.0), (10, 300.0), (11, 800.0)]:
            _write_morphology(detailed, skeleton_id, y)
            _write_morphology(backbones, skeleton_id, y)
        connectors = pd.DataFrame([
            (101, 1, "[10, 11]", 1004, "[10004, 11004]"),
            (102, 1, "[10, 10]", 1010, "[10010, 10011]"),
            (103, 1, "[11]", 1017, "[11017]"),
            (201, 2, "[10]", 2005, "[10005]"),
            (202, 2, "[10, 11]", 2011, "[10011, 11011]"),
            (203, 2, "[11, 11]", 2018, "[11018, 11019]"),
        ], columns=[
            "connector_id", "presynaptic_to", "postsynaptic_to",
            "presynaptic_to_node", "postsynaptic_to_node",
        ])
        connector_path = root / "connectors.csv"
        connectors.to_csv(connector_path, index=False)
        celltypes = pd.DataFrame([
            ("pre", "[1, 2]"), ("target_a", "[10]"), ("target_b", "[11]")
        ], columns=["name", "skids"])
        celltype_path = root / "celltypes.csv"
        celltypes.to_csv(celltype_path, index=False)
        hemisphere_path = root / "hemisphere.csv"
        pd.DataFrame([(1, "left"), (2, "right")], columns=["skid", "hemisphere"]).to_csv(
            hemisphere_path, index=False
        )
        pair_path = root / "pairs.csv"
        pd.DataFrame([(1, 2)], columns=["leftid", "rightid"]).to_csv(pair_path, index=False)

        config = ModelConfig(paths=DataPaths(
            connectors=str(connector_path),
            detailed_nodes=str(root / "unused.csv"),
            target_backbones=str(root / "unused_backbones.csv"),
            celltypes=str(celltype_path),
            hemispheres=str(hemisphere_path),
            bilateral_pairs=str(pair_path),
            morphology_cache=str(detailed),
            backbone_cache=str(backbones),
        ))
        config.training.verbose = False
        config.site.candidate_spacing_um = 0.5
        config.site.selection = "topk"
        config.degree.maximum = 4
        config.partner.prediction_samples = 5
        repository = DataRepository(config.paths)
        model = PolyadicAssignmentModel(config)
        summary = model.fit(repository, [1, 2])
        assert summary["train_neurons"] == 2
        result = model.predict(repository, 1, 3, {10: 3, 11: 2})
        result.validate(3, {10: 3, 11: 2})
        assert result.assignment_matrix.sum() == 5
        assert result.assignment_samples.shape == (5, 3, 2)
        assert not result.assignment_summary.empty
        assert len(result.sites) == 3
        exact_histogram = model.predict(
            repository, 1, 3, {10: 3, 11: 2},
            degree_mode="exact_histogram", degree_histogram={1: 1, 2: 2},
        )
        assert sorted(exact_histogram.sites["predicted_degree"]) == [1, 2, 2]
        summary_result = model.predict(
            repository, 1, 3, {10: 3, 11: 2},
            degree_mode="supplied_summary",
            degree_summary=DegreeSummary(1, 1, 2, 2, 3),
        )
        summary_result.validate(3, {10: 3, 11: 2})
        saved = root / "model.pkl"
        model.save(saved)
        loaded = PolyadicAssignmentModel.load(saved)
        loaded_result = loaded.predict(
            repository, 1, 3, {10: 3, 11: 2}, exact_site_node_ids=[1004, 1010, 1017]
        )
        loaded_result.validate(3, {10: 3, 11: 2})

        # Disabling only the backbones keeps constant, uniform availability fields.
        fallback_config = ModelConfig.from_dict(config.to_dict())
        fallback_config.features.use_target_backbones = False
        fallback_config.training.verbose = False
        fallback_model = PolyadicAssignmentModel(fallback_config)
        fallback_model.fit(repository, [1, 2])
        fallback = fallback_model.predict(repository, 1, 3, {10: 3, 11: 2})
        fallback.validate(3, {10: 3, 11: 2})
        assert fallback.diagnostics["target_backbones_available"] == 0

        # Additive-only scores contain no identifiable site-target interaction.
        additive_scores = (
            np.asarray([1.0, 2.0, 3.0])[:, None]
            + np.asarray([4.0, 5.0])[None, :]
        )
        residual, strength = residualize_partner_scores(additive_scores)
        assert strength < 1e-6
        assert np.array_equal(residual, np.zeros_like(residual))

        # The random null preserves both exact margins.
        null_assignment = sample_partner_assignment(
            np.asarray([2, 2, 1]),
            np.asarray([3, 2]),
            np.zeros((3, 2)),
            np.random.default_rng(123),
            information_mix=0.0,
        )
        assert np.array_equal(null_assignment.sum(axis=1), [2, 2, 1])
        assert np.array_equal(null_assignment.sum(axis=0), [3, 2])
        null_assignment_with_irrelevant_scores = sample_partner_assignment(
            np.asarray([2, 2, 1]),
            np.asarray([3, 2]),
            np.asarray([[100.0, -100.0], [-50.0, 50.0], [20.0, -20.0]]),
            np.random.default_rng(123),
            information_mix=0.0,
        )
        assert np.array_equal(null_assignment, null_assignment_with_irrelevant_scores)

        # Even extreme learned scores may relabel targets but cannot change any
        # site's null-sampled multiplicity profile when the safeguard is on.
        informed_assignment = sample_partner_assignment(
            np.asarray([2, 2, 1]),
            np.asarray([3, 2]),
            np.asarray([[100.0, -100.0], [-50.0, 50.0], [20.0, -20.0]]),
            np.random.default_rng(123),
            information_mix=1.0,
            preserve_null_repeat_mass=True,
            mcmc_steps_per_contact=100,
        )
        null_profiles = [sorted(row[row > 0].tolist()) for row in null_assignment]
        informed_profiles = [sorted(row[row > 0].tolist()) for row in informed_assignment]
        assert informed_profiles == null_profiles
        assert np.array_equal(informed_assignment.sum(axis=0), [3, 2])

        # Target-celltype features include site x target interactions, not only
        # target main effects that would disappear under exact target totals.
        pair_record = partner_feature_record(
            {"compartment": "axon", "relative_path": 0.5, "presynaptic_celltype": "pre"},
            0, 10, 3, 5, {10: np.asarray([1.0])}, {10: True}, "target_a", -1.0, config,
        )
        assert pair_record["compartment_x_target_celltype"] == "axon|target_a"
        assert pair_record["relative_path_x_target_celltype=target_a"] == 0.5


if __name__ == "__main__":
    run_smoke_test()
    print("approach1 smoke test passed")
