"""Dependency-light end-to-end smoke test for Approach 2."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from models.approach1.config import DataPaths
from models.approach1.data import DataRepository
from models.approach2.config import Approach2Config
from models.approach2.simulator import CONDITIONS, run_attribution_experiment


def _write_morphology(cache: Path, skeleton_id: int, y: float) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    n = 12
    node_ids = np.arange(skeleton_id * 1000, skeleton_id * 1000 + n, dtype=np.int64)
    np.savez_compressed(
        cache / f"{skeleton_id}.npz",
        node_id=node_ids,
        parent_node_id=np.concatenate([[-1], node_ids[:-1]]),
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
        _write_morphology(detailed, 1, 0.0)
        _write_morphology(backbones, 10, 200.0)
        _write_morphology(backbones, 11, 1000.0)
        _write_morphology(backbones, 12, 2000.0)
        connectors = pd.DataFrame([
            (101, 1, "[10, 10]", 1002, "[10002, 10003]"),
            (102, 1, "[10, 11]", 1004, "[10004, 11004]"),
            (103, 1, "[11, 12]", 1006, "[11006, 12006]"),
            (104, 1, "[12]", 1008, "[12008]"),
        ], columns=[
            "connector_id", "presynaptic_to", "postsynaptic_to",
            "presynaptic_to_node", "postsynaptic_to_node",
        ])
        connector_path = root / "connectors.csv"
        connectors.to_csv(connector_path, index=False)
        config = Approach2Config(paths=DataPaths(
            connectors=str(connector_path),
            detailed_nodes=str(root / "unused.csv"),
            target_backbones=str(root / "unused_backbones.csv"),
            celltypes=None,
            hemispheres=None,
            bilateral_pairs=None,
            morphology_cache=str(detailed),
            backbone_cache=str(backbones),
        ))
        config.sampler.prediction_samples = 5
        config.sampler.mcmc_steps_per_contact = 10
        config.sampler.null_burnin_steps_per_contact = 20
        repository = DataRepository(config.paths)
        result = run_attribution_experiment(repository, 1, config)
        result.validate()
        assert set(result.samples) == set(CONDITIONS)
        for samples in result.samples.values():
            assert samples.shape == (5, 4, 3)
            assert np.all(samples.sum(axis=2) == np.asarray([2, 2, 2, 1]))
            assert np.all(samples.sum(axis=1) == np.asarray([3, 2, 2]))
        assert result.geometry.backbone_available.all()
        assert len(result.condition_summary) == 4
        pair_rows = result.pair_attribution
        assert np.allclose(
            pair_rows["geometry_shapley_contribution"]
            + pair_rows["cotargeting_shapley_contribution"],
            pair_rows["combined_mean"] - pair_rows["null_mean"],
        )
        output = root / "output"
        result.save(output)
        assert (output / "pair_attribution.csv").exists()
        assert (output / "assignment_samples.npz").exists()


if __name__ == "__main__":
    run_smoke_test()
    print("approach2 smoke test passed")
