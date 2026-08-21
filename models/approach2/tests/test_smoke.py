"""Dependency-light end-to-end smoke test for Approach 2."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from models.approach1.config import DataPaths
from models.approach1.data import DataRepository
from models.approach2.config import Approach2Config
from models.approach2.propensity import mean_pair_count_matrix
from models.approach2.sampler import validate_assignment
from models.approach2.simulator import (
    CONDITIONS,
    permutation_expectation_samples,
    run_attribution_experiment,
)


def _write_morphology(
    cache: Path,
    skeleton_id: int,
    y: float,
    x_start: float = 0.0,
    step: float = 500.0,
) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    n = 12
    node_ids = np.arange(skeleton_id * 1000, skeleton_id * 1000 + n, dtype=np.int64)
    np.savez_compressed(
        cache / f"{skeleton_id}.npz",
        node_id=node_ids,
        parent_node_id=np.concatenate([[-1], node_ids[:-1]]),
        skeleton_id=np.full(n, skeleton_id, dtype=np.int64),
        compartment=np.asarray(["axon"] * n, dtype="U8"),
        x=x_start + np.arange(n, dtype=float) * step,
        y=np.full(n, y, dtype=float),
        z=np.zeros(n, dtype=float),
        rank=np.arange(n, dtype=np.int64),
        branch_id=np.zeros(n, dtype=np.int64),
        step_dist=np.concatenate([[np.nan], np.full(n - 1, step)]),
        cum_dist=np.arange(n, dtype=float) * step,
    )


# Which sites fall inside the local maximum target distance, per target. Equal-length
# backbones would put every site at the same distance from a target, and column
# centring erases a constant column, so the backbones cover different stretches of x.
WITHIN_THRESHOLD = np.array([
    [True, False, True],
    [True, True, True],
    [False, True, True],
    [False, True, False],
])


def run_smoke_test() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        detailed = root / "detailed"
        backbones = root / "backbones"
        _write_morphology(detailed, 1, 0.0)
        _write_morphology(backbones, 10, 200.0, x_start=0.0, step=100.0)
        _write_morphology(backbones, 11, 200.0, x_start=2900.0, step=100.0)
        _write_morphology(backbones, 12, 200.0, x_start=1200.0, step=100.0)
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
        # The backbone variant is read from the file name, and the threshold comes
        # from this neuron's own row. A single row is its own median, so the median
        # floor in local_max_target_distance_um leaves the value alone.
        local_max_path = root / "local_max_distances.csv"
        pd.DataFrame({"presynaptic_to": [1], "Sover1": [1500.0]}).to_csv(
            local_max_path, index=False
        )

        def make_config() -> Approach2Config:
            config = Approach2Config(paths=DataPaths(
                connectors=str(connector_path),
                detailed_nodes=str(root / "unused.csv"),
                target_backbones=str(root / "unused_backbones_Sover1.csv"),
                celltypes=None,
                hemispheres=None,
                bilateral_pairs=None,
                morphology_cache=str(detailed),
                backbone_cache=str(backbones),
                local_max_dist=str(local_max_path),
            ))
            config.sampler.prediction_samples = 5
            config.sampler.mcmc_steps_per_contact = 10
            config.sampler.null_burnin_steps_per_contact = 20
            config.cotarget.n_perm_expectation = 17
            return config

        config = make_config()
        repository = DataRepository(config.paths)
        result = run_attribution_experiment(repository, 1, config)
        result.validate()

        # Exact margins hold in every condition.
        assert set(result.samples) == set(CONDITIONS)
        for samples in result.samples.values():
            assert samples.shape == (5, 4, 3)
            assert np.all(samples.sum(axis=2) == np.asarray([2, 2, 2, 1]))
            assert np.all(samples.sum(axis=1) == np.asarray([3, 2, 2]))

        # Geometry actually discriminates: each target is reachable from some sites
        # and not others, so no column is constant and the scores carry information.
        assert result.geometry.backbone_available.all()
        assert result.geometry.distance_threshold_um == 1.5
        within = result.geometry.distances_um <= result.geometry.distance_threshold_um
        assert np.array_equal(within, WITHIN_THRESHOLD)
        assert result.geometry.raw_score_sd > 0.0
        assert np.all(result.geometry.scores[within] > 0.0)
        assert np.all(result.geometry.scores[~within] < 0.0)

        # The co-targeting expectation is its own set of permutations, sized by
        # cotarget.n_perm_expectation rather than by sampler.prediction_samples.
        draws = permutation_expectation_samples(result.inputs, config)
        assert draws.shape == (17, 4, 3)
        for draw in draws:
            validate_assignment(
                draw, result.inputs.site_degrees, result.inputs.target_counts
            )
        assert np.allclose(
            result.propensity_expectation,
            mean_pair_count_matrix(draws, config.cotarget.include_self_repeats),
        )
        assert np.array_equal(result.cotarget_scores, result.cotarget_scores.T)

        # Those scores are settled before any condition is simulated, so changing
        # the sampler cannot move them. Shortening the null burn-in changes the null
        # samples themselves, which is what makes this check meaningful.
        shorter = make_config()
        shorter.sampler.null_burnin_steps_per_contact = 2
        other = run_attribution_experiment(repository, 1, shorter)
        assert not np.array_equal(result.samples["null"], other.samples["null"])
        assert np.array_equal(result.cotarget_scores, other.cotarget_scores)
        assert np.array_equal(result.propensity_expectation, other.propensity_expectation)

        assert len(result.condition_summary) == 4
        assert result.diagnostics["cotargeting_n_perm_expectation"] == 17
        assert "cotargeting_expectation" not in result.diagnostics

        output = root / "output"
        result.save(output)
        for condition in CONDITIONS:
            assert (output / f"assignments_{condition}.csv").exists()
            assert (output / f"assignment_summary_{condition}.csv").exists()
        for name in (
            "fixed_sites.csv", "targets.csv", "geometry_scores.csv",
            "cotarget_scores.csv", "observed_assignments.csv",
            "condition_summary.csv", "assignment_samples.npz",
        ):
            assert (output / name).exists()
        saved = json.loads((output / "diagnostics.json").read_text())
        assert saved["cotargeting_n_perm_expectation"] == 17
        assert saved["n_contacts"] == 7


if __name__ == "__main__":
    run_smoke_test()
    print("approach2 smoke test passed")
