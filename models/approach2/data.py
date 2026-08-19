"""Fixed observed inputs, using the validated Approach 1 data repository."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from models.approach1.data import DataRepository


@dataclass
class FixedNeuronInputs:
    skeleton_id: int
    sites: pd.DataFrame
    site_degrees: np.ndarray
    target_ids: np.ndarray
    target_counts: np.ndarray
    observed_assignment: np.ndarray

    def validate(self) -> None:
        n_sites = len(self.site_degrees)
        n_targets = len(self.target_ids)
        if len(self.sites) != n_sites:
            raise ValueError("Site table and degree vector differ in length")
        if self.observed_assignment.shape != (n_sites, n_targets):
            raise ValueError("Observed assignment has the wrong shape")
        if np.any(self.site_degrees < 1) or np.any(self.target_counts < 1):
            raise ValueError("Every fixed site degree and target count must be positive")
        if not np.array_equal(self.observed_assignment.sum(axis=1), self.site_degrees):
            raise ValueError("Observed rows do not match the fixed degrees")
        if not np.array_equal(self.observed_assignment.sum(axis=0), self.target_counts):
            raise ValueError("Observed columns do not match the target totals")


def load_observed_inputs(repository: DataRepository, skeleton_id: int) -> FixedNeuronInputs:
    """Read exact connector sites, degrees, identities, and totals for one neuron."""
    neuron = repository.connectors.neuron(int(skeleton_id))
    rows = neuron.rows.copy()
    site_node_ids = rows["presynaptic_to_node"].astype(int).to_numpy()
    degrees = rows["postsynaptic_to"].map(len).to_numpy(dtype=int)
    target_ids = np.asarray(sorted(neuron.target_counts), dtype=int)
    target_to_column = {int(target): index for index, target in enumerate(target_ids)}
    assignment = np.zeros((len(rows), len(target_ids)), dtype=int)
    for site_index, targets in enumerate(rows["postsynaptic_to"]):
        for target, count in Counter(int(value) for value in targets).items():
            assignment[site_index, target_to_column[target]] = int(count)
    target_counts = assignment.sum(axis=0)

    morphology = repository.morphologies.load(int(skeleton_id)).nodes.set_index("node_id")
    missing = sorted(set(site_node_ids) - set(morphology.index.astype(int)))
    if missing:
        raise ValueError(f"Fixed presynaptic nodes are absent from the morphology: {missing[:20]}")
    site_geometry = morphology.loc[site_node_ids].reset_index()
    sites = pd.DataFrame({
        "site_id": np.arange(len(rows), dtype=int),
        "connector_id": rows["connector_id"].astype(int).to_numpy(),
        "presynaptic_node_id": site_node_ids,
        "degree": degrees,
        "compartment": site_geometry["compartment"].astype(str).to_numpy(),
        "x": site_geometry["x"].astype(float).to_numpy(),
        "y": site_geometry["y"].astype(float).to_numpy(),
        "z": site_geometry["z"].astype(float).to_numpy(),
    })
    result = FixedNeuronInputs(
        skeleton_id=int(skeleton_id),
        sites=sites,
        site_degrees=degrees,
        target_ids=target_ids,
        target_counts=target_counts,
        observed_assignment=assignment,
    )
    result.validate()
    return result
