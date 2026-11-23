''' this is to try develop and examine an alternative geometry approach 
which rather than relying on the location of neuronal backbone, 
uses locations of postsynaptic sites on target neurons. 
The idea is that it might better treat scenarios where neurons only really form postsynaptic sites
in some locations across the neuron, and the locaiton thresholds for distance may not be as critical. 


also optional: keep compartments fixed?

Potential downside: too biased towards existing synapses? '''


from pathlib import Path
import os
import sys
import ast

import pandas as pd
from itertools import chain
from itertools import product
import numpy as np 
import matplotlib.pyplot as plt


# Ensure the repository is importable and relative data paths work.
REPO_ROOT = Path("/Users/nessn/git-repos/polyadic-assignment-model")
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)
from models.validate_predictions_functions import get_target_co_occurrence
from models.approach1.data import DataRepository
from models.approach2 import Approach2Config, run_attribution_experiment
from models.approach2.geometry import geometry_scores
from models.approach2.data import FixedNeuronInputs, load_observed_inputs

config = Approach2Config.load(
    REPO_ROOT / "models/approach2/example_config.json"
)


repository = DataRepository(config.paths)

celltype_df = pd.read_csv('data/celltype_df.csv')
celltype_df['skids'] = celltype_df['skids'].apply(ast.literal_eval)

skids = celltype_df[celltype_df.name == 'MBONs']['skids'].values[0]

#%% pick an example neuron & run it 5 times 

config.sampler.prediction_samples = 5 # run fewer samples while testing.

ex_sk = skids[4]
# get the real stuff 
inputs_sk = load_observed_inputs(repository, int(ex_sk))

#%% for geometry input - get connectors and nodes with their xyz pos 

nodes = pd.read_csv(config.paths.detailed_nodes)
con = pd.read_csv(config.paths.connectors)
con['postsynaptic_to_node'] = con['postsynaptic_to_node'].apply(ast.literal_eval)
con['postsynaptic_to'] = con['postsynaptic_to'].apply(ast.literal_eval)
cone = con.explode(['postsynaptic_to_node', 'postsynaptic_to'])

targets = inputs_sk.target_ids

tpost_nodes = nodes[nodes['skeleton_id'].isin(targets)]
rel_syn_nodes = cone[cone['postsynaptic_to'].isin(targets)]['postsynaptic_to_node'].tolist()
tpost_snodes = tpost_nodes[tpost_nodes['node_id'].isin(rel_syn_nodes)]


#%% local geometry functions using postsynaptic sites instead of backbones

COMPARTMENT_LETTERS = {"axon": "A", "dendrite": "D"}
COMPARTMENT_PAIRS = ("AA", "AD", "DA", "DD", "other")


def _nearest_with_index(points: np.ndarray, target: np.ndarray, chunk_size: int = 2048):
    """Return the distance to, and row index of, the nearest target point for each point."""
    if len(target) == 0:
        return np.full(len(points), np.inf), np.full(len(points), -1, dtype=int)
    try:
        from scipy.spatial import cKDTree  # type: ignore

        dist, idx = cKDTree(target).query(points, workers=-1)
        return dist, idx.astype(int)
    except ImportError:
        dist = np.full(len(points), np.inf, dtype=float)
        idx = np.full(len(points), -1, dtype=int)
        for start in range(0, len(points), chunk_size):
            block = points[start : start + chunk_size]
            squared = np.sum((block[:, None, :] - target[None, :, :]) ** 2, axis=2)
            best = np.argmin(squared, axis=1)
            dist[start : start + len(block)] = np.sqrt(squared[np.arange(len(block)), best])
            idx[start : start + len(block)] = best
        return dist, idx


def nearest_postsyn_sites(
    inputs: FixedNeuronInputs,
    postsyn_nodes: pd.DataFrame,
    coordinate_scale_to_um: float,
    compartment: str | None = None,
    exclude_node_ids: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance from each presynaptic site to the nearest postsynaptic site on each target.

    ``postsyn_nodes`` holds all postsynaptic nodes on the targets (from any presynaptic
    partner), with node_id, skeleton_id, compartment and x, y, z columns. If
    ``compartment`` is given ("axon" or "dendrite"), only postsynaptic sites in that
    compartment are considered. ``exclude_node_ids`` are dropped from the pool, e.g.
    the target-side nodes of this neuron's own synapses (see ``own_partner_node_ids``).
    A target with no remaining sites gets distance inf and nearest node id -1.

    Returns (distances_um, nearest_node_ids), both shaped (n_sites, n_targets).
    """
    n_sites = len(inputs.site_degrees)
    n_targets = len(inputs.target_ids)
    distances = np.full((n_sites, n_targets), np.inf, dtype=float)
    nearest_ids = np.full((n_sites, n_targets), -1, dtype=np.int64)

    site_points = inputs.sites[["x", "y", "z"]].to_numpy(float) * coordinate_scale_to_um
    pool = postsyn_nodes.drop_duplicates("node_id")
    if exclude_node_ids:
        pool = pool[~pool["node_id"].isin(exclude_node_ids)]
    if compartment is not None:
        pool = pool[pool["compartment"] == compartment]
    by_target = dict(tuple(pool.groupby("skeleton_id")))

    for target_index, target_id in enumerate(inputs.target_ids):
        target_nodes = by_target.get(int(target_id))
        if target_nodes is None or target_nodes.empty:
            continue
        target_points = target_nodes[["x", "y", "z"]].to_numpy(float) * coordinate_scale_to_um
        dist, idx = _nearest_with_index(site_points, target_points)
        distances[:, target_index] = dist
        nearest_ids[:, target_index] = target_nodes["node_id"].to_numpy()[idx]
    return distances, nearest_ids


def postsyn_geometry_by_compartment(
    inputs: FixedNeuronInputs,
    postsyn_nodes: pd.DataFrame,
    coordinate_scale_to_um: float,
    exclude_node_ids: set[int] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Nearest postsynaptic site distances for any, axonal and dendritic target sites."""
    return {
        key: nearest_postsyn_sites(
            inputs, postsyn_nodes, coordinate_scale_to_um, compartment, exclude_node_ids
        )
        for key, compartment in (("any", None), ("axon", "axon"), ("dendrite", "dendrite"))
    }


def own_partner_node_ids(inputs: FixedNeuronInputs, exploded_connectors: pd.DataFrame) -> set[int]:
    """Target-side postsynaptic nodes that are only used by this neuron's own connectors.

    A node that also receives a synapse from another presynaptic neuron is kept, since
    it is evidence of postsynaptic machinery there independent of this neuron.
    """
    own = exploded_connectors["connector_id"].isin(inputs.sites["connector_id"])
    own_nodes = set(exploded_connectors.loc[own, "postsynaptic_to_node"].astype(int))
    other_nodes = set(exploded_connectors.loc[~own, "postsynaptic_to_node"].dropna().astype(int))
    return own_nodes - other_nodes


def existing_compartment_pairs(
    inputs: FixedNeuronInputs,
    exploded_connectors: pd.DataFrame,
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    """Count each target's observed synapses by presynaptic-postsynaptic compartment pair.

    The first letter is the presynaptic compartment (on this neuron), the second the
    postsynaptic compartment (on the target): AA, AD, DA, DD. Anything not axon or
    dendrite, or a postsynaptic node missing from ``nodes``, counts as "other".
    """
    pre_compartment = dict(zip(inputs.sites["connector_id"], inputs.sites["compartment"]))
    syn = exploded_connectors[
        exploded_connectors["connector_id"].isin(pre_compartment)
    ][["connector_id", "postsynaptic_to", "postsynaptic_to_node"]].copy()
    syn = syn[syn["postsynaptic_to"].isin(inputs.target_ids)]

    post_compartment = nodes.drop_duplicates("node_id").set_index("node_id")["compartment"]
    pre = syn["connector_id"].map(pre_compartment).map(COMPARTMENT_LETTERS)
    post = syn["postsynaptic_to_node"].astype(int).map(post_compartment).map(COMPARTMENT_LETTERS)
    syn["pair"] = (pre + post).where(pre.notna() & post.notna(), "other")

    counts = pd.crosstab(syn["postsynaptic_to"].astype(int), syn["pair"])
    counts = counts.reindex(index=inputs.target_ids, columns=list(COMPARTMENT_PAIRS), fill_value=0)
    counts.index.name = "target_id"
    return counts


#%% test on the example neuron

scale = config.geometry.coordinate_scale_to_um
postsyn_geom = postsyn_geometry_by_compartment(inputs_sk, tpost_snodes, scale)
# note postsyn_geom returns dist which is (site, dist to nearest node on target) in order target_ids, 
# and ids, which returns (site, node_id of the nearest node on the target)

pair_counts = existing_compartment_pairs(inputs_sk, cone, nodes)

# sanity check: compartment pair counts should add up to the observed target totals
assert np.array_equal(pair_counts.sum(axis=1).to_numpy(), inputs_sk.target_counts)

for key, (dist, ids) in postsyn_geom.items():
    n_inf_targets = int(np.all(np.isinf(dist), axis=0).sum())
    finite = dist[np.isfinite(dist)]
    print(f"{key}: median nearest distance {np.median(finite):.2f} um, "
          f"{n_inf_targets}/{dist.shape[1]} targets without such sites")
print(pair_counts.sum())

# distance from each site to the nearest site on the targets it actually contacts
observed = inputs_sk.observed_assignment > 0
d_any = postsyn_geom["any"][0]
print(f"observed partners: median {np.median(d_any[observed]):.2f} um | "
      f"non-partners: median {np.median(d_any[~observed]):.2f} um")

# same again, without the target-side nodes of this neuron's own synapses
own_nodes = own_partner_node_ids(inputs_sk, cone)
postsyn_geom_excl = postsyn_geometry_by_compartment(inputs_sk, tpost_snodes, scale, own_nodes)
print(f"excluding {len(own_nodes)} own partner nodes:")
for key, (dist, ids) in postsyn_geom_excl.items():
    n_inf_targets = int(np.all(np.isinf(dist), axis=0).sum())
    print(f"  {key}: observed partners median {np.median(dist[observed]):.2f} um | "
          f"non-partners median {np.median(dist[~observed]):.2f} um | "
          f"{n_inf_targets}/{dist.shape[1]} targets without such sites")


#%% per-target overview: observed sites vs distance distributions

DIST_THRESHOLDS_UM = (5, 10, 15)
COMBOS = COMPARTMENT_PAIRS[:4]
POST_KEY = {"A": "axon", "D": "dendrite"}

COLOR_OTHER = "#a3a29b"      # sites not observed to target this target
COLOR_OBSERVED = "#2a78d6"   # sites observed to target this target
COLOR_CRITICAL = "#d03b3b"   # combo not possible / reach below observed

geom_plot = postsyn_geom_excl  # swap for postsyn_geom to include own partner nodes


def combo_distances(inputs: FixedNeuronInputs, geom: dict) -> dict[str, np.ndarray]:
    """Site x target distances per compartment combo (pre letter, post letter).

    Sites whose own compartment does not match the combo's first letter are nan;
    targets without postsynaptic sites in the combo's second compartment are inf.
    """
    pre = inputs.sites["compartment"].map(COMPARTMENT_LETTERS).to_numpy()
    out = {}
    for combo in COMBOS:
        dist = geom[POST_KEY[combo[1]]][0].copy()
        dist[pre != combo[0], :] = np.nan
        out[combo] = dist
    return out


def observed_site_combos(
    inputs: FixedNeuronInputs, exploded_connectors: pd.DataFrame, nodes: pd.DataFrame
) -> dict[str, np.ndarray]:
    """Boolean site x target matrices: was site i observed to target j with this combo."""
    site_index = dict(zip(inputs.sites["connector_id"], inputs.sites["site_id"]))
    target_index = {int(t): j for j, t in enumerate(inputs.target_ids)}
    pre = inputs.sites["compartment"].map(COMPARTMENT_LETTERS).to_numpy()
    post_compartment = nodes.drop_duplicates("node_id").set_index("node_id")["compartment"]

    syn = exploded_connectors[exploded_connectors["connector_id"].isin(site_index)]
    out = {combo: np.zeros(inputs.observed_assignment.shape, dtype=bool) for combo in COMBOS}
    for connector_id, target, post_node in syn[
        ["connector_id", "postsynaptic_to", "postsynaptic_to_node"]
    ].itertuples(index=False):
        i, j = site_index[connector_id], target_index.get(int(target))
        post = COMPARTMENT_LETTERS.get(post_compartment.get(int(post_node)))
        if j is None or post is None or not isinstance(pre[i], str):
            continue
        out[pre[i] + post][i, j] = True
    return out


def plot_target_distance_strips(
    ax, dist: np.ndarray, observed: np.ndarray, order: np.ndarray, labels: list[str],
    title: str, min_dist_um: float = 0.05,
):
    """One row per target: every site's distance (grey) with observed sites on top (blue).

    Rows where no site can reach the target (all nan/inf) are marked 'not possible'.
    The observed site count for the row is written at the right edge.
    """
    rng = np.random.default_rng(0)
    for row, j in enumerate(order):
        col, obs = dist[:, j], observed[:, j]
        usable = np.isfinite(col)
        if not usable.any():
            ax.axhspan(row - 0.4, row + 0.4, color=COLOR_CRITICAL, alpha=0.12, lw=0)
            ax.text(0.5, row, "not possible", transform=ax.get_yaxis_transform(),
                    ha="center", va="center", fontsize=7, color=COLOR_CRITICAL)
        else:
            x = np.clip(col, min_dist_um, None)
            jitter = rng.uniform(-0.25, 0.25, len(col))
            ax.scatter(x[usable & ~obs], row + jitter[usable & ~obs], s=6,
                       color=COLOR_OTHER, alpha=0.6, lw=0)
            ax.scatter(x[usable & obs], row + jitter[usable & obs], s=14,
                       color=COLOR_OBSERVED, edgecolor="white", lw=0.5, zorder=3)
        ax.text(1.01, row, str(int(obs.sum())), transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=7,
                color=COLOR_CRITICAL if obs.any() and not usable[obs].all() else "0.2")
    for t in DIST_THRESHOLDS_UM:
        ax.axvline(t, color="0.75", lw=0.8, ls="--", zorder=0)
    ax.set_xscale("log")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlabel("distance to nearest postsynaptic site (µm)")
    ax.set_title(title, fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def reach_counts(
    dist: np.ndarray, target_ids: np.ndarray, thresholds=DIST_THRESHOLDS_UM
) -> pd.DataFrame:
    """Number of sites within each distance threshold of each target."""
    return pd.DataFrame(
        {f"<={t}um": (dist <= t).sum(axis=0) for t in thresholds},
        index=pd.Index(target_ids, name="target_id"),
    )


dist_any = geom_plot["any"][0]
dist_combo = combo_distances(inputs_sk, geom_plot)
obs_any = inputs_sk.observed_assignment > 0
obs_combo = observed_site_combos(inputs_sk, cone, nodes)

n_obs_sites = obs_any.sum(axis=0)
order = np.lexsort((inputs_sk.target_ids, -n_obs_sites))  # most-targeted first
row_labels = [str(inputs_sk.target_ids[j]) for j in order]

# 1) ignoring compartments
fig, ax = plt.subplots(figsize=(7, 0.18 * len(order) + 1.5))
plot_target_distance_strips(ax, dist_any, obs_any, order, row_labels,
                            f"{ex_sk}: all sites → nearest postsynaptic site per target")
ax.text(1.01, 1.0, "obs.\nsites", transform=ax.transAxes, fontsize=7, va="bottom")
fig.tight_layout()

# 2) by compartment combo (pre-post)
fig, axes = plt.subplots(1, 4, figsize=(18, 0.18 * len(order) + 1.5), sharey=True)
for ax, combo in zip(axes, COMBOS):
    plot_target_distance_strips(ax, dist_combo[combo], obs_combo[combo], order, row_labels,
                                f"{combo} ({combo[0]} site → {POST_KEY[combo[1]]} postsyn)")
    ax.text(1.01, 1.0, "obs.\nsites", transform=ax.transAxes, fontsize=7, va="bottom")
fig.suptitle(f"{ex_sk}: distance distributions by compartment combo", fontsize=11)
fig.tight_layout()

# 3) reach counts per threshold, without and with the compartment requirement
# with requirement: a site counts if it reaches the target through any combo
# that the target is observed with for this neuron
observed_combos = {combo: obs_combo[combo].any(axis=0) for combo in COMBOS}
dist_compat = np.full_like(dist_any, np.inf)
for combo in COMBOS:
    allowed = np.where(observed_combos[combo][None, :], dist_combo[combo], np.inf)
    dist_compat = np.fmin(dist_compat, np.nan_to_num(allowed, nan=np.inf))

reach_any = reach_counts(dist_any, inputs_sk.target_ids)
reach_compat = reach_counts(dist_compat, inputs_sk.target_ids)
reach_by_combo = pd.concat(
    {combo: reach_counts(dist_combo[combo], inputs_sk.target_ids) for combo in COMBOS}, axis=1
)
reach_summary = pd.concat(
    {"observed": pd.DataFrame({"n_sites": n_obs_sites, "n_synapses": inputs_sk.target_counts},
                              index=reach_any.index),
     "any compartment": reach_any, "observed combos": reach_compat},
    axis=1,
).iloc[order]

with pd.option_context("display.max_rows", None, "display.width", 200):
    print(reach_summary)
    print(reach_by_combo.iloc[order])
for name, reach in (("any compartment", reach_any), ("observed combos", reach_compat)):
    short = (reach.to_numpy() < n_obs_sites[:, None]).sum(axis=0)
    print(f"{name}: targets with fewer reachable than observed sites "
          + ", ".join(f"{c} {n}" for c, n in zip(reach.columns, short)))

heat = reach_summary.drop(columns=("observed", "n_synapses"))
fig, ax = plt.subplots(figsize=(6, 0.18 * len(order) + 1.8))
values = heat.to_numpy(dtype=float)
ax.imshow(values, aspect="auto", cmap="Blues", vmin=0, vmax=values.max())
need = heat[("observed", "n_sites")].to_numpy()
for r in range(values.shape[0]):
    for c in range(values.shape[1]):
        short = c > 0 and values[r, c] < need[r]
        ax.text(c, r, int(values[r, c]), ha="center", va="center", fontsize=6,
                color=COLOR_CRITICAL if short else ("white" if values[r, c] > 0.6 * values.max() else "0.15"),
                fontweight="bold" if short else "normal")
ax.set_xticks(range(values.shape[1]))
ax.set_xticklabels(["observed\nsites"] + [f"any\n{c[1]}" for c in heat.columns[1:4]]
                   + [f"obs. combos\n{c[1]}" for c in heat.columns[4:]], fontsize=7)
ax.xaxis.tick_top()
ax.set_yticks(range(len(order)))
ax.set_yticklabels(row_labels, fontsize=6)
ax.axvline(0.5, color="white", lw=3)
ax.axvline(3.5, color="white", lw=3)
ax.set_title(f"{ex_sk}: sites within reach (red = fewer than observed)", fontsize=10, pad=30)
fig.tight_layout()
plt.show()


# %%
