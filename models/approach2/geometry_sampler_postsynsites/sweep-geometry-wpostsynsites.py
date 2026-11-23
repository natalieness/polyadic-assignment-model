''' sweep the gibbs sampler with postsynaptic-site geometry over example MBONs.

Geometry scores come from the distance of each presynaptic site to the nearest
postsynaptic site on each target (excluding the target-side nodes of the neuron's
own synapses), turned into scores the same way as geometry.py: binary-and-soft
kernel at a distance threshold, column centred, standardised and clipped.

Sweep over:
-- distance threshold (um)
-- wrong compartment penalty: subtracted (in score units) where a site is only in
   reach of a target through a compartment combo the target does not use for its
   other inputs. 0 = no compartment constraint.
'''

from pathlib import Path
import os
import sys
import ast

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Ensure the repository is importable and relative data paths work.
REPO_ROOT = Path("/Users/nessn/git-repos/polyadic-assignment-model")
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)
from models.approach1.data import DataRepository
from models.approach2 import Approach2Config
from models.approach2.data import FixedNeuronInputs, load_observed_inputs
from models.approach2.sampler import gibbs_target_swaps
from models.approach2.simulator import _rng

config = Approach2Config.load(
    REPO_ROOT / "models/approach2/example_config.json"
)
# sampler settings as the defaults in sweep_geometry_sampler.py
config.sampler.temperature = 0.1
config.sampler.geometry_weight = 2.0
config.sampler.mcmc_steps_per_contact = 20

repository = DataRepository(config.paths)

celltype_df = pd.read_csv('data/celltype_df.csv')
celltype_df['skids'] = celltype_df['skids'].apply(ast.literal_eval)
skids = celltype_df[celltype_df.name == 'MBONs']['skids'].values[0]

SWEEP_SKIDS = [int(sk) for sk in skids[:15]]
N_SAMPLES = 10
DIST_THRESHOLDS_UM = (2, 5, 10)
COMPARTMENT_PENALTIES = (0.0, 2.0, 10.0, 20.0, 40.0)  # 0 = no compartment constraint
MIN_COMBO_FRAC = 0.05  # combo counts as allowed for a target above this input fraction

COMPARTMENT_LETTERS = {"axon": "A", "dendrite": "D"}
COMBOS = ("AA", "AD", "DA", "DD")
POST_KEY = {"A": "axon", "D": "dendrite"}

#%% load nodes and connectors once

nodes = pd.read_csv(config.paths.detailed_nodes)
con = pd.read_csv(config.paths.connectors)
con['postsynaptic_to_node'] = con['postsynaptic_to_node'].apply(ast.literal_eval)
con['postsynaptic_to'] = con['postsynaptic_to'].apply(ast.literal_eval)
cone = con.explode(['postsynaptic_to_node', 'postsynaptic_to']).dropna(
    subset=['postsynaptic_to_node', 'postsynaptic_to']
)
cone['postsynaptic_to_node'] = cone['postsynaptic_to_node'].astype('int64')
cone['postsynaptic_to'] = cone['postsynaptic_to'].astype('int64')

node_compartment = nodes.drop_duplicates('node_id').set_index('node_id')['compartment']
# every node that is postsynaptic in some connector
postsyn_nodes_all = nodes[nodes['node_id'].isin(cone['postsynaptic_to_node'])]


#%% functions

def _nearest_distances(points: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(target) == 0:
        return np.full(len(points), np.inf)
    from scipy.spatial import cKDTree  # type: ignore

    return cKDTree(target).query(points, workers=-1)[0]


def nearest_postsyn_distances(
    inputs: FixedNeuronInputs,
    postsyn_nodes: pd.DataFrame,
    coordinate_scale_to_um: float,
    compartment: str | None = None,
    exclude_node_ids: set[int] | None = None,
) -> np.ndarray:
    """(n_sites, n_targets) distance to the nearest postsynaptic site on each target.

    Targets without (remaining) postsynaptic sites in the pool get inf.
    """
    distances = np.full(inputs.observed_assignment.shape, np.inf, dtype=float)
    site_points = inputs.sites[["x", "y", "z"]].to_numpy(float) * coordinate_scale_to_um
    pool = postsyn_nodes.drop_duplicates("node_id")
    if exclude_node_ids:
        pool = pool[~pool["node_id"].isin(exclude_node_ids)]
    if compartment is not None:
        pool = pool[pool["compartment"] == compartment]
    by_target = dict(tuple(pool.groupby("skeleton_id")))
    for target_index, target_id in enumerate(inputs.target_ids):
        target_nodes = by_target.get(int(target_id))
        if target_nodes is None:
            continue
        target_points = target_nodes[["x", "y", "z"]].to_numpy(float) * coordinate_scale_to_um
        distances[:, target_index] = _nearest_distances(site_points, target_points)
    return distances


def own_partner_node_ids(inputs: FixedNeuronInputs, exploded_connectors: pd.DataFrame) -> set[int]:
    """Target-side postsynaptic nodes that are only used by this neuron's own connectors."""
    own = exploded_connectors["connector_id"].isin(inputs.sites["connector_id"])
    own_nodes = set(exploded_connectors.loc[own, "postsynaptic_to_node"])
    other_nodes = set(exploded_connectors.loc[~own, "postsynaptic_to_node"])
    return own_nodes - other_nodes


def population_allowed_combos(
    inputs: FixedNeuronInputs,
    exploded_connectors: pd.DataFrame,
    min_frac: float = MIN_COMBO_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Which compartment combos each target uses for inputs from *other* neurons.

    Returns (allowed, fraction), both targets x combos. A target without other
    inputs allows every combo.
    """
    syn = exploded_connectors[
        exploded_connectors["postsynaptic_to"].isin(inputs.target_ids)
        & ~exploded_connectors["connector_id"].isin(inputs.sites["connector_id"])
    ]
    pre = syn["presynaptic_to_node"].map(node_compartment).map(COMPARTMENT_LETTERS)
    post = syn["postsynaptic_to_node"].map(node_compartment).map(COMPARTMENT_LETTERS)
    combo = pre + post
    frac = pd.crosstab(syn["postsynaptic_to"], combo, normalize="index")
    frac = frac.reindex(index=inputs.target_ids, columns=list(COMBOS), fill_value=0.0)
    allowed = frac >= min_frac
    allowed[~allowed.any(axis=1)] = True
    return allowed, frac


def prepare_neuron(skeleton_id: int) -> dict:
    """Distances (with and without own partner nodes), site compartments, allowed combos."""
    inputs = load_observed_inputs(repository, skeleton_id)
    scale = config.geometry.coordinate_scale_to_um
    postsyn_nodes = postsyn_nodes_all[postsyn_nodes_all["skeleton_id"].isin(inputs.target_ids)]
    own_nodes = own_partner_node_ids(inputs, cone)
    allowed, frac = population_allowed_combos(inputs, cone)
    return {
        "inputs": inputs,
        "dist_full": nearest_postsyn_distances(inputs, postsyn_nodes, scale),
        "dist_excl": {
            key: nearest_postsyn_distances(inputs, postsyn_nodes, scale, compartment, own_nodes)
            for key, compartment in (("any", None), ("axon", "axon"), ("dendrite", "dendrite"))
        },
        "pre": inputs.sites["compartment"].map(COMPARTMENT_LETTERS).to_numpy(),
        "allowed": allowed,
        "combo_frac": frac,
    }


def wrong_compartment_mask(prep: dict, threshold_um: float) -> np.ndarray:
    """Site/target pairs in reach only through combos the target does not allow."""
    dist = prep["dist_excl"]
    in_reach_allowed = np.zeros(prep["inputs"].observed_assignment.shape, dtype=bool)
    for combo in COMBOS:
        site_ok = (prep["pre"] == combo[0])[:, None]
        target_ok = prep["allowed"][combo].to_numpy()[None, :]
        in_reach_allowed |= site_ok & target_ok & (dist[POST_KEY[combo[1]]] <= threshold_um)
    return (dist["any"] <= threshold_um) & ~in_reach_allowed


def postsyn_geometry_scores(
    dist: np.ndarray,
    threshold_um: float,
    wrong_compartment: np.ndarray | None = None,
    penalty: float = 0.0,
) -> np.ndarray:
    """Binary-and-soft kernel scores as in geometry.py, minus the compartment penalty."""
    geo = config.geometry
    with np.errstate(invalid="ignore"):
        kernel = (dist <= threshold_um) * np.exp(-0.5 * (dist / threshold_um) ** 2)
    kernel = np.nan_to_num(kernel)
    available = np.isfinite(dist).any(axis=0)
    scores = kernel - kernel.mean(axis=0, keepdims=True)
    raw_sd = float(np.std(scores[:, available])) if available.any() else 0.0
    if raw_sd <= geo.uninformative_threshold:
        scores.fill(0.0)
    elif geo.standardize_scores:
        scores /= raw_sd
    if geo.score_clip is not None:
        scores = np.clip(scores, -geo.score_clip, geo.score_clip)
    scores[:, ~available] = 0.0
    if penalty and wrong_compartment is not None:
        scores = scores - penalty * wrong_compartment
    return scores


def run_chain(
    start: np.ndarray, inputs: FixedNeuronInputs, scores: np.ndarray,
    rng: np.random.Generator, geometry_weight: float, steps_per_contact: int,
) -> np.ndarray:
    """One geometry-only gibbs chain from ``start`` (geometry_weight 0 gives the null)."""
    sampler = config.sampler
    return gibbs_target_swaps(
        start,
        inputs.site_degrees,
        inputs.target_counts,
        scores,
        np.zeros((len(inputs.target_ids),) * 2),
        rng,
        geometry_weight=geometry_weight,
        cotargeting_weight=0.0,
        temperature=sampler.temperature,
        geometry_repeat_discount=config.geometry.repeat_discount,
        cotarget_aggregation=config.cotarget.aggregation,
        cotarget_site_clip=config.cotarget.site_score_clip,
        self_repeat_penalty=0.0,
        preserve_multiplicity_profile=sampler.preserve_multiplicity_profile,
        max_same_target_per_site=sampler.max_same_target_per_site,
        steps=steps_per_contact * int(inputs.site_degrees.sum()),
    )


def contact_distances(assignment: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Distance per contact, counting a site that hits a target twice twice."""
    hit = assignment > 0
    return np.repeat(dist[hit], assignment[hit])


def assignment_metrics(
    assignment: np.ndarray, observed: np.ndarray, dist: np.ndarray,
    threshold_um: float, wrong: np.ndarray,
) -> dict:
    n_contacts = assignment.sum()
    d = contact_distances(assignment, dist)
    return {
        "median_dist_um": float(np.median(d)),
        "frac_within_threshold": float(np.mean(d <= threshold_um)),
        "overlap_with_observed": float(np.minimum(assignment, observed).sum() / n_contacts),
        "frac_wrong_compartment": float((assignment * wrong).sum() / n_contacts),
    }


#%% run the sweep

records = []
contact_dists = {}  # (skid, label) -> contact distances, label is a reference or (thr, penalty)
for sk in SWEEP_SKIDS:
    prep = prepare_neuron(sk)
    inputs = prep["inputs"]
    observed = inputs.observed_assignment
    dist = prep["dist_excl"]["any"]
    print(f"{sk}: {len(inputs.site_degrees)} sites, {observed.sum()} contacts, "
          f"{len(inputs.target_ids)} targets")

    zero_scores = np.zeros(observed.shape)
    nulls = [
        run_chain(observed, inputs, zero_scores, _rng(config, sk, s, 10), 0.0,
                  config.sampler.null_burnin_steps_per_contact)
        for s in range(N_SAMPLES)
    ]
    contact_dists[(sk, "observed (own syn. included)")] = contact_distances(observed, prep["dist_full"])
    contact_dists[(sk, "observed")] = contact_distances(observed, dist)
    contact_dists[(sk, "null")] = np.concatenate([contact_distances(n, dist) for n in nulls])

    for thr in DIST_THRESHOLDS_UM:
        wrong = wrong_compartment_mask(prep, thr)
        for label, assignments in (("observed", [observed]), ("null", nulls)):
            metrics = pd.DataFrame([assignment_metrics(a, observed, dist, thr, wrong) for a in assignments])
            records.append({"skid": sk, "threshold_um": thr, "penalty": np.nan, "condition": label,
                            **metrics.mean().to_dict()})
        for penalty in COMPARTMENT_PENALTIES:
            scores = postsyn_geometry_scores(dist, thr, wrong, penalty)
            samples = [
                run_chain(nulls[s], inputs, scores, _rng(config, sk, s, 20),
                          config.sampler.geometry_weight, config.sampler.mcmc_steps_per_contact)
                for s in range(N_SAMPLES)
            ]
            contact_dists[(sk, (thr, penalty))] = np.concatenate(
                [contact_distances(a, dist) for a in samples]
            )
            metrics = pd.DataFrame([assignment_metrics(a, observed, dist, thr, wrong) for a in samples])
            records.append({"skid": sk, "threshold_um": thr, "penalty": penalty, "condition": "geometry",
                            **metrics.mean().to_dict()})
            print(f"  thr {thr}um, penalty {penalty}: "
                  + ", ".join(f"{k} {v:.2f}" for k, v in metrics.mean().items()))

results_df = pd.DataFrame(records)
with pd.option_context("display.max_rows", None, "display.width", 200):
    print(results_df.groupby(["threshold_um", "condition", "penalty"], dropna=False)
          .mean(numeric_only=True).drop(columns="skid").round(3))


#%% plot: contact distance distributions (ECDF) per neuron and threshold

COLOR_OBS_FULL = "#52514e"
COLOR_OBS = "#0b0b0b"
COLOR_NULL = "#a3a29b"
PENALTY_COLORS = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")  # sequential blue, low -> high penalty
MIN_PLOT_UM, MAX_PLOT_UM = 0.05, 200.0


def plot_ecdf(ax, values: np.ndarray, **kwargs):
    x = np.clip(np.sort(values), MIN_PLOT_UM, MAX_PLOT_UM)  # inf (no site left) ends up at the right edge
    ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", **kwargs)


fig, axes = plt.subplots(len(SWEEP_SKIDS), len(DIST_THRESHOLDS_UM),
                         figsize=(4 * len(DIST_THRESHOLDS_UM), 2.6 * len(SWEEP_SKIDS)),
                         sharex=True, sharey=True)
for r, sk in enumerate(SWEEP_SKIDS):
    for c, thr in enumerate(DIST_THRESHOLDS_UM):
        ax = axes[r, c]
        plot_ecdf(ax, contact_dists[(sk, "observed (own syn. included)")], color=COLOR_OBS_FULL,
                  lw=1.5, ls=":", label="observed (own synapse included)")
        plot_ecdf(ax, contact_dists[(sk, "observed")], color=COLOR_OBS, lw=2,
                  label="observed (own synapse excluded)")
        plot_ecdf(ax, contact_dists[(sk, "null")], color=COLOR_NULL, lw=2, label="null")
        for penalty, color in zip(COMPARTMENT_PENALTIES, PENALTY_COLORS):
            plot_ecdf(ax, contact_dists[(sk, (thr, penalty))], color=color, lw=1.5,
                      label="sampled, no compartment constraint" if penalty == 0
                      else f"sampled, wrong compartment penalty {penalty:g}")
        ax.axvline(thr, color="0.75", lw=0.8, ls="--", zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(MIN_PLOT_UM, MAX_PLOT_UM)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if r == 0:
            ax.set_title(f"threshold {thr} µm", fontsize=10)
        if c == 0:
            ax.set_ylabel(f"{sk}\nfrac. contacts ≤ x", fontsize=8)
        if r == len(SWEEP_SKIDS) - 1:
            ax.set_xlabel("site → nearest postsyn. site on target (µm)", fontsize=8)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0.05, 1, 1))


#%% plot: summary metrics vs penalty, one line per threshold (mean over neurons)

THRESHOLD_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots 1-3
metric_labels = {
    "median_dist_um": "median contact distance (µm)",
    "frac_within_threshold": "frac. contacts within threshold",
    "overlap_with_observed": "overlap with observed",
    "frac_wrong_compartment": "frac. contacts wrong compartment",
}
summary = results_df.groupby(["threshold_um", "condition", "penalty"], dropna=False).mean(numeric_only=True)
x = np.arange(len(COMPARTMENT_PENALTIES))
fig, axes = plt.subplots(1, len(metric_labels), figsize=(4 * len(metric_labels), 3.2))
for ax, (metric, label) in zip(axes, metric_labels.items()):
    for thr, color in zip(DIST_THRESHOLDS_UM, THRESHOLD_COLORS):
        geo = summary.loc[(thr, "geometry")][metric].reindex(COMPARTMENT_PENALTIES)
        ax.plot(x, geo.to_numpy(), color=color, lw=2, marker="o", ms=5, label=f"{thr} µm")
        ax.axhline(summary.loc[(thr, "observed")][metric].iloc[0], color=color, lw=1, ls="--")
        ax.axhline(summary.loc[(thr, "null")][metric].iloc[0], color=color, lw=1, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:g}" for p in COMPARTMENT_PENALTIES])
    ax.set_xlabel("wrong compartment penalty (0 = none)")
    ax.set_title(label, fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].legend(title="threshold", fontsize=8, title_fontsize=8, frameon=False)
fig.text(0.5, -0.02, "dashed = observed (own synapse excluded), dotted = null; mean over "
         f"{len(SWEEP_SKIDS)} MBONs", ha="center", fontsize=8)
fig.tight_layout()
plt.show()

# %%
