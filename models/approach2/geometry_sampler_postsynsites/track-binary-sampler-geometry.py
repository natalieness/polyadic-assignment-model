''' binary (hard constraint) swap sampler with tracked contacts.

Instead of the probabilistic gibbs sampler: every observed contact gets an id and
keeps its target. Random swap proposals pick two contacts (different targets,
different sites) and exchange their sites. A swap is accepted, always, only if both
contacts are allowed at their new site; otherwise it is rejected. Margins (site
degrees, target totals) are preserved by construction, and the multiplicity profile
of each site as well if the sweep config preserves it.

Allowed = the target has a postsynaptic site (own partner nodes excluded) within
the distance threshold of the site. With the compartment maintenance constraint the
contact must also keep its original combo: the new site has the original presynaptic
compartment and the postsynaptic site within reach is in the original synapse's
postsynaptic compartment.

Chains start from the observed assignment and run the same number of proposals as
the gibbs geometry chain (mcmc_steps_per_contact x contacts). An unconstrained run
(every swap allowed) with the same number of proposals is the reference.

Uses the neuron data, distances and helpers of sweep-geometry-wpostsynsites.py by
running that script's setup section (everything before its "run the sweep" cell).
'''

from pathlib import Path
import ast
from itertools import chain

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.spatial.distance import cdist

REPO_ROOT = Path("/Users/nessn/git-repos/polyadic-assignment-model")
SWEEP_SCRIPT = REPO_ROOT / "models/approach2/geometry_sampler_postsynsites/sweep-geometry-wpostsynsites.py"
sweep = {"__name__": "sweep_setup", "__file__": str(SWEEP_SCRIPT)}
exec(compile(SWEEP_SCRIPT.read_text().split("#%% run the sweep")[0], str(SWEEP_SCRIPT), "exec"), sweep)
config = sweep["config"]

N_SAMPLES = sweep["N_SAMPLES"]
STEPS_PER_CONTACT = config.sampler.mcmc_steps_per_contact  # same proposals as the gibbs geometry chain
DIST_THRESHOLDS_UM = (2, 5, 10, 40)
COMPARTMENT_CONSTRAINT = (False, True)
PRESERVE_MULTIPLICITY_PROFILE = config.sampler.preserve_multiplicity_profile

celltype_df = pd.read_csv('data/celltype_df.csv')
celltype_df['skids'] = celltype_df['skids'].apply(ast.literal_eval)
n_per_ct = 4
SWEEP_SKIDS = [int(sk) for sk in chain.from_iterable(
    celltype_df.loc[i, 'skids'][:n_per_ct] for i in range(len(celltype_df))
)]
SAVE_DIR = None  # e.g. Path("model_out/track_binary") to save tables and figures

COMPARTMENT_LETTERS = sweep["COMPARTMENT_LETTERS"]
POST_KEY = sweep["POST_KEY"]


#%% functions

def initial_contacts(observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Site and target of every observed contact, one entry (id) per contact."""
    sites, targets = np.nonzero(observed)
    counts = observed[sites, targets]
    return np.repeat(sites, counts), np.repeat(targets, counts)


def original_post_compartments(inputs, contact_site: np.ndarray, contact_target: np.ndarray) -> np.ndarray:
    """Post compartment letter of each contact's real synapse.

    Copies of one target at one site are interchangeable, so the letters of that
    cell's synapses are shared out over its contacts in order.
    """
    site_index = dict(zip(inputs.sites["connector_id"], inputs.sites["site_id"]))
    target_index = {int(t): j for j, t in enumerate(inputs.target_ids)}
    cone = sweep["cone"]
    syn = cone[cone["connector_id"].isin(site_index)]
    cells: dict[tuple[int, int], list[str]] = {}
    for connector_id, target, post_node in syn[
        ["connector_id", "postsynaptic_to", "postsynaptic_to_node"]
    ].itertuples(index=False):
        j = target_index.get(int(target))
        if j is None:
            continue
        compartment = sweep["node_compartment"].get(int(post_node))
        cells.setdefault((site_index[connector_id], j), []).append(COMPARTMENT_LETTERS.get(compartment, "?"))
    return np.array([cells[(i, j)].pop() for i, j in zip(contact_site, contact_target)])


def nearest_post_compartment(prep: dict, sites: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Compartment letter of the nearest postsynaptic site on the target from each site."""
    d_axon = prep["dist_excl"]["axon"][sites, targets]
    d_dendrite = prep["dist_excl"]["dendrite"][sites, targets]
    letters = np.where(d_axon < d_dendrite, "A", "D")
    return np.where(np.isinf(d_axon) & np.isinf(d_dendrite), "?", letters)


def allowed_positions(
    prep: dict, threshold_um: float, keep_compartment: bool,
    contact_target: np.ndarray, required_pre: np.ndarray, required_post: np.ndarray,
) -> np.ndarray:
    """(n_contacts, n_sites) bool: may contact c sit at site s."""
    dist = prep["dist_excl"]
    if not keep_compartment:
        return (dist["any"][:, contact_target] <= threshold_um).T
    pre = np.array([p if isinstance(p, str) else "?" for p in prep["pre"]])
    allowed = np.zeros((len(contact_target), len(pre)), dtype=bool)
    for c, (target, req_pre, req_post) in enumerate(zip(contact_target, required_pre, required_post)):
        post_dist = dist[POST_KEY[req_post]] if req_post in POST_KEY else dist["any"]  # unknown post: any
        allowed[c] = (pre == req_pre) & (post_dist[:, target] <= threshold_um)
    return allowed


def binary_swaps(
    contact_site: np.ndarray, contact_target: np.ndarray, allowed: np.ndarray,
    n_sites: int, n_targets: int, steps: int, rng: np.random.Generator,
    preserve_multiplicity_profile: bool,
) -> tuple[np.ndarray, int]:
    """Propose random contact pairs; exchange their sites whenever both new positions are allowed.

    Returns the final site of every contact and the number of accepted swaps.
    """
    site = contact_site.tolist()
    target = contact_target.tolist()
    allowed_rows = allowed.tolist()
    counts = np.zeros((n_sites, n_targets), dtype=int)
    np.add.at(counts, (contact_site, contact_target), 1)
    counts = counts.tolist()
    accepted = 0
    for a, b in rng.integers(0, len(site), size=(int(steps), 2)).tolist():
        i, j = site[a], site[b]
        t_a, t_b = target[a], target[b]
        if i == j or t_a == t_b:
            continue
        if not (allowed_rows[a][j] and allowed_rows[b][i]):
            continue
        # site i loses one t_a and gains one t_b (site j the reverse): the multiset of
        # non-zero counts only stays the same if the two counts trade places
        if preserve_multiplicity_profile and not (
            counts[i][t_a] == counts[i][t_b] + 1 and counts[j][t_b] == counts[j][t_a] + 1
        ):
            continue
        site[a], site[b] = j, i
        counts[i][t_a] -= 1
        counts[i][t_b] += 1
        counts[j][t_b] -= 1
        counts[j][t_a] += 1
        accepted += 1
    return np.array(site), accepted


def condition_label(threshold_um: float, keep_compartment: bool) -> str:
    if not np.isfinite(threshold_um):
        return "unconstrained"
    return f"{threshold_um:g} µm" + (" + comp." if keep_compartment else "")


def run_neuron(sk: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-contact table and per-chain table (acceptance) for every condition."""
    prep = sweep["prepare_neuron"](sk)
    inputs = prep["inputs"]
    observed = inputs.observed_assignment
    n_sites, n_targets = observed.shape
    contact_site, contact_target = initial_contacts(observed)
    n_contacts = len(contact_site)
    xyz = inputs.sites[["x", "y", "z"]].to_numpy(float) * config.geometry.coordinate_scale_to_um
    site_dist = cdist(xyz, xyz)
    dist_any = prep["dist_excl"]["any"]
    pre = np.array([p if isinstance(p, str) else "?" for p in prep["pre"]])
    required_pre = pre[contact_site]
    required_post = original_post_compartments(inputs, contact_site, contact_target)
    post_nearest_at_original = nearest_post_compartment(prep, contact_site, contact_target)
    steps = STEPS_PER_CONTACT * n_contacts

    conditions = [(np.inf, False)] + [
        (thr, keep) for thr in DIST_THRESHOLDS_UM for keep in COMPARTMENT_CONSTRAINT
    ]
    contact_rows, chain_rows = [], []
    for thr, keep in conditions:
        allowed = allowed_positions(prep, thr, keep, contact_target, required_pre, required_post)
        allowed_at_start = allowed[np.arange(n_contacts), contact_site]
        for smp in range(N_SAMPLES):
            rng = np.random.default_rng([sk, smp, int(min(thr, 1e6)), int(keep)])
            final_site, accepted = binary_swaps(
                contact_site, contact_target, allowed, n_sites, n_targets, steps, rng,
                PRESERVE_MULTIPLICITY_PROFILE,
            )
            final = np.zeros_like(observed)
            np.add.at(final, (final_site, contact_target), 1)
            assert np.array_equal(final.sum(axis=1), inputs.site_degrees)
            if PRESERVE_MULTIPLICITY_PROFILE:
                assert all(sorted(f[f > 0]) == sorted(o[o > 0]) for f, o in zip(final, observed))
            label = condition_label(thr, keep)
            chain_rows.append({
                "skid": sk, "condition": label, "threshold_um": thr, "keep_compartment": keep,
                "sample": smp, "proposals": steps, "accepted": accepted,
                "acceptance_rate": accepted / steps if steps else np.nan,
                "frac_contacts_allowed_at_start": allowed_at_start.mean(),
            })
            contact_rows.append(pd.DataFrame({
                "skid": sk, "condition": label, "threshold_um": thr, "keep_compartment": keep,
                "sample": smp, "contact_id": np.arange(n_contacts),
                "target_id": inputs.target_ids[contact_target],
                "original_site": contact_site, "final_site": final_site,
                "displacement_um": site_dist[contact_site, final_site],
                "dist_original_um": dist_any[contact_site, contact_target],
                "dist_final_um": dist_any[final_site, contact_target],
                "pre_original": required_pre, "post_original": required_post,  # real synapse
                "post_nearest_at_original": post_nearest_at_original,
                "pre_final": pre[final_site],
                "post_final": nearest_post_compartment(prep, final_site, contact_target),
                "allowed_at_start": allowed_at_start,
                "allowed_at_end": allowed[np.arange(n_contacts), final_site],
            }))
    return pd.concat(contact_rows, ignore_index=True), pd.DataFrame(chain_rows)


def finite_stats(values: pd.Series, prefix: str) -> dict:
    finite = values[np.isfinite(values)]
    return {f"{prefix}_mean": finite.mean(), f"{prefix}_median": values.median(), f"{prefix}_max": finite.max()}


def neuron_overview(per_contact: pd.DataFrame, per_chain: pd.DataFrame) -> pd.DataFrame:
    """One row per neuron and condition (plus 'observed'), chains pooled."""
    acceptance = per_chain.groupby(["skid", "condition"])["acceptance_rate"].mean()
    rows = []
    for (sk, label), d in per_contact.groupby(["skid", "condition"]):
        rows.append({
            "skid": sk, "condition": label,
            **finite_stats(d["dist_final_um"], "dist"),
            "displacement_mean": d["displacement_um"].mean(),
            "displacement_median": d["displacement_um"].median(),
            "displacement_max": d["displacement_um"].max(),
            "pct_at_original_site": 100 * (d["displacement_um"] == 0).mean(),
            "pct_same_combo": 100 * ((d["pre_final"] == d["pre_original"])
                                     & (d["post_final"] == d["post_original"])).mean(),
            "pct_acceptance": 100 * acceptance.loc[(sk, label)],
        })
    # observed reference: every condition carries the same original positions, take one chain.
    # its "same combo" is how often the nearest-site rule agrees with the real synapse,
    # i.e. the best a contact that never moves can score
    reference = per_contact[(per_contact["condition"] == "unconstrained") & (per_contact["sample"] == 0)]
    for sk, d in reference.groupby("skid"):
        rows.append({
            "skid": sk, "condition": "observed", **finite_stats(d["dist_original_um"], "dist"),
            "pct_same_combo": 100 * (d["post_nearest_at_original"] == d["post_original"]).mean(),
        })
    return pd.DataFrame(rows)


#%% run

per_contact, per_chain = [], []
for sk in SWEEP_SKIDS:
    try:
        contacts, chains = run_neuron(sk)
    except (KeyError, ValueError) as error:
        print(f"{sk}: skipped ({error})")
        continue
    per_contact.append(contacts)
    per_chain.append(chains)
    print(sk, "done", flush=True)
per_contact_df = pd.concat(per_contact, ignore_index=True)
per_chain_df = pd.concat(per_chain, ignore_index=True)
neuron_df = neuron_overview(per_contact_df, per_chain_df)
run_skids = list(per_contact_df["skid"].unique())

condition_order = ["observed", "unconstrained"] + [
    condition_label(thr, keep) for thr in DIST_THRESHOLDS_UM for keep in COMPARTMENT_CONSTRAINT
]
with pd.option_context("display.width", 220, "display.max_rows", None):
    print(neuron_df.drop(columns="skid").groupby("condition").mean()
          .reindex(condition_order).round(2).to_string())

if SAVE_DIR is not None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    per_contact_df.to_csv(SAVE_DIR / "binary_per_contact.csv", index=False)
    per_chain_df.to_csv(SAVE_DIR / "binary_per_chain.csv", index=False)
    neuron_df.to_csv(SAVE_DIR / "binary_neuron_overview.csv", index=False)


#%% plot 1: per presynaptic neuron (chains pooled), x = condition, bar = mean over neurons

COLOR_OBSERVED = "#0b0b0b"
COLOR_UNCONSTRAINED = "#a3a29b"
THRESHOLD_COLORS = dict(zip(DIST_THRESHOLDS_UM, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")))  # categorical 1-4


def condition_style(label: str) -> tuple[str, bool]:
    """Colour and fill: threshold sets the colour, compartment constraint draws it hollow."""
    if label == "observed":
        return COLOR_OBSERVED, True
    if label == "unconstrained":
        return COLOR_UNCONSTRAINED, True
    thr = float(label.split(" µm")[0])
    return THRESHOLD_COLORS[thr], "comp." not in label


def plot_neuron_metric(ax, column: str, log: bool = False):
    """Strip of neurons per condition, with a bar at the mean over neurons."""
    rng = np.random.default_rng(0)
    for xi, label in enumerate(condition_order):
        values = neuron_df.loc[neuron_df["condition"] == label, column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        color, filled = condition_style(label)
        ax.bar(xi, values.mean(), width=0.6, color=color, alpha=0.25, lw=0, zorder=1)
        ax.scatter(xi + rng.uniform(-0.18, 0.18, len(values)), values, s=16,
                   color=color if filled else "white", edgecolor=color, lw=1.0, zorder=3)
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(FormatStrFormatter("%g"))
        ax.tick_params(axis="y", which="both", labelsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


panels = [
    [("dist_mean", "mean (finite)", True), ("dist_median", "median", True), ("dist_max", "max (finite)", True)],
    [("displacement_mean", "mean", True), ("displacement_median", "median", True),
     ("displacement_max", "max", True)],
    [("pct_at_original_site", "% contacts at original site", False),
     ("pct_same_combo", "% contacts keeping compartment combo", False),
     ("pct_acceptance", "% swap proposals accepted", False)],
]
row_labels = ["site-nearest ps (µm)", "displacement from original (µm)", "%"]
fig, axes = plt.subplots(len(panels), 3, figsize=(15, 3.6 * len(panels)), sharex=True, squeeze=False)
for r, row in enumerate(panels):
    for c, (column, title, log) in enumerate(row):
        plot_neuron_metric(axes[r, c], column, log)
        axes[r, c].set_title(title, fontsize=9)
    axes[r, 0].set_ylabel(row_labels[r], fontsize=8)
for ax in axes[-1]:
    ax.set_xticks(range(len(condition_order)))
    ax.set_xticklabels(condition_order, rotation=45, ha="right", fontsize=8)
fig.suptitle(f"binary swap sampler, per presynaptic neuron: one dot per neuron ({N_SAMPLES} chains "
             f"pooled), bar = mean over {len(run_skids)} neurons; hollow = compartment kept", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97))


#%% plot 2: displacement ECDF, all neurons and chains pooled, one panel per threshold

MIN_PLOT_UM = 0.05


def plot_ecdf(ax, values: np.ndarray, **kwargs):
    x = np.sort(np.clip(values, MIN_PLOT_UM, None))  # stayed/returned (0) drawn at the left edge
    ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", **kwargs)


fig, axes = plt.subplots(1, len(DIST_THRESHOLDS_UM), figsize=(3.8 * len(DIST_THRESHOLDS_UM), 3.4), sharey=True)
unconstrained = per_contact_df.loc[per_contact_df["condition"] == "unconstrained", "displacement_um"]
for ax, thr in zip(axes, DIST_THRESHOLDS_UM):
    plot_ecdf(ax, unconstrained.to_numpy(), color=COLOR_UNCONSTRAINED, lw=2, label="unconstrained")
    for keep in COMPARTMENT_CONSTRAINT:
        label = condition_label(thr, keep)
        values = per_contact_df.loc[per_contact_df["condition"] == label, "displacement_um"].to_numpy()
        plot_ecdf(ax, values, color=THRESHOLD_COLORS[thr], lw=2, ls="--" if keep else "-", label=label)
    ax.set_xscale("log")
    ax.set_title(f"threshold {thr:g} µm", fontsize=10)
    ax.set_xlabel(f"original → final site (µm)\n(stayed/returned drawn at {MIN_PLOT_UM})", fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("frac. original contacts ≤ x")
fig.suptitle(f"binary swap sampler: displacement of each original contact ({len(run_skids)} neurons pooled)",
             fontsize=11)
fig.tight_layout()

if SAVE_DIR is not None:
    for number in plt.get_fignums():
        plt.figure(number).savefig(SAVE_DIR / f"binary_fig{number}.png", dpi=120, bbox_inches="tight")
plt.show()

# %%
