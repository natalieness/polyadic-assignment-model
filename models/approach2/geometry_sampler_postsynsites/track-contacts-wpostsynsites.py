''' track each original connector-site-target contact through the gibbs chains.

Every observed contact gets a unique id. A tracked copy of the reference gibbs loop
(geometry only) moves ids along with the counts: when a 2x2 update moves k copies of
target A from site 1 to site 2 (and k of B back), k of the A ids at site 1 travel to
site 2. Which copies move is drawn from a separate rng, so the count chain is
identical to models.approach2.sampler.gibbs_target_swaps for the same seed (asserted).

Ids start on the observed assignment and are carried through the null burn-in into
the geometry chain, so an id at its original site has either stayed or returned.
A contact never changes target, so its displacement is the distance between its
original and final presynaptic site.

Uses the same neurons, sweep values, scores and sampler settings as
sweep-geometry-wpostsynsites.py by running that script's setup section (everything
before its "run the sweep" cell). Repeats per neuron are N_SAMPLES from that script.

Optionally (INCLUDE_BACKBONE_GEOMETRY) also tracks the original backbone geometry
sampler: geometry.py scores and the sampler settings of BACKBONE_CONFIG_PATH, with its
own null burn-in, asserted equal to the chains run_attribution_experiment runs.
'''

from pathlib import Path
from math import exp
import ast
from itertools import chain

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import FormatStrFormatter
from scipy.spatial.distance import cdist

REPO_ROOT = Path("/Users/nessn/git-repos/polyadic-assignment-model")
SWEEP_SCRIPT = REPO_ROOT / "models/approach2/geometry_sampler_postsynsites/sweep-geometry-wpostsynsites.py"
sweep = {"__name__": "sweep_setup", "__file__": str(SWEEP_SCRIPT)}
exec(compile(SWEEP_SCRIPT.read_text().split("#%% run the sweep")[0], str(SWEEP_SCRIPT), "exec"), sweep)
config, _rng = sweep["config"], sweep["_rng"]
#SWEEP_SKIDS = sweep["SWEEP_SKIDS"]
N_SAMPLES = 2 #sweep["N_SAMPLES"]
DIST_THRESHOLDS_UM = sweep["DIST_THRESHOLDS_UM"]
COMPARTMENT_PENALTIES = sweep["COMPARTMENT_PENALTIES"]

from models.approach2.sampler import (  # noqa: E402  (repo path set up by the sweep setup)
    _UniformStream, _distinct_pair, _static_row_part, _moving_row_objective, gibbs_target_swaps,
)
from models.approach2.geometry import geometry_scores  # noqa: E402

celltype_df = pd.read_csv('data/celltype_df.csv')
celltype_df['skids'] = celltype_df['skids'].apply(ast.literal_eval)

n_per_ct = 4
skids = [celltype_df.loc[i, 'skids'][:n_per_ct] for i in range(len(celltype_df))]

SWEEP_SKIDS = list(chain.from_iterable(skids)) #SWEEP_SKIDS[-1]
EXAMPLE_SKID = SWEEP_SKIDS[0]
SAVE_DIR = None  # e.g. Path("model_out/track_contacts") to save tables and figures

# optional: also track the original backbone geometry sampler (geometry.py scores and
# sampler settings of the config run_and_save.py / run_on_neurons.py use), with its own
# null chains so the result is exactly run_attribution_experiment's geometry condition
INCLUDE_BACKBONE_GEOMETRY = True
BACKBONE_CONFIG_PATH = REPO_ROOT / "models/approach2/sover1_config.json"
backbone_config = sweep["Approach2Config"].load(BACKBONE_CONFIG_PATH)
backbone_repository = sweep["DataRepository"](backbone_config.paths)


#%% functions

def tracked_gibbs(assignment, tokens, geometry_scores, rng, *, geometry_weight, steps, id_rng,
                  run_config=None):
    """Geometry-only copy of gibbs_target_swaps_nonumba that also moves contact ids.

    ``tokens[site][target]`` is the list of contact ids in that cell. ``run_config``
    supplies the sampler settings (the sweep config by default). Returns the new
    assignment and tokens (inputs are not modified).
    """
    run_config = run_config or config
    s = run_config.sampler
    result = np.asarray(assignment, dtype=int).copy()
    tokens = [[list(cell) for cell in row] for row in tokens]
    sites = np.flatnonzero(result.sum(axis=1) > 0).tolist()
    geometry_rows = np.asarray(geometry_scores, dtype=float).tolist()
    row_scores = dict(
        geometry_weight=geometry_weight, cotargeting_weight=0.0,
        geometry_repeat_discount=run_config.geometry.repeat_discount,
        cotarget_aggregation=run_config.cotarget.aggregation,
        cotarget_site_clip=run_config.cotarget.site_score_clip,
        self_repeat_penalty=0.0, cotarget_include_self=False,
    )
    uniforms = _UniformStream(rng)
    for _ in range(int(steps)):
        ls, rs = _distinct_pair(sites, uniforms.next(), uniforms.next())
        left_row, right_row = result[ls], result[rs]
        pooled = np.flatnonzero(left_row + right_row).tolist()
        if len(pooled) < 2:
            continue
        lt, rt = _distinct_pair(pooled, uniforms.next(), uniforms.next())
        lv, rv = left_row.tolist(), right_row.tolist()
        left_subtotal = lv[lt] + lv[rt]
        right_subtotal = rv[lt] + rv[rt]
        left_target_subtotal = lv[lt] + rv[lt]
        lower = max(0, left_target_subtotal - right_subtotal)
        upper = min(left_subtotal, left_target_subtotal)
        if upper < lower:
            continue
        l_static = _static_row_part(lv, pooled, lt, rt, None, False)
        r_static = _static_row_part(rv, pooled, lt, rt, None, False)
        left_profile = sorted(v for v in (lv[lt], lv[rt]) if v)
        right_profile = sorted(v for v in (rv[lt], rv[rt]) if v)
        candidates, objectives = [], []
        for new_ll in range(lower, upper + 1):
            new_lr = left_subtotal - new_ll
            new_rl = left_target_subtotal - new_ll
            new_rr = right_subtotal - new_rl
            if s.max_same_target_per_site is not None and (
                max(l_static[1], new_ll, new_lr) > s.max_same_target_per_site
                or max(r_static[1], new_rl, new_rr) > s.max_same_target_per_site
            ):
                continue
            if s.preserve_multiplicity_profile and (
                sorted(v for v in (new_ll, new_lr) if v) != left_profile
                or sorted(v for v in (new_rl, new_rr) if v) != right_profile
            ):
                continue
            objective = _moving_row_objective(
                new_ll, new_lr, geometry_rows[ls][lt], geometry_rows[ls][rt],
                l_static[0], l_static[2], l_static[3], l_static[4], 0.0,
                l_static[5], l_static[6], 0.0, 0.0, **row_scores,
            ) + _moving_row_objective(
                new_rl, new_rr, geometry_rows[rs][lt], geometry_rows[rs][rt],
                r_static[0], r_static[2], r_static[3], r_static[4], 0.0,
                r_static[5], r_static[6], 0.0, 0.0, **row_scores,
            )
            candidates.append((new_ll, new_lr, new_rl, new_rr))
            objectives.append(objective)
        if len(candidates) < 2:
            continue
        largest = max(objectives)
        weights = [exp((o - largest) / s.temperature) for o in objectives]
        threshold = uniforms.next() * sum(weights)
        cumulative, chosen = 0.0, len(weights) - 1
        for index, weight in enumerate(weights):
            cumulative += weight
            if threshold < cumulative:
                chosen = index
                break
        new_ll, new_lr, new_rl, new_rr = candidates[chosen]

        # move ids: k copies of lt go left->right and k of rt right->left (or reverse)
        k = lv[lt] - new_ll
        if k > 0:
            _move_ids(tokens, ls, rs, lt, k, id_rng)
            _move_ids(tokens, rs, ls, rt, k, id_rng)
        elif k < 0:
            _move_ids(tokens, rs, ls, lt, -k, id_rng)
            _move_ids(tokens, ls, rs, rt, -k, id_rng)
        result[ls, lt], result[ls, rt] = new_ll, new_lr
        result[rs, lt], result[rs, rt] = new_rl, new_rr
    return result, tokens


def _move_ids(tokens, from_site, to_site, target, k, id_rng):
    cell = tokens[from_site][target]
    for _ in range(k):
        tokens[to_site][target].append(cell.pop(int(id_rng.integers(len(cell)))))


def initial_tokens(observed: np.ndarray):
    """Unique id per observed contact; returns tokens and the (site, target) of each id."""
    tokens = [[[] for _ in range(observed.shape[1])] for _ in range(observed.shape[0])]
    origin = []
    for i, j in zip(*np.nonzero(observed)):
        for _ in range(observed[i, j]):
            tokens[i][j].append(len(origin))
            origin.append((i, j))
    return tokens, np.array(origin)


def final_sites(tokens, n_contacts: int) -> tuple[np.ndarray, np.ndarray]:
    """Final (site, target) of each contact id."""
    site = np.full(n_contacts, -1)
    target = np.full(n_contacts, -1)
    for i, row in enumerate(tokens):
        for j, ids in enumerate(row):
            site[ids] = i
            target[ids] = j
    return site, target


def original_post_compartments(inputs, origin: np.ndarray) -> np.ndarray:
    """Post compartment letter of each contact id's real synapse.

    Copies of one target at one site are interchangeable, so the letters of that
    cell's synapses are shared out over its ids in order.
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
        cells.setdefault((site_index[connector_id], j), []).append(
            sweep["COMPARTMENT_LETTERS"].get(compartment, "?")
        )
    return np.array([cells[(i, j)].pop() for i, j in origin])


def nearest_post_compartment(prep: dict, sites: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Compartment letter of the nearest postsynaptic site on the target from each site."""
    d_axon = prep["dist_excl"]["axon"][sites, targets]
    d_dendrite = prep["dist_excl"]["dendrite"][sites, targets]
    letters = np.where(d_axon < d_dendrite, "A", "D")
    return np.where(np.isinf(d_axon) & np.isinf(d_dendrite), "?", letters)


def allowed_combo_distances(prep: dict) -> np.ndarray:
    """(n_sites, n_targets) distance to the nearest postsynaptic site in an allowed combo.

    A site of compartment X may use the target's compartment-Y sites only if combo XY is
    allowed for that target (sweep's population rule); inf if no allowed site exists.
    """
    dist = prep["dist_excl"]
    out = np.full(dist["any"].shape, np.inf)
    for combo in sweep["COMBOS"]:
        ok = (prep["pre"] == combo[0])[:, None] & prep["allowed"][combo].to_numpy()[None, :]
        out = np.fmin(out, np.where(ok, dist[sweep["POST_KEY"][combo[1]]], np.inf))
    return out


def backbone_reference_chain(start, inputs, scores, rng, geometry_weight, steps_per_contact) -> np.ndarray:
    """A null or geometry chain as run_attribution_experiment runs it with backbone_config."""
    bb = backbone_config
    return gibbs_target_swaps(
        start, inputs.site_degrees, inputs.target_counts, scores,
        np.zeros((len(inputs.target_ids),) * 2), rng,
        geometry_weight=geometry_weight,
        cotargeting_weight=0.0,
        temperature=bb.sampler.temperature,
        geometry_repeat_discount=bb.geometry.repeat_discount if geometry_weight else 0.0,
        cotarget_aggregation=bb.cotarget.aggregation,
        cotarget_site_clip=bb.cotarget.site_score_clip,
        self_repeat_penalty=bb.sampler.self_repeat_penalty if geometry_weight else 0.0,
        preserve_multiplicity_profile=bb.sampler.preserve_multiplicity_profile,
        max_same_target_per_site=bb.sampler.max_same_target_per_site,
        steps=steps_per_contact * int(inputs.site_degrees.sum()),
    )


def track_neuron(sk: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-contact displacement for the null and every (threshold, penalty) condition.

    Returns the per-contact table and the site coordinates (um) for plotting.
    """
    prep = sweep["prepare_neuron"](sk)
    inputs = prep["inputs"]
    observed = inputs.observed_assignment
    n_contacts = int(observed.sum())
    dist = prep["dist_excl"]["any"]
    xyz = inputs.sites[["x", "y", "z"]].to_numpy(float) * config.geometry.coordinate_scale_to_um
    site_dist = cdist(xyz, xyz)
    tokens0, origin = initial_tokens(observed)
    zero = np.zeros(observed.shape)

    nulls = []
    for smp in range(N_SAMPLES):
        a, t = tracked_gibbs(observed, tokens0, zero, _rng(config, sk, smp, 10), geometry_weight=0.0,
                             steps=config.sampler.null_burnin_steps_per_contact * n_contacts,
                             id_rng=np.random.default_rng([sk, smp, 99]))
        assert np.array_equal(a, sweep["run_chain"](observed, inputs, zero, _rng(config, sk, smp, 10),
                                                    0.0, config.sampler.null_burnin_steps_per_contact))
        nulls.append((a, t))
    conditions = {("null", np.nan, np.nan): [t for _, t in nulls]}
    for thr in DIST_THRESHOLDS_UM:
        wrong = sweep["wrong_compartment_mask"](prep, thr)
        for penalty in COMPARTMENT_PENALTIES:
            scores = sweep["postsyn_geometry_scores"](dist, thr, wrong, penalty)
            token_samples = []
            for smp, (a, t) in enumerate(nulls):
                g, gt = tracked_gibbs(a, t, scores, _rng(config, sk, smp, 20),
                                      geometry_weight=config.sampler.geometry_weight,
                                      steps=config.sampler.mcmc_steps_per_contact * n_contacts,
                                      id_rng=np.random.default_rng([sk, smp, 98]))
                assert np.array_equal(g, sweep["run_chain"](
                    a, inputs, scores, _rng(config, sk, smp, 20),
                    config.sampler.geometry_weight, config.sampler.mcmc_steps_per_contact))
                token_samples.append(gt)
            conditions[("geometry", thr, penalty)] = token_samples

    if INCLUDE_BACKBONE_GEOMETRY:
        # its own null burn-in (run_attribution_experiment's, with backbone_config),
        # then its geometry chain; both checked against the real sampler
        bb = backbone_config
        bb_weight = bb.sampler.geometry_weight if bb.geometry.enabled else 0.0
        scores = geometry_scores(backbone_repository, inputs, bb.geometry).scores
        token_samples = []
        for smp in range(N_SAMPLES):
            a, t = tracked_gibbs(observed, tokens0, zero, _rng(bb, sk, smp, 10), geometry_weight=0.0,
                                 steps=bb.sampler.null_burnin_steps_per_contact * n_contacts,
                                 id_rng=np.random.default_rng([sk, smp, 97]), run_config=bb)
            assert np.array_equal(a, backbone_reference_chain(
                observed, inputs, zero, _rng(bb, sk, smp, 10), 0.0, bb.sampler.null_burnin_steps_per_contact))
            g, gt = tracked_gibbs(a, t, scores, _rng(bb, sk, smp, 20), geometry_weight=bb_weight,
                                  steps=bb.sampler.mcmc_steps_per_contact * n_contacts,
                                  id_rng=np.random.default_rng([sk, smp, 96]), run_config=bb)
            assert np.array_equal(g, backbone_reference_chain(
                a, inputs, scores, _rng(bb, sk, smp, 20), bb_weight, bb.sampler.mcmc_steps_per_contact))
            token_samples.append(gt)
        conditions[("backbone", np.nan, np.nan)] = token_samples

    pre = np.array([p if isinstance(p, str) else "?" for p in prep["pre"]])
    post_original = original_post_compartments(inputs, origin)
    post_nearest_at_original = nearest_post_compartment(prep, origin[:, 0], origin[:, 1])
    dist_allowed = allowed_combo_distances(prep)

    rows = []
    for (condition, thr, penalty), token_samples in conditions.items():
        for smp, tokens in enumerate(token_samples):
            site, target = final_sites(tokens, n_contacts)
            assert np.array_equal(target, origin[:, 1])  # ids never change target
            rows.append(pd.DataFrame({
                "skid": sk, "condition": condition, "threshold_um": thr, "penalty": penalty,
                "sample": smp, "contact_id": np.arange(n_contacts),
                "target_id": inputs.target_ids[origin[:, 1]],
                "original_site": origin[:, 0], "final_site": site,
                "displacement_um": site_dist[origin[:, 0], site],
                "pre_original": pre[origin[:, 0]],
                "post_original": post_original,  # real synapse
                "post_nearest_at_original": post_nearest_at_original,
                "pre_final": pre[site],
                "post_final": nearest_post_compartment(prep, site, target),  # nearest site rule
                # site -> nearest postsyn site on the target, without / with compartment restraint
                "dist_original_um": dist[origin[:, 0], origin[:, 1]],
                "dist_original_allowed_um": dist_allowed[origin[:, 0], origin[:, 1]],
                "dist_final_um": dist[site, target],
                "dist_final_allowed_um": dist_allowed[site, target],
            }))
    return pd.concat(rows, ignore_index=True), xyz


def summarise(per_contact: pd.DataFrame) -> pd.DataFrame:
    """Displacement metrics per neuron and condition."""
    def metrics(d: pd.Series) -> pd.Series:
        moved = d[d > 0]
        return pd.Series({
            "frac_at_original_site": float(np.mean(d == 0)),
            "median_displacement_um": float(np.median(d)),
            "median_displacement_if_moved_um": float(np.median(moved)) if len(moved) else 0.0,
            "frac_within_2um": float(np.mean(d <= 2)),
            "frac_within_10um": float(np.mean(d <= 10)),
        })
    return (per_contact.groupby(["skid", "condition", "threshold_um", "penalty"], dropna=False)
            ["displacement_um"].apply(metrics).unstack().reset_index())


#%% run

per_contact, site_xyz = [], {}
for sk in SWEEP_SKIDS:
    table, site_xyz[sk] = track_neuron(sk)
    per_contact.append(table)
    print(sk, "done", flush=True)
per_contact_df = pd.concat(per_contact, ignore_index=True)
results_df = summarise(per_contact_df)

with pd.option_context("display.width", 200, "display.max_rows", None):
    print(results_df.groupby(["condition", "threshold_um", "penalty"], dropna=False)
          .mean(numeric_only=True).drop(columns="skid").round(3).to_string())

if SAVE_DIR is not None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    per_contact_df.to_csv(SAVE_DIR / "displacement_per_contact.csv", index=False)
    results_df.to_csv(SAVE_DIR / "displacement_results.csv", index=False)


#%% compartment matching overview
# original combo: original site compartment + post compartment of the real synapse
# final combo: final site compartment + compartment of the nearest postsynaptic site on
# the target from the final site (a sampled contact has no real postsynaptic node)

def compartment_overview(per_contact: pd.DataFrame) -> pd.DataFrame:
    """Fraction of contacts keeping / changing compartments, per condition (all neurons pooled)."""
    d = per_contact
    same_pre = d["pre_final"] == d["pre_original"]
    same_post = d["post_final"] == d["post_original"]
    flags = pd.DataFrame({
        "same_combo": same_pre & same_post,
        "changed_post_only": same_pre & ~same_post,
        "changed_pre_only": ~same_pre & same_post,
        "changed_both": ~same_pre & ~same_post,
        "matched_comp_original (AA/DD)": d["pre_original"] == d["post_original"],
        "matched_comp_final (AA/DD)": d["pre_final"] == d["post_final"],
    })
    keys = [d["condition"], d["threshold_um"], d["penalty"]]
    out = flags.groupby(keys, dropna=False).mean()
    out.insert(0, "contacts_per_chain", d.groupby(keys, dropna=False).size() / N_SAMPLES)
    return out


def combo_transitions(per_contact: pd.DataFrame, condition: str, thr=np.nan, penalty=np.nan) -> pd.DataFrame:
    """Original -> final combo counts, mean per chain, all neurons pooled."""
    sel = per_contact["condition"] == condition
    if condition == "geometry":
        sel &= (per_contact["threshold_um"] == thr) & (per_contact["penalty"] == penalty)
    d = per_contact[sel]
    table = pd.crosstab(d["pre_original"] + d["post_original"], d["pre_final"] + d["post_final"],
                        rownames=["original"], colnames=["final"], margins=True)
    return (table / N_SAMPLES).round(1)


# how often the nearest-site rule agrees with the real synapse at the original site:
# the best "same post compartment" a contact that stays put can score under this rule
observed_once = per_contact_df[(per_contact_df["condition"] == "null") & (per_contact_df["sample"] == 0)]
rule_agreement = (observed_once["post_nearest_at_original"] == observed_once["post_original"]).mean()
print(f"nearest-site rule matches the real post compartment for {rule_agreement:.1%} of observed contacts")

compartment_df = compartment_overview(per_contact_df)
with pd.option_context("display.width", 220, "display.max_rows", None):
    print(compartment_df.round(3).to_string())
    for args, label in (((("null",)), "null"),
                        ((("geometry", DIST_THRESHOLDS_UM[0], 0.0)), f"geometry {DIST_THRESHOLDS_UM[0]} µm, no penalty"),
                        ((("geometry", DIST_THRESHOLDS_UM[0], COMPARTMENT_PENALTIES[-1])),
                         f"geometry {DIST_THRESHOLDS_UM[0]} µm, penalty {COMPARTMENT_PENALTIES[-1]:g}")):
        print(f"\n{label}: original -> final combo (contacts per chain, {len(SWEEP_SKIDS)} neurons pooled)")
        print(combo_transitions(per_contact_df, *args).to_string())

if SAVE_DIR is not None:
    compartment_df.to_csv(SAVE_DIR / "compartment_overview.csv")


#%% plot 1: displacement ECDF per neuron, null vs geometry

COLOR_NULL = "#a3a29b"
COLOR_BACKBONE = "#eda100"  # categorical slot 4, after the three thresholds
BACKBONE_LABEL = f"backbone geometry ({BACKBONE_CONFIG_PATH.stem})"
THRESHOLD_COLORS = dict(zip(DIST_THRESHOLDS_UM, ("#2a78d6", "#eb6834", "#1baf7a")))  # categorical 1-3
MIN_PLOT_UM = 0.05


def condition_displacements(sk, condition, thr=np.nan, penalty=np.nan) -> np.ndarray:
    sel = (per_contact_df["skid"] == sk) & (per_contact_df["condition"] == condition)
    if condition == "geometry":
        sel &= (per_contact_df["threshold_um"] == thr) & (per_contact_df["penalty"] == penalty)
    return per_contact_df.loc[sel, "displacement_um"].to_numpy()


def plot_ecdf(ax, values: np.ndarray, **kwargs):
    x = np.sort(np.clip(values, MIN_PLOT_UM, None))  # stayed/returned (0) drawn at the left edge
    ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", **kwargs)


N_COLS = min(5, len(SWEEP_SKIDS))
n_rows = int(np.ceil(len(SWEEP_SKIDS) / N_COLS))
fig, axes = plt.subplots(n_rows, N_COLS, figsize=(3.6 * N_COLS, 3.4 * n_rows), sharey=True, squeeze=False)
for ax in axes.flat[len(SWEEP_SKIDS):]:
    ax.set_visible(False)
for ax, sk in zip(axes.flat, SWEEP_SKIDS):
    plot_ecdf(ax, condition_displacements(sk, "null"), color=COLOR_NULL, lw=2, label="null")
    for thr, color in THRESHOLD_COLORS.items():
        plot_ecdf(ax, condition_displacements(sk, "geometry", thr, 0.0), color=color, lw=1.8,
                  label=f"geometry {thr} µm")
    plot_ecdf(ax, condition_displacements(sk, "geometry", DIST_THRESHOLDS_UM[0], COMPARTMENT_PENALTIES[-1]),
              color=THRESHOLD_COLORS[DIST_THRESHOLDS_UM[0]], lw=1.2, ls="--",
              label=f"geometry {DIST_THRESHOLDS_UM[0]} µm, penalty {COMPARTMENT_PENALTIES[-1]:g}")
    if INCLUDE_BACKBONE_GEOMETRY:
        plot_ecdf(ax, condition_displacements(sk, "backbone"), color=COLOR_BACKBONE, lw=1.8,
                  label=BACKBONE_LABEL)
    ax.set_xscale("log")
    ax.set_title(str(sk), fontsize=10)
    ax.set_xlabel(f"original → final site (µm)\n(stayed/returned drawn at {MIN_PLOT_UM})", fontsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
for ax in axes[:, 0]:
    ax.set_ylabel("frac. original contacts ≤ x")
axes[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
fig.suptitle("how far does each original contact end up from its own site?", fontsize=11)
fig.tight_layout()


#%% plot 2: summary metrics vs penalty, one line per threshold (mean over neurons)

metric_labels = {
    "frac_at_original_site": "frac. contacts at original site",
    "median_displacement_um": "median displacement (µm)",
    "frac_within_2um": "frac. contacts within 2 µm of origin",
    "frac_within_10um": "frac. contacts within 10 µm of origin",
}
summary = results_df.groupby(["condition", "threshold_um", "penalty"], dropna=False).mean(numeric_only=True)
null_summary = results_df[results_df["condition"] == "null"].mean(numeric_only=True)
x = np.arange(len(COMPARTMENT_PENALTIES))
fig, axes = plt.subplots(1, len(metric_labels), figsize=(4 * len(metric_labels), 3.2))
for ax, (metric, label) in zip(axes, metric_labels.items()):
    for thr, color in THRESHOLD_COLORS.items():
        geo = summary.loc[("geometry", thr)][metric].reindex(COMPARTMENT_PENALTIES)
        ax.plot(x, geo.to_numpy(), color=color, lw=2, marker="o", ms=5, label=f"{thr} µm")
    ax.axhline(null_summary[metric], color=COLOR_NULL, lw=1.5, ls=":", label="null")
    if INCLUDE_BACKBONE_GEOMETRY:
        backbone_summary = results_df[results_df["condition"] == "backbone"].mean(numeric_only=True)
        ax.axhline(backbone_summary[metric], color=COLOR_BACKBONE, lw=1.5, ls="--", label=BACKBONE_LABEL)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:g}" for p in COMPARTMENT_PENALTIES])
    ax.set_xlabel("wrong compartment penalty (0 = none)")
    ax.set_title(label, fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].legend(title="threshold", fontsize=8, title_fontsize=8, frameon=False)
fig.text(0.5, -0.02, f"mean over {len(SWEEP_SKIDS)} MBONs, {N_SAMPLES} chains each",
         ha="center", fontsize=8)
fig.tight_layout()


#%% plot 3: where contacts move on the example neuron (one chain, 2D projection)

def project_2d(xyz: np.ndarray) -> np.ndarray:
    """Project site coordinates onto their two main axes of variation."""
    centred = xyz - xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return centred @ vt[:2].T

EXAMPLE_SKID = list(site_xyz.keys())[0]
xy = project_2d(site_xyz[EXAMPLE_SKID])
panels = [("null", np.nan, np.nan, "null"),
          ("geometry", DIST_THRESHOLDS_UM[0], 0.0, f"geometry {DIST_THRESHOLDS_UM[0]} µm")]
if INCLUDE_BACKBONE_GEOMETRY:
    panels.append(("backbone", np.nan, np.nan, "backbone geometry"))
fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 3.6), sharex=True, sharey=True)
for ax, (condition, thr, penalty, title) in zip(axes, panels):
    sel = ((per_contact_df["skid"] == EXAMPLE_SKID) & (per_contact_df["condition"] == condition)
           & (per_contact_df["sample"] == 0))
    if condition == "geometry":
        sel &= (per_contact_df["threshold_um"] == thr) & (per_contact_df["penalty"] == penalty)
    contacts = per_contact_df[sel]
    moved = contacts[contacts["final_site"] != contacts["original_site"]]
    stayed = contacts[contacts["final_site"] == contacts["original_site"]]
    ax.scatter(xy[:, 0], xy[:, 1], s=10, color=COLOR_NULL, lw=0, zorder=1, label="presynaptic sites")
    segments = np.stack([xy[moved["original_site"]], xy[moved["final_site"]]], axis=1)
    ax.add_collection(LineCollection(segments, colors="#2a78d6", lw=0.6, alpha=0.35, zorder=2,
                                     label="original → final site"))
    ax.scatter(xy[stayed["original_site"], 0], xy[stayed["original_site"], 1], s=22,
               color="#eb6834", edgecolor="white", lw=0.5, zorder=3, label="at original site")
    frac = len(stayed) / len(contacts)
    ax.set_title(f"{title}: {frac:.0%} at original site, median move "
                 f"{np.median(contacts['displacement_um']):.1f} µm", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("site position, main axis (µm)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("site position, second axis (µm)")
axes[0].legend(fontsize=8, frameon=False, loc="best")
fig.suptitle(f"{[0]}: movement of each original contact (chain 0)", fontsize=11)
fig.tight_layout()


#%% plot 4: compartment matching, original vs final combo

CHANGE_COLUMNS = {
    "same_combo": ("same combo", "#2a78d6"),
    "changed_post_only": ("changed post only", "#eb6834"),
    "changed_pre_only": ("changed pre only", "#1baf7a"),
    "changed_both": ("changed pre and post", "#eda100"),
}

# 4a: stacked bar per condition
bars = compartment_df.reset_index()
bars["label"] = [
    f"geometry {t:g} µm, penalty {p:g}" if c == "geometry" else (BACKBONE_LABEL if c == "backbone" else c)
    for c, t, p in zip(bars["condition"], bars["threshold_um"], bars["penalty"])
]
bars = pd.concat([bars[bars["condition"] == c] for c in ("null", "backbone", "geometry")])
y = np.arange(len(bars))
fig, ax = plt.subplots(figsize=(8, 0.35 * len(bars) + 1.6))
left = np.zeros(len(bars))
for column, (label, color) in CHANGE_COLUMNS.items():
    values = bars[column].to_numpy()
    ax.barh(y, values, left=left, color=color, edgecolor="white", lw=1, height=0.7, label=label)
    left += values
for row, (_, r) in enumerate(bars.iterrows()):
    ax.text(r["same_combo"] / 2, row, f"{r['same_combo']:.0%}", ha="center", va="center",
            fontsize=7, color="white")
    ax.text(1.01, row, f"{r['matched_comp_final (AA/DD)']:.0%}", ha="left", va="center", fontsize=7,
            transform=ax.get_yaxis_transform())
ax.text(1.01, -0.9, "AA/DD\nfinal", ha="left", va="bottom", fontsize=7, transform=ax.get_yaxis_transform())
ax.set_yticks(y)
ax.set_yticklabels(bars["label"], fontsize=8)
ax.set_ylim(len(bars) - 0.5, -0.5)
ax.set_xlim(0, 1)
ax.set_xlabel("fraction of original contacts")
ax.set_title(f"does a contact keep its compartment combo? (AA/DD originally "
             f"{bars['matched_comp_original (AA/DD)'].iloc[0]:.0%}; nearest-site rule "
             f"agrees with real synapse for {rule_agreement:.0%})", fontsize=9)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(ncol=4, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()

# 4b: original -> final transition heatmaps, row normalised, every condition
# first row: reference conditions; then one row per threshold, one column per penalty
reference_panels = [(("null",), "null")]
if INCLUDE_BACKBONE_GEOMETRY:
    reference_panels.append((("backbone",), BACKBONE_LABEL))
n_cols = max(len(COMPARTMENT_PENALTIES), len(reference_panels))
transition_panels = {(0, c): panel for c, panel in enumerate(reference_panels)}
for r, thr in enumerate(DIST_THRESHOLDS_UM, start=1):
    for c, penalty in enumerate(COMPARTMENT_PENALTIES):
        transition_panels[(r, c)] = (("geometry", thr, penalty), f"geometry {thr:g} µm, penalty {penalty:g}")
combos = list(sweep["COMBOS"])
fig, axes = plt.subplots(1 + len(DIST_THRESHOLDS_UM), n_cols,
                         figsize=(3.4 * n_cols, 3.3 * (1 + len(DIST_THRESHOLDS_UM))), squeeze=False)
for position, ax in np.ndenumerate(axes):
    if position not in transition_panels:
        ax.set_visible(False)
for (r, c), (args, title) in transition_panels.items():
    ax = axes[r, c]
    counts = combo_transitions(per_contact_df, *args).drop(index="All", columns="All")
    counts = counts.reindex(index=combos, columns=combos, fill_value=0.0)
    shares = counts.div(counts.sum(axis=1), axis=0).fillna(0.0)
    ax.imshow(shares.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    for r in range(len(combos)):
        for c in range(len(combos)):
            share = shares.iloc[r, c]
            ax.text(c, r, f"{share:.0%}\n({counts.iloc[r, c]:.0f})", ha="center", va="center",
                    fontsize=7, color="white" if share > 0.6 else "0.15")
    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels(combos)
    ax.set_yticks(range(len(combos)))
    ax.set_yticklabels(combos)
    ax.set_xlabel("final combo", fontsize=8)
    ax.set_ylabel("original combo" if c == 0 else "", fontsize=8)
    ax.set_title(title, fontsize=9)
fig.suptitle(f"original → final compartment combo (share of row; contacts per chain in brackets, "
             f"{len(SWEEP_SKIDS)} MBONs pooled)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97))


#%% plot 5: per presynaptic neuron, all chains pooled
# distance from each contact's site to the target's nearest postsynaptic site (own
# partner nodes excluded), without / with compartment restraint (allowed combos only;
# inf where none exists, so mean and max are over finite values), displacement from
# the original site, and % of contacts at their original site

def condition_label(condition, thr, penalty) -> str:
    if condition == "geometry":
        return f"geometry {thr:g} µm, penalty {penalty:g}"
    return BACKBONE_LABEL if condition == "backbone" else condition


def finite_stats(values: pd.Series, prefix: str) -> dict:
    finite = values[np.isfinite(values)]
    return {
        f"{prefix}_mean": finite.mean(), f"{prefix}_median": values.median(),
        f"{prefix}_max": finite.max(), f"{prefix}_frac_inf": 1 - len(finite) / len(values),
    }


def neuron_overview(per_contact: pd.DataFrame) -> pd.DataFrame:
    """One row per neuron and condition (plus 'observed'), chains pooled."""
    rows = []
    for (sk, condition, thr, penalty), d in per_contact.groupby(
        ["skid", "condition", "threshold_um", "penalty"], dropna=False
    ):
        rows.append({
            "skid": sk, "condition": condition_label(condition, thr, penalty),
            **finite_stats(d["dist_final_um"], "dist"),
            **finite_stats(d["dist_final_allowed_um"], "dist_allowed"),
            "displacement_mean": d["displacement_um"].mean(),
            "displacement_median": d["displacement_um"].median(),
            "displacement_max": d["displacement_um"].max(),
            "pct_at_original_site": 100 * (d["displacement_um"] == 0).mean(),
        })
    # observed reference: every condition carries the same original distances, take one chain
    for sk, d in per_contact[(per_contact["condition"] == "null") & (per_contact["sample"] == 0)].groupby("skid"):
        rows.append({
            "skid": sk, "condition": "observed",
            **finite_stats(d["dist_original_um"], "dist"),
            **finite_stats(d["dist_original_allowed_um"], "dist_allowed"),
        })
    return pd.DataFrame(rows)


neuron_df = neuron_overview(per_contact_df)
with pd.option_context("display.width", 250, "display.max_rows", None):
    print(neuron_df.round(2).to_string(index=False))
if SAVE_DIR is not None:
    neuron_df.to_csv(SAVE_DIR / "neuron_overview.csv", index=False)

# conditions shown in the plot (all are in neuron_df): label -> (colour, marker, filled)
NEURON_PLOT_CONDITIONS = {
    "observed": ("#0b0b0b", "o", True),
    "null": (COLOR_NULL, "o", True),
    **{condition_label("geometry", thr, 0.0): (color, "o", True) for thr, color in THRESHOLD_COLORS.items()},
    condition_label("geometry", DIST_THRESHOLDS_UM[0], COMPARTMENT_PENALTIES[-1]):
        (THRESHOLD_COLORS[DIST_THRESHOLDS_UM[0]], "o", False),
}
if INCLUDE_BACKBONE_GEOMETRY:
    NEURON_PLOT_CONDITIONS[BACKBONE_LABEL] = (COLOR_BACKBONE, "D", True)

panel_rows = [
    ("dist", "site-nearest ps (µm)", True),
    ("dist_allowed", "site-nearest ps comp. (µm)", True),
    ("displacement", "displacement from original (µm)", True),
]
stats = ("mean", "median", "max")
plot_conditions = list(NEURON_PLOT_CONDITIONS)
x = np.arange(len(plot_conditions))
fig, axes = plt.subplots(len(panel_rows) + 1, len(stats),
                         figsize=(max(12, 0.6 * len(plot_conditions) * len(stats)), 3.4 * (len(panel_rows) + 1)),
                         sharex=True, squeeze=False)


def plot_neuron_metric(ax, column):
    """Strip of neurons per condition, with a bar at the mean over neurons."""
    rng = np.random.default_rng(0)
    for xi, label in zip(x, plot_conditions):
        color, marker, filled = NEURON_PLOT_CONDITIONS[label]
        values = neuron_df.loc[neuron_df["condition"] == label, column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        ax.bar(xi, values.mean(), width=0.6, color=color, alpha=0.25, lw=0, zorder=1)
        ax.scatter(xi + rng.uniform(-0.18, 0.18, len(values)), values, s=18, marker=marker,
                   color=color if filled else "white", edgecolor=color, lw=1.0, zorder=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


for r, (prefix, ylabel, log) in enumerate(panel_rows):
    for c, stat in enumerate(stats):
        ax = axes[r, c]
        plot_neuron_metric(ax, f"{prefix}_{stat}")
        if log:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
            ax.yaxis.set_minor_formatter(FormatStrFormatter("%g"))
            ax.tick_params(axis="y", which="both", labelsize=7)
        ax.set_title(f"{stat}{' (finite)' if prefix != 'displacement' and stat != 'median' else ''}",
                     fontsize=9)
        if c == 0:
            ax.set_ylabel(ylabel, fontsize=8)
plot_neuron_metric(axes[-1, 0], "pct_at_original_site")
axes[-1, 0].set_title("contacts at original site", fontsize=9)
axes[-1, 0].set_ylabel("% of contacts", fontsize=8)
for ax in axes[-1, 1:]:
    ax.set_visible(False)
for ax in axes[-1, :1].tolist() + axes[-2, 1:].tolist():
    ax.set_xticks(x)
    ax.set_xticklabels(plot_conditions, rotation=45, ha="right", fontsize=7)
    ax.tick_params(labelbottom=True)
fig.suptitle(f"per presynaptic neuron: one dot per neuron ({N_SAMPLES} chains pooled), "
             f"bar = mean over {len(SWEEP_SKIDS)} neurons", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97))


#%% plot 6: % of site-target pairings outside the allowed range, per presynaptic neuron
# a pairing is outside if the target's nearest postsynaptic site (own partner nodes
# excluded) is further than the threshold from the contact's site; with compartment
# restraint only postsynaptic sites in allowed combos count. inf (none) = outside.

OUTSIDE_THRESHOLDS_UM = DIST_THRESHOLDS_UM


def outside_overview(per_contact: pd.DataFrame) -> pd.DataFrame:
    """% of contacts outside each threshold, per neuron and condition (plus 'observed')."""
    def pct_outside(dist: pd.Series, dist_allowed: pd.Series) -> dict:
        return {
            **{f"pct_outside_{t:g}": 100 * (dist > t).mean() for t in OUTSIDE_THRESHOLDS_UM},
            **{f"pct_outside_comp_{t:g}": 100 * (dist_allowed > t).mean() for t in OUTSIDE_THRESHOLDS_UM},
        }
    rows = []
    for (sk, condition, thr, penalty), d in per_contact.groupby(
        ["skid", "condition", "threshold_um", "penalty"], dropna=False
    ):
        rows.append({"skid": sk, "condition": condition_label(condition, thr, penalty),
                     **pct_outside(d["dist_final_um"], d["dist_final_allowed_um"])})
    for sk, d in per_contact[(per_contact["condition"] == "null") & (per_contact["sample"] == 0)].groupby("skid"):
        rows.append({"skid": sk, "condition": "observed",
                     **pct_outside(d["dist_original_um"], d["dist_original_allowed_um"])})
    return pd.DataFrame(rows)


neuron_df = neuron_df.drop(columns=[c for c in neuron_df if c.startswith("pct_outside")]).merge(
    outside_overview(per_contact_df), on=["skid", "condition"], how="left"
)
with pd.option_context("display.width", 220, "display.max_rows", None):
    print(neuron_df.groupby("condition")[[c for c in neuron_df if c.startswith("pct_outside")]]
          .mean().reindex(plot_conditions).round(1).to_string())

fig, axes = plt.subplots(2, len(OUTSIDE_THRESHOLDS_UM),
                         figsize=(max(12, 0.6 * len(plot_conditions) * len(OUTSIDE_THRESHOLDS_UM)), 7.5),
                         sharex=True, sharey=True, squeeze=False)
for c, t in enumerate(OUTSIDE_THRESHOLDS_UM):
    plot_neuron_metric(axes[0, c], f"pct_outside_{t:g}")
    plot_neuron_metric(axes[1, c], f"pct_outside_comp_{t:g}")
    axes[0, c].set_title(f"threshold {t:g} µm", fontsize=9)
for r, ylabel in enumerate(("% outside (any comp.)", "% outside (allowed comp. only)")):
    axes[r, 0].set_ylabel(ylabel, fontsize=8)
for ax in axes[-1]:
    ax.set_xticks(x)
    ax.set_xticklabels(plot_conditions, rotation=45, ha="right", fontsize=7)
fig.suptitle(f"site-target pairings outside the threshold: one dot per neuron ({N_SAMPLES} chains "
             f"pooled), bar = mean over {len(SWEEP_SKIDS)} neurons", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))

if SAVE_DIR is not None:
    for number in plt.get_fignums():
        plt.figure(number).savefig(SAVE_DIR / f"track_contacts_fig{number}.png", dpi=120, bbox_inches="tight")
plt.show()

# %%
