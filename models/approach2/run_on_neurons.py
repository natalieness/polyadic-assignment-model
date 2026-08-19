from pathlib import Path
import os
import sys
import ast

import pandas as pd


# Ensure the repository is importable and relative data paths work.
REPO_ROOT = Path("/Users/nessn/git-repos/polyadic-assignment-model")
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)
from models.validate_predictions_functions import get_target_co_occurrence
from models.approach1.data import DataRepository
from models.approach2 import Approach2Config, run_attribution_experiment



config = Approach2Config.load(
    REPO_ROOT / "models/approach2/sover1_config.json"
)

#%% Use fewer samples while testing.
config.sampler.prediction_samples = 10

# try different settings for geometry weight 
# config.geometry.bandwidth_um = 10.0

# examine effect of different sampler 
config.sampler.mcmc_steps_per_contact = 200
config.sampler.temperature = 0.1
config.sampler.geometry_weight = 2.0
    # "temperature": 1.0,
    # "mcmc_steps_per_contact": 20,
    # "max_null_restarts": 100


repository = DataRepository(config.paths)

celltype_df = pd.read_csv('data/celltype_df.csv')
celltype_df['skids'] = celltype_df['skids'].apply(ast.literal_eval)

skids = celltype_df[celltype_df.name == 'MBONs']['skids'].values[0]

conditions = {}
null_samples = {}
geometry_samples = {}
cotargeting_samples = {}
combined_samples = {}
targets = {}

for e, sk in enumerate(skids[:5]):
    print(f'Running neuron # {e+1}')

    result = run_attribution_experiment(
        repository,
        skeleton_id=sk,
        config=config,
    )

    # Four-row summary: null, geometry, co-targeting and combined.
    conditions[sk] = result.condition_summary

    # One row per target pair with the attribution results.
    # pair_df = result.pair_attribution
    targets[sk] = result.inputs.target_ids

    # Exact integer samples, each shaped: samples × sites × targets.
    null_samples[sk] = result.samples["null"]
    geometry_samples[sk] = result.samples["geometry"]
    cotargeting_samples[sk] = result.samples["cotargeting"]
    combined_samples[sk] = result.samples["combined"]

#%%
cdf = pd.concat(conditions, names=["skeleton_id"]).reset_index(level='skeleton_id').reset_index(drop=True)


# %%
from itertools import chain
import numpy as np 
def get_assignment_matrix(hyperedges):
    u_targets = set(list(chain.from_iterable(hyperedges)))
    u_targets = list(u_targets)
    n_targets = len(u_targets)
    target_map = {t:i for i,t in enumerate(u_targets)}
    n_sites = len(hyperedges)
    X = np.zeros((n_sites, n_targets), dtype=int)
    for i, edge in enumerate(hyperedges):
        for t in edge:
            X[i, target_map[t]] += 1
    return X, u_targets



def get_co_targeting_matrix(con_sub, skeleton_id, presyn_col='presynaptic_to', postsyn_col='postsynaptic_to'):
    # assume postsynaptic_to already a list 
    hyperedges = con_sub.loc[con_sub[presyn_col]==skeleton_id][postsyn_col].tolist()
    assignment_matrix, u_targets = get_assignment_matrix(hyperedges)
    co_targeting_matrix = get_target_co_occurrence(assignment_matrix)
    return co_targeting_matrix, u_targets

def compute_cosine(real_flat, padj_perm):
    pred_flat = padj_perm[np.triu_indices_from(padj_perm, k=1)]
    dot_product = np.dot(real_flat, pred_flat)
    norm_real = np.linalg.norm(real_flat)
    norm_pred = np.linalg.norm(pred_flat)
    if norm_real == 0 or norm_pred == 0:
        return np.nan
    else:
        return (dot_product / (norm_real * norm_pred))

con = pd.read_csv('data/con_short_2026_correctnodes.csv')
con['postsynaptic_to'] = con['postsynaptic_to'].apply(ast.literal_eval)
import matplotlib.pyplot as plt
import seaborn as sns
cos_sims = {}
# padj_perms = {}
for e, sk in enumerate(skids[:5]):
    padj_real, u_targets = get_co_targeting_matrix(con, sk)
    padj_real = pd.DataFrame(padj_real, index=u_targets, columns=u_targets)

    ordered_targets = targets[sk]
    padj_real_reindexed = padj_real.reindex(index=ordered_targets, columns=ordered_targets)
    real_flat = padj_real_reindexed.values[np.triu_indices_from(padj_real_reindexed, k=1)]

    cos_sims[sk]  = {}
    cos_sims[sk]['null'] = []
    cos_sims[sk]['geometry'] = []
    cos_sims[sk]['cotargeting'] = []
    cos_sims[sk]['combined'] = []
    padj_perms_null = []
    padj_perms_combined = []
    for nsamp, gsamp, cosamp, combsamp in zip(null_samples[sk], geometry_samples[sk], cotargeting_samples[sk], combined_samples[sk]):
        padj_perm_nsamp = get_target_co_occurrence(nsamp)
        padj_perm_gsamp = get_target_co_occurrence(gsamp)
        padj_perm_cosamp = get_target_co_occurrence(cosamp)
        padj_perm_combsamp = get_target_co_occurrence(combsamp)
        padj_perms_null.append(padj_perm_nsamp)
        padj_perms_combined.append(padj_perm_combsamp)
        # compute cosine similarity
        cos_sims[sk]['null'].append(compute_cosine(real_flat, padj_perm_nsamp))
        cos_sims[sk]['geometry'].append(compute_cosine(real_flat, padj_perm_gsamp))
        cos_sims[sk]['cotargeting'].append(compute_cosine(real_flat, padj_perm_cosamp))
        cos_sims[sk]['combined'].append(compute_cosine(real_flat, padj_perm_combsamp))
    mean_null = np.mean(np.stack(padj_perms_null), axis=0)
    mean_combined = np.mean(np.stack(padj_perms_combined), axis=0)

    fig, ax = plt.subplots(1,3, figsize=(15, 5))
    sns.heatmap(padj_real_reindexed, ax=ax[0])
    sns.heatmap(mean_null, ax=ax[1])
    sns.heatmap(mean_combined, ax=ax[2])

        
cs = (
    pd.concat(
        {
            sk: pd.DataFrame(similarities)
            for sk, similarities in cos_sims.items()
        },
        names=["sk", "sample"]
    )
    .reset_index()
)


### plot the actual padj matrices

# %% check quality of sampler results in terms of geometric adherence 

ex_sk = skids[4]
print(f'Checking geometric adherence for skeleton {ex_sk}')
print(f"Final average cos sims: {cs.groupby('sk').mean()}")

geo = geometry_samples[ex_sk]

geo1 = geo[0] # shape (connectors, targets)

from models.approach2.geometry import geometry_scores
from models.approach2.data import FixedNeuronInputs, load_observed_inputs

inputs_sk = load_observed_inputs(repository, int(ex_sk))
geometry_scores = geometry_scores(
    repository=DataRepository(config.paths),
    inputs=inputs_sk,
    config=config.geometry,
)


#%% get observed distances from backbones 
obs_ass = inputs_sk.observed_assignment
distances_um = geometry_scores.distances_um


def get_observed_distances(obs_ass, distances_um):
    # get any double instances twice 
    obs_distances = (obs_ass > 0) * distances_um
    obs_distances_chain = obs_distances[obs_distances > 0].flatten()
    n_obs = obs_ass[obs_ass > 0].flatten()
    obs_distances_repeated = np.repeat(obs_distances_chain, n_obs)
    return obs_distances_repeated

# get real 
dist_real = get_observed_distances(obs_ass, distances_um)
# get distances from each geometry sample 
dist_geo_samples = [get_observed_distances(geo_ass, distances_um) for geo_ass in geo]


# log_scale = True
# for s, dist_geo in enumerate(dist_geo_samples):
#     fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#     max_max = max([dist_real.max()] + [d.max() for d in dist_geo_samples])
#     bins = np.linspace(0, max_max, 30)
#     ax.hist(dist_real, bins=bins, alpha=0.5, label='Observed', density=False)
#     ax.hist(dist_geo, bins=bins, alpha=0.5, label=f'Geometry Sample {s+1}', density=False)
#     ax.legend()
#     if log_scale:
#         ax.set_yscale('log')
#         ax.set_ylabel('log(count)')
#     else:
#         ax.set_ylabel('count')
#     ax.set_xlabel('Distance (um)')
#%%
# examine correlation between dist and scores 

geo_scores = geometry_scores.scores
geo_scores_flat = geo_scores.flatten()
geo_dists_flat = distances_um.flatten()
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.scatter(geo_dists_flat, geo_scores_flat, alpha=0.5, s=2)
ax.set_xlabel('Distance (um)')
ax.set_ylabel('Geometry Score')

# by site 
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
for site_idx in range(1):
    print(distances_um[site_idx, :].shape)
    ax.scatter(distances_um[site_idx, :], geo_scores[site_idx, :], alpha=0.5)
ax.set_title('Geometry Scores vs Distances for an example site')
ax.set_xlabel('Distance (um)')
ax.set_ylabel('Geometry Score')
# target-specific columns-centered geometry scores

# by target 
xmax = np.nanmax(distances_um)
ymax = np.nanmax(geo_scores)
ymin = np.nanmin(geo_scores)
for site_idx in range(2):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    print(distances_um[:, site_idx].shape)
    ax.scatter(distances_um[:, site_idx], geo_scores[:, site_idx], alpha=0.5)
    ax.set_title('Geometry Scores vs Distances for an example target')
    ax.set_xlabel('Distance (um)')
    ax.set_ylabel('Geometry Score')
    ax.set_xlim(0, xmax)
    ax.set_ylim(ymin, ymax)

# %%
# score the geomery samples by mean, median max dist 
print(f"Observed distances: mean {np.mean(dist_real):.2f}, median {np.median(dist_real):.2f}, max {np.max(dist_real):.2f}")
dist_geo_chain = list(chain.from_iterable(dist_geo_samples))
print(f"Geometry sample distances: mean {np.mean(dist_geo_chain):.2f}, median {np.median(dist_geo_chain):.2f}, max {np.max(dist_geo_chain):.2f}")
print(f"cos sim: {cs.mean(axis=0)['geometry']}")

# assessment could include fraction within observed
obs_max = np.max(dist_real)
geo_within_obs = [len(dist_geo[dist_geo <= obs_max]) / len(dist_geo) for dist_geo in dist_geo_samples]
print(f"Fraction of geometry sample distances within observed max: {np.median(geo_within_obs):.2f} (median), {np.mean(geo_within_obs):.2f} (mean)")

# %%
