from pathlib import Path
import os
import sys
import ast
import time 

import pandas as pd
import numpy as np
from itertools import chain


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
config.sampler.random_seed = 84
config.cotarget.score_clip = None # try removing score clip 

# try different settings for geometry weight 
# config.geometry.bandwidth_um = 10.0

# examine effect of different sampler 
# config.sampler.mcmc_steps_per_contact = 200
# config.sampler.temperature = 0.1
# config.sampler.geometry_weight = 2.0
    # "temperature": 1.0,
    # "mcmc_steps_per_contact": 20,
    # "max_null_restarts": 100
config.cotarget.include_self_repeats = True

repository = DataRepository(config.paths)

celltype_df = pd.read_csv('data/celltype_df.csv')
celltype_df['skids'] = celltype_df['skids'].apply(ast.literal_eval)
neuron_names = pd.read_csv('data/skid_to_name_wMB.csv')
neuron_names['skid'] = neuron_names['skid'].astype(int)
skid_to_name = dict(zip(neuron_names['skid'], neuron_names['name']))


skids = celltype_df[celltype_df.name == 'MBONs']['skids'].values[0]
# for sensories try just orns 
# skids = [sk for sk, nam in skid_to_name.items() if ' ORN' in nam]

conditions = {}
null_samples = {}
geometry_samples = {}
cotargeting_samples = {}
combined_samples = {}
targets = {}

for e, sk in enumerate(skids[:5]):
    stime = time.time()
    try:
        result = run_attribution_experiment(
            repository,
            skeleton_id=sk,
            config=config,
        )
    except Exception as ex:
        print(f"Error processing skeleton {sk}: {ex}")
        continue

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
    print(f'Finished neuron # {e+1} in {time.time() - stime:.2f} s')

#%%
cdf = pd.concat(conditions, names=["skeleton_id"]).reset_index(level='skeleton_id').reset_index(drop=True)


### legacy plotting geometry scores vs distances etc etc 


ex_sk = skids[4]
print(f'Checking geometric adherence for skeleton {ex_sk}')
print(f"Final average cos sims: {cs.groupby('sk').mean()}")

geo = combined_samples[ex_sk]

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
for site_idx in range(1):
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


# assessment could include fraction within observed
obs_max = np.max(dist_real)
geo_within_obs = [len(dist_geo[dist_geo <= obs_max]) / len(dist_geo) for dist_geo in dist_geo_samples]
print(f"Fraction of geometry sample distances within observed max: {np.median(geo_within_obs):.2f} (median), {np.mean(geo_within_obs):.2f} (mean)")


# %%
