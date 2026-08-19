
''' sweep over parameter space to assess geometry sampler performance '''

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

def run_experiment_for_sk(repository, sk, config):
    result = run_attribution_experiment(
        repository,
        skeleton_id=sk,
        config=config,
    )

    return result.samples['geometry'], result.samples['null']

def get_observed_distances(obs_ass, distances_um):
    # get any double instances twice 
    obs_distances = (obs_ass > 0) * distances_um
    obs_distances_chain = obs_distances[obs_distances > 0].flatten()
    n_obs = obs_ass[obs_ass > 0].flatten()
    obs_distances_repeated = np.repeat(obs_distances_chain, n_obs)
    return obs_distances_repeated


'''' SWEEP over:
-- sampler.temperature 
-- sampler.geometry_weight 
-- sampler.mcmc_steps_per_contact
-- sampler.max_null_restarts 

'''

possible_sweep_vals = [
    [0.05, 0.1], # temp
    [1.0, 2.0], # geometry weight
    [20,  200, 500], # mcmc steps
    [100], # max null restarts
]
default_vals = [0.1, 2.0, 20, 100]

# get unique combinations of sweep values
sweep_type = 'control' 
if sweep_type == 'sweep':
    sweep_vals = list(product(*possible_sweep_vals))
elif sweep_type == 'control':
    
    # combine all default, with altering all others
    sweep_vals = []
    for i, vals in enumerate(possible_sweep_vals):
        for val in vals:
            new_val = default_vals.copy()
            new_val[i] = val
            sweep_vals.append(tuple(new_val))

print(f"Running {len(sweep_vals)} combinations of sweep values.")

# get the real stuff 
inputs_sk = load_observed_inputs(repository, int(ex_sk))
geometry_scores = geometry_scores(
    repository=DataRepository(config.paths),
    inputs=inputs_sk,
    config=config.geometry,
)
obs_ass = inputs_sk.observed_assignment
distances_um = geometry_scores.distances_um
geo_scores = geometry_scores.scores
# get real 
dist_real = get_observed_distances(obs_ass, distances_um)
obs_max = np.max(dist_real)

evalres = []
for e, sweep_val in enumerate(sweep_vals):
    print(f"Running sweep {e+1}/{len(sweep_vals)}: {sweep_val}")
    # set all the config values to the sweep value
    config.sampler.temperature = sweep_val[0]
    config.sampler.geometry_weight = sweep_val[1]
    config.sampler.mcmc_steps_per_contact = sweep_val[2]
    config.sampler.max_null_restarts = sweep_val[3]

    geos, nulls = run_experiment_for_sk(repository, ex_sk, config)

    # get distances from each geometry sample 
    dist_geo_samples = [get_observed_distances(geo_ass, distances_um) for geo_ass in geos]

    dist_geo_chain = list(chain.from_iterable(dist_geo_samples))

    geo_within_obs = [len(dist_geo[dist_geo <= obs_max]) / len(dist_geo) for dist_geo in dist_geo_samples]

    evalres.append({'mean': np.mean(dist_geo_chain), 'median': np.median(dist_geo_chain),  'max': np.max(dist_geo_chain), 'min': np.min(dist_geo_chain), 'frac_within_obs': np.mean(geo_within_obs)})

results_df = pd.DataFrame(evalres)

#%%
sweep_val_arr = np.array(sweep_vals)
results_df['temp'] = sweep_val_arr[:, 0]
results_df['geo_weight'] = sweep_val_arr[:, 1]
results_df['mcmcsteps'] = sweep_val_arr[:, 2]
results_df['null_restarts'] = sweep_val_arr[:, 3]

# gotta find a way to filter for all the default ones only 
for i, col in enumerate(['temp', 'geo_weight', 'mcmcsteps', 'null_restarts']):
    other_cols = [c for j, c in enumerate(['temp', 'geo_weight', 'mcmcsteps', 'null_restarts']) if j != i]
    results_df[f'all_default_except_{col}'] = np.all([(results_df[c] == default_vals[i_c]) for i_c, c in enumerate(['temp', 'geo_weight', 'mcmcsteps', 'null_restarts']) if c != col], axis=0)


fig, ax = plt.subplots(2, 2, figsize=(6, 5))
ax = ax.flatten()
for i, col in enumerate(['temp', 'geo_weight', 'mcmcsteps', 'null_restarts']):
    sub = results_df[results_df[f'all_default_except_{col}']]
    ax[i].scatter(sub[col], sub['frac_within_obs'], s=20)
    ax[i].set_xlabel(col)
    ax[i].set_ylabel('mean frac within obs')
fig.tight_layout()

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
# fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# for site_idx in range(1):
#     print(distances_um[site_idx, :].shape)
#     ax.scatter(distances_um[site_idx, :], geo_scores[site_idx, :], alpha=0.5)
# ax.set_title('Geometry Scores vs Distances for an example site')
# ax.set_xlabel('Distance (um)')
# ax.set_ylabel('Geometry Score')
# # target-specific columns-centered geometry scores

# # by target 
# xmax = np.nanmax(distances_um)
# ymax = np.nanmax(geo_scores)
# ymin = np.nanmin(geo_scores)
# for site_idx in range(2):
#     fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#     print(distances_um[:, site_idx].shape)
#     ax.scatter(distances_um[:, site_idx], geo_scores[:, site_idx], alpha=0.5)
#     ax.set_title('Geometry Scores vs Distances for an example target')
#     ax.set_xlabel('Distance (um)')
#     ax.set_ylabel('Geometry Score')
#     ax.set_xlim(0, xmax)
#     ax.set_ylim(ymin, ymax)

# %%
a = np.array([4]*3+[30]*3)
b = np.array([4]*5+[30]*1)
c = np.array([4]*1+[30]*5)
array = [a, b, c]
ra = []
for arr in array:
    kernel = (arr <= (10)).astype(float)
    ra.append(kernel - kernel.mean())

# %%
