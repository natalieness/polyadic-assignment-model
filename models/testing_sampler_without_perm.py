from pathlib import Path
import os
import sys
import ast
import time 

import pandas as pd
from itertools import chain
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# Ensure the repository is importable and relative data paths work.
REPO_ROOT = Path("/Users/nessn/git-repos/polyadic-assignment-model")
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)
from models.validate_predictions_functions import get_target_co_occurrence
from models.padj_functions import get_assignment_matrix, get_co_targeting_matrix, compute_cosine
from models.approach1.data import DataRepository
from models.approach2 import Approach2Config, run_attribution_experiment
from models.approach2.simulator import run_sampler_with_permutation


# get config and repository for sampler
config = Approach2Config.load(
    REPO_ROOT / "models/approach2/sover1_config.json"
)
repository = DataRepository(config.paths)

#%% Run on neurons
# get basic neuron details 
celltype_df = pd.read_csv('data/celltype_df.csv')
celltype_df['skids'] = celltype_df['skids'].apply(ast.literal_eval)
ct_df_long = celltype_df.explode('skids')
skid_to_celltype = dict(zip(ct_df_long['skids'], ct_df_long['name']))
neuron_names = pd.read_csv('data/skid_to_name_wMB.csv')
neuron_names['skid'] = neuron_names['skid'].astype(int)
skid_to_name = dict(zip(neuron_names['skid'], neuron_names['name']))
all_neurons = neuron_names['skid'].values # all neurons of interest
hemi_df = pd.read_csv('data/skid_hemisphere.csv')
hemi_df['skid'] = hemi_df['skid'].astype(int)
left_neus = hemi_df.loc[hemi_df['hemisphere']=='left']['skid'].values
right_neus = hemi_df.loc[hemi_df['hemisphere']=='right']['skid'].values
center_neus = hemi_df.loc[hemi_df['hemisphere']=='center']['skid'].values

pairs = pd.read_csv('data/pairs-2022-02-14.csv')
pairs['leftid'] = pairs['leftid'].astype(int)
pairs['rightid'] = pairs['rightid'].astype(int)
pairs_grouped = pairs.groupby('leftid')['rightid'].apply(list)
pairs_grouped = pairs_grouped.reset_index()
pairs_grouped['pair_id'] = pairs_grouped.index
pairs_grouped_e = pairs_grouped.explode('rightid')
skid_to_pair = dict(zip(pairs_grouped_e['rightid'], pairs_grouped_e['pair_id']))
skid_to_pair.update(dict(zip(pairs_grouped['leftid'], pairs_grouped['pair_id'])))

con = pd.read_csv('data/con_short_2026_correctnodes.csv')
con['postsynaptic_to'] = con['postsynaptic_to'].apply(ast.literal_eval)

# decide which neurons to run on
skids = all_neurons
# skids = list(chain.from_iterable([celltype_df[celltype_df.name == ct_nam]['skids'].values[0] for ct_nam in ['LHNs','PNs',  'MBINs']]))
# skids = celltype_df[celltype_df.name == 'MBONs']['skids'].values[0]
# for sensories try just orns 
# skids = [sk for sk, nam in skid_to_name.items() if ' ORN' in nam]
print(f'Running on {len(skids)} neurons')

#%%

skid = skids[20]
temp = 1.0
steps = 5
config.sampler.temperature = temp
config.sampler.mcmc_steps_per_contact = steps
config.sampler.null_burnin_steps_per_contact = steps
result = run_sampler_with_permutation(repository, skid, config)
result_att = run_attribution_experiment(repository, skid, config)

#%%
def get_observed_distances(obs_ass, distances_um):
    # get any double instances twice 
    obs_distances = (obs_ass > 0) * distances_um
    obs_distances_chain = obs_distances[obs_distances > 0].flatten()
    n_obs = obs_ass[obs_ass > 0].flatten()
    obs_distances_repeated = np.repeat(obs_distances_chain, n_obs)
    return obs_distances_repeated

for res, res_name in zip([result, result_att], ['Result from null', 'Result from real']):
    dists = res.geometry.distances_um # sites x targets 

    obs = res.inputs.observed_assignment
    null = res.samples['null']
    geo = res.samples['geometry']
    co = res.samples['cotargeting']

    dist_real = get_observed_distances(obs, dists)
    dist_null = np.stack([get_observed_distances(n, dists) for n in null])
    dist_geo = np.stack([get_observed_distances(n, dists) for n in geo])
    dist_co = np.stack([get_observed_distances(n, dists) for n in co])

    max_real = np.max(dist_real)

    sampled_dists = [dist_null, dist_geo, dist_co]
    max_dist = np.max(sampled_dists)
    bins = np.linspace(0, max_dist, 30)
    fig, ax = plt.subplots(1, 3, figsize=(8, 3))
    for a, comp_d, comp_name in zip(ax, [dist_null, dist_geo, dist_co], ['Null', 'Geometry', 'Cotargeting']):
        frac_within = (comp_d.flatten() <= max_real).sum() / comp_d.flatten().shape[0]
        a.set_title(f'{comp_name}, frac within: {frac_within:.3f}')
        sns.histplot(dist_real, ax=a, bins=bins, stat='density', label='Observed', color='blue')
        sns.histplot(comp_d.flatten(), ax=a, bins=bins, stat='density', label='Sample', color='orange')
    fig.suptitle(f'{res_name}, temp {temp}, skeleton {skid}')
    fig.tight_layout()


# %%
# get local max target distances for each neuron
lmt = pd.read_csv('data/max_presyn_to_backbone_distances_by_skid.csv')
lmt['presynaptic_to'] = lmt['presynaptic_to'].astype(int)

# %%
