from pathlib import Path
import os
import sys
import ast
import time 

import pandas as pd
from itertools import chain
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm

# Ensure the repository is importable and relative data paths work.
REPO_ROOT = Path("/Users/nessn/git-repos/polyadic-assignment-model")
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)


### for figs 
FIG_OUT = 'sample_output_analysis/figs_shapley_sampled/'
mpl.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 18,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 2,
    'xtick.major.width': 2,
    'ytick.major.width': 2,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    "font.family": ["DejaVu Sans"],      # exact family
    "font.sans-serif": ["DejaVu Sans"],  # avoid fallback stack error in adobe
    "mathtext.fontset": "dejavusans", 
    'svg.fonttype': 'none',
    'text.usetex': False,
    'savefig.dpi': 300,
    'savefig.transparent': True,
})

# coolwarm's midpoint is light grey; force it to true white so zero-difference cells
# read as empty rather than as a measured value on a transparent background
_base_cw = mpl.colormaps['coolwarm']
coolwarm_white = LinearSegmentedColormap.from_list(
    'coolwarm_white',
    [_base_cw(0.0), _base_cw(0.25), (1, 1, 1), _base_cw(0.75), _base_cw(1.0)], N=256,
)
coolwarm_white.set_bad('lightgrey')  # keep NaNs distinguishable from true zeros

from perm_functions.polyadic_edge_permutation import get_perms

from models.padj_functions import get_co_targeting_matrix, compute_cosine
from models.approach2.propensity import pair_count_matrix, mean_pair_count_matrix
from models.approach2.simulator import _decode_array
# NOTE; main difference between pair_count_matric and get_target_co_occurrence is
# pair_count _matrix only counts 1 co-targeting instance if one target occurs >1 in a site. 

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
skid_to_hemi = dict(zip(hemi_df['skid'], hemi_df['hemisphere']))

adj = pd.read_csv('data/adj.csv')
adj.set_index('sources', inplace=True)
adj = adj.astype(int)
adj.columns = adj.columns.astype(int)
adj.index = adj.index.astype(int)

flow_neurons = pd.read_csv('data/flow_neurons.csv')
flow_neurons['skid'] = flow_neurons['skid'].astype(int)
flow_neurons['flow_score'] = flow_neurons['flow_score'].astype(float)
skid_to_flow = dict(zip(flow_neurons['skid'], flow_neurons['flow_score']))


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

#for fields in con.columns:
presyn_col = 'presynaptic_to'
postsyn_col = 'postsynaptic_to'

def get_inhibitory_neurons(threshold=0.0, verbose=False):
    # load neurotransmitter predictions
    neurons_nt = pd.read_csv('data/20260205_corrected_nts.csv')

    # inhibitory neurons include GABA and Glutamate
    inhibitory_nts = ['GABA', 'Glutamate']

    # first filter by confidence threshold
    neurons_nt_filtered = neurons_nt[neurons_nt['neurotransmitter_confidence'] >= threshold]
    inhibitory_neurons = neurons_nt_filtered[neurons_nt_filtered['predicted_neurotransmitter'].isin(inhibitory_nts)]['skeleton_id'].to_list()
    if verbose:
        print(f'Found {len(inhibitory_neurons)} inhibitory neurons with confidence >= {threshold}')
    return inhibitory_neurons

inhibitory_neurons = get_inhibitory_neurons(threshold=0.0, verbose=True)

# sensory modality 
sensory_neurons = pd.read_csv('data/sensory_neurons.csv')
sensory_neurons['skids'] = sensory_neurons['skids'].apply(ast.literal_eval)
skid_to_mod = dict()
for _, row in sensory_neurons.iterrows():
    for skid in row['skids']:
        skid_to_mod[skid] = row['modality']

sensory_neurons['2nd_order'] = sensory_neurons['2nd_order'].apply(ast.literal_eval)
skid_to_2ndorder = dict()

# ensure the dictionary is populated before iterating
for _, row in sensory_neurons.iterrows():
    for skid in row['2nd_order']:
        if skid in skid_to_2ndorder:
            skid_to_2ndorder[skid] = 'mixed'
        else:
            skid_to_2ndorder[skid] = row['modality']



#%% load geometry samples and real samples

from models.approach2.propensity import pair_count_matrix, mean_pair_count_matrix

# get cotargeting numbers for each neuron 

import json
import os
# do this on average geometry sample vs real observed assignment for all postsynaptic neurons. 
INPUT = 'out/approach2/shapley_run_with_individual_prop_no_multiples_temp1_weights2/'

def load_geo_samples(INPUT):
    files = [f for f in os.listdir(INPUT) if f.endswith('.json')]
    n_files = len(files)
    geo_samples = {}
    real_samples = {}
    target_skids = {}
    null_samples = {}
    ind_samples = {}
    combined_samples = {}
    for e, f in enumerate(files):
        if e % 100 == 0:
            print(f'Loading file {e+1}/{n_files}')
        with open(os.path.join(INPUT, f), 'r') as infile:
            data = json.load(infile)
            f_skid = int(f.replace('.json', ''))
            if e == 0:
                print(f'Loading first file: {f}')
                # get parameters 
                print(data['inputs'])

            null_samples[f_skid] = np.array(data['samples']['null']['data'])
            geo_samples[f_skid] = np.array(data['samples']['geometry']['data'])
            real_samples[f_skid] = np.array(data['inputs']['observed_assignment']['data'])
            target_skids[f_skid] = np.array(data['inputs']['target_ids']['data'])
            ind_samples[f_skid] = np.array(data['samples']['cotargeting']['data'])
            combined_samples[f_skid] = np.array(data['samples']['combined']['data'])

    return geo_samples, real_samples, target_skids, null_samples, ind_samples, combined_samples

geo_samples, real_samples, target_skids, null_samples, individual_samples, combined_samples = load_geo_samples(INPUT)


all_presyn = list(target_skids.keys())

def assigned_to_cooc(S):
    # S is  (n_sample, sites, targets)
    # this is a batched matmul; einsum computes it with its own generic loop, while @
    # dispatches to BLAS - but only for floats, so the cast is what makes it ~30x faster
    S = np.asarray(S, dtype=np.float64)
    cooc_all = S.transpose(0, 2, 1) @ S                # (n_samples, targets, targets)
    self_pairs = (S * (S - 1) / 2).sum(axis=1)         # (n_samples, targets)
    idx = np.arange(S.shape[2])
    cooc_all[:, idx, idx] = self_pairs
    return cooc_all

#%% 

kc_skids = celltype_df[celltype_df['name']=='MBONs']['skids'].values[0]
kc_skids = [skid for skid in kc_skids if skid in all_presyn]

kc_geo_samples = {skid: geo_samples[skid] for skid in kc_skids}
kc_real_samples = {skid: real_samples[skid] for skid in kc_skids}

# get average co-occurrence matrix for KCs
kc_geo_cooc = {skid: assigned_to_cooc(kc_geo_samples[skid]) for skid in kc_skids}
kc_real_cooc = {skid: assigned_to_cooc(kc_real_samples[skid][None, ...])[0] for skid in kc_skids}

avg_kc_geo_cooc = {skid: np.mean(kc_geo_cooc[skid], axis=0) for skid in kc_skids}

# %% plot some example

# reindex to all possible targets
kc_target_skids = [target_skids[skid] for skid in kc_skids]
all_targets = np.unique(np.concatenate(kc_target_skids))

def reindex_cooc(cooc, original_targets, all_targets):
    idx_map = {t: i for i, t in enumerate(original_targets)}
    n_all = len(all_targets)
    cooc_reindexed = np.zeros((n_all, n_all))
    for i, t1 in enumerate(all_targets):
        if t1 not in idx_map:
            continue
        for j, t2 in enumerate(all_targets):
            if t2 not in idx_map:
                continue
            cooc_reindexed[i, j] = cooc[idx_map[t1], idx_map[t2]]
    return cooc_reindexed

kc_real_cooc_reindexed = {skid: reindex_cooc(kc_real_cooc[skid], target_skids[skid], all_targets) for skid in kc_skids}
kc_geo_cooc_reindexed = {skid: reindex_cooc(avg_kc_geo_cooc[skid], target_skids[skid], all_targets) for skid in kc_skids}

kc_real_cooc_sum = sum(kc_real_cooc_reindexed.values())
kc_geo_cooc_sum = sum(kc_geo_cooc_reindexed.values())

summed_real = pd.DataFrame(kc_real_cooc_sum, index=all_targets, columns=all_targets)
summed_geo = pd.DataFrame(kc_geo_cooc_sum, index=all_targets, columns=all_targets)

# %%

# aggregate by celltype 
target_ct = [skid_to_celltype.get(s, 'unknown') for s in all_targets]

def aggregate_by_group(mat, groups, type='sum'):
    # group both axes; relabelling and using level= avoids the deprecated axis=1 groupby
    mat = mat.set_axis(groups, axis=0).set_axis(groups, axis=1)
    if type == 'sum':
        return mat.groupby(level=0).sum().T.groupby(level=0).sum().T
    elif type == 'mean':
        return mat.groupby(level=0).mean().T.groupby(level=0).mean().T
    else:
        raise ValueError(f"Unsupported aggregation type: {type}")

real_ct_sum = aggregate_by_group(summed_real, target_ct)
geo_ct_sum = aggregate_by_group(summed_geo, target_ct)

def plot_heatmap_sidebyside(real_ct_sum, geo_ct_sum):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.heatmap(real_ct_sum, ax=axes[0], cmap='viridis')
    axes[0].set_title('Real Co-occurrence by Cell Type')
    sns.heatmap(geo_ct_sum, ax=axes[1], cmap='viridis')
    axes[1].set_title('Geo Co-occurrence by Cell Type')
    plt.tight_layout()

def plot_heatmap_difference(diff_ct_sum, skid_to_name=skid_to_name, tickfont=8, figsize=(7, 6),
                            cmap=coolwarm_white, log=False, linthresh=1.0, vmax=None):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # symmetric limits so equal colour distance means equal magnitude either side of zero
    v = vmax if vmax is not None else np.nanmax(np.abs(diff_ct_sum.values))
    if log:
        # log-scaled tails, linear within +/-linthresh, so zero stays defined and centred
        norm_kwargs = dict(norm=SymLogNorm(linthresh=linthresh, vmin=-v, vmax=v))
    else:
        norm_kwargs = dict(vmin=-v, vmax=v, center=0)
    sns.heatmap(diff_ct_sum, ax=ax, cmap=cmap, yticklabels=1, xticklabels=1, **norm_kwargs)
    ax.set_title('Difference to geometry sampled')
    ax.set_xticklabels([skid_to_name.get(s, s) for s in diff_ct_sum.columns], rotation=90, fontsize=tickfont)
    ax.set_yticklabels([skid_to_name.get(s, s) for s in diff_ct_sum.index], rotation=0, fontsize=tickfont)

    plt.tight_layout()
    return fig


plot_heatmap_sidebyside(real_ct_sum, geo_ct_sum)

real_ct_mean = aggregate_by_group(summed_real, target_ct, type='mean')
geo_ct_mean = aggregate_by_group(summed_geo, target_ct, type='mean')

# plot mean co-occurrence by cell type
plot_heatmap_sidebyside(real_ct_mean, geo_ct_mean)

ct_sum_diff = real_ct_sum - geo_ct_sum
plot_heatmap_difference(ct_sum_diff, tickfont=14)

# subgroup mb neurons further by DANs and OANs 

#%% find interesting ones 
diff = summed_real - summed_geo
# get top 30 idx, col combinations
top_30_idx_col = diff.unstack().abs().sort_values(ascending=False).head(30).index.tolist()
named_top_30_idx_col = [(skid_to_name.get(idx, idx), skid_to_name.get(col, col)) for idx, col in top_30_idx_col]


# %% plot summed_real and summed_geo and diff side by side with specific neurons only 


include_only_x = [  'MBINs']
targets_filtered_x = [s for s in all_targets if skid_to_celltype.get(s, 'unknown') in include_only_x]

include_only_y = [ 'MBINs']
targets_filtered_y = [s for s in all_targets if skid_to_celltype.get(s, 'unknown') in include_only_y]

colorbar_scale_log = True 
order_by_compartment = True
def name_to_comp(name):
    x = name.replace('MBON-', '').replace(' right', '').replace(' left', '').replace('MBIN-', '').replace('DAN-', '').replace('OAN-', ' ')
    return x



# order by cell type
targets_filtered_x.sort(key=lambda s: include_only_x.index(skid_to_celltype.get(s, 'unknown')))
targets_filtered_y.sort(key=lambda s: include_only_y.index(skid_to_celltype.get(s, 'unknown')))

targets_filtered_x.sort(key=lambda s: name_to_comp(skid_to_name.get(s, s)))
targets_filtered_y.sort(key=lambda s: name_to_comp(skid_to_name.get(s, s)))

summed_real_filtered = summed_real.loc[targets_filtered_x, targets_filtered_y]
summed_geo_filtered = summed_geo.loc[targets_filtered_x, targets_filtered_y]
diff_filtered = summed_real_filtered - summed_geo_filtered
fig = plot_heatmap_difference(diff_filtered, figsize=(5, 7))

exclude_specific = [8980589, 16223537]
targets_filtered_x = [s for s in targets_filtered_x if s not in exclude_specific]
targets_filtered_y = [s for s in targets_filtered_y if s not in exclude_specific]

summed_real_filtered = summed_real.loc[targets_filtered_x, targets_filtered_y]
summed_geo_filtered = summed_geo.loc[targets_filtered_x, targets_filtered_y]
diff_filtered = summed_real_filtered - summed_geo_filtered
fig = plot_heatmap_difference(diff_filtered, figsize=(6,6))


# %%
