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

#%% examine all neurons and left/right split pairs for all options 

# get real vs sampled cosine similarlity for each neuron & sample type & get average and sd of that 

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

def cosine_upper_triangle(matrix_a, matrix_b):
    # k=0 keeps the diagonal, so same-target repeats count towards the similarity.
    if matrix_a.shape != matrix_b.shape:
        raise ValueError("Both matrices must have the same shape.")
    if matrix_a.ndim != 2 or matrix_a.shape[0] != matrix_a.shape[1]:
        raise ValueError("Both matrices must be square.")
    triangle = np.triu_indices_from(matrix_a, k=0)
    flat_a = matrix_a[triangle]
    flat_b = matrix_b[triangle]
    norm_a = np.linalg.norm(flat_a)
    norm_b = np.linalg.norm(flat_b)
    if norm_a == 0 or norm_b == 0:
        return np.nan
    else:
        return float(np.dot(flat_a, flat_b) / (norm_a * norm_b))

def cosine_upper_triangle_batch(stack, reference):
    """Cosine similarity of each (n, k, k) matrix's upper triangle against one (k, k) reference."""
    rows, cols = np.triu_indices(reference.shape[0], k=0)
    flat_stack = stack[:, rows, cols]          # (n, K)
    flat_ref = reference[rows, cols]           # (K,)
    stack_norms = np.linalg.norm(flat_stack, axis=1)
    ref_norm = np.linalg.norm(flat_ref)
    with np.errstate(invalid='ignore', divide='ignore'):
        result = (flat_stack @ flat_ref) / (stack_norms * ref_norm)
    result[(stack_norms == 0) | (ref_norm == 0)] = np.nan
    return result


# for each presynaptic neuron, get co-padj for each 
only_major_targets = False # compute cosine similarity only for targets with more than major_threshold synapses
major_threshold = 3 

# get major targets from adj 
if only_major_targets:
    adj_bin = (adj >= major_threshold)
    major_targets = adj_bin.apply(lambda row: row.index[row].tolist(), axis=1).to_dict()



from concurrent.futures import ThreadPoolExecutor

# the neurons are independent, so this runs over a thread pool. threads rather than
# processes because the heavy work is numpy, which releases the GIL, and processes
# would have to pickle the whole sample dicts to every worker. each task only touches
# its own arrays and returns them, so there is no shared state to lock.
def process_pre(pre):
    # stack real sample, null_samples, and geo_samples for this presynaptic neuron for efficient computation
    # real has shape (x,y) all others already have shapes with (n_samples, x, y) so need to expand real_samples to match
    real_samples_expanded = real_samples[pre][None, :, :]           # (1, 58, 47)
    n_sampled = null_samples[pre].shape[0] # how many are sampled in the null and geo versions
    stacked_samples = np.concatenate([real_samples_expanded, null_samples[pre], geo_samples[pre], individual_samples[pre], combined_samples[pre]], axis=0)
    padj_stacked = assigned_to_cooc(stacked_samples)
    present_targets = list(target_skids[pre])
    nan_row = {'pre': pre, 'null_sims': np.nan, 'geo_sims': np.nan,
               'individual_sims': np.nan, 'combined_sims': np.nan}

    # if using only major_targets, can sort all of them together here
    if only_major_targets:
        major_targs = set(major_targets.get(pre, []))  # set, so membership below is O(1)
        if len(major_targs) <= 2:
            return pre, None, None, nan_row
        present_idx = [i for i, t in enumerate(present_targets) if t in major_targs]
        present_targets = [present_targets[i] for i in present_idx]
        padj_stacked = padj_stacked[:, present_idx, :][:, :, present_idx]

    # take an average for each type of sample; offsets named once rather than
    # re-deriving the same running sums four times
    o1 = 1 + n_sampled
    o2 = o1 + geo_samples[pre].shape[0]
    o3 = o2 + individual_samples[pre].shape[0]
    avg_sample_padjs = np.stack([np.nanmean(padj_stacked[a:b], axis=0)
                                 for a, b in [(1, o1), (o1, o2), (o2, o3), (o3, None)]])
    css = cosine_upper_triangle_batch(avg_sample_padjs, padj_stacked[0])   # (n_samples - 1,)
    return pre, padj_stacked, present_targets, {
        'pre': pre,
        'null_sims': css[0],
        'geo_sims': css[1],
        'individual_sims': css[2],
        'combined_sims': css[3]
    }

n_presyn = len(all_presyn)
print(f"Processing {n_presyn} presynaptic neurons")
with ThreadPoolExecutor(max_workers=8) as ex:
    out = list(ex.map(process_pre, all_presyn))   # map keeps input order, so this stays deterministic

# keyed by skid rather than position, so a skipped neuron cannot shift the others
padj_stacks = {pre: stack for pre, stack, _, _ in out}
padj_targets = {pre: targs for pre, _, targs, _ in out}
sims = pd.DataFrame([row for *_, row in out])

#%% filter and labelling 

# drop nans and keep for reference
nan_pres = sims[sims.isna().any(axis=1)]['pre']
sims = sims.dropna()

sims['name'] = sims['pre'].map(skid_to_name)
sims['pairid'] = sims['pre'].map(skid_to_pair)
sims['celltype'] = sims['pre'].map(skid_to_celltype)
sims['hemi'] = sims['pre'].map(skid_to_hemi)

# remove RGNs
sims = sims[sims['celltype'] != 'RGNs']

# chaacterise neurons by their inputs and outputs
sims['n_syns_total'] = sims['pre'].map(lambda pre: real_samples[pre].sum())
sims['n_syns_sites'] = sims['pre'].map(lambda pre: real_samples[pre].shape[0])
sims['u_outputs'] = sims['pre'].map(lambda pre: real_samples[pre].shape[1])


# axon and dendrite structure
import pymaid
from pymaid_creds import url, name, password, token
rm = pymaid.CatmaidInstance(url, token, name, password)

anns = ['mw contralateral axon', 'mw contralateral dendrite', 'mw bilateral axon', 'mw bilateral dendrite', 'mw ipsilateral axon','mw ipsilateral dendrite']

anns_dict = {ann: pymaid.get_skids_by_annotation(ann) for ann in anns}

sims['axon_structure'] = sims['pre'].apply(lambda x: 'bilateral' if x in anns_dict['mw bilateral axon'] else ('contralateral' if x in anns_dict['mw contralateral axon'] else ('ipsilateral' if x in anns_dict['mw ipsilateral axon'] else 'other')))
sims['dendrite_structure'] = sims['pre'].apply(lambda x: 'bilateral' if x in anns_dict['mw bilateral dendrite'] else ('contralateral' if x in anns_dict['mw contralateral dendrite'] else ('ipsilateral' if x in anns_dict['mw ipsilateral dendrite'] else 'other')))



# %% classification by performance 

# parameters 
sufficiency_threshold = 0.9 # threshold for considering a sample's performance as sufficient proportion of best peforming sample 
partial_contribution_threshold = 0.1 

# for each neuron, get Qnull, Qgeo, Qind, Qcomb, and Qstar - best performance 
sims['Qstar'] = sims[['null_sims', 'geo_sims', 'individual_sims', 'combined_sims']].max(axis=1)
sims['Qstar_cat'] = sims[['null_sims', 'geo_sims', 'individual_sims', 'combined_sims']].idxmax(axis=1)

# %% overview of best cosine similarity

fig, ax = plt.subplots(1,1, figsize=(5,5))
ax.hist(sims['Qstar'], bins=20, color='green', alpha=0.4)
ax.set_xlim(-0.05,1.05)
ax.set_xlabel('Best similarity to real co-targeting')
ax.set_ylabel('# of presynaptic neurons')

bc = ['#85d12e', '#d22d86','#2d87d2']
partial_bc = '#8332cd'

# # other - probably plot all together 
fig, ax = plt.subplots(1,1, figsize=(5,5))
for col, title, clr in zip(['null_sims', 'geo_sims', 'combined_sims'], ['null', 'geometry', '+ co-targeting propensity'], bc):
    ax.hist(sims[col], bins=20, alpha=0.6, label=title, color=clr)
ax.set_xlim(-0.05,1.05)
ax.set_xticks(np.arange(0, 1.1, 1))
ax.set_xlabel('Similarity to real co-targeting')
ax.set_ylabel('# of presynaptic neurons')
ax.legend(bbox_to_anchor=(1, 1), loc='upper left')

fig.savefig(f'{FIG_OUT}/sampled_sim_to_real_hist.svg', bbox_inches='tight')

# %% calculating categories of sufficiency and geometry share

# null sufficient if null >= Qstar*sufficiency_threshold
sims['null_sufficient'] = sims['null_sims'] >= sims['Qstar'] * sufficiency_threshold
### NOTE: change geo sufficient to having to be either the highest Qstar or account for more than a certain threshold of remaining potential
# to allow accurate estimating in cases where difference between null and Qstar is >10% but still small

#### ABSOLUTE GEO CONTRIBUTION
# improvement explained by geometry 
# i.e. is it neglible or partially explanatory? 
# this does not work well when Qstar is or is very close to 0 
sims['geo_abs'] = (sims['geo_sims'] - sims['null_sims'])/(sims['Qstar'] - sims['null_sims']) 
sims['geo_sufficient'] = sims['geo_abs'] >= sufficiency_threshold
# remove null sufficient neurons from geo sufficient 
sims.loc[sims['null_sufficient'], 'geo_sufficient'] = False # geo not important if null is already pretty good
sims.loc[sims['Qstar_cat'] == 'geo_sims', 'geo_sufficient'] = True # geo is sufficient if it is the best contributor

# if Qstar == null OR null > geo, set geo_abs to 0
sims.loc[(sims['null_sufficient'] == True) | (sims['null_sims'] > sims['geo_sims']), 'geo_abs'] = 0


#### RELATIVE GEO CONTRIBUTION --- LEAVE OUT, doesnt make sense because of the hierarchical design
# relative to the propensity & combined contributions
# note, this is only of interest when Qstar != null and 
sims['geo_rel'] = 0.5*((sims['geo_sims'] - sims['null_sims'])+(sims['combined_sims'] - sims['individual_sims']))
sims['ind_rel'] = 0.5*((sims['individual_sims'] - sims['null_sims'])+(sims['combined_sims'] - sims['geo_sims']))
sims['geometry_share'] = sims['geo_rel'] / (sims['geo_rel'] + sims['ind_rel'])
# a negative geometry contribution isn't really interesting, so clip it? 
sims['geometry_share'] = sims['geometry_share'].clip(lower=0)

#%% overview of categories and geometry share 

sims['category'] = np.where(sims['null_sufficient'], 'null sufficient',
                            np.where(sims['geo_sufficient'], 'geometry sufficient',
                                     #np.where(sims['geo_abs'] >= 0.5, 'geometry explains >=50%',
                                        np.where(sims['geo_abs'] >= 0.1, 'geometry partially explanatory',
                                              'geometry neglible')))

fig, ax = plt.subplots(1,1, figsize=(4,5))
cat_counts = sims['category'].value_counts()
cat_counts = cat_counts.reindex(['null sufficient', 'geometry sufficient', 'geometry partially explanatory', 'geometry neglible'], fill_value=0) #  'geometry explains >=50%',
ax.bar(cat_counts.index, cat_counts.values, color=[bc[0], bc[1], partial_bc, bc[2]], alpha=0.8, width=0.95)
ax.set_ylabel('# of presynaptic neurons')
ax.set_xticklabels(cat_counts.index, rotation=35, ha='right', rotation_mode='anchor')
fig.savefig(f'{FIG_OUT}/category_overview.svg', bbox_inches='tight')

sims['category_v2'] = np.where(sims['null_sufficient'], 'null sufficient',
                            np.where(sims['geo_sufficient'], 'geometry sufficient',
                                     np.where(sims['geo_abs'] >= 0.5, 'geometry explains >=50%',
                                        np.where(sims['geo_abs'] >= 0.1, 'geometry explains <50%',
                                              'geometry neglible'))))


fig, ax = plt.subplots(1,1, figsize=(4,5))
cat_counts = sims['category_v2'].value_counts()
cat_counts = cat_counts.reindex(['null sufficient', 'geometry sufficient','geometry explains >=50%', 'geometry explains <50%', 'geometry neglible'], fill_value=0) #  'geometry explains >=50%',
ax.bar(cat_counts.index, cat_counts.values, color=[bc[0], bc[1], partial_bc, partial_bc, bc[2]], alpha=0.8, width=0.95)
ax.set_ylabel('# of presynaptic neurons')
ax.set_xticklabels(cat_counts.index, rotation=35, ha='right', rotation_mode='anchor')
fig.savefig(f'{FIG_OUT}/category_overview_detailed.svg', bbox_inches='tight')

#%% plot geo abs contribution by category

def identity_fit(x, y):
    """R^2 of the line y = x, and a swap-symmetric Deming (orthogonal) fit.

    R^2 is measured about the identity line rather than a fitted one, so it can
    go negative: that means y = x describes the data worse than the grand mean.
    Deming assumes equal error variance in x and y, so unlike OLS its slope is
    not attenuated by noise in x and does not depend on which side is plotted
    on which axis.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    sxy = ((x - mx) * (y - my)).mean()

    grand_mean = np.r_[x, y].mean()
    ss_resid = ((y - x) ** 2).sum()
    ss_total = ((x - grand_mean) ** 2).sum() + ((y - grand_mean) ** 2).sum()
    r2_identity = 1 - ss_resid / ss_total

    slope = ((vy - vx) + np.sqrt((vy - vx) ** 2 + 4 * sxy ** 2)) / (2 * sxy)
    return r2_identity, slope, my - slope * mx

# remove null sufficient category for this plot

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
bins = np.arange(0, 1.05, 0.05)
alpha = 0.7

# colour bars along the same axis the categories sit on: blue (geometry neglible)
# at 0, purple (partially explanatory) through the middle, pink (geometry
# sufficient) at 1
geo_grad = mpl.colors.LinearSegmentedColormap.from_list('geo_grad', [bc[2], partial_bc, bc[1]])
counts, edges, patches = ax.hist(sims['geo_abs'], bins=bins, alpha=alpha, color='darkgray')
# normalise to the bin edges, not the data, so the gradient always spans the full
# 0-1 axis and stays comparable across plots
# norm = plt.Normalize(edges[0], edges[-1])
# for patch, centre in zip(patches, (edges[:-1] + edges[1:]) / 2):
#     patch.set_facecolor(geo_grad(norm(centre)))
# ax.hist(sims[sims['category']=='geometry sufficient']['geo_abs'], bins=bins, label='geometry sufficient', alpha=alpha)
# ax.hist(sims[sims['category']=='geometry partially explanatory']['geo_abs'], bins=bins, label='geometry partially explanatory', alpha=alpha)
# ax.hist(sims[sims['category']=='geometry neglible']['geo_abs'], bins=bins, label='geometry neglible', alpha=alpha)
# ax.legend()
ax.set_xticks([0,1])
ax.set_ylim([0,355])

# add patches for sufficient and neglible categories
from matplotlib.patches import Rectangle
# y > 1 is above the top spine; x and y both in axes fractions (0–1)
ax.add_patch(Rectangle((0.04, 1.02), 0.1, 0.03,
                       transform=ax.transAxes, clip_on=False,
                       color=bc[2], alpha=0.8))
ax.add_patch(Rectangle((0.86, 1.02), 0.1, 0.03,
                       transform=ax.transAxes, clip_on=False,
                       color=bc[1], alpha=0.8))
# label above 
ax.text(0.9, 1.06, 'sufficient', transform=ax.transAxes, ha='center', va='bottom', color=bc[1])
ax.text(0.1, 1.06, 'neglible', transform=ax.transAxes, ha='center', va='bottom', color=bc[2])


ax.set_ylabel('# of presynaptic neurons')
ax.set_xlabel('Relative improvement \nexplained by geometry')
fig.savefig(f'{FIG_OUT}/geo_abs_hist.svg', bbox_inches='tight')

#%%
lr = (sims[(sims['pairid'] >= 0) & (sims['hemi'].isin(['left', 'right'])) & (sims['category'].isin(['geometry sufficient', 'geometry partially explanatory', 'geometry neglible']))]
      .drop_duplicates(subset=['pairid', 'hemi'])
      .pivot(index='pairid', columns='hemi', values='geo_abs')
      .dropna(subset=['left', 'right']))

scatter_color = 'dimgray'
error_lw = 2

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.scatter(lr['left'], lr['right'], s=5, alpha=0.4, color=scatter_color)
# get fit 
r2_identity, slope, intercept = identity_fit(lr['left'], lr['right'])
lims = [min(lr['left'].min(), lr['right'].min()), max(lr['left'].max(), lr['right'].max())]
ax.plot(lims, lims, color='gray', linestyle='-', label='y=x', linewidth=error_lw)
ax.plot(lims, intercept + slope * np.array(lims), color='black', linestyle='--', label=f'Deming regression', linewidth=error_lw)
ax.text(0.03, 1.0, f'$R²_{{y=x}}$ = {r2_identity:.2f}', 
        transform=ax.transAxes, verticalalignment='top')
ax.set_xlim([lims[0]-0.05, lims[1]+0.05])
ax.set_ylim([lims[0]-0.05, lims[1]+0.05])
ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
ax.set_xlabel('Improvement by geometry (L)')
ax.set_ylabel('Improvement by geometry (R)')
# ax.set_title('Geo_abs comparison by pairid: Left vs Right')
fig.savefig(f'{FIG_OUT}/improvement_by_geo_LvsR.svg', bbox_inches='tight')

lr = (sims[(sims['pairid'] >= 0) & (sims['hemi'].isin(['left', 'right'])) ]
      .drop_duplicates(subset=['pairid', 'hemi'])
      .pivot(index='pairid', columns='hemi', values='geo_sims')
      .dropna(subset=['left', 'right']))
# and instead for just geometry-induced similarity 
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.scatter(lr['left'], lr['right'], s=5, alpha=0.4, color=scatter_color)
# get fit 
r2_identity, slope, intercept = identity_fit(lr['left'], lr['right'])
lims = [min(lr['left'].min(), lr['right'].min()), max(lr['left'].max(), lr['right'].max())]
ax.plot(lims, lims, color='gray', linestyle='-', label='y=x', linewidth=error_lw)
ax.plot(lims, intercept + slope * np.array(lims), color='black', linestyle='--', label=f'Deming regression', linewidth=error_lw)
ax.text(0.03, 1.0, f'$R²_{{y=x}}$ = {r2_identity:.2f}', 
        transform=ax.transAxes, verticalalignment='top')
ax.set_xlim([lims[0]-0.05, lims[1]+0.05])
ax.set_ylim([lims[0]-0.05, lims[1]+0.05])
ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
ax.set_xlabel('geometry vs real similarity (L)')
ax.set_ylabel('geometry vs real similarity (R)')
fig.savefig(f'{FIG_OUT}/geometry_vs_real_similarity_LvsR.svg', bbox_inches='tight')

# and for null 

lr = (sims[(sims['pairid'] >= 0) & (sims['hemi'].isin(['left', 'right'])) ]
      .drop_duplicates(subset=['pairid', 'hemi'])
      .pivot(index='pairid', columns='hemi', values='null_sims')
      .dropna(subset=['left', 'right']))
# and instead for just geometry-induced similarity 
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.scatter(lr['left'], lr['right'], s=5, alpha=0.4, color=scatter_color)
# get fit 
r2_identity, slope, intercept = identity_fit(lr['left'], lr['right'])
lims = [min(lr['left'].min(), lr['right'].min()), max(lr['left'].max(), lr['right'].max())]
ax.plot(lims, lims, color='gray', linestyle='-', label='y=x', linewidth=error_lw)
ax.plot(lims, intercept + slope * np.array(lims), color='black', linestyle='--', label=f'Deming regression', linewidth=error_lw)
ax.text(0.03, 1.0, f'$R²_{{y=x}}$ = {r2_identity:.2f}', 
        transform=ax.transAxes, verticalalignment='top')
ax.set_xlim([lims[0]-0.05, lims[1]+0.05])
ax.set_ylim([lims[0]-0.05, lims[1]+0.05])
ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
ax.set_xlabel('null vs real similarity (L)')
ax.set_ylabel('null vs real similarity (R)')
fig.savefig(f'{FIG_OUT}/null_vs_real_similarity_LvsR.svg', bbox_inches='tight')


#%% cell type split 
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
# sort by median geo_abs within each cell type
sims_ = sims.copy()
sims_['celltype'] = pd.Categorical(sims_['celltype'], 
                                  categories=sims_.groupby('celltype')['geo_abs'].median().sort_values().index,
                                  ordered=True)
sns.stripplot(data=sims_, x='celltype', y ='geo_abs', ax=ax, alpha=0.3, s=4, jitter=0.2, zorder=1)
#sns.pointplot(data=sims_, x='celltype', y='geo_abs', ax=ax, ci='sd', color='black', alpha=0.6)
sns.boxplot(data=sims_, x='celltype', y='geo_abs', ax=ax, whis=np.inf, color='black', width=0.3, zorder=2, boxprops=dict(alpha=.3))
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_ylabel('Relative improvement explained by geometry')
ax.set_xlabel('Cell type')

#%% quick LR comparison of category by pairid Left vs right 

# confusion table of left vs right by pairid in categories
# pivot on pairid rather than sorting each hemisphere separately: not every pairid
# has both halves in the output, so two sorted arrays would silently misalign
lr = (sims[(sims['pairid'] >= 0) & (sims['hemi'].isin(['left', 'right']))]
      .drop_duplicates(subset=['pairid', 'hemi'])
      .pivot(index='pairid', columns='hemi', values='category')
      .dropna(subset=['left', 'right']))

left_pairs = lr['left'].values
right_pairs = lr['right'].values
print(f'{len(lr)} pairids with both hemispheres (of {sims["pairid"].nunique()} total)')

cat_order = ['null sufficient', 'geometry sufficient',
             'geometry partially explanatory (>=10%)', 'geometry neglible']
conf = pd.crosstab(lr['left'], lr['right']).reindex(index=cat_order, columns=cat_order, fill_value=0)

# raw agreement is inflated when one category dominates, so report kappa too
n = conf.values.sum()
po = np.trace(conf.values) / n
pe = (conf.values.sum(axis=1) * conf.values.sum(axis=0)).sum() / n**2
kappa = (po - pe) / (1 - pe)
print(f'left/right agreement: {po:.3f}   Cohen kappa: {kappa:.3f}')

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.heatmap(conf, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
ax.set_xlabel('right hemisphere')
ax.set_ylabel('left hemisphere')
ax.set_title(f'L/R category agreement: {po:.2f} (kappa {kappa:.2f})')
plt.tight_layout()

#%% what about where combined is worse than individual?
#### this is probably not that interesting - just use geo share without anything in addition vs best available performance
# get a sub without null or geometry sufficiency
# sub_x = sims[~sims['null_sufficient'] & ~sims['geo_sufficient']]
# sub_x['combined_result'] = np.where(sub_x['combined_sims'] <= 1.0*sub_x['geo_sims'], 'combined worse than geometry alone', 
#                                     np.where(sub_x['combined_sims'] <= 1.0 * sub_x['individual_sims'], 'combined worse than propensity alone',
#                                      'combined better'))
# sub_x['combined_result'].value_counts()

# fig, ax = plt.subplots(1,1, figsize=(5,5))
# combined_counts = sub_x['combined_result'].value_counts()
# ax.bar(combined_counts.index, combined_counts.values)
# ax.set_ylabel('# of presynaptic neurons')
# ax.set_xticklabels(combined_counts.index, rotation=45, ha='right')
# ax.set_title('Where geometry is not sufficient, does adding geometry help?')

# # take these out 

# #%% examine geometry contribution depending on condition

# ### if individual > combined, how much does geometry alone achieve? 
# # sub_ind_gt_comb = sub_x[sub_x['combined_result'] == 'combined >=10% worse']
# # # geo abs answers this already? 
# # fig, ax = plt.subplots(1,1, figsize=(5,5))
# # ax.hist(sub_ind_gt_comb['geo_sims'].dropna(), bins=np.arange(0, 1.05, 0.05), alpha=0.3, color='blue')
# # ax.set_xlabel('Geometry contribution')
# # ax.set_ylabel('# of presynaptic neurons')
# # ax.set_title('Geometry contribution where combined worse')

# #%%
# #### geometry share (relevant only for those neurons where Qstar != null & Qstar != geo) --> how much of the combined effect is attributable to geometry?
# # sims.loc[sims['null_sufficient'], 'geometry_share'] = np.nan
# # sims.loc[sims['geo_sufficient'], 'geometry_share'] = np.nan
# sub = sub_x[sub_x['combined_result'] == 'combined better'] # already excludes null sufficient, geo sufficient and where ind >> combined

# # 
# fig, ax = plt.subplots(1,1, figsize=(5,5))
# alpha = 0.3
# bins = np.arange(0, 1.05, 0.05)
# ax.hist(sub['geometry_share'].dropna(), bins=bins, alpha=alpha, color='blue')
# ax.set_xlabel('Relative geometry contribution to combined effect')
# ax.set_ylabel('# of presynaptic neurons')
# ax.set_title('Where combined is better than either alone, how much does geometry contribute to combined effect?')

# %% feature / celltype attribution 

# add mod category 
sims['modality'] = sims['pre'].apply(lambda x: skid_to_mod.get(x, 'not sensory'))

sims['2nd_mod'] = sims.apply(lambda row: f"2nd order {skid_to_2ndorder[row['pre']]}" if (row['pre'] in skid_to_2ndorder) & (row['modality'] == 'not sensory') else row['modality'] if row['modality'] != 'not sensory' else 'other', axis=1)

# do this for categories null, geo sufficient and geo neglible 

nulls = sims[sims['category'] == 'null sufficient']
geos = sims[sims['category'] == 'geometry sufficient']
no_geos = sims[sims['category'] == 'geometry neglible']

# use raw numbers and fractions of cell types
ct_counts = sims['celltype'].value_counts().to_dict()
mod_counts = sims['modality'].value_counts().to_dict()

# for samp, samp_name in zip([nulls, geos, no_geos], ['Null sufficient', 'Geometry sufficient', 'Geometry neglible']):

#     fig, ax = plt.subplots(1,2, figsize=(8,4))
#     ax[0].bar(samp['celltype'].value_counts().index, samp['celltype'].value_counts().values)
#     ax[1].bar(samp['celltype'].value_counts().index, samp['celltype'].value_counts().values / [ct_counts[i] for i in samp['celltype'].value_counts().index])
#     ax[0].set_ylabel('# of presynaptic neurons')
#     ax[1].set_ylabel('Fraction of presynaptic neurons')
#     for a in ax:
#         a.set_xticklabels(a.get_xticklabels(), rotation=45, ha='right')
#     fig.suptitle(samp_name)
#     fig.tight_layout()

# # just modality for sensories 

# for samp, samp_name in zip([nulls, geos, no_geos], ['Null sufficient', 'Geometry sufficient', 'Geometry neglible']):
#     fig, ax = plt.subplots(1,2, figsize=(8,4))
#     ax[0].bar(samp['modality'].value_counts().index, samp['modality'].value_counts().values)
#     ax[1].bar(samp['modality'].value_counts().index, samp['modality'].value_counts().values / [mod_counts[i] for i in samp['modality'].value_counts().index])
#     ax[0].set_ylabel('# of presynaptic neurons')
#     ax[1].set_ylabel('Fraction of presynaptic neurons')
#     for a in ax:
#         a.set_xticklabels(a.get_xticklabels(), rotation=45, ha='right')
#     fig.suptitle(samp_name)
#     fig.tight_layout()

# use fractions of cell types for all 

def plot_ct_fraction_for_samp(samp, ct_counts, ct_plot_threshold, samp_color, samp_ticks, ycol='celltype', figwidth=4, show_raw=False, sort_by_fraction=True):
    samp = samp[samp[ycol].map(samp[ycol].value_counts()) > ct_plot_threshold]
    fig, ax = plt.subplots(1,1, figsize=(figwidth,5))
    xvals = samp[ycol].value_counts().index
    yvals = samp[ycol].value_counts().values / [ct_counts[i] for i in samp[ycol].value_counts().index if ct_counts[i] > ct_plot_threshold]
    if sort_by_fraction:
        sorted_indices = np.argsort(yvals)[::-1]
        xvals = xvals[sorted_indices]
        yvals = yvals[sorted_indices]
    ax.bar(xvals, yvals, width=0.9, color=samp_color)
    ax.set_ylabel(f'Fraction of {ycol}')
    ax.set_yticks(samp_ticks)
    ax.set_ylim(0, samp_ticks[-1])
    fs = 17 if figwidth <=5 else 15
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', rotation_mode='anchor', fontsize=fs)
    # add total numbers 
    if show_raw:
        for i, v in enumerate(samp[ycol].value_counts().values):
            ax.text(i, v / [ct_counts[i] for i in samp[ycol].value_counts().index if ct_counts[i] > ct_plot_threshold][i] + 0.01, str(v), ha='center')
    #fig.tight_layout()
    return fig

# single stack raw numbers bargraph 

def stack_colors(cmap_name, n):
    """n colors from cmap_name: qualitative maps are indexed, continuous ones sampled dark -> light."""
    cmap = plt.get_cmap(cmap_name)
    if isinstance(cmap, mpl.colors.ListedColormap) and cmap.N <= 20:
        return [cmap(i % cmap.N) for i in range(n)]
    # keep away from the near-white end so the smallest segments stay visible
    return [cmap(t) for t in np.linspace(0.85, 0.25, n)]


def single_column_stack_bargraph(data, cat_col, cats=None, cmap_name='Blues', bar_label='',
                                 normalize=True, annotate=False, annotate_threshold=0, figwidth=2.5):
    """One bar, stacked by the categories in cat_col.

    normalize=True gives each category's share of the column (segments sum to 1),
    normalize=False gives raw counts. Categories stack largest at the bottom, and
    counts are only written into segments with more than annotate_threshold neurons.
    """
    counts = data[cat_col].value_counts()
    if cats is None:
        cats = counts.index.tolist()  # value_counts is already largest -> smallest
    counts = counts.reindex(cats, fill_value=0)
    values = counts.values / counts.values.sum() if normalize else counts.values.astype(float)

    fig, ax = plt.subplots(1, 1, figsize=(figwidth, 5))
    bottom = 0.0
    for cat, value, color in zip(cats, values, stack_colors(cmap_name, len(cats))):
        ax.bar([0], [value], width=0.6, bottom=bottom, color=color, label=cat)
        if annotate and counts[cat] > annotate_threshold:
            ax.text(0, bottom + value / 2, str(counts[cat]), ha='center', va='center', fontsize=14)
        bottom += value
    ax.set_xticks([0])
    ax.set_xticklabels([bar_label])
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(0, bottom)
    ax.set_ylabel('Fraction of presynaptic neurons' if normalize else '# of presynaptic neurons')
    # reverse so the legend reads top-to-bottom in the same order as the stack
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    return fig

####### null ######
# only include cell type categories with more than 5 presynaptic neurons?
ct_plot_threshold = 1


samp = nulls.copy()
samp_color = bc[0]
samp_ticks = np.arange(0, 0.21, 0.1)

fig = plot_ct_fraction_for_samp(samp, ct_counts, ct_plot_threshold, samp_color, samp_ticks)
fig.savefig(f'{FIG_OUT}/ct_fractions_null_over_threshold_{ct_plot_threshold}.svg', bbox_inches='tight')

fig = single_column_stack_bargraph(samp, cat_col='celltype', cmap_name='tab10',
                                   bar_label='Null sufficient', normalize=True,
                                   annotate=True, annotate_threshold=3)
fig.savefig(f'{FIG_OUT}/ct_count_null_over_threshold_{ct_plot_threshold}_stacked.svg', bbox_inches='tight')

samp_ticks = np.arange(0, 1.1, 0.2)

fig = plot_ct_fraction_for_samp(samp, mod_counts, ct_plot_threshold, samp_color, samp_ticks, figwidth=5, ycol='modality')
fig.savefig(f'{FIG_OUT}/mod_fractions_null_over_threshold_{ct_plot_threshold}.svg', bbox_inches='tight')

#%%
samp = geos.copy()
samp_color = bc[1]
samp_ticks = np.arange(0, 0.21, 0.1)

fig = plot_ct_fraction_for_samp(samp, ct_counts, ct_plot_threshold, samp_color, samp_ticks)
fig.savefig(f'{FIG_OUT}/ct_fractions_geos_over_threshold_{ct_plot_threshold}.svg', bbox_inches='tight')

samp = no_geos.copy()
samp_color = bc[2]
samp_ticks = np.arange(0, 0.31, 0.1)

fig = plot_ct_fraction_for_samp(samp, ct_counts, ct_plot_threshold, samp_color, samp_ticks, figwidth=7)
fig.savefig(f'{FIG_OUT}/ct_fractions_no_geos_over_threshold_{ct_plot_threshold}.svg', bbox_inches='tight')





#%% stacked histogram for null sufficient sensory modalities

sims_sub = sims.copy()
sims_sub['special_sensory'] = sims_sub.apply(lambda row: row['modality'] if (row['modality'] in ['olfactory', 'visual', 'thermo-warm', 'thermo-cold']) else 'other sensory' if row['celltype']=='sensories' else 'non-sensory', axis=1)


hue_col, x_col = 'special_sensory', 'null_sims'
sub = sims_sub[[x_col, hue_col]].dropna(subset=[x_col])

# seaborn computes one set of bins over all the data, then splits it by hue;
# ax.hist would otherwise bin each group separately and the stack wouldn't line up
bins = np.histogram_bin_edges(sub[x_col], bins='auto')

cats = ['other sensory',
 'olfactory',
 'thermo-cold',
 'thermo-warm',
 'visual',
 'non-sensory',] #sorted(sub[hue_col].unique())
colors = plt.get_cmap('tab20').colors

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.hist([sub.loc[sub[hue_col] == c, x_col] for c in cats],
        bins=bins, stacked=True, label=cats, alpha=0.6,
        color=[colors[i % len(colors)] for i in range(len(cats))])

# fig, ax = plt.subplots(1, 1, figsize=(5,5))
# sns.histplot(data=sims_sub, x='null_sims', hue='special_sensory', multiple='stack', palette='tab20', ax=ax)
ax.set_xlabel('Co-targeting similarity to null')
ax.set_ylabel('# of presynaptic neurons')
ax.legend(bbox_to_anchor=(1, 1), loc='upper left')

# %% CHECK SPECIFIC CASES 
sub = sims[sims['modality']=='visual']

fig, ax = plt.subplots(1,1, figsize=(5,5))
for col in ['null_sims', 'geo_sims', 'individual_sims', 'combined_sims']:
    ax.hist(sub[col], bins=20, alpha=0.4, label=col)
ax.set_xlim(0.7,1.01)
ax.set_xlabel('Similarity to real co-targeting')
ax.set_ylabel('# of presynaptic neurons')
ax.legend()

# cell features dont seem super important 
# metrics = ['n_syns_total', 'n_syns_sites', 'u_outputs']
# fig, ax = plt.subplots(1,3, figsize=(12,4))
# for i, metric in enumerate(metrics):
#     ax[i].scatter(sub[metric], sub['geo_abs'])  
#     ax[i].set_xlabel(metric)
#     ax[i].set_ylabel('geo_abs')
# fig.tight_layout()

#%% dvncs 

dnv_null = sims[(sims['celltype']=='dVNCs') & (sims['category']=='null sufficient')]['name'].tolist()
dnv_geo = sims[(sims['celltype']=='dVNCs') & (sims['category']=='geometry sufficient')]['name'].tolist()
dnv_no_geo = sims[(sims['celltype']=='dVNCs') & (sims['category']=='geometry neglible')]['name'].tolist()
dnv_partial = sims[(sims['celltype']=='dVNCs') & (sims['category']=='geometry partially explanatory (>=10%)')]['name'].tolist()

#%% what subgroups are geometry sufficient??
sub = geos.copy()
metrics = ['n_syns_total', 'n_syns_sites', 'u_outputs']
fig, ax = plt.subplots(1,3, figsize=(12,4))
for i, metric in enumerate(metrics):
    ax[i].scatter(sub[metric], sub['geo_abs'], s=2)  
    ax[i].set_xlabel(metric)
    ax[i].set_ylabel('geo_abs')
fig.tight_layout()

#%% examine dvncs 
skid_groups_OI = pd.read_csv('data/skid_groups_of_interest.csv')
skid_groups_OI['skids'] = skid_groups_OI['skids'].apply(ast.literal_eval)
skid_to_group = {}
for _, row in skid_groups_OI.iterrows():
    for skid in row['skids']:
        skid_to_group[skid] = row['group_name']

sims['beh_group'] = sims['pre'].map(skid_to_group)
sims['beh_group'] = sims['beh_group'].fillna('other')
sims['beh_group'] = sims['beh_group'].apply(lambda x: x.replace('_left', '').replace('_right', ''))


sns.countplot(data=sims[sims['celltype']=='dVNCs'], x='beh_group', hue='category')

#%% what causes no geometric contribution?? 


#%% other categories 

# structural 


# stacked histogram of geo_abs with hue of axon or dendrite structure 
fig, ax = plt.subplots(1,2, figsize=(8,4))
alpha=0.5
sns.histplot(data=sims, x='geo_abs', hue='axon_structure', multiple='stack', ax=ax[0], alpha=alpha)
sns.histplot(data=sims, x='geo_abs', hue='dendrite_structure', multiple='stack', ax=ax[1], alpha=alpha)
ax[0].set_title('Axon Structure')
ax[1].set_title('Dendrite Structure')
fig.tight_layout()

# x = structure, y = category freq 
fig, ax = plt.subplots(1,2, figsize=(8,6))

sns.countplot(data=sims, x='axon_structure', hue='category', ax=ax[0], alpha=alpha)
sns.countplot(data=sims, x='dendrite_structure', hue='category', ax=ax[1], alpha=alpha)
ax[0].set_title('Axon Structure')
ax[1].set_title('Dendrite Structure')
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha='right')
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha='right')
fig.tight_layout()

fig, ax = plt.subplots(1, 2, figsize=(8, 6))

for a, struct, title in zip(ax,
                            ['axon_structure', 'dendrite_structure'],
                            ['Axon Structure', 'Dendrite Structure']):
    frac = (pd.crosstab(sims['category'], sims[struct], normalize='index')
              .reset_index()
              .melt(id_vars='category', value_name='fraction'))
    sns.barplot(data=frac, x='category', y='fraction', hue=struct, ax=a, alpha=alpha)
    a.set_title(title)
    a.set_ylabel('fraction within category')
    a.set_xticklabels(a.get_xticklabels(), rotation=45, ha='right')
fig.tight_layout()

# x = structure, y = fraction of that structure in each category
fig, ax = plt.subplots(1, 2, figsize=(10, 6))

for a, struct, title in zip(ax,
                            ['axon_structure', 'dendrite_structure'],
                            ['Axon Structure', 'Dendrite Structure']):
    frac = (pd.crosstab(sims[struct], sims['category'], normalize='index')
              .reset_index()
              .melt(id_vars=struct, value_name='fraction'))
    sns.barplot(data=frac, x=struct, y='fraction', hue='category', ax=a, alpha=alpha)
    a.set_title(title)
    a.set_xlabel('structure')
    a.set_ylabel('fraction within structure')
    a.set_xticklabels(a.get_xticklabels(), rotation=45, ha='right')
    a.legend(title='category', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
fig.tight_layout()

#%% does structure actually predict geo_abs? tests, not just plots

# The stacked histograms above confound group size with distribution shape, and there
# was no test behind them. This cell answers the question properly, working around
# three things that break a naive Kruskal-Wallis here:
#   - geo_abs is zero-and-one inflated (clamped to 0 above, exactly 1 when geo is Qstar),
#     so it is split into "does geometry contribute at all" and "if so, how much"
#   - each pairid contributes two near-identical neurons, so permutation moves whole
#     pairids rather than rows
#   - axon laterality is near-deterministic within some cell types, so labels are only
#     shuffled within a celltype; the gap to the unstratified p is the confound's size
# from structure_stats import analyse_structure, print_report, cramers_v

structure_results = {}
for struct in ['axon_structure', 'dendrite_structure']:
    structure_results[struct] = analyse_structure(
        sims, struct, value_col='geo_abs', stratum_col='celltype',
        pair_col='pairid', exclude=('other',), n_perm=10000, seed=0)
    print_report(structure_results[struct])

# descriptive plots for the structure comparison

# ECDF instead of a stacked histogram: normalised per group, so the shapes stay
# comparable when one structure has several times more neurons than another
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for a, struct, title in zip(ax,
                            ['axon_structure', 'dendrite_structure'],
                            ['Axon Structure', 'Dendrite Structure']):
    sub = sims[sims[struct] != 'other']
    sns.ecdfplot(data=sub, x='geo_abs', hue=struct, ax=a)
    a.set_title(f'{title}  (p={structure_results[struct]["partA"]["p"]:.4f} contribute, '
                f'{structure_results[struct]["partB"]["p"]:.4f} magnitude)'
                if structure_results[struct]['partB'] else title)
    a.set_xlabel('geo_abs')
fig.tight_layout()

# the two point masses, which any binning hides
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for a, struct, title in zip(ax,
                            ['axon_structure', 'dendrite_structure'],
                            ['Axon Structure', 'Dendrite Structure']):
    summ = structure_results[struct]['summary'].melt(
        id_vars=[struct], value_vars=['frac_zero', 'frac_one'], var_name='mass',
        value_name='fraction')
    sns.barplot(data=summ, x=struct, y='fraction', hue='mass', ax=a, alpha=alpha)
    a.set_title(title)
    a.set_ylabel('fraction of neurons at the boundary')
    a.set_xticklabels(a.get_xticklabels(), rotation=45, ha='right')
fig.tight_layout()

# how tangled structure and celltype are, side by side for both annotations
for struct in ['axon_structure', 'dendrite_structure']:
    v = cramers_v(sims['celltype'], sims[struct])
    print(f"Cramer's V(celltype, {struct}) = {v:.3f}")

# %% summary graphs

# null sufficient mainly sensory neurons - counts for maybe sensories, PNs, LNs, MBONs, 'other'
# subcategories modalities show almost all olfactory, thermo, some visual 

# geometry sufficient - mostly PNs and dVNCs, dSEZs

# geometry neglible - mostly pre-dvncs, sensories, LHNs, KCs, LNs, pre-dSEZs; for sensories modality mostly visual 


# structure contributes beyond cell type 
# --> bilateral axon significantly increases geometric contribution, even in celltype-stratified test, i.e. contributes beyond cell type. 
# dendrite structure does not significantly contribute. 
# show curve of cumulative proportion of ipsilateral, contralateral and bilateral axonvs geo absolute contribution 


#%% plot an example padj to illustrate 

diff_only = True

for ex_pre in [4235139, 14541927]: # add more examples as needed
    # ex_pre = 4235139 #3813487 #5946241
    # 14541927 nice illustration but co-targeting number not very high 
    # 3813487 nice but geo and comb very similar & lots of targets 
    ex_padjs = padj_stacks[ex_pre]
    n_sampled = int((len(ex_padjs)-1)/4)

    xpadj_real = ex_padjs[0]
    xpadj_null = np.mean(ex_padjs[1:n_sampled+1], axis=0)
    xpadj_geo = np.mean(ex_padjs[n_sampled+1:2*n_sampled+1], axis=0)
    xpadj_ind = np.mean(ex_padjs[2*n_sampled+1:3*n_sampled+1], axis=0)
    xpadj_comb = np.mean(ex_padjs[3*n_sampled+1:4*n_sampled+1], axis=0)

    xsims = sims[sims['pre']==ex_pre][['null_sims', 'geo_sims', 'individual_sims', 'combined_sims']]

    panels = [xpadj_real, xpadj_null, xpadj_geo, xpadj_comb]
    if diff_only:
        panels = [xpadj_real, xpadj_null-xpadj_real, xpadj_geo-xpadj_real, xpadj_comb-xpadj_real]
    titles = ['Real', 'Null', 'Geometry', '+ co-targeting propensity']
    panel_sims = [None,
                xsims['null_sims'].values[0],
                xsims['geo_sims'].values[0],
                xsims['combined_sims'].values[0]]

    # one scale across every panel, otherwise imshow renormalises each one and the
    # colours cannot be compared between them
    vmin = min(p.min() for p in panels)
    vmax = max(p.max() for p in panels)
    if diff_only:
        # just real
        vmin, vmax = panels[0].min(), panels[0].max()
        mindiff = min(p.min() for p in panels[1:])
        maxdiff = max(p.max() for p in panels[1:])
        minmaxdiff = max(abs(mindiff), abs(maxdiff))

    for cmap in ['magma']: # magma looks cooler maybe keep in mind

        fig, ax = plt.subplots(1, 4, figsize=(15, 4.5))
        for e, a, padj, title, sim in zip(range(len(ax)), ax, panels, titles, panel_sims):
            if diff_only and e != 0:
                im = a.imshow(padj, cmap='coolwarm', vmin=-minmaxdiff, vmax=minmaxdiff)
            else:
                im = a.imshow(padj, cmap=cmap, vmin=vmin, vmax=vmax)
            a.set_title(f'{title}\n ' if sim is None else f'{title}\n$s = {sim:.2f}$',  linespacing=1.5)
            a.set_xlabel('Postsynaptic targets')
            a.set_xticks([])
            a.set_yticks([])
        ax[0].set_ylabel('Postsynaptic targets')
        fig.tight_layout()
        # one colorbar, stealing space from all four axes so it sits at the far right
        cbar = fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.011, pad=0.03,
                    label='# of co-targeting events')

        cbar.outline.set_linewidth(1)
        cbar.ax.tick_params(width=1)
        # for heatmaps, change axes linewidth locally 
        for a in ax:
            for spine in a.spines.values():
                spine.set_linewidth(1)
        fig.suptitle(f'Presynaptic neuron: {skid_to_name[ex_pre]}', y=1.09, fontsize=20)
        if diff_only:
            fig.savefig(f'{FIG_OUT}/padj_sampled_{ex_pre}_diff.svg', bbox_inches='tight')
        else:
            fig.savefig(f'{FIG_OUT}/padj_sampled_{ex_pre}.svg', bbox_inches='tight')
        #fig.savefig(f'{FIG_OUT}/padj_sampled_{ex_pre}.svg', bbox_inches='tight')

#%% same neuron, left vs right: real padj with targets ordered by pairid



# the two homologues have different target sets, so ordering both by pairid is what
# makes the panels comparable - homologous targets end up at roughly the same
# position in each matrix. targets without a pairid have nothing to align to, so
# they go to the end rather than being interleaved arbitrarily by skid.
def sort_targets_by_pair(skids):
    skids = np.asarray(skids).astype(int)
    order = sorted(range(len(skids)),
                   key=lambda i: (skid_to_pair.get(skids[i], np.inf), skids[i]))
    return np.array(order)

pair_examples = [4235139, 14541927]
# plot left first regardless of the order listed above
pair_examples = sorted(pair_examples, key=lambda s: skid_to_hemi.get(s, 'z'))

panels, titles = [], []
for ex_pre in pair_examples:
    # padj_targets, not target_skids: with only_major_targets the padj is already
    # subset to the major targets, so the full target list would mis-order it
    order = sort_targets_by_pair(padj_targets[ex_pre])
    panels.append(padj_stacks[ex_pre][0][np.ix_(order, order)])
    titles.append(skid_to_hemi.get(ex_pre, '').capitalize())

# one scale across both panels, otherwise imshow renormalises each one
vmin = min(p.min() for p in panels)
vmax = max(p.max() for p in panels)

fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
for a, padj, title in zip(ax, panels, titles):
    im = a.imshow(padj, cmap='magma', vmin=vmin, vmax=vmax)
    a.set_title(title)#, fontsize=16)
    a.set_xticks([])
    a.set_yticks([])
    a.set_xlabel('Postsynaptic targets')#, fontsize=14)
ax[0].set_ylabel('Postsynaptic targets') #, fontsize=14)
fig.tight_layout()
# one colorbar, stealing space from both axes so it sits at the far right
cbar = fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.023, pad=0.03,
             label='# of co-targeting events')
cbar.outline.set_linewidth(1)
cbar.ax.tick_params(width=1)
# for heatmaps, change axes linewidth locally 
for a in ax:
    for spine in a.spines.values():
        spine.set_linewidth(1)

fig.suptitle(f"Presynaptic neuron: {skid_to_name[pair_examples[0]].rsplit(' ', 1)[0]}",
             y=1.07, fontsize=20)

# pad to include the suptitle in the saved figure


fig.savefig(f'{FIG_OUT}/padj_LR_presyn_{pair_examples[0]}.svg', bbox_inches='tight')

# # the two homologues have different target sets, so ordering both by pairid is what
# # makes the panels comparable - homologous targets end up at the same position in
# # each matrix. a pair targeted by only one of the two gets an all-zero row/column in
# # the other, so the panels stay index-for-index comparable. a neuron can target both
# # homologues of a pair, so slots are keyed (pairid, rank) rather than by pairid alone.
# # targets without a pairid have nothing to align to and go at the end, per neuron.
# def pair_slots(skids):
#     """slot key -> position in this neuron's padj, for its targets that have a pairid."""
#     slot_to_idx = {}
#     seen = {}
#     for idx in np.argsort(np.asarray(skids).astype(int)):
#         pid = skid_to_pair.get(int(skids[idx]))
#         if pid is None:
#             continue
#         rank = seen.get(pid, 0)
#         seen[pid] = rank + 1
#         slot_to_idx[(pid, rank)] = int(idx)
#     return slot_to_idx

# def unpaired_idx(skids):
#     skids = np.asarray(skids).astype(int)
#     return [int(i) for i in np.argsort(skids) if skids[i] not in skid_to_pair]

# pair_examples = [4235139, 14541927]
# # plot left first regardless of the order listed above
# pair_examples = sorted(pair_examples, key=lambda s: skid_to_hemi.get(s, 'z'))

# slot_maps = [pair_slots(target_skids[p]) for p in pair_examples]
# all_slots = sorted(set().union(*[set(m) for m in slot_maps]))

# panels, titles = [], []
# for ex_pre, slot_map in zip(pair_examples, slot_maps):
#     ex_idx = np.where(np.array(all_presyn) == ex_pre)[0][0]
#     real = padj_stacks[ex_idx][0]

#     # None marks a slot this neuron has no target for, which stays zero
#     source = [slot_map.get(slot) for slot in all_slots] + unpaired_idx(target_skids[ex_pre])
#     present = [k for k, src in enumerate(source) if src is not None]
#     src = [source[k] for k in present]

#     padded = np.zeros((len(source), len(source)), dtype=real.dtype)
#     padded[np.ix_(present, present)] = real[np.ix_(src, src)]
#     panels.append(padded)
#     titles.append(skid_to_hemi.get(ex_pre, '').capitalize())

# # one scale across both panels, otherwise imshow renormalises each one
# vmin = min(p.min() for p in panels)
# vmax = max(p.max() for p in panels)

# fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
# for a, padj, title in zip(ax, panels, titles):
#     im = a.imshow(padj, cmap='magma', vmin=vmin, vmax=vmax)
#     a.set_title(title, fontsize=14)
#     a.axis('off')
# fig.tight_layout()
# # one colorbar, stealing space from both axes so it sits at the far right
# fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.023, pad=0.03,
#              label='# of co-targeting events')
# fig.suptitle(f"Real co-targeting: {skid_to_name[pair_examples[0]].rsplit(' ', 1)[0]}",
#              y=1.05, fontsize=16)
# %%
