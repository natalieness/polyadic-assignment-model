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

# Ensure the repository is importable and relative data paths work.
REPO_ROOT = Path("/Users/nessn/git-repos/polyadic-assignment-model")
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

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
INPUT = 'out/approach2/geo_and_full_individual_cotargeting/'

def load_geo_samples(INPUT):
    files = [f for f in os.listdir(INPUT) if f.endswith('.json')]
    n_files = len(files)
    geo_samples = {}
    real_samples = {}
    target_skids = {}
    null_samples = {}
    ind_samples = {}
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

    return geo_samples, real_samples, target_skids, null_samples, ind_samples

geo_samples, real_samples, target_skids, null_samples, individual_samples = load_geo_samples(INPUT)

#%% examine all neurons and left/right split pairs for all options 

# get real vs sampled cosine similarlity for each neuron & sample type & get average and sd of that 

all_presyn = list(target_skids.keys())

null_cos_sims = {}
geo_cos_sims = {}
individual_cos_sims = {}

def assigned_to_cooc(S):
    # S is  (n_sample, sites, targets)
    cooc_all = np.einsum('kij,kil->kjl', S, S)         # (n_samples, targets, targets)
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
results = []
padj_stacks = []
n_presyn = len(all_presyn)
for e, pre in enumerate(all_presyn):
    if e % 10 == 0:
        print(f"Processing presynaptic neuron {e+1}/{n_presyn}")
    # stack real sample, null_samples, and geo_samples for this presynaptic neuron for efficient computation 
    # rela has shape (x,y) all others already have shapes with (n_samples, x, y) so need to expand real_samples to match
    real_samples_expanded = real_samples[pre][None, :, :]           # (1, 58, 47)
    n_sampled = null_samples[pre].shape[0] # how many are sampled in the null and geo versions
    stacked_samples = np.concatenate([real_samples_expanded, null_samples[pre], geo_samples[pre], individual_samples[pre]], axis=0)
    padj_stacked = assigned_to_cooc(stacked_samples)
    padj_stacks.append(padj_stacked)

    sims = cosine_upper_triangle_batch(padj_stacked[1:], padj_stacked[0])   # (n_samples - 1,)
    results.append({
        'pre': pre,
        'null_sims': sims[:n_sampled],
        'geo_sims': sims[n_sampled:n_sampled + geo_samples[pre].shape[0]],
        'individual_sims': sims[n_sampled + geo_samples[pre].shape[0]:]
    })


cossims = pd.DataFrame(results)

# %% 
cossims['mean_null'] = cossims['null_sims'].apply(np.mean)
cossims['mean_geo'] = cossims['geo_sims'].apply(np.mean)
cossims['mean_individual'] = cossims['individual_sims'].apply(np.mean)

cossims['sd_null'] = cossims['null_sims'].apply(np.std)
cossims['sd_geo'] = cossims['geo_sims'].apply(np.std)
cossims['sd_individual'] = cossims['individual_sims'].apply(np.std)

cossims['name'] = cossims['pre'].map(skid_to_name)
cossims['pairid'] = cossims['pre'].map(skid_to_pair)
cossims['celltype'] = cossims['pre'].map(skid_to_celltype)
cossims['hemi'] = cossims['pre'].map(skid_to_hemi)

# remove RGNs
cossims = cossims[cossims['celltype'] != 'RGNs']

# chaacterise neurons by their inputs and outputs
cossims['n_syns_total'] = cossims['pre'].map(lambda pre: real_samples[pre].sum())
cossims['n_syns_sites'] = cossims['pre'].map(lambda pre: real_samples[pre].shape[0])
cossims['u_outputs'] = cossims['pre'].map(lambda pre: real_samples[pre].shape[1])

# cossims_full = cossims.copy()
#### further filtering to remove neurons for which it just doesnt make sense cause they donthave enough sites or targets 
min_n_syns_sites = 5
min_u_outputs = 3
min_n_syns_total = 10
cossims = cossims[(cossims['n_syns_sites'] >= min_n_syns_sites) & (cossims['u_outputs'] >= min_u_outputs) & (cossims['n_syns_total'] >= min_n_syns_total)]

print(f'Filtered out {cossims_full.shape[0] - cossims.shape[0]} out of {cossims_full.shape[0]}')


# scale results to theoretical limit 
cossims['mean_null_frac'] = cossims['mean_null'] / cossims['mean_individual']
cossims['mean_geo_frac'] = cossims['mean_geo'] / cossims['mean_individual']

#%% maybe as a reference for what's a " good " similarity score - check L vs R similarity scores in real sample 

skid_to_index = {skid: i for i, skid in enumerate(all_presyn)}


def cooc_by_pairid(S, targets, combined):
    tp = [skid_to_pair.get(ski, 'ind') for ski in targets]
    # sum the site x target matrix over columns sharing a pair id, then count
    S_merged = pd.DataFrame(S.T).groupby(tp).sum().reindex(combined, fill_value=0).values.T
    return assigned_to_cooc(S_merged[None])[0]

#### NOTE: suboptimal here to use 'ind' to avoid missing pair ids mainly for kcs 

pair_cos = []

# iterate through left ids 
for id_row in pairs.index:
    left = pairs.loc[id_row, 'leftid']
    right = pairs.loc[id_row, 'rightid']

    left_targets = target_skids[left] if left in target_skids else None
    right_targets = target_skids[right] if right in target_skids else None
    if left_targets is None or right_targets is None:
        pair_cos.append(np.nan)
        continue

    combined = np.unique([skid_to_pair.get(s, 'ind') for s in left_targets]
                    + [skid_to_pair.get(s, 'ind') for s in right_targets])
    padj_left = cooc_by_pairid(real_samples[left], left_targets, combined)
    padj_right = cooc_by_pairid(real_samples[right], right_targets, combined)
    #normalize to the same number of outputs 
    padj_left_norm = padj_left / np.linalg.norm(padj_left) if np.linalg.norm(padj_left) > 0 else padj_left
    padj_right_norm = padj_right / np.linalg.norm(padj_right) if np.linalg.norm(padj_right) > 0 else padj_right
    cs_local = cosine_upper_triangle(
        padj_left_norm,
        padj_right_norm,
    )

    pair_cos.append(cs_local)

pair_cos = pd.DataFrame(pair_cos, columns=['pair_cossim'])
pair_cos['leftid'] = pairs['leftid'].values
pair_cos['rightid'] = pairs['rightid'].values

fig, ax = plt.subplots(1, 1, figsize=(5,5))
alpha = 0.7
bins = np.arange(0, 1.05, 0.05)
color_ = ["#35d0ff"]
ax.hist(pair_cos['pair_cossim'].dropna(), bins=bins, alpha=alpha, color=color_[0], label='bilateral pair')
ax.legend()

pair_cos['celltype'] = pair_cos['leftid'].map(skid_to_celltype)


#%% plots 

##### NOTE: need more samples for each here!! n should probably be like 100 

# separate plots for usability 
alpha = 0.7
bins = np.arange(0, 1.05, 0.05)
colors = ["#ffee35", "#3ed900", "#ff5733"]
fig, ax = plt.subplots(1, 1, figsize=(5,5))
ax.hist(cossims['mean_null'].dropna(), bins=bins, alpha=alpha, label='null', color=colors[0])
fig, ax = plt.subplots(1, 1, figsize=(5,5))
ax.hist(cossims['mean_geo'].dropna(), bins=bins, alpha=alpha, label='geometry', color=colors[1])
fig, ax = plt.subplots(1, 1, figsize=(5,5))
ax.hist(cossims['mean_null'].dropna(), bins=bins, alpha=alpha, label='null', color=colors[0])
ax.hist(cossims['mean_geo'].dropna(), bins=bins, alpha=alpha, label='geometry', color=colors[1])
ax.hist(cossims['mean_individual'].dropna(), bins=bins, alpha=alpha, label='individual', color=colors[2])

# optional by celltype 
bins_frac = np.arange(0, 1.65, 0.05)
alpha=0.4
for celltype, group in cossims.groupby('celltype'):
    fig, ax = plt.subplots(1, 1, figsize=(4,4))
    ax.hist(group['mean_null_frac'].dropna(), bins=bins_frac, alpha=alpha, label='null', color=colors[0])
    ax.hist(group['mean_geo_frac'].dropna(), bins=bins_frac, alpha=alpha, label='geometry', color=colors[1])
    #ax.hist(group['mean_individual'].dropna(), bins=bins, alpha=alpha, label='individual', color=colors[2])
    ax.vlines(0.9, ymin=0, ymax=ax.get_ylim()[1], colors='black', linestyles='dashed')
    ax.set_title(f'Celltype: {celltype}')
    ax.legend()


# %% reliability L-R for null and geo sampled cos similarity 

pre_to_null = dict(zip(cossims['pre'], cossims['mean_null']))
pre_to_geo = dict(zip(cossims['pre'], cossims['mean_geo']))
pre_to_individual = dict(zip(cossims['pre'], cossims['mean_individual']))


paired_cos = pair_cos.copy()
paired_cos['null_L'] = paired_cos['leftid'].map(pre_to_null)
paired_cos['geo_L'] = paired_cos['leftid'].map(pre_to_geo)
paired_cos['null_R'] = paired_cos['rightid'].map(pre_to_null)
paired_cos['geo_R'] = paired_cos['rightid'].map(pre_to_geo)
paired_cos['individual_L'] = paired_cos['leftid'].map(pre_to_individual)
paired_cos['individual_R'] = paired_cos['rightid'].map(pre_to_individual)

paired_cos.dropna(inplace=True, subset=['null_L', 'geo_L', 'null_R', 'geo_R', 'individual_L', 'individual_R'])

alpha=0.7
size = 8
linecolor = 'black'
lw = 2


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


for col, lab, color in [('null', 'null', colors[0]), ('geo', 'geometry', colors[1]), ('individual', 'individual', colors[2])]:
    L, R = paired_cos[f'{col}_L'], paired_cos[f'{col}_R']
    r2_identity, slope, intercept = identity_fit(L, R)

    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    ax.scatter(L, R, alpha=alpha, s=size, color=color)

    # equal limits so the identity line actually sits at 45 degrees
    lims = [min(L.min(), R.min()), max(L.max(), R.max())]
    ax.plot(lims, lims, color=linecolor, linestyle='--', lw=lw, label='y = x')
    ax.plot(lims, intercept + slope * np.array(lims), color=linecolor,
            linestyle='-', lw=lw, alpha=0.6, label='Deming regression')

    ax.text(0.03, 0.97,
            f'$R^2_{{y=x}}$ = {r2_identity:.3f}',
            transform=ax.transAxes, ha='left', va='top', fontsize=12)

    ax.set_xlim(lims[0]-0.05, lims[1]+0.05)
    ax.set_ylim(lims[0]-0.05, lims[1]+0.05)
    ax.set_aspect('equal')
    ax.set_xlabel(f'similarity to {lab} (L)')
    ax.set_ylabel(f'similarity to {lab} (R)')
    ax.legend(fontsize=8, loc='lower right')

#%% what makes a neuron have a lower theoretical limit on drawing a good sample? 

covars = [ 'n_syns_total', 'n_syns_sites', 'u_outputs']
cossims_clean = cossims.dropna(subset=['mean_individual'])
logx = False

fig, ax = plt.subplots(1,3,figsize=(12,4))
ax = ax.flatten()
for axs, cov in zip(ax, covars):
    axs.scatter(cossims_clean[cov], cossims_clean['mean_individual'], alpha=0.4, s=3)
    axs.set_xlabel(cov)
    if logx:
        axs.set_xscale('log')
        axs.set_xlabel(f'log({cov})')
    axs.set_ylabel('mean_individual')


#%% stacked cell type hist for theoretically possible 

fig, ax = plt.subplots(1,1,figsize=(5,5))
sns.histplot(data=cossims, x='mean_individual', hue='celltype', multiple='stack', palette='tab20', ax=ax)

#%% characterise which neurons are within a certain percentage or higher than theoretical limit 
threshold_frac = 0.9
cossims['null_above_rel_threshold'] = cossims['mean_null_frac'] >= threshold_frac
cossims['geo_above_rel_threshold'] = cossims['mean_geo_frac'] >= threshold_frac

# %% also label cossim with sneosry modality for each sensory cell type 
cossims['modality'] = cossims['pre'].apply(lambda x: skid_to_mod.get(x, 'not sensory'))
cossims['2nd_order_modality'] = cossims['pre'].apply(lambda x: skid_to_2ndorder.get(x, 'none'))
# histogram with modality coloured 
# fig, ax = plt.subplots(1, 1, figsize=(5,5))
# sns.stripplot(data=cossims, x='mean_null', hue='modality', palette='tab20', ax=ax, dodge=True)
# ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
spn = 'olfactory, thermo or visual'
special_cats = {'olfactory':spn, 'thermo-warm':spn, 'thermo-cold':spn, 'visual':spn, 'not sensory':'not sensory'}
cossims['special_sensory'] = cossims['modality'].apply(lambda x: special_cats.get(x, 'other sensory'))

spn_2nd = '2nd order olfactory, thermo or visual'
special_cats_2nd = {'olfactory':spn_2nd, 'thermo-warm':spn_2nd, 'thermo-cold':spn_2nd, 'visual':spn_2nd, 'none':'none'}
cossims['special_2nd_order_sensory'] = cossims['2nd_order_modality'].apply(lambda x: special_cats_2nd.get(x, 'other 2nd order sensory'))
cossims['special_combined'] = cossims.apply(lambda row: row['special_2nd_order_sensory'] if row['2nd_order_modality'] != 'none' else row['special_sensory'], axis=1)

fig, ax = plt.subplots(1, 1, figsize=(5,5))
sns.histplot(data=cossims, x='mean_null', hue='special_sensory', multiple='stack', palette='tab10', ax=ax)

fig, ax = plt.subplots(1, 1, figsize=(5,5))
sns.histplot(data=cossims, x='mean_null', hue='special_combined', multiple='stack', palette='tab10', ax=ax)

# relative
fig, ax = plt.subplots(1, 1, figsize=(5,5))
sns.histplot(data=cossims, x='mean_null_frac', hue='special_combined', multiple='stack', palette='tab10', ax=ax)


# plot a bar graph instead 
threshold = 0.75 
cossims_above_null = cossims[cossims['mean_null'] > threshold]

fig, ax = plt.subplots(1, 1, figsize=(5,5))
sns.countplot(data=cossims_above_null, x='special_combined', palette='tab10', ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# use relative frac instead 
fig, ax = plt.subplots(1, 1, figsize=(5,5))
sns.countplot(data=cossims[cossims['null_above_rel_threshold']], x='special_combined', palette='tab10', ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

### use fraction of each subtype instead 
countdata = cossims[cossims['null_above_rel_threshold']].groupby('special_combined').size()
maxpercat = cossims.groupby('special_combined').size()
countdata = countdata / maxpercat
countdata = countdata.reset_index().rename(columns={0:'fraction'})
# order by size 
countdata = countdata.sort_values('fraction', ascending=False)
fig, ax = plt.subplots(1, 1, figsize=(5,5))
sns.barplot(data=countdata, x='special_combined', y='fraction', palette='tab10', ax=ax, order=countdata['special_combined'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_ylabel('Fraction of cell type')

#%% stripplot of all instead 

def half_violin_strip(data, x, y, ax, palette='tab10', order=None,
                      side='right', gap=0.05, jitter=0.12, size=3,
                      violin_alpha=0.4,
                      strip_color=None, inner='box', **violin_kw):
    """Violin clipped to one side of each category, with the raw points beside it."""
    if order is None:
        order = list(pd.unique(data[x].dropna()))

    n_before = len(ax.collections)
    sns.violinplot(data=data, x=x, y=y, hue=x, order=order, hue_order=order, alpha=violin_alpha,
                   palette=palette, ax=ax, inner=inner, legend=False, **violin_kw)
    # seaborn draws one filled collection per category, centred on its tick
    for centre, coll in enumerate(ax.collections[n_before:]):
        for path in coll.get_paths():
            v = path.vertices
            v[:, 0] = (np.clip(v[:, 0], centre, None) if side == 'right'
                       else np.clip(v[:, 0], None, centre))

    n_before = len(ax.collections)
    strip_kw = (dict(color=strip_color) if strip_color else
                dict(hue=x, hue_order=order, palette=palette, legend=False))
    sns.stripplot(data=data, x=x, y=y, order=order, ax=ax, size=size,
                  jitter=jitter / 2, **strip_kw)
    shift = -(gap + jitter / 2) if side == 'right' else (gap + jitter / 2)
    for coll in ax.collections[n_before:]:
        coll.set_offsets(coll.get_offsets() + [shift, 0])
    return ax

toplot = 'mean_null_frac'
fig, ax = plt.subplots(1, 1, figsize=(5,5))
order = ['olfactory, thermo or visual', '2nd order olfactory, thermo or visual', 'other sensory', 'other 2nd order sensory', 'not sensory']
half_violin_strip(cossims, 'special_combined',toplot, ax=ax, palette='tab10', order=order)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# %%
# cossims['null_above_threshold'] = cossims['mean_null'] > threshold
# cossims['geo_above_threshold'] = cossims['mean_geo'] > threshold
# cossims['geo_improvement'] = cossims['mean_geo'] - cossims['mean_null']
# geo_improve_threshold = 0.05
# cossims['geo_improvement_above_threshold'] = cossims['geo_improvement'] > geo_improve_threshold

# cossims['geo_above_threshold_only'] = cossims['geo_above_threshold'] & ~cossims['null_above_threshold']

# %% which neurons are improved by geometry? 

# fully 
fig, ax = plt.subplots(1, 1, figsize=(8,5))
bins = np.arange(0, 1.05, 0.05)  # example bin edges from 0 to 1 with step 0.05
# plot hist without those above the null threshold
ax.hist(cossims[~cossims['null_above_rel_threshold']]['mean_geo_frac'], bins=bins)

fig, ax = plt.subplots(1, 1, figsize=(8,5))
s= 2
sns.swarmplot(data=cossims[~cossims['null_above_rel_threshold']], x='celltype', y='mean_geo_frac', palette='tab20', ax=ax, size=s)

# and with modlaity 
fig, ax = plt.subplots(1, 1, figsize=(8,5))
s= 2
sns.swarmplot(data=cossims[~cossims['null_above_rel_threshold']], x='modality', y='mean_geo_frac', palette='tab20', ax=ax, size=s)




fig, ax = plt.subplots(1, 1, figsize=(8,5))
counts_above = [cossims['null_above_rel_threshold'].sum(), cossims[~cossims['null_above_rel_threshold']]['geo_above_rel_threshold'].sum()] 
n_above = sum(counts_above)
counts_above.append(len(cossims) - n_above)
ax.bar(['null above threshold', 'geo above threshold', 'not above threshold'], counts_above)


#%% 

cossims['geo_improvement'] = cossims['mean_geo_frac'] - cossims['mean_null_frac']
cossims['proportional_geo_improvement'] = cossims['geo_improvement'] / (1 - cossims['mean_null_frac'])
# partial improvement above threshold 

fig, ax = plt.subplots(1, 1, figsize=(8,5))
s= 2
sns.swarmplot(data=cossims, x='celltype', y='geo_improvement', palette='tab20', ax=ax, size=s)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

fig, ax = plt.subplots(1, 1, figsize=(8,5))
s= 2
sns.swarmplot(data=cossims, x='celltype', y='proportional_geo_improvement', palette='tab20', ax=ax, size=s)
ax.set_ylim(-2,2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

fig, ax = plt.subplots(1, 1, figsize=(8,5))
ax.hist(cossims['proportional_geo_improvement'], bins=np.arange(0, 1.05, 0.05))
ax.set_xlabel('Proportional Geo Improvement')
ax.set_ylabel('Frequency')
ax.set_title('Histogram of Proportional Geo Improvement')

#%% LR symmetry of geo improvement 

fig, ax = plt.subplots(1, 1, figsize=(8,5))
# %%
