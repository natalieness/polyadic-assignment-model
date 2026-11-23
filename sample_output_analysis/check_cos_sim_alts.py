''' comparing alternatives to cosine similarity for computing polyadic co-targeting similaity across neurons'''


from pathlib import Path
import os
import sys
import ast
import time 

import pandas as pd
from itertools import chain
import numpy as np
from scipy import sparse
from scipy.special import rel_entr
from scipy.stats import rankdata
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

def flatten_upper_triangle(stack, reference):
    """Flatten a (n, k, k) stack and one (k, k) reference to (n, K) and (K,) upper triangles."""
    rows, cols = np.triu_indices(reference.shape[0], k=0)
    return stack[:, rows, cols].astype(float), reference[rows, cols].astype(float)


def _cosine_rows(flat_stack, flat_ref):
    """Cosine similarity of each row of (n, K) against one (K,) vector; nan for zero-norm rows."""
    stack_norms = np.linalg.norm(flat_stack, axis=1)
    ref_norm = np.linalg.norm(flat_ref)
    with np.errstate(invalid='ignore', divide='ignore'):
        result = (flat_stack @ flat_ref) / (stack_norms * ref_norm)
    result[(stack_norms == 0) | (ref_norm == 0)] = np.nan
    return result


def _to_distribution(x):
    """Normalise along the last axis so it sums to 1; all-zero vectors become nan."""
    total = x.sum(axis=-1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        dist = x / total
    return np.where(total == 0, np.nan, dist)


def _pearson_rows(flat_stack, flat_ref):
    """Pearson r is cosine similarity after mean-centring, so reuse the cosine core."""
    centred_stack = flat_stack - flat_stack.mean(axis=1, keepdims=True)
    centred_ref = flat_ref - flat_ref.mean()
    return _cosine_rows(centred_stack, centred_ref)


def _hellinger_rows(p_stack, q_ref):
    """Hellinger distance in [0, 1] between each row distribution and the reference distribution."""
    bhattacharyya = np.sqrt(p_stack) @ np.sqrt(q_ref)
    return np.sqrt(np.clip(1.0 - bhattacharyya, 0.0, 1.0))


def _jensen_shannon_rows(p_stack, q_ref, base=2.0):
    """Jensen-Shannon divergence; base 2 bounds it in [0, 1]. rel_entr handles the 0 log 0 terms."""
    mixture = 0.5 * (p_stack + q_ref)
    divergence = 0.5 * (rel_entr(p_stack, mixture).sum(axis=1)
                        + rel_entr(q_ref, mixture).sum(axis=1))
    return np.clip(divergence, 0.0, np.log(2.0)) / np.log(base)


def _weighted_jaccard_rows(flat_stack, flat_ref):
    """sum(min) / sum(max) per row; assumes non-negative weights (true for co-occurrence counts)."""
    intersection = np.minimum(flat_stack, flat_ref).sum(axis=1)
    union = np.maximum(flat_stack, flat_ref).sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        result = intersection / union
    result[union == 0] = np.nan
    return result


def cosine_upper_triangle_batch(stack, reference):
    """Cosine similarity of each (n, k, k) matrix's upper triangle against one (k, k) reference."""
    return _cosine_rows(*flatten_upper_triangle(stack, reference))


def pearson_upper_triangle_batch(stack, reference):
    """Pearson correlation of each upper triangle against the reference upper triangle."""
    return _pearson_rows(*flatten_upper_triangle(stack, reference))


def spearman_upper_triangle_batch(stack, reference):
    """Spearman correlation: Pearson on average-tied ranks of the upper triangles."""
    flat_stack, flat_ref = flatten_upper_triangle(stack, reference)
    return _pearson_rows(rankdata(flat_stack, axis=1), rankdata(flat_ref))


def hellinger_upper_triangle_batch(stack, reference):
    """Hellinger distance between upper triangles treated as probability distributions."""
    flat_stack, flat_ref = flatten_upper_triangle(stack, reference)
    return _hellinger_rows(_to_distribution(flat_stack), _to_distribution(flat_ref))


def jensen_shannon_upper_triangle_batch(stack, reference, base=2.0):
    """Jensen-Shannon divergence between upper triangles treated as probability distributions."""
    flat_stack, flat_ref = flatten_upper_triangle(stack, reference)
    return _jensen_shannon_rows(_to_distribution(flat_stack), _to_distribution(flat_ref), base=base)


def weighted_jaccard_upper_triangle_batch(stack, reference):
    """Weighted (Ruzicka) Jaccard between upper triangles, keeping the raw counts as weights."""
    return _weighted_jaccard_rows(*flatten_upper_triangle(stack, reference))


METRIC_NAMES = ('cosine', 'pearson', 'spearman', 'hellinger', 'jsd', 'weighted_jaccard')


def compare_upper_triangle_batch(stack, reference, base=2.0):
    """All metrics in METRIC_NAMES at once, flattening and normalising only once.

    Returns a dict of (n,) arrays. Cosine/Pearson/Spearman/weighted Jaccard are
    similarities (higher = closer); Hellinger and JSD are divergences (lower = closer).
    """
    flat_stack, flat_ref = flatten_upper_triangle(stack, reference)
    p_stack = _to_distribution(flat_stack)
    q_ref = _to_distribution(flat_ref)
    return {
        'cosine': _cosine_rows(flat_stack, flat_ref),
        'pearson': _pearson_rows(flat_stack, flat_ref),
        'spearman': _pearson_rows(rankdata(flat_stack, axis=1), rankdata(flat_ref)),
        'hellinger': _hellinger_rows(p_stack, q_ref),
        'jsd': _jensen_shannon_rows(p_stack, q_ref, base=base),
        'weighted_jaccard': _weighted_jaccard_rows(flat_stack, flat_ref),
    }


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

    metrics = compare_upper_triangle_batch(padj_stacked[1:], padj_stacked[0])  # each (n_samples - 1,)
    n_geo = geo_samples[pre].shape[0]
    slices = {'null': slice(0, n_sampled),
              'geo': slice(n_sampled, n_sampled + n_geo),
              'individual': slice(n_sampled + n_geo, None)}
    row = {'pre': pre}
    for metric, values in metrics.items():
        for sample_type, sl in slices.items():
            row[f'{sample_type}_{metric}'] = values[sl]
    results.append(row)


cossims = pd.DataFrame(results)

#%% take average of each 
cols = [col for col in cossims.columns if col not in ['pre']]
cossims_avg = cossims.copy()
for col in cols:
    cossims_avg[col] = cossims_avg[col].apply(np.mean)

#%% plot res 

sample_type = 'individual'
def_metric = 'cosine'
metrics = ['pearson', 'spearman', 'hellinger', 'jsd', 'weighted_jaccard']

for metric in metrics:
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    ax.hist(cossims_avg[f'{sample_type}_{metric}'], bins=np.arange(0, 1.05, 0.05), label=metric,alpha=0.3)
    ax.hist(cossims_avg[f'{sample_type}_{def_metric}'], bins=np.arange(0, 1.05, 0.05), label=def_metric, alpha=0.3)
    ax.set_xlabel(f'{sample_type.capitalize()} {metric.capitalize()}')
    ax.set_ylabel('Frequency')
    ax.legend()

# %%
