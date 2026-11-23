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

#%% create kernels for multivariate regression rsa

# kernel should be constructed as distance matrix, where closer to 0 means more similar, and larger values mean less similar.
''' 
Kernels: 
- hemisphere 
- celltype membership 
- signal flow hierarchical value 
- outgoing connectivity 
- incoming connectivity
- inhibitory vs excitatory 
'''

ct_order = sorted(set(skid_to_celltype.values()))


def _group_codes(neurons, groups):
    """Label each neuron with the index of the group it belongs to, -1 if in none.

    Later groups win, so pass them in reverse priority order.
    """
    codes = np.full(len(neurons), -1, dtype=np.int64)
    for code, group in enumerate(groups):
        codes[np.isin(neurons, np.asarray(group))] = code
    return codes


def get_hemi_kernel(neurons, left_neus=left_neus, right_neus=right_neus, center_neus=center_neus):
    """Create a kernel matrix for hemisphere of neurons."""
    neurons = np.asarray(neurons)
    # priority left > right > center, so build in the reverse order
    codes = _group_codes(neurons, (center_neus, right_neus, left_neus))
    # neurons with no hemisphere (-1) are never "same", matching the loop version
    same = (codes[:, None] == codes[None, :]) & (codes[:, None] >= 0)
    return 1.0 - same

def _celltype_labels(neurons, skid_to_celltype):
    return np.array([skid_to_celltype.get(s, 'NA') for s in neurons], dtype=object)

def get_celltype_kernel(neurons, skid_to_celltype=skid_to_celltype, ct_order=ct_order):
    """Create a kernel matrix for cell type of neurons.
    same/ not same kernel is to examine whether category membership matters in general """
    codes = pd.factorize(_celltype_labels(neurons, skid_to_celltype))[0]
    return (codes[:, None] != codes[None, :]).astype(float)

def get_categorical_celltype_kernel(neurons, skid_to_celltype=skid_to_celltype, ct_order=ct_order):
    """Create a kernel matrix for cell type of neurons.
    categorical kernel is to examine whether specific category membership matters """
    labels = _celltype_labels(neurons, skid_to_celltype)
    member = np.stack([labels == ct for ct in ct_order])  # (n_ct, n)
    return 1.0 - (member[:, :, None] & member[:, None, :])


def get_flow_distance_kernel(neurons, skid_to_flow=skid_to_flow):
    """Create a kernel matrix for flow distance of neurons."""
    flow = np.array([skid_to_flow.get(s, np.nan) for s in neurons], dtype=float)
    return np.abs(flow[:, None] - flow[None, :])

def weighted_jaccard(i, j):
    intersection = np.sum(np.minimum(i, j))
    union = np.sum(np.maximum(i, j))
    if union == 0:
        return 0
    else:
        return 1 - (intersection / union)

def _pairwise_min_sums(X):
    """All-pairs sum_k min(X[i,k], X[j,k]) for non-negative X.

    min(a, b) is only non-zero where both entries are non-zero, so each column
    only contributes an outer minimum over its own non-zero support. Cost is
    sum_k (non-zeros in column k)^2, i.e. set by how sparse the connectivity is
    rather than by how many distinct weights it holds.
    """
    n, m = X.shape
    out = np.zeros((n, n))
    for k in range(m):
        idx = np.nonzero(X[:, k])[0]
        if idx.size == 0:
            continue
        v = X[idx, k]
        # idx has no repeats, so this fancy-indexed += accumulates correctly
        out[np.ix_(idx, idx)] += np.minimum(v[:, None], v[None, :])
    return out

def _weighted_jaccard_matrix(X):
    """Row-wise weighted Jaccard distances, vectorised over all pairs."""
    X = np.asarray(X, dtype=float)
    intersection = _pairwise_min_sums(X)
    # sum(min) + sum(max) == sum(a) + sum(b), so the union comes for free
    totals = X.sum(axis=1)
    union = totals[:, None] + totals[None, :] - intersection
    kernel = np.zeros_like(intersection)
    nz = union != 0
    kernel[nz] = 1.0 - intersection[nz] / union[nz]
    return kernel

def _row_fractions(X):
    """Scale each row to sum to 1, leaving all-zero rows as zeros."""
    X = np.asarray(X, dtype=float)
    totals = X.sum(axis=1, keepdims=True)
    return np.divide(X, totals, out=np.zeros_like(X), where=totals != 0)

def get_similarity_of_inputs_and_outputs(neurons, adj=adj, frac_inputs=True):
    ''' Create a kernel matrix for similarity of inputs of neurons.
    Using weighted Jaccard similarity.
    Frac_inputs: if True, use fraction of inputs, if False, use absolute number of inputs.
    '''
    adj_reindexed = adj.loc[neurons, neurons].to_numpy()
    # rows are what gets compared: columns of adj are a neuron's inputs, rows its outputs
    adj_inputs = adj_reindexed.T
    adj_outputs = adj_reindexed
    if frac_inputs:
        # normalize each neuron's own vector to sum to 1, so distances reflect the
        # fraction of its inputs/outputs rather than the absolute synapse counts
        adj_inputs = _row_fractions(adj_inputs)
        adj_outputs = _row_fractions(adj_outputs)
    return _weighted_jaccard_matrix(adj_inputs), _weighted_jaccard_matrix(adj_outputs)


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

def get_exc_inh_kernel(neurons):
    inhibitory_neurons = get_inhibitory_neurons(threshold=0.0, verbose=False)
    is_inh = np.isin(np.asarray(neurons), np.asarray(inhibitory_neurons))
    return (is_inh[:, None] != is_inh[None, :]).astype(float)



# %% prep kernels 

def vectorize_and_remove_diagonal(kernel):
    ''' For each kernel, take the upper triangle and vectorize it, removing the diagonal.'''
    if kernel.ndim == 2:
        triu_indices = np.triu_indices(kernel.shape[0], k=1)
        return kernel[triu_indices]
    elif kernel.ndim == 3:
        print(f'Kernel has shape {kernel.shape}, vectorizing each slice and removing diagonal.')
        n_kernels = kernel.shape[0]
        vectorized_kernels = []
        for i in range(n_kernels):
            triu_indices = np.triu_indices(kernel.shape[1], k=1)
            vectorized_kernels.append(kernel[i][triu_indices])
        final = np.array(vectorized_kernels).T
        print(f'Final vectorized kernel shape: {final.shape}')
        return final
    else:
        raise ValueError("Kernel must be 2D or 3D array.")

def normalize_kernel(kernel, norm='zscore'):
    if norm == 'zscore':
        mean = np.mean(kernel)
        std = np.std(kernel)
        if std == 0:
            print("Warning: Standard deviation is zero, returning original kernel.")
            return kernel
        return (kernel - mean) / std
    elif norm == 'minmax':
        min_val = np.min(kernel)
        max_val = np.max(kernel)
        if max_val - min_val == 0:
            print("Warning: Max and min are equal, returning original kernel.")
            return kernel
        return (kernel - min_val) / (max_val - min_val)
    else:
        raise ValueError("Normalization method must be 'zscore' or 'minmax'.")


kernels = {}
for kernel_name, func in zip(
    ['hemi', 'ct', 'ct_cat', 'flow', 'inputs', 'outputs', 'inhibitory'],
    [get_hemi_kernel, get_celltype_kernel, get_categorical_celltype_kernel, get_flow_distance_kernel, get_similarity_of_inputs_and_outputs, get_similarity_of_inputs_and_outputs, get_exc_inh_kernel]
):
    print(f'Creating kernel: {kernel_name}')
    if kernel_name in ['inputs', 'outputs']:
        if 'inputs' in kernels.keys() and 'outputs' in kernels.keys():
            continue  # already computed both
        kernels['inputs'], kernels['outputs'] = func(all_neurons)
    else:
        kernels[kernel_name] = func(all_neurons)

norm = 'zscore'
# keep the raw vectors as well as the normalized ones: normalize_kernel z-scores with a
# global mean, so one NaN marks the whole column and there is no way to tell afterwards
# which pairs were actually missing. The regression below masks on kernels_vec.
kernels_vec = {}
kernels_norm = {}
for kernel_name, kernel in kernels.items():
    print(f'Normalizing and vectorizing kernel: {kernel_name}')
    kernel_vec = vectorize_and_remove_diagonal(kernel)
    kernels_vec[kernel_name] = kernel_vec
    kernels_norm[kernel_name] = normalize_kernel(kernel_vec, norm=norm) # ca


# %% load RSA result 
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
    for e, f in enumerate(files):
        if e % 100 == 0:
            print(f'Loading file {e+1}/{n_files}')
        with open(os.path.join(INPUT, f), 'r') as infile:
            data = json.load(infile)
            f_skid = int(f.replace('.json', ''))
            geo_samples[f_skid] = np.array(data['samples']['geometry']['data'])
            real_samples[f_skid] = np.array(data['inputs']['observed_assignment']['data'])
            target_skids[f_skid] = np.array(data['inputs']['target_ids']['data'])

    return geo_samples, real_samples, target_skids

geo_samples, real_samples, target_skids = load_geo_samples(INPUT)

#%% get cotargeting enrichment over geo samples for each neuron, then get some neuron x neuron cotargeting enrichment matrix.

from models.approach2.propensity import pair_count_matrix, mean_pair_count_matrix

# get cotargeting enrichment for each neuron using hyperedges

def _global_positions(local_ids, index_neurons):
    """Map a neuron's own target ids onto positions in the shared index.

    Returned positions follow the order of ``index_neurons`` as given, so the
    pooled matrix lines up with the kernels, which are built on that same order.
    """
    order = np.argsort(index_neurons)
    sorted_index = index_neurons[order]
    slot = np.searchsorted(sorted_index, local_ids)
    if np.any(slot >= len(sorted_index)) or np.any(sorted_index[slot] != local_ids):
        missing = np.setdiff1d(local_ids, index_neurons)
        raise KeyError(f'{len(missing)} target ids are absent from index_neurons, e.g. {missing[:5]}')
    return order[slot]


def build_global_cotargeting(geo_samples, real_samples, target_skids,
                             index_neurons=all_neurons, include_self_repeats=False,
                             keep_perm_stack=True):
    """Pool every presynaptic neuron's sites into one global co-targeting matrix.

    pair_count_matrix is presence.T @ presence, a Gram matrix over sites, and Gram
    matrices are additive over rows. Pooling sites across presynaptic neurons is
    therefore exactly summing each neuron's own matrix scattered into the shared
    target index -- concatenating all ~140k sites into one assignment matrix gives
    an identical answer for far more memory.

    Summing over neurons and averaging over permutations both being linear, they
    commute, so ``expected`` does not depend on how permutations are paired across
    neurons (they are independent draws, so any pairing is arbitrary). The pairing
    only matters for spread, which is why ``keep_perm_stack`` builds the per-perm
    stack needed for a z score. That stack is n_perm x G x G int32, so roughly
    350 MB at G = 2956 and n_perm = 10.
    """
    index_neurons = np.asarray(index_neurons)
    n = len(index_neurons)
    observed = np.zeros((n, n), dtype=np.int64)
    expected = np.zeros((n, n), dtype=np.float64)
    # pair_count_matrix overwrites the diagonal, so keep the per-target site counts
    # here -- they are needed for any odds-ratio style normalisation later.
    site_counts = np.zeros(n, dtype=np.int64)
    perm_stack = None
    n_sites = 0

    for e, skid in enumerate(geo_samples):
        if e % 250 == 0:
            print(f'Pooling neuron {e+1}/{len(geo_samples)}')
        pos = _global_positions(np.asarray(target_skids[skid]), index_neurons)
        grid = np.ix_(pos, pos)

        assignment = np.asarray(real_samples[skid])
        observed[grid] += pair_count_matrix(assignment, include_self_repeats)
        site_counts[pos] += (assignment > 0).sum(axis=0)
        n_sites += assignment.shape[0]

        samples = np.asarray(geo_samples[skid])
        if keep_perm_stack:
            if perm_stack is None:
                perm_stack = np.zeros((samples.shape[0], n, n), dtype=np.int32)
            for p in range(samples.shape[0]):
                perm_stack[p][grid] += pair_count_matrix(samples[p], include_self_repeats).astype(np.int32)
        else:
            expected[grid] += mean_pair_count_matrix(samples, include_self_repeats)

    if keep_perm_stack:
        expected = perm_stack.mean(axis=0)

    return {'index': index_neurons, 'observed': observed, 'expected': expected,
            'perm_stack': perm_stack, 'site_counts': site_counts, 'n_sites': n_sites}


def global_cotargeting_enrichment(pooled, smoothing=1.0):
    """Log enrichment and null z score for the pooled co-targeting matrix.

    ``smoothing`` matches CotargetConfig so pooled scores stay on the same scale as
    the per-neuron ones from estimate_individual_propensity. It is not a minor
    regulariser here: most supported pairs are seen 0 or 1 times, so the pseudocount
    is comparable to the counts themselves and sets the spread of the result.

    ``support`` flags pairs that some presynaptic neuron could actually co-target.
    Unsupported pairs are structurally zero rather than unenriched, so they must be
    masked out before the RSA rather than fed in as a large block of zeros.
    """
    observed = pooled['observed']
    expected = pooled['expected']
    support = (observed + expected) > 0
    log_enrichment = np.log((observed + smoothing) / (expected + smoothing))

    z = np.zeros_like(log_enrichment)
    if pooled['perm_stack'] is not None:
        sd = pooled['perm_stack'].astype(np.float64).std(axis=0)
        # pairs whose null never varied have no scale to divide by; leave them at 0
        np.divide(observed - expected, sd, out=z, where=sd > 0)

    return {'log_enrichment': log_enrichment, 'z': z, 'support': support}


# def get_cotargeting_enrichment(geo_samples, real_samples, target_skids, all_neurons, include_self_repeats=False, smoothing=1.0):
#     global_cotarget = np.zeros((len(all_neurons), len(all_neurons)))

#     cotargeting_enrichment = {}
#     for skid in geo_samples.keys():

#         geo_sample = geo_samples[skid]
#         real_sample = real_samples[skid]
#         target_skid = target_skids[skid]

#         observed = pair_count_matrix(real_sample, include_self_repeats)
#         expected = mean_pair_count_matrix(geo_sample, include_self_repeats)
#         raw = np.log((observed + smoothing) / (expected + smoothing))

''' for global co-target 
- can just remove some presynaptic neurons 
- and the impossible target pairs should be auto-removed further down with the support mask.
'''
presyn_neurons = celltype_df[celltype_df['name']=='KCs']['skids'].values[0]  # all_neurons
# filter 
geo_samples_f = {k: v for k, v in geo_samples.items() if k in presyn_neurons}
real_samples_f = {k: v for k, v in real_samples.items() if k in presyn_neurons}
target_skids_f = {k: v for k, v in target_skids.items() if k in presyn_neurons}

global_cotarget = build_global_cotargeting(geo_samples_f, real_samples_f, target_skids_f, index_neurons=all_neurons, include_self_repeats=False, keep_perm_stack=True)
log_enrichment, zscored_enrichment, support = global_cotargeting_enrichment(global_cotarget, smoothing=0.5).values()

# left un-normalized: the two models below are fitted on different pair subsets, so each
# needs its own mean/sd rather than statistics inherited from the full set.
y_raw = vectorize_and_remove_diagonal(log_enrichment)
support_vec = vectorize_and_remove_diagonal(support).astype(bool)

# y should be a dissimilarity matrix, so the log enrichment is negated to match the kernel distances.
y_raw = -y_raw

# %%

def finite_pair_mask(features, kernels_vec, y_raw=None):
    """Pairs that are finite across every selected feature (and y, if given).

    Runs on the RAW vectors, not kernels_norm: normalize_kernel z-scores with a global
    mean, so a single NaN there marks the whole column and this mask would come back
    empty. Handles 3D kernels (ct_cat), which vectorize to n_pairs x n_categories.
    """
    n_pairs = len(next(iter(kernels_vec.values())))
    mask = np.ones(n_pairs, dtype=bool)
    for feat in features:
        v = np.asarray(kernels_vec[feat], dtype=float)
        ok = np.isfinite(v)
        if ok.ndim > 1:
            ok = ok.all(axis=tuple(range(1, ok.ndim)))
        mask &= ok
    if y_raw is not None:
        mask &= np.isfinite(y_raw)
    return mask


def kernel_column_names(feature, n_cols, ct_order=ct_order):
    """Column labels for one kernel; 3D kernels get one label per category."""
    if n_cols == 1:
        return [feature]
    labels = ct_order if (feature == 'ct_cat' and len(ct_order) == n_cols) else range(n_cols)
    return [f'{feature}[{lab}]' for lab in labels]


def build_design(features, kernels_vec, y_raw, keep, norm=norm, standardize_dummies=False):
    """Subset to ``keep``, then scale on the surviving pairs only.

    Order matters: subsetting first means each column's mean/sd describe the pairs
    actually fitted. 3D kernels (ct_cat) expand to one 0/1 column per category, coded
    0 when both neurons are in that category and 1 otherwise, so the reference level is
    "the pair spans two cell types" and the intercept absorbs it.

    Those indicator columns are left on their raw scale: a one-sd change in a dummy that
    is on for ~0.01% of pairs is not an interpretable contrast, whereas the raw
    coefficient reads directly as the mean y difference between cross-category and
    within-category pairs. The 1D kernels are still standardized so they stay mutually
    comparable (the old normalize_kernel-then-StandardScaler was a z-score either way).

    Returns the design, y, and the column names to pass to ``summary(xname=...)``.
    """
    cols, names, is_dummy = [], [], []
    for f in features:
        v = np.asarray(kernels_vec[f], dtype=float)[keep]
        dummy = v.ndim > 1
        if not dummy:
            v = v[:, None]
        cols.append(v)
        names.extend(kernel_column_names(f, v.shape[1]))
        is_dummy.extend([dummy] * v.shape[1])

    X = np.hstack(cols)
    is_dummy = np.asarray(is_dummy)

    # a category with no surviving within-category pair is constant on this subset: it
    # carries no information, and StandardScaler would quietly turn it into an all-zero
    # column that OLS reports as a 0 coefficient with a nan standard error.
    live = X.std(axis=0) > 0
    if not live.all():
        print(f'  dropping {int((~live).sum())} constant column(s): '
              f'{[n for n, k in zip(names, live) if not k]}')
    X, is_dummy = X[:, live], is_dummy[live]
    names = [n for n, k in zip(names, live) if k]

    scale = np.ones(len(names), dtype=bool) if standardize_dummies else ~is_dummy
    if scale.any():
        X[:, scale] = StandardScaler().fit_transform(X[:, scale])

    y = normalize_kernel(y_raw[keep], norm=norm)
    return sm.add_constant(X), y, names


import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# X: n_pairs x n_features
# y: n_pairs

exclude_nan_pairs = True
use_ct_cat = False  # one column per cell type instead of a single same/different column

# ct == sum_c (1 - ct_cat[:, c]) by construction, so the two are exactly collinear: fit
# one or the other, never both. They also differ on the neurons with no cell type --
# ct_cat has them share nothing, while ct scores two unlabelled neurons as same type.
features = ['hemi', 'ct_cat', 'flow', 'inputs', 'outputs', 'inhibitory'] if use_ct_cat else ['hemi', 'ct', 'flow', 'inputs', 'outputs', 'inhibitory']

n_pairs = len(y_raw)
keep_nan = finite_pair_mask(features, kernels_vec, y_raw) if exclude_nan_pairs \
    else np.ones(n_pairs, dtype=bool)

if exclude_nan_pairs:
    n_removed = int((~keep_nan).sum())
    print(f'exclude_nan_pairs: removed {n_removed} / {n_pairs} pairs '
          f'({100 * n_removed / n_pairs:.2f}%) with NaN across {features}')
    for feat in features:  # attribute the loss, so a new NaN source is visible
        v = np.asarray(kernels_vec[feat], dtype=float)
        bad = ~np.isfinite(v)
        if bad.ndim > 1:
            bad = bad.any(axis=tuple(range(1, bad.ndim)))
        if bad.any():
            print(f'  {feat}: {int(bad.sum())} pairs NaN')

# unsupported pairs are structurally zero rather than unenriched -- fitting them is
# fitting "do these two neurons share any presynaptic partner", so report both.
keep_sup = keep_nan & support_vec
print(f'support: {int(keep_sup.sum())} / {int(keep_nan.sum())} retained pairs are supported '
      f'({100 * keep_sup.sum() / max(keep_nan.sum(), 1):.2f}%)')

models = {}
for label, keep in (('ALL PAIRS', keep_nan), ('SUPPORTED PAIRS ONLY', keep_sup)):
    print(f'\n{"=" * 70}\n{label}: n = {int(keep.sum())}\n{"=" * 70}')
    X_design, y_fit, xnames = build_design(features, kernels_vec, y_raw, keep)
    models[label] = sm.OLS(y_fit, X_design).fit()
    print(models[label].summary(xname=['const'] + xnames))



# %%
