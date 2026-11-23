from pathlib import Path
import os
import sys
import ast
import time 
import json

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

#%% get cotargeting propensity scores 



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



def estimate_individual_propensity_rich(
    observed_assignment: np.ndarray,
    expected_samples: np.ndarray,
    include_self: bool = True,
    smoothing: float = 1.0,
    shrinkage: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate shrunk log enrichment relative to a simulated expectation."""
    observed = pair_count_matrix(observed_assignment, include_self)
    expected = mean_pair_count_matrix(expected_samples, include_self)
    expected_sd = np.std([pair_count_matrix(sample, include_self) for sample in expected_samples], axis=0)
    raw = np.log((observed + smoothing) / (expected + smoothing))
    delta = observed - expected
    if not include_self:
        np.fill_diagonal(raw, 0.0)
    return raw, delta, observed, expected, expected_sd

def to_supplied_format(propensity_scores, delta, observed, expected, expected_sd, u_targets, skeleton_id=None, as_df=True):
    """Convert a propensity score matrix to a DataFrame with the expected columns."""
    # dont need duplicates so take just upper triangle 
    target_a, target_b = np.meshgrid(u_targets, u_targets)
    mask = np.triu(np.ones_like(target_a, dtype=bool), k=0)
    out = np.array([target_a[mask], target_b[mask], propensity_scores[mask], delta[mask], observed[mask], expected[mask], expected_sd[mask]]).T
    if skeleton_id is not None:
        pre_col = np.full(out.shape[0], skeleton_id, dtype=int)
        out = np.column_stack((pre_col, out))
    if as_df:
        columns = ['pre', 'target1', 'target2', 'score', 'delta', 'observed', 'expected', 'expected_sd'] if skeleton_id is not None else ['target_a', 'target_b', 'score', 'delta', 'observed', 'expected', 'expected_sd']
        out = pd.DataFrame(out, columns=columns)
    return out


### generate relative to geometry null 
rng = np.random.default_rng(23)
prop_tables = []

all_presynaptic = con[presyn_col].unique()
n_pre = len(all_presynaptic)

# load geometry samples
INPUT = 'out/approach2/geo_and_full_individual_cotargeting/'
geo_samples, real_samples, target_skids = load_geo_samples(INPUT)

include_self = True
smoothing = 1.0
shrinkage = 10.0

for e, skeleton_id in enumerate(all_presynaptic[:]):
    if e % 100 == 0:
        print(f"Processing skeleton {e+1}/{n_pre}")
    hyperedges = con.loc[con[presyn_col]==skeleton_id][postsyn_col].tolist()
    real_assignment, u_targets = real_samples[skeleton_id], target_skids[skeleton_id]

    expected_assignment = geo_samples[skeleton_id] # n_samples x n_sites x n_targets

    propensity_scores, delta, observed, expected, expected_sd = estimate_individual_propensity_rich(
        observed_assignment=real_assignment,
        expected_samples=expected_assignment,
        include_self=include_self,
        smoothing=smoothing,
        shrinkage=shrinkage,
    ) # gives n_target x n_target matrix of scores

    prop_table = to_supplied_format(propensity_scores, delta, observed, expected, expected_sd, u_targets, skeleton_id=skeleton_id, as_df=False)
    prop_tables.append(prop_table)

all_prop_table = np.vstack(prop_tables)
all_prop_table = pd.DataFrame(all_prop_table, columns=['pre', 'target1', 'target2', 'score', 'delta', 'observed', 'expected', 'expected_sd'])

# filter out score = 0.0 since this is the default anyways 
#all_prop_table = all_prop_table.loc[all_prop_table['delta'] != 0.0]

#%% generate an averaged prop_table 

# targets need to be symmetric 
# give each target combo an identifier that works both ways 
all_prop_table['target_min'] = all_prop_table[['target1', 'target2']].min(axis=1)
all_prop_table['target_max'] = all_prop_table[['target1', 'target2']].max(axis=1)
avg_prop_table = all_prop_table.groupby(['target_min', 'target_max']).agg({'score':'mean','delta':'mean','observed':'sum','expected':'sum','expected_sd':'mean', 'target1':'count'}).reset_index().rename(columns={'target1':'n_presynaptic'})
avg_prop_table.rename(columns={'target_min':'target1','target_max':'target2', 'score':'mean_score', 'delta':'mean_delta', 'expected_sd':'mean_expected_sd'}, inplace=True)
avg_prop_table['total_delta'] = avg_prop_table['observed'] - avg_prop_table['expected']
pseudo=0.5
avg_prop_table['fold_enrichment'] = (avg_prop_table['observed']+ pseudo) / (avg_prop_table['expected']+ pseudo) # avoid divide by zero
avg_prop_table['log2_enrichment'] = np.log2(avg_prop_table['fold_enrichment'])

#%% get neuron metadata
adj = pd.read_csv('data/adj.csv')
adj.set_index('sources', inplace=True)
adj = adj.astype(int)
adj.columns = adj.columns.astype(int)
adj.index = adj.index.astype(int)

source, target = np.meshgrid(adj.index.values, adj.columns.values, indexing='ij')
edges = np.array([source.flatten(), target.flatten(), adj.values.flatten()])
edges = pd.DataFrame(edges.T, columns=['source', 'target', 'weight'])
edges_nonzero = edges.loc[edges['weight'] > 0].reset_index(drop=True)

# presynaptic neuron metadata 
pre = pd.DataFrame({'pre_id': edges['source'].unique()})
pre['pre_celltype'] = pre['pre_id'].apply(lambda x: skid_to_celltype.get(x, 'NA'))
pre['pre_ei'] = pre['pre_id'].map(lambda x: 'I' if x in inhibitory_neurons else 'E')
pre['pre_hemisphere'] = pre['pre_id'].map(lambda x: 'L' if x in left_neus else ('R' if x in right_neus else 'C'))
pre['pre_hierarchy'] = pre['pre_id'].map(skid_to_flow)
# input neus 
pre['pre_input_neus'] = pre['pre_id'].apply(lambda x: edges_nonzero.loc[edges_nonzero['target']==x]['source'].count())
pre['pre_output_neus'] = pre['pre_id'].apply(lambda x: edges_nonzero.loc[edges_nonzero['source']==x]['target'].count())
pre['pre_input_synapses'] = pre['pre_id'].apply(lambda x: edges_nonzero.loc[edges_nonzero['target']==x]['weight'].sum())
pre['pre_output_synapses'] = pre['pre_id'].apply(lambda x: edges_nonzero.loc[edges_nonzero['source']==x]['weight'].sum())
pre[['pre_input_neus', 'pre_output_neus', 'pre_input_synapses', 'pre_output_synapses']].fillna(0, inplace=True)

# postsynaptic neuron metadata
post = pd.DataFrame({'neuron_id': edges['target'].unique()})
post['celltype'] = post['neuron_id'].apply(lambda x: skid_to_celltype.get(x, 'NA'))
post['ei'] = post['neuron_id'].map(lambda x: 'I' if x in inhibitory_neurons else 'E')
post['hierarchy'] = post['neuron_id'].map(skid_to_flow)
post['hemisphere'] = post['neuron_id'].map(lambda x: 'L' if x in left_neus else ('R' if x in right_neus else 'C'))
post['input_neus'] = post['neuron_id'].apply(lambda x: edges_nonzero.loc[edges_nonzero['target']==x]['source'].count())
post['output_neus'] = post['neuron_id'].apply(lambda x: edges_nonzero.loc[edges_nonzero['source']==x]['target'].count())
post['input_synapses'] = post['neuron_id'].apply(lambda x: edges_nonzero.loc[edges_nonzero['target']==x]['weight'].sum())
post['output_synapses'] = post['neuron_id'].apply(lambda x: edges_nonzero.loc[edges_nonzero['source']==x]['weight'].sum())
post[['input_neus', 'output_neus', 'input_synapses', 'output_synapses']].fillna(0, inplace=True)

#%% get structurally possible target pairs 
from scipy.sparse import csr_matrix

# grouped_targets = edges_nonzero.groupby('source')['target'].apply(list).reset_index()
# grouped_hyperedges = grouped_targets['target'].tolist()

# def edges_to_cotarget_num(edges):
#     e = edges.loc[edges['weight'] > 0]
#     e = e.drop_duplicates(['source', 'target'])        # == unique_within=True

#     rows, _      = pd.factorize(e['source'])
#     cols, labels = pd.factorize(e['target'])
#     H = csr_matrix((np.ones(len(e), np.int32), (rows, cols)),
#                 shape=(rows.max() + 1, cols.max() + 1))
#     C = (H.T @ H).toarray()
#     np.fill_diagonal(C, 0)
#     return C, labels


def edges_to_cotarget_num_and_weight(edges):
    """Target x target co-occurrence counts and summed min-weight.

    C[a, b] = number of sources projecting to both a and b
    W[a, b] = sum over those sources of min(weight[s, a], weight[s, b])
    """
    e = edges.loc[edges['weight'] > 0]
    e = e.drop_duplicates(['source', 'target'])        # == unique_within=True

    rows, _      = pd.factorize(e['source'])
    cols, labels = pd.factorize(e['target'])
    w = e['weight'].to_numpy()
    shape = (rows.max() + 1, cols.max() + 1)

    H = csr_matrix((np.ones(len(e), np.int32), (rows, cols)), shape=shape)
    C = (H.T @ H).toarray()

    # sort so each source occupies one contiguous block of rows
    order = np.argsort(rows, kind='stable')
    cols, w, rows = cols[order], w[order], rows[order]
    deg    = np.bincount(rows, minlength=shape[0])
    starts = np.concatenate([[0], np.cumsum(deg)[:-1]])

    # every (i, j) pair within every block, without a Python loop
    block  = deg * deg
    g      = np.repeat(np.arange(len(deg)), block)
    local  = np.arange(block.sum()) - np.repeat(np.concatenate([[0], np.cumsum(block)[:-1]]), block)
    left   = starts[g] + local // deg[g]
    right  = starts[g] + local %  deg[g]

    # csr_matrix sums duplicate (row, col) entries, doing the accumulation for us
    W = csr_matrix((np.minimum(w[left], w[right]), (cols[left], cols[right])),
                   shape=(shape[1], shape[1])).toarray()

    np.fill_diagonal(C, 0)
    np.fill_diagonal(W, 0)
    return C, W, labels


C_counts, W, target_labels = edges_to_cotarget_num_and_weight(edges)

pre_pair_numbers = pd.DataFrame(C_counts, index=target_labels, columns=target_labels)
pre_pair_min_shared_weights = pd.DataFrame(W, index=target_labels, columns=target_labels)

possible_pairs = pre_pair_numbers[pre_pair_numbers > 0].stack().reset_index()
possible_pairs['target_min'] = possible_pairs[['level_0', 'level_1']].min(axis=1)
possible_pairs['target_max'] = possible_pairs[['level_0', 'level_1']].max(axis=1)
possible_pairs.drop_duplicates(subset=['target_min', 'target_max'], inplace=True)
possible_pairs = possible_pairs[['target_min', 'target_max']]

avg_prop_table.reset_index(drop=True, inplace=True)
avg_prop_table['target_min'] = avg_prop_table[['target1', 'target2']].min(axis=1)
avg_prop_table['target_max'] = avg_prop_table[['target1', 'target2']].max(axis=1)
avg_prop_table.drop_duplicates(subset=['target_min', 'target_max'], inplace=True)
all_prop_2 = possible_pairs.merge(avg_prop_table, left_on=['target_min', 'target_max'], right_on=['target_min', 'target_max'], how='left')

all_prop_2.fillna({'mean_score':0.0, 'mean_delta':0.0, 'observed':0.0, 'expected':0.0, 'mean_expected_sd':0.0, 'n_presynaptic':0, 'total_delta':0.0, 'fold_enrichment':1.0, 'log2_enrichment':0.0}, inplace=True)

# %% categorise by structural possibility & observations 

#%%
fig, ax = plt.subplots(1,1,figsize=(4,4))
ax.scatter(all_prop_2['expected'], all_prop_2['observed'], alpha=0.5, s=2)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Expected (log)')
ax.set_ylabel('Observed (log)')
max_val = max(all_prop_2['expected'].max(), all_prop_2['observed'].max())
ax.set_xlim(1e-1, max_val)
ax.set_ylim(1e-1, max_val)

all_prop_2['group_cat'] = all_prop_2.apply(lambda row: 'possible_and_observed' if (row['observed'] > 1 and row['expected'] > 1.0) else ('low_expected_and_never_observed' if (row['expected'] <= 1.0 and row['observed']==0) else ('low_expected_but_observed' if (row['expected'] <= 1.0 and row['observed']>0 ) else 'expected_but_not_observed')), axis=1)

#%%
fig,ax = plt.subplots(1,3,figsize=(10,6))
sns.countplot(data=all_prop_2, x='group_cat', ax=ax[0])
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha='right')
sns.barplot(x=[1,2], y=[len(adj)**2 - len(adj), len(all_prop_2)], ax=ax[1], color='lightgrey', label=['theoretically possible pairs', 'structurally possible pairs'])
bins = np.arange(-0.5, all_prop_2[all_prop_2['group_cat'] == 'low_expected_but_observed']['observed'].max()+0.5, 1)
ax[2].hist(all_prop_2[all_prop_2['group_cat'] == 'low_expected_but_observed']['observed'], bins=bins)
ax[2].set_xlabel('Observed co-targeting count')
ax[2].set_ylabel('count')
ax[2].set_title('Observed counts for low expected (<=1) but observed pairs')
ax[2].set_xticks(np.arange(0, all_prop_2[all_prop_2['group_cat'] == 'low_expected_but_observed']['observed'].max()+1, 1))

fig.tight_layout()
# --> should i focus just on possible_and_observed + expected_but_not_observed?
only_expected = all_prop_2[all_prop_2['expected'] > 1.0]

# %% 

''' Among anatomically possible target pairs with enough null opportunity to be informative, 
 which pairwise biological features predict enrichment or depletion of co-targeting relative to the null?
'''

#%% get metadata 

data = only_expected[['target_min', 'target_max', 'observed', 'expected', 'fold_enrichment', 'log2_enrichment']].copy()

# celltype pair 
ct_names = sorted(list(set(skid_to_celltype.values()))) + ['NA']
data['target_min_ct'] = data['target_min'].map(lambda x: skid_to_celltype.get(x, 'NA'))
data['target_max_ct'] = data['target_max'].map(lambda x: skid_to_celltype.get(x, 'NA'))
ct_pairs = data[['target_min_ct', 'target_max_ct']].apply(lambda row: tuple(sorted([row['target_min_ct'], row['target_max_ct']])), axis=1)
data['ct_pair'] = [f'{a}_{b}' for a, b in ct_pairs]

# E/I pair 
data['target_min_ei'] = data['target_min'].map(lambda x: 'I' if x in inhibitory_neurons else 'E')
data['target_max_ei'] = data['target_max'].map(lambda x: 'I' if x in inhibitory_neurons else 'E')
ei_pairs = data[['target_min_ei', 'target_max_ei']].apply(lambda row: tuple(sorted([row['target_min_ei'], row['target_max_ei']])), axis=1)
data['ei_pair'] = [f'{a}{b}' for a, b in ei_pairs]

# hierarchy mean
data['target_min_flow'] = data['target_min'].map(lambda x: skid_to_flow.get(x, np.nan))
data['target_max_flow'] = data['target_max'].map(lambda x: skid_to_flow.get(x, np.nan))
data['mean_hierarchy'] = data[['target_min_flow', 'target_max_flow']].mean(axis=1)
# hierarchy distance
data['hierarchy_distance'] = (data['target_min_flow'] - data['target_max_flow']).abs()


# input/output neuron and synapse counts
target_mins = data['target_min'].values
target_maxs = data['target_max'].values
input_neu_diff = [abs(post.loc[post['neuron_id']==t1]['input_neus'].values[0] - post.loc[post['neuron_id']==t2]['input_neus'].values[0]) if t1 in post['neuron_id'].values and t2 in post['neuron_id'].values else np.nan for t1, t2 in zip(target_mins, target_maxs)]
output_neu_diff = [abs(post.loc[post['neuron_id']==t1]['output_neus'].values[0] - post.loc[post['neuron_id']==t2]['output_neus'].values[0]) if t1 in post['neuron_id'].values and t2 in post['neuron_id'].values else np.nan for t1, t2 in zip(target_mins, target_maxs)]
input_syn_diff = [abs(post.loc[post['neuron_id']==t1]['input_synapses'].values[0] - post.loc[post['neuron_id']==t2]['input_synapses'].values[0]) if t1 in post['neuron_id'].values and t2 in post['neuron_id'].values else np.nan for t1, t2 in zip(target_mins, target_maxs)]
output_syn_diff = [abs(post.loc[post['neuron_id']==t1]['output_synapses'].values[0] - post.loc[post['neuron_id']==t2]['output_synapses'].values[0]) if t1 in post['neuron_id'].values and t2 in post['neuron_id'].values else np.nan for t1, t2 in zip(target_mins, target_maxs)]
data['input_neu_diff'] = input_neu_diff
data['output_neu_diff'] = output_neu_diff
data['input_syn_diff'] = input_syn_diff
data['output_syn_diff'] = output_syn_diff

# input & output cosine OR jaccard on input/output fraction normalised data 
all_targets = np.unique(np.concatenate([data['target_min'].values,
                                        data['target_max'].values])).astype(int)

input_meta  = adj.loc[:, all_targets].T   # rows = target, cols = presynaptic partners
output_meta = adj.loc[all_targets, :]     # rows = target, cols = postsynaptic partners

def cosine_gram(df):
    """Row-wise cosine similarity matrix; all-zero rows yield NaN."""
    X = df.to_numpy(dtype=float)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = np.divide(X, n, out=np.zeros_like(X), where=n > 0)
    S = Xn @ Xn.T
    dead = (n.ravel() == 0)
    S[dead, :] = np.nan
    S[:, dead] = np.nan
    return S

target_to_index = {t: i for i, t in enumerate(all_targets)}
ia = data['target_min'].astype(int).map(target_to_index).to_numpy()
ib = data['target_max'].astype(int).map(target_to_index).to_numpy()

data['input_cosine']  = cosine_gram(input_meta)[ia, ib]
data['output_cosine'] = cosine_gram(output_meta)[ia, ib]


def weighted_jaccard_pairs(df, ia, ib, chunk=1000):
    """Ruzicka similarity sum(min) / sum(max) on input/output fraction rows.

    Unlike cosine this is not scale invariant, so the L1 normalisation is what
    makes it a comparison of connectivity *distributions* rather than of raw
    synapse counts. Only the requested pairs are evaluated, in column-pruned
    chunks, since the full n_target x n_target min/max cannot be formed by a
    single matrix product the way the cosine gram can.
    """
    X = df.to_numpy(dtype=float)
    s = X.sum(axis=1, keepdims=True)
    Xn = np.divide(X, s, out=np.zeros_like(X), where=s > 0)
    Xn = Xn[:, Xn.any(axis=0)]        # all-zero columns add nothing to min or max
    dead = (s.ravel() == 0)

    out = np.empty(len(ia), dtype=float)
    for start in range(0, len(ia), chunk):
        sl = slice(start, start + chunk)
        a, b = Xn[ia[sl]], Xn[ib[sl]]
        inter = np.minimum(a, b).sum(axis=1)
        union = np.maximum(a, b).sum(axis=1)
        out[sl] = np.divide(inter, union, out=np.full(inter.shape, np.nan), where=union > 0)
    out[dead[ia] | dead[ib]] = np.nan  # a neuron with no partners is undefined, not dissimilar
    return out

data['input_jaccard_w']  = weighted_jaccard_pairs(input_meta, ia, ib)
data['output_jaccard_w'] = weighted_jaccard_pairs(output_meta, ia, ib)

# also add bilateral pairs? 
data['target_min_pair'] = data['target_min'].apply(lambda x: str(skid_to_pair.get(x, 'unpaired')))
data['target_max_pair'] = data['target_max'].apply(lambda x: str(skid_to_pair.get(x, 'unpaired')))

# %% Process features 

cat_features = ['ct_pair', 'ei_pair']
cont_features = ['mean_hierarchy', 'hierarchy_distance', 'input_cosine', 'output_cosine', 'input_jaccard_w', 'output_jaccard_w', 'input_neu_diff', 'output_neu_diff', 'input_syn_diff', 'output_syn_diff']
cont_features_z = [f'{cf}_z' for cf in cont_features]

# zscore continuous features
for cf in cont_features:
    data[f'{cf}_z'] = (data[cf] - data[cf].mean()) / data[cf].std()


# %% assess correlations 
from scipy.spatial.distance import correlation
# continuous x continuous spearman & pearsons 
# also add distance correlation?
def distance_correlation(x, y, max_n=3000, rng=np.random.default_rng(0)):
    """Szekely-Rizzo distance correlation; NaN rows dropped, subsampled if large."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return np.nan
    if len(x) > max_n:                       # O(n^2) memory - must subsample
        idx = rng.choice(len(x), max_n, replace=False)
        x, y = x[idx], y[idx]

    def centred(v):
        d = np.abs(v[:, None] - v[None, :])
        return d - d.mean(0)[None, :] - d.mean(1)[:, None] + d.mean()

    A, B = centred(x), centred(y)
    dcov2 = (A * B).mean()
    denom = np.sqrt((A * A).mean() * (B * B).mean())
    return np.sqrt(max(dcov2, 0.0) / denom) if denom > 0 else np.nan


cont_corrs = []
for i, f1 in enumerate(cont_features_z):
    for j, f2 in enumerate(cont_features_z):
        if j <= i:
            continue
        a, b = data[f1].to_numpy(dtype=float), data[f2].to_numpy(dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)     # one mask for all three metrics
        cont_corrs.append({
            'feature1': f1,
            'feature2': f2,
            'n_used': int(ok.sum()),
            'spearman_rho': data.loc[ok, [f1, f2]].corr(method='spearman').iloc[0, 1],
            'pearson_r': data.loc[ok, [f1, f2]].corr(method='pearson').iloc[0, 1],
            'distance_corr': distance_correlation(a[ok], b[ok]),
        })

cont_corrs = pd.DataFrame(cont_corrs)
# plot quick heatmap 
cont_corrs_pivot = cont_corrs.pivot(index='feature1', columns='feature2', values='spearman_rho')
order = cont_corrs_pivot.count(axis=1).sort_values(ascending=False).index
cont_corrs_pivot = cont_corrs_pivot.reindex(order, axis=0)
cont_corrs_pivot = cont_corrs_pivot.reindex(order, axis=1, fill_value=np.nan)
vals = cont_corrs_pivot.values
vals = np.nan_to_num(vals, nan=0.0)  # fill NaN with 0 for plotting
vals = (vals + vals.T)  # make symmetric
vals[np.diag_indices_from(vals)] = np.nan
cont_corrs_pivot = pd.DataFrame(vals, index=cont_corrs_pivot.index, columns=cont_corrs_pivot.columns)

fig, ax = plt.subplots(1,1,figsize=(8,6))
sns.heatmap(cont_corrs_pivot, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)

# categorical x categorical: cramer's V
from scipy.stats.contingency import association
from scipy.stats import chi2_contingency
cats = data[['ct_pair', 'ei_pair']]
ctpair_to_cat = {cat: i for i, cat in enumerate(cats['ct_pair'].unique())}
eipair_to_cat = {cat: i for i, cat in enumerate(cats['ei_pair'].unique())}
cats['ct_pair_cat'] = cats['ct_pair'].map(ctpair_to_cat)
cats['ei_pair_cat'] =cats['ei_pair'].map(eipair_to_cat)

def cramers_v(a, b, bias_correction=True):
    """Cramer's V between two categorical Series, with optional Bergsma correction."""
    tab = pd.crosstab(a, b)
    tab = tab.loc[tab.sum(axis=1) > 0, tab.sum(axis=0) > 0]
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan
    chi2 = chi2_contingency(tab.to_numpy(), correction=False).statistic
    n = tab.to_numpy().sum()
    phi2 = chi2 / n
    r, k = tab.shape
    if not bias_correction:
        return np.sqrt(phi2 / (min(r, k) - 1))
    phi2 = max(0.0, phi2 - (r - 1) * (k - 1) / (n - 1))
    r = r - (r - 1) ** 2 / (n - 1)
    k = k - (k - 1) ** 2 / (n - 1)
    return np.sqrt(phi2 / (min(r, k) - 1))

cat_corr = cramers_v(cats['ct_pair_cat'], cats['ei_pair_cat'], bias_correction=False)
cat_corr_corrected = cramers_v(cats['ct_pair_cat'], cats['ei_pair_cat'], bias_correction=True)

# %% fit the model 

import statsmodels.api as sm
import statsmodels.formula.api as smf

# get log(expected)
data['log_expected'] = np.log(data['expected'])

# first model: observed ~ offset(log_expected) + features + random_effect_target1 + random_effect_target2

# need to find a way for target effect to be symmetric, so that target1 and target2 are interchangeable.
target_ids = np.unique(np.concatenate([data['target_min'].values, data['target_max'].values])).astype(int)
t_to_idx = {t: i for i, t in enumerate(target_ids)}
target_min_idx = data['target_min'].astype(int).map(t_to_idx).to_numpy()
target_max_idx = data['target_max'].astype(int).map(t_to_idx).to_numpy()

# %% bayesian NB GLM with offset
import pymc as pm
import arviz as az
from patsy import dmatrix

# pick one similarity metric per side if cosine and weighted jaccard are near-redundant (|spearman| > 0.8)
def pick_similarity_features(side, thresh=0.8):
    m1 = (cont_corrs['feature1'] == f'{side}_cosine_z') & (cont_corrs['feature2'] == f'{side}_jaccard_w_z')
    m2 = (cont_corrs['feature1'] == f'{side}_jaccard_w_z') & (cont_corrs['feature2'] == f'{side}_cosine_z')
    rho = cont_corrs.loc[m1 | m2, 'spearman_rho'].iloc[0]
    print(f'{side}: cosine vs jaccard spearman rho = {rho:.3f} -> ' + ('keeping cosine only' if abs(rho) > thresh else 'keeping both'))
    return [f'{side}_cosine_z'] if abs(rho) > thresh else [f'{side}_cosine_z', f'{side}_jaccard_w_z']
# and only using neu diff z, not syn diff z
model_cont = ['mean_hierarchy_z', 'hierarchy_distance_z'] + pick_similarity_features('input') + pick_similarity_features('output') + ['input_neu_diff_z', 'output_neu_diff_z']

mdata = data.dropna(subset=model_cont).reset_index(drop=True).copy()
print(f'dropped {len(data) - len(mdata)} of {len(data)} pairs with NaN features, {len(mdata)} remain')
mdata['same_ct'] = ((mdata['target_min_ct'] == mdata['target_max_ct']) & (mdata['target_min_ct'] != 'NA')).astype(float)
mdata['same_pair'] = (mdata['target_min_pair'] == mdata['target_max_pair']).astype(float)

# indices rebuilt on mdata (NaN rows dropped, so data-level indices would misalign)
ct_levels = np.sort(mdata['ct_pair'].unique())
ct_idx = mdata['ct_pair'].map({c: i for i, c in enumerate(ct_levels)}).to_numpy()
ei_levels = np.sort(mdata['ei_pair'].unique())
ei_idx = mdata['ei_pair'].map({c: i for i, c in enumerate(ei_levels)}).to_numpy()
### use bilateral pair as random effect
min_col, max_col = 'target_min_pair', 'target_max_pair'
u_model_targets = np.unique(np.concatenate([mdata[min_col].to_numpy(), mdata[max_col].to_numpy()]))
imin = mdata[min_col].map({t: i for i, t in enumerate(u_model_targets)}).to_numpy()
imax = mdata[max_col].map({t: i for i, t in enumerate(u_model_targets)}).to_numpy()
y_obs = mdata['observed'].to_numpy(dtype=float).round().astype(int)
log_expected_off = np.log(mdata['expected'].to_numpy(dtype=float))
print(f'{len(ct_levels)} ct_pair levels, {len(u_model_targets)} unique targets, {y_obs.sum()} total co-targeting events, {np.mean(y_obs == 0):.1%} zero pairs')

# %% model factory: observed ~ NB(mu, alpha), log(mu) = log(expected) + eta -> exp(eta) is fold-enrichment over the geometry null
# ct_pair enters as a hierarchical (partially pooled) effect so sparse levels shrink to 0; target effects are
# multi-membership (one shared vector u indexed by both target_min and target_max), symmetric by construction.
def build_model(fixed_formula='1', family='nb', use_ei=False, use_ct=False, use_target_re=False):
    dm = dmatrix(fixed_formula, mdata, return_type='dataframe')
    X = dm.to_numpy()
    coords = {'coef': list(dm.columns), 'ei_level': ei_levels, 'ct_level': ct_levels, 'target': u_model_targets}
    with pm.Model(coords=coords) as model:
        b = pm.Normal('b', 0, 2, dims='coef')
        eta = pm.math.dot(X, b)
        if use_ei:
            # few levels -> sigma_ei is weakly identified and leans on its prior; effects are deviations from the
            # intercept (shrunk to 0), not contrasts vs a reference level as with C(ei_pair)
            sigma_ei = pm.HalfNormal('sigma_ei', 1.0)
            a_ei = pm.Deterministic('a_ei', sigma_ei * pm.Normal('z_ei', 0, 1, dims='ei_level'), dims='ei_level')
            eta = eta + a_ei[ei_idx]
        if use_ct:
            sigma_ct = pm.HalfNormal('sigma_ct', 1.0)
            a_ct = pm.Deterministic('a_ct', sigma_ct * pm.Normal('z_ct', 0, 1, dims='ct_level'), dims='ct_level')
            eta = eta + a_ct[ct_idx]
        if use_target_re:
            sigma_u = pm.HalfNormal('sigma_u', 1.0)
            u = pm.Deterministic('u', sigma_u * pm.Normal('z_u', 0, 1, dims='target'), dims='target')
            eta = eta + u[imin] + u[imax]
        mu = pm.math.exp(log_expected_off + eta)
        if family == 'nb':
            alpha = pm.Gamma('alpha', 2.0, 0.1)  # NB dispersion, larger alpha -> closer to poisson
            pm.NegativeBinomial('y', mu=mu, alpha=alpha, observed=y_obs)
        else:
            pm.Poisson('y', mu=mu, observed=y_obs)
    return model

def fit_model(model, **kw):
    with model:
        return pm.sample(draws=1000, tune=1000, chains=4, cores=4, target_accept=0.9, idata_kwargs={'log_likelihood': True}, random_seed=23, **kw)  # if slow: pip install nutpie, pass nuts_sampler='nutpie'

# posterior-mean linear predictor split into components (fixed, ei_pair, ct_pair, target re) for residuals & variance partition
def eta_components(name, idata):
    dm = dmatrix(specs[name]['fixed_formula'], mdata, return_type='dataframe')
    b_hat = idata.posterior['b'].mean(('chain', 'draw')).to_numpy()
    comps = {'fixed': dm.to_numpy() @ b_hat}
    if 'a_ei' in idata.posterior:
        comps['ei_pair'] = idata.posterior['a_ei'].mean(('chain', 'draw')).to_numpy()[ei_idx]
    if 'a_ct' in idata.posterior:
        comps['ct_pair'] = idata.posterior['a_ct'].mean(('chain', 'draw')).to_numpy()[ct_idx]
    if 'u' in idata.posterior:
        u_hat = idata.posterior['u'].mean(('chain', 'draw')).to_numpy()
        comps['target_re'] = u_hat[imin] + u_hat[imax]
    return comps, dm

# %% fit the model ladder, compare with LOO
f_main = '1 + ' + ' + '.join(model_cont)  # categorical features enter as hierarchical effects, continuous features as fixed effects
# with no C(ei_pair) or C(ei_pair) main effect in the formula, patsy codes the interaction full-rank: one cosine slope
f_m4a = f_main + ' + C(ei_pair):(' + ' + '.join([f for f in model_cont if 'cosine' in f]) + ')'
f_m4c = f_main + ' + C(ct_pair):(' + ' + '.join([f for f in model_cont if 'cosine' in f]) + ')'
f_m4aa = f_main + ' + C(target_re):(' + ' + '.join([f for f in model_cont if 'cosine' in f]) + ')'
f_m4b = f_m4a + ' + mean_hierarchy_z:hierarchy_distance_z + input_cosine_z:output_cosine_z + same_ct:(input_cosine_z + output_cosine_z)'
save_to_disk = True
only_run_nonexistent = True

specs = {}
# base enrichment from geometry null, no features
specs['M0_null'] = dict(fixed_formula='1', family='nb')
# continuous features only, no categorical effects - trial with poisson to see if it fits better than NB
# specs['M1_pois'] = dict(fixed_formula=f_main, family='poisson') # this is much worse
specs['M1_main'] = dict(fixed_formula=f_main, family='nb')
# add categoricals one by one 
specs['M2_ei'] = dict(fixed_formula=f_main, family='nb', use_ei=True)
specs['M2_ct'] = dict(fixed_formula=f_main, family='nb', use_ei=False, use_ct=True)
specs['M2_targetre'] = dict(fixed_formula=f_main, family='nb', use_ei=False, use_ct=False, use_target_re=True)
# add both categoricals
specs['M3_ei_ct'] = dict(fixed_formula=f_main, family='nb', use_ei=True, use_ct=True)
# add target random effect to shrink sparse targets
specs['M4_targetre'] = dict(fixed_formula=f_main, family='nb', use_ei=False, use_ct=True, use_target_re=True)
# target re and interactions 
specs['M4aa_targetre_x_sim'] = dict(fixed_formula=f_m4aa, family='nb', use_ei=False, use_ct=False, use_target_re=True)
# add interactions between ei_pair and continuous features, then add more interactions
# could use celltype etc instead later
specs['M4a_ei_x_sim'] = dict(fixed_formula=f_m4a, family='nb', use_ei=True, use_ct=True, use_target_re=True)
specs['M4c_ct_x_sim'] = dict(fixed_formula=f_m4c, family='nb', use_ei=False, use_ct=True, use_target_re=True)

# specs['M4b_full_int'] = dict(fixed_formula=f_m4b, family='nb', use_ei=True, use_ct=True, use_target_re=True)

output_dir = 'model_out/'
models, idatas = {}, {}
for name, spec in specs.items():
    if only_run_nonexistent and os.path.exists(f'{output_dir}{name}.nc'):
        print(f'{name} already exists, skipping')
        models[name] = build_model(**spec)
        idatas[name] = az.from_netcdf(f'{output_dir}{name}.nc')
        continue
    t0 = time.time()
    models[name] = build_model(**spec)
    idatas[name] = fit_model(models[name])
    if save_to_disk:
        idatas[name].to_netcdf(f'{output_dir}{name}.nc')
    print(f'{name} fitted in {time.time() - t0:.0f}s')

loo_comp = az.compare(idatas, ic='loo')
print(loo_comp[['rank', 'elpd_loo', 'elpd_diff', 'p_loo', 'weight', 'se', 'dse']])
loo_comp
''' rank: model rank, 0=best, elpd: expected log predictive density, higher is better, unless scale=neg_log,
 p_loo: effective number of parameters, weight: model weight (~ how likely the model is relative to others, redundant predictive power causes low weight if stacking),
   se: standard error of elpd_diff, dse: standard error of p_loo difference '''

# for name, idata in idatas.items():
#     idata.to_netcdf(f'{output_dir}{name}.nc')

# %% convergence diagnostics + posterior predictive checks
### checking posteriors to determine whether you can trust inference
for name, idata in idatas.items():
    s = az.summary(idata, var_names=['b'], kind='diagnostics')
    print(f"{name}: max rhat {s['r_hat'].max():.3f}, min ess_bulk {s['ess_bulk'].min():.0f}, divergences {int(idata.sample_stats['diverging'].sum())}")
''' 
Note; this tells us how trustworthy the posterior sampling is, not how well model predicts the data. 
max rhat should be 1.0-1.01, > 1.01 investigate, > 1.05 concerning,
 min ess_bulk (ESS = effective sample size) , high  means coefficients are sampled efficiently,
 divergences: counts NUTS/HMC divergent transitions, ideally 0, more means sampler is struggling with regions of posterior geometry,
 if many , posterior summaries may be biased or unreliable. 
 '''

# diagnosis for which parameter is causing low ESS/borderling r-hat 
# model_to_inspect = 'M4a_ei_x_sim'
# s = az.summary(idatas[model_to_inspect],kind='diagnostics')
# s.sort_values('ess_bulk').head(20)

# az.plot_trace(idatas[model_to_inspect],var_names=['b', 'a_ct', 'z_ct'])
# az.summary(
#     idatas[model_to_inspect],
#     var_names=['b', 'sigma_ct', 'alpha'],
#     kind='all'
# )

# or diagnose due to warning in loo 
loo_m4a = az.loo(
    idatas['M4a_ei_x_sim'],
    pointwise=True
)

print(loo_m4a)

k = loo_m4a.pareto_k.values
np.nanmax(k)
bad_idx = np.where(k > 0.7)[0]
print(bad_idx)
print(k[bad_idx])
mdata.iloc[bad_idx]

#### --> m3_ei_ct not useable, m2 ct and ei are borderline but look potentially usable
#%% check EI and ct overlap 
tab = pd.crosstab(mdata["ei_pair"], mdata["ct_pair"])
# Number of CT categories observed for each EI category
ei_degree = (tab > 0).sum(axis=1)
# Number of EI categories observed for each CT category
ct_degree = (tab > 0).sum(axis=0)
cell_n = mdata.groupby(["ei_pair", "ct_pair"
                        ]).size()

print("CT categories per EI category:")
print(ei_degree.describe())

print("\nEI categories per CT category:")
print(ct_degree.describe())

print("\nObservations per EI × CT cell:")
print(cell_n.describe())

print("\nFraction of joint cells with one observation:")
print((cell_n == 1).mean())

#%% choose a main inference model and generate new posterior predictive replicated datasets to generate new hypothetical y
# should fit data as well as possible 
best_name = 'M2_targetre'  # main inference model; revisit after inspecting loo_comp
# best_name = 'M1_main'
with models[best_name]:
    ppc = pm.sample_posterior_predictive(idatas[best_name], random_seed=23)
y_rep = ppc.posterior_predictive['y'].stack(s=('chain', 'draw')).to_numpy()  # n_obs x n_draws
print(f'observed zero fraction {np.mean(y_obs == 0):.3f}, ppc (predicted) {np.mean(y_rep == 0):.3f} +/- {np.std((y_rep == 0).mean(axis=0)):.3f}')
print(f'observed var/mean {y_obs.var() / y_obs.mean():.2f}, ppc {np.mean(y_rep.var(axis=0) / y_rep.mean(axis=0)):.2f}')
''' zero fraction mismatch could suggest missing heterogeneity, missing predictors etc etc '''
''' var/mean checks dispersion - NB should have var/mean > 1, Poisson = 1, underdispersion < 1
    if predited var/mean much lower --> predicted samples are much less variable than reality, if much higher --> more variable '''

#%% getting fitted expected counts and Pearson residuals for offset adequacy check
comps_best, dm_best = eta_components(best_name, idatas[best_name]) # components of linear predictor
eta_hat = np.sum(list(comps_best.values()), axis=0) # sum of components = posterior mean linear predictor
mu_hat = np.exp(log_expected_off + eta_hat) # posterior mean expected counts, conditional expected count with negative binomial distribution around expectation
alpha_hat = idatas[best_name].posterior['alpha'].mean().item() # negative binomial dispersion
''' posterior mean of alpha. Larger alpha -> closer to Poisson, smaller alpha -> more overdispersed.'''
pearson_res = (y_obs - mu_hat) / np.sqrt(mu_hat + mu_hat**2 / alpha_hat) # pearson residual
''' residual = 0 close to expectation, positive = more observed than expected, negative = less observed than expected.
    larger magnitudes are more extreme deviations from expectation. '''

fig, ax = plt.subplots(1, 2, figsize=(9, 4))
bins = np.arange(-0.5, max(30, y_obs.max()) + 1.5, 1)
ax[0].hist(y_obs, bins=bins, density=True, alpha=0.6, label='observed')
ax[0].hist(y_rep[:, ::40].ravel(), bins=bins, density=True, histtype='step', color='k', label='ppc')
ax[0].set_yscale('log')
ax[0].set_xlabel('co-targeting count')
ax[0].legend()
ax[1].scatter(log_expected_off, pearson_res, s=2, alpha=0.3)
ax[1].axhline(0, color='grey', ls='--', lw=0.8)
ax[1].set_xlabel('log(expected) offset')
ax[1].set_ylabel('pearson residual')
ax[1].set_title("offset adequacy check \n does model error depend on baseline expected count?")
fig.tight_layout()

# TODO: inspect histogram of residuals 
# e.g. shifted above/below 0 --> model tends to under/over predict, 
# strong positive tail, model misses large counts
# unusually meany large residuals --> dispersion or mean structure not adequate
# multimodality --> missing heterogeneity, missing predictors, missing interactions, etc

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.hist(pearson_res, bins=100, density=True, alpha=0.6)
ax.set_xlabel('pearson residual')
fig.tight_layout()

# %% which features matter: posterior rate ratios + shrunk ei_pair/ct_pair effects
best = idatas[best_name]
print(f'note: ei_pair vs ct_pair association (bias-corrected cramers v) = {cat_corr_corrected:.2f} - interpret their coefficients jointly')
az.plot_forest(best, var_names=['b'], transform=np.exp, combined=True, hdi_prob=0.94, figsize=(6, 5))
plt.axvline(1.0, color='grey', lw=0.8, ls='--')
plt.title(f'{best_name}: rate ratio vs geometry null')
plt.tight_layout()
''' For forest plot
transform np.exp converts log to rate ratios, so 1 = no effect, >1 = enrichment, <1 = depletion
1.2 = a one-unit increase in feature is associated with a 20% increase in co-targeting rate 
relative to the geometry null, holding other features constant
interval width = 94% highest density interval (HDI) of posterior
'''

''' b intercept: 0.8 = the model expects about 20% less counts than the offset geometry null alone would predict.'''

if 'a_ei' in best.posterior:
    az.plot_forest(best, var_names=['a_ei'], transform=np.exp, combined=True, hdi_prob=0.94, figsize=(5, 2.5))
    plt.axvline(1.0, color='grey', lw=0.8, ls='--')
    plt.title(f'{best_name}: ei_pair rate ratio (shrunk deviations from intercept)')
    plt.tight_layout()

# hierarchical effect for ei_pair categories. e.g. 0.9 means 10% less than baseline implied by the rest of the model '''

# for ct: precalculate and sort from most negative to most positive for readability 
# but keep on log scale, so 0 = no effect, positive = enrichment, negative = depletion
a_mean = best.posterior['a_ct'].mean(('chain', 'draw')).to_numpy()
a_hdi = az.hdi(best, var_names=['a_ct'])['a_ct'].to_numpy()
order = np.argsort(a_mean)
fig, ax = plt.subplots(figsize=(6, max(6, len(ct_levels) * 0.12)))
ax.errorbar(a_mean[order], np.arange(len(order)), xerr=[a_mean[order] - a_hdi[order, 0], a_hdi[order, 1] - a_mean[order]], fmt='o', ms=2, lw=0.5)
ax.set_yticks(np.arange(len(order)))
ax.set_yticklabels(ct_levels[order], fontsize=10)
ax.axvline(0, color='grey', ls='--', lw=0.8)
ax.set_xlabel('log enrichment beyond fixed effects (a_ct, shrunk)')
fig.tight_layout()
# interpret as: positive val, after accounting for offset, fixed effects, EI effects etc, 
# CT pair X_X still tends to have counts above or below model's baseline expectation

#%% target random effects 
if 'u' in best.posterior:
    u_sum = az.summary(best, var_names=["u"],hdi_prob=0.94)
    u_sum.sort_values("mean", inplace=True)

fig, ax = plt.subplots(1,1, figsize=(6,4))
ax.hist(u_sum["mean"], bins=60, density=False, alpha=0.6)
ax.set_xlabel("target random effect (u)")
ax.set_yscale("log")
ax.set_ylabel("count (log)")

pair_to_skids ={}
for sk, pair in skid_to_pair.items():
    if pair not in pair_to_skids:
        pair_to_skids[pair] = []
    pair_to_skids[pair].append(sk)

u_sum["pair"] = u_sum.index.map(lambda x: x.replace('u[','').replace(']','').replace('unpaired', '-1' ))
u_sum['pair_skids'] = u_sum['pair'].map(lambda x: pair_to_skids.get(int(x), 'unpaired'))
u_sum['pair_names'] = u_sum['pair_skids'].map(lambda x: [skid_to_name.get(sk, 'NA') for sk in x] if isinstance(x, list) else ['unpaired'])

# %% how much enrichment do features explain: pseudo-R2 ladder, variance partition, predicted vs observed
ll = {name: idata.log_likelihood['y'].mean(('chain', 'draw')).sum().item() for name, idata in idatas.items()}
print('mcfadden pseudo-R2 vs M0_null (fraction of excess-over-null deviance explained):')
for name in idatas:
    if name != 'M0_null':
        print(f'  {name}: {1 - ll[name] / ll["M0_null"]:.3f}')

names_f = np.array(dm_best.columns)
b_hat = best.posterior['b'].mean(('chain', 'draw')).to_numpy()
Xf = dm_best.to_numpy()
groups = {'continuous': [i for i, c in enumerate(names_f) if ':' not in c and c != 'Intercept'], 'interactions': [i for i, c in enumerate(names_f) if ':' in c]}
eta_parts = {g: Xf[:, ix] @ b_hat[ix] for g, ix in groups.items() if len(ix)}
if 'ei_pair' in comps_best:
    eta_parts['ei_pair'] = comps_best['ei_pair']
if 'ct_pair' in comps_best:
    eta_parts['ct_pair'] = comps_best['ct_pair']
if 'target_re' in comps_best:
    eta_parts['target_re'] = comps_best['target_re']
var_eta = np.var(eta_hat)
print(f'{best_name} variance partition of eta (log-enrichment scale), var(eta) = {var_eta:.3f}:')
for g, p in eta_parts.items():
    print(f'  {g}: sd {np.std(p):.3f}, share {np.var(p) / var_eta:.1%}')

# CT-pair membership creates more heterogeneity in predicted enrichment than the continuous features do. (sd higher)
# share 43% and 92% mean ct and continuous features are negatively correlated across observations 
# because var(eta)= var(c) + var(ct) + 2cov(c,ct). partially cancl each other out

fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(eta_hat / np.log(2), mdata['log2_enrichment'], s=2, alpha=0.3)
r = np.corrcoef(eta_hat / np.log(2), mdata['log2_enrichment'])[0, 1]
lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lims, lims, color='grey', ls='--', lw=0.8)
ax.set_xlabel('predicted log2 enrichment (features)')
ax.set_ylabel('observed log2 enrichment')
ax.set_title(f'{best_name}: r = {r:.2f}, r2 = {r**2:.2f}')
fig.tight_layout()

# %% frequentist NB cross-check (no random effects, alpha estimated by ML) - should agree with M1_main in sign/magnitude

X_sm = dmatrix(f_main, mdata, return_type='dataframe')
nb_res = sm.NegativeBinomial(y_obs, X_sm, offset=log_expected_off).fit(maxiter=200, disp=0)
print(nb_res.summary())
b_bayes = idatas['M1_main'].posterior['b'].mean(('chain', 'draw')).to_numpy()
cross = pd.DataFrame({'coef': list(X_sm.columns), 'statsmodels_nb': nb_res.params.to_numpy()[:len(X_sm.columns)], 'pymc_M1_main': b_bayes})
print(cross.to_string(index=False))

# %%
