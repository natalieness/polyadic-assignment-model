''' This is to generate some supplied co-targeting rules that are not directly 
derived from the observed neuron data, but are instead based on some other criteria.'''

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

from perm_functions.polyadic_edge_permutation import get_perms

from models.padj_functions import get_co_targeting_matrix, compute_cosine
from models.approach2.propensity import pair_count_matrix, mean_pair_count_matrix
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

#%% local params
presyn_col = 'presynaptic_to'
postsyn_col = 'postsynaptic_to'

# params 
include_self = True
smoothing = 1.0
shrinkage = 10.0

all_presynaptic = con[presyn_col].unique()
n_pre = len(all_presynaptic)

#%% 

# a local estimator of propensity, independent of config rules 
def estimate_individual_propensity_local(
    observed_assignment: np.ndarray,
    expected_samples: np.ndarray,
    include_self: bool = True,
    smoothing: float = 1.0,
    shrinkage: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate shrunk log enrichment relative to a simulated expectation."""
    observed = pair_count_matrix(observed_assignment, include_self)
    expected = mean_pair_count_matrix(expected_samples, include_self)
    raw = np.log((observed + smoothing) / (expected + smoothing))
    support = observed + expected
    weight = support / (support + shrinkage)
    shrunk = weight * raw
    if not include_self:
        np.fill_diagonal(shrunk, 0.0)
    return shrunk # standardization happens on import 

from models.approach2.sampler import sample_random_null
def permutation_expectation_samples_local(
    site_degrees: np.ndarray,
    target_counts: np.ndarray,
    rng: np.random.Generator,
    max_same_target_per_site: int = None,
    max_null_restarts: int = 100,
    n_perm_expectation: int = 100,
) -> np.ndarray:
    """Draw independent margin-preserving shuffles for the co-targeting expectation.

    Each draw shuffles labelled target tokens into the fixed site slots, so the site
    degrees and target totals stay exact while the pairing is random. Nothing here
    reads the simulated conditions: the expectation is not the null or geometry
    chain reused, so it carries no MCMC autocorrelation and its draw count is set by
    ``cotarget.n_perm_expectation`` rather than by ``sampler.prediction_samples``.
    """
    return np.stack([
        sample_random_null(
            site_degrees,
            target_counts,
            rng,
            max_same_target_per_site=max_same_target_per_site,
            max_restarts=max_null_restarts,
        )
        for index in range(int(n_perm_expectation))
    ])

def to_supplied_format(propensity_scores, u_targets, skeleton_id=None, as_df=True):
    """Convert a propensity score matrix to a DataFrame with the expected columns."""
    # dont need duplicates so take just upper triangle 
    target_a, target_b = np.meshgrid(u_targets, u_targets)
    mask = np.triu(np.ones_like(target_a, dtype=bool), k=0)
    out = np.array([target_a[mask], target_b[mask], propensity_scores[mask]]).T
    if skeleton_id is not None:
        pre_col = np.full(out.shape[0], skeleton_id, dtype=int)
        out = np.column_stack((pre_col, out))
    if as_df:
        columns = ['pre', 'target_a', 'target_b', 'score'] if skeleton_id is not None else ['target_a', 'target_b', 'score']
        out = pd.DataFrame(out, columns=columns)
    return out


def blank_supplied_format(all_neurons):
    """Create a DataFrame with all possible target pairs and a score of 0.0."""
    target_a, target_b = np.meshgrid(all_neurons, all_neurons)
    upper_triangle_mask = np.triu(np.ones_like(target_a, dtype=bool), k=0)
    blank_scores = pd.DataFrame(
        {'target_a': target_a[upper_triangle_mask],
        'target_b': target_b[upper_triangle_mask],
        'score': 0.0}
    )
    return blank_scores


def blank_supplied_format_restricted(con):
    """Create a DataFrame with all possible target pairs, taking into account presynaptic to postsynaptic and a score of 0.0."""
    pre_tables = []
    pre_skids = con[presyn_col].unique()
    for pre_skid in pre_skids:
        postsyns = con.loc[con[presyn_col] == pre_skid, postsyn_col].explode().unique()
        target_a, target_b = np.meshgrid(postsyns, postsyns)
        upper_triangle_mask = np.triu(np.ones_like(target_a, dtype=bool), k=0)
        blank_scores = pd.DataFrame(
            {'pre': pre_skid,
            'target_a': target_a[upper_triangle_mask],
            'target_b': target_b[upper_triangle_mask],
            'score': 0.0}
        )
        pre_tables.append(blank_scores)
    blank_scores = pd.concat(pre_tables, ignore_index=True)
    return blank_scores

def get_assignment_matrix(hyperedges, u_targets=None):
    if u_targets is None:
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

''' Co-targeting propensity rules should be in the shape of: 
pre,target_a,target_b,score
pre column is optional, but if present used to filter
'''
#%% first try to just recreate exactly what the native implementation does, 
# to make sure the imported propensity functions are working as expected. 

####### REGULAR NEURON-WISE PROPENSITY ESTIMATION #######

rng = np.random.default_rng(23)
prop_tables = []

for e, skeleton_id in enumerate(all_presynaptic[:1]):
    if e % 100 == 0:
        print(f"Processing skeleton {e+1}/{n_pre}")
    hyperedges = con.loc[con[presyn_col]==skeleton_id][postsyn_col].tolist()
    real_assignment, u_targets = get_assignment_matrix(hyperedges)

    expected_assignment = permutation_expectation_samples_local(
        site_degrees=real_assignment.sum(axis=1),
        target_counts=real_assignment.sum(axis=0),
        rng=rng,
        max_same_target_per_site=None,
        max_null_restarts=100,
        n_perm_expectation=100,
    )

    propensity_scores = estimate_individual_propensity_local(
        observed_assignment=real_assignment,
        expected_samples=expected_assignment,
        include_self=include_self,
        smoothing=smoothing,
        shrinkage=shrinkage,
    ) # gives n_target x n_target matrix of scores

    prop_table = to_supplied_format(propensity_scores, u_targets, skeleton_id=skeleton_id, as_df=False)
    prop_tables.append(prop_table)

all_prop_table = np.vstack(prop_tables)
all_prop_table = pd.DataFrame(all_prop_table, columns=['pre', 'target_a', 'target_b', 'score'])

# filter out score = 0.0 since this is the default anyways 
all_prop_table = all_prop_table.loc[all_prop_table['score'] != 0.0]
# all_prop_table.to_csv('data/cotargeting_rules/neuronwise_cotargeting.csv', index=False)


# %%
##### AVERAGE CELLTYPE-BASED PROPENSITY ESTIMATION #####

# all celltypes, independent of presynaptic neuron 
# need to get actual full permutation model to estimate the expected co-targeting

con_copy = con.copy()
rng = np.random.default_rng(23)
perms = get_perms(con_copy, n_perms=100, rng=rng, connector_col='connector_id', 
                  pre_col='presynaptic_to', post_col='postsynaptic_to', 
                  allow_self_cotarget=True)

# map cell type to index
ct_names = list(set(skid_to_celltype.values())) + ['NA'] # add fallback 
ct_to_ct_idx = {ct: i for i, ct in enumerate(ct_names)}

hyperedges = con_copy[postsyn_col].tolist()
hyperedges_ct = [[ct_to_ct_idx[skid_to_celltype.get(skid, 'NA')] for skid in edge] for edge in hyperedges]

# final expected assignment should be shape (n_perms, n_sites, n_celltypes)
# right now nested list should be fine, i.e. list of perms, with list of sites 

hyperedges_perms = [perm_i[postsyn_col].tolist() for perm_i in perms]
def map_sublists_to_celltype_idx(hyperedges, skid_to_celltype=skid_to_celltype, ct_to_ct_idx=ct_to_ct_idx, fail_cond='NA'):
    return [[ct_to_ct_idx[skid_to_celltype.get(skid, fail_cond)] for skid in edge] for edge in hyperedges]
hyperedges_perms_ct = [map_sublists_to_celltype_idx(perm_i) for perm_i in hyperedges_perms]

def get_assignment_matrix(hyperedges, u_targets=None):
    if u_targets is None:
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

real_assignment, u_targets = get_assignment_matrix(hyperedges_ct, u_targets=ct_to_ct_idx.values())
expected_assignment = [get_assignment_matrix(perm_i, u_targets=u_targets)[0] for perm_i in hyperedges_perms_ct]
expected_assignment = np.stack(expected_assignment, axis=0)

propensity_scores_ct = estimate_individual_propensity_local(
    observed_assignment=real_assignment,
    expected_samples=expected_assignment,
    include_self=include_self,
    smoothing=smoothing,
    shrinkage=shrinkage,
) # gives n_target x n_target matrix of scores

# need to now map this to some overall all possible targets with all possible targets score 
ct_scores = blank_supplied_format(all_neurons)
ct_scores['score'] = ct_scores.apply(
    lambda row: propensity_scores_ct[ct_to_ct_idx[skid_to_celltype.get(row['target_a'], 'NA')],
                                     ct_to_ct_idx[skid_to_celltype.get(row['target_b'], 'NA')]], axis=1
                                     )

# drop zero scores to reduce file size
ct_scores = ct_scores.loc[ct_scores['score'] != 0.0]
# ct_scores.to_csv('data/cotargeting_rules/broad_ct_cotargeting.csv', index=False)


#%% AVERGAE CELL TYPE PROPENSITY BUT PRE-SYNAPTIC CELLTYPE SPECIFIC 
# OR PRESYNAPTIC CELL SPECIFIC 

con_copy = con.copy()
rng = np.random.default_rng(23)
if not perms:
    perms = get_perms(con_copy, n_perms=100, rng=rng, connector_col='connector_id', 
                    pre_col='presynaptic_to', post_col='postsynaptic_to', 
                    allow_self_cotarget=True)

# map cell type to index - done above
# ct_names = list(set(skid_to_celltype.values())) + ['NA'] # add fallback 
# ct_to_ct_idx = {ct: i for i, ct in enumerate(ct_names)}


# presynaptic cell type specific propensity estimation
pre_ct_scores = blank_supplied_format_restricted(con)

for pre_ct in ct_names:
    pre_skids = [sk for sk in all_presynaptic if skid_to_celltype.get(sk, 'NA') == pre_ct]
    con_sub = con_copy.loc[con_copy[presyn_col].isin(pre_skids)]
    hyperedges = con_sub[postsyn_col].tolist()
    hyperedges_ct = [[ct_to_ct_idx[skid_to_celltype.get(skid, 'NA')] for skid in edge] for edge in hyperedges]

    # final expected assignment should be shape (n_perms, n_sites, n_celltypes)
    # right now nested list should be fine, i.e. list of perms, with list of sites 

    hyperedges_perms = [perm_i[perm_i[presyn_col].isin(pre_skids)][postsyn_col].tolist() for perm_i in perms]
    hyperedges_perms_ct = [map_sublists_to_celltype_idx(perm_i) for perm_i in hyperedges_perms]

    real_assignment, u_targets = get_assignment_matrix(hyperedges_ct, u_targets=ct_to_ct_idx.values())
    expected_assignment = [get_assignment_matrix(perm_i, u_targets=u_targets)[0] for perm_i in hyperedges_perms_ct]
    expected_assignment = np.stack(expected_assignment, axis=0)

    propensity_scores_ct = estimate_individual_propensity_local(
        observed_assignment=real_assignment,
        expected_samples=expected_assignment,
        include_self=include_self,
        smoothing=smoothing,
        shrinkage=shrinkage,
    ) # gives n_target x n_target matrix of scores
    # get just the relevant postsyn neurons for this cell type 
    relevant_post_skids = con_sub[postsyn_col].explode().unique()

    # map to target for each possible presynaptic neuron of that cell type
    for pre_skid in pre_skids:
        pre_ct_scores.loc[pre_ct_scores['pre']==pre_skid, 'score'] = pre_ct_scores.loc[pre_ct_scores['pre']==pre_skid].apply(
            lambda row: propensity_scores_ct[ct_to_ct_idx[skid_to_celltype.get(row['target_a'], 'NA')],
                                            ct_to_ct_idx[skid_to_celltype.get(row['target_b'], 'NA')]], axis=1
                                            )


    pre_ct_scores['score'] = pre_ct_scores.apply(
        lambda row: propensity_scores_ct[ct_to_ct_idx[skid_to_celltype.get(row['target_a'], 'NA')],
                                        ct_to_ct_idx[skid_to_celltype.get(row['target_b'], 'NA')]], axis=1
                                        )
    # drop zero scores to reduce file size
    pre_ct_scores = pre_ct_scores.loc[pre_ct_scores['score'] != 0.0]

# pre_ct_scores.to_csv('data/cotargeting_rules/ct_to_ct_cotargeting.csv', index=False)

# %% bilateral pair averaged propensity table

# Local copies of the propensity counters, used only by this cell. Identical output to
# the imported versions, but the assignment matrix here is ~1758 targets wide and only
# ~0.1% dense, so:
#  - counting from the targets present at each site instead of X.T @ X avoids the dense
#    matmul. numpy has no BLAS kernel for integer dtypes, so that matmul falls back to a
#    naive C loop and is ~200x slower than accumulating the present targets directly.
#  - accumulating the permutation mean in place avoids holding all n_perms count matrices
#    (100 x 1758 x 1758 int64 = 2.5 GB) before averaging them.
def pair_count_matrix_local(assignment, include_self_repeats=False):
    """Number of sites containing each unordered pair of targets, counted sparsely."""
    assignment = np.asarray(assignment)
    result = np.zeros((assignment.shape[1],) * 2, dtype=np.int64)
    for row in assignment:
        present = np.flatnonzero(row > 0)
        result[np.ix_(present, present)] += 1
    if include_self_repeats:
        np.fill_diagonal(result, (assignment >= 2).sum(axis=0))
    else:
        np.fill_diagonal(result, 0)
    return result

def mean_pair_count_matrix_local(samples, include_self_repeats=False):
    samples = np.asarray(samples)
    if samples.ndim != 3:
        raise ValueError("Samples must have shape samples x sites x targets")
    total = np.zeros((samples.shape[2],) * 2, dtype=np.int64)
    for sample in samples:
        total += pair_count_matrix_local(sample, include_self_repeats)
    return total / len(samples)

def estimate_individual_propensity_sparse(
    observed_assignment,
    expected_samples,
    include_self=True,
    smoothing=1.0,
    shrinkage=10.0,
):
    """estimate_individual_propensity_local, on the sparse counters above."""
    observed = pair_count_matrix_local(observed_assignment, include_self)
    expected = mean_pair_count_matrix_local(expected_samples, include_self)
    raw = np.log((observed + smoothing) / (expected + smoothing))
    support = observed + expected
    weight = support / (support + shrinkage)
    shrunk = weight * raw
    if not include_self:
        np.fill_diagonal(shrunk, 0.0)
    return shrunk

con_copy = con.copy()
rng = np.random.default_rng(23)

try:
    perms[0]
except:
    perms = get_perms(con_copy, n_perms=100, rng=rng, connector_col='connector_id', 
                    pre_col='presynaptic_to', post_col='postsynaptic_to', 
                    allow_self_cotarget=True)

# map all neurons pre and post to their bilateral pair: skid_to_pair
# paired neurons collapse onto their pair_id, unpaired neurons keep their own skid.
# both share one key space, so unpaired skids are shifted above the pair_id range -
# skid 29 and pair_id 29 are different neurons and must not share a key.
PAIR_ID_OFFSET = max(skid_to_pair.values()) + 1

def target_key(skid, skid_to_pair=skid_to_pair, offset=PAIR_ID_OFFSET):
    """Bilateral-pair key for a skid: its pair_id, or its offset skid if unpaired."""
    return skid_to_pair.get(skid, skid + offset)

all_post = set(con_copy[postsyn_col].explode().unique())
assert min(all_post) >= 0, 'negative skids would fall back into the pair_id range'

nonpaired_targets = sorted(sk for sk in all_post if sk not in skid_to_pair)
all_targets = sorted(set(skid_to_pair.values())) + [target_key(sk) for sk in nonpaired_targets]
pair_to_pair_idx = {key: i for i, key in enumerate(all_targets)}
assert len(pair_to_pair_idx) == len(all_targets), 'target keys collide, indices would be wrong'
u_indices = list(range(len(pair_to_pair_idx)))

# collapse the two lookups (skid -> key -> index) into one, so the offset is applied
# in exactly one place and never again while scoring
skid_to_target_idx = {sk: pair_to_pair_idx[target_key(sk)] for sk in all_post}

def map_sublists_to_pair_idx(hyperedges, skid_to_target_idx=skid_to_target_idx):
    return [[skid_to_target_idx[skid] for skid in edge] for edge in hyperedges]

# the permutation shuffles postsynaptic partners within a presynaptic neuron, so the
# pre column and the target label pool are unchanged - group positions computed here
# stay valid for every perm, and no perm can introduce an unmapped skid
pre_pair_of_skid = {sk: target_key(sk) for sk in all_presynaptic}
all_pre_pairs = sorted(set(pre_pair_of_skid.values()))
con_groups = con_copy.groupby(con_copy[presyn_col].map(pre_pair_of_skid)).indices
con_posts = con_copy[postsyn_col].to_numpy()
perm_groups = [perm_i.groupby(perm_i[presyn_col].map(pre_pair_of_skid)).indices for perm_i in perms]
perm_posts = [perm_i[postsyn_col].to_numpy() for perm_i in perms]

# presynaptic cell type specific propensity estimation
pair_scores = blank_supplied_format_restricted(con)

# resolve every target skid to its matrix index once for the whole table, rather than
# once per row inside the loop
a_map = pair_scores['target_a'].map(skid_to_target_idx)
b_map = pair_scores['target_b'].map(skid_to_target_idx)
assert not a_map.isna().any() and not b_map.isna().any(), 'target skid missing from skid_to_target_idx'
a_idx = a_map.to_numpy(dtype=np.intp)
b_idx = b_map.to_numpy(dtype=np.intp)
score_groups = pair_scores.groupby(pair_scores['pre'].map(pre_pair_of_skid)).indices
scores = np.zeros(len(pair_scores), dtype=float)

for e, pre_pair in enumerate(all_pre_pairs):
    print(f"Processing  pair {e+1}/{len(all_pre_pairs)}: {pre_pair}")
    hyperedges = con_posts[con_groups[pre_pair]]
    hyperedges_ct = map_sublists_to_pair_idx(hyperedges)


    hyperedges_perms = [posts[groups[pre_pair]] for posts, groups in zip(perm_posts, perm_groups)]
    hyperedges_perms_ct = [map_sublists_to_pair_idx(perm_i) for perm_i in hyperedges_perms]


    real_assignment, u_targets = get_assignment_matrix(hyperedges_ct, u_targets=u_indices)
    expected_assignment = [get_assignment_matrix(perm_i, u_targets=u_targets)[0] for perm_i in hyperedges_perms_ct]
    expected_assignment = np.stack(expected_assignment, axis=0)
    st4 = time.time()

    propensity_scores_ct = estimate_individual_propensity_sparse(
        observed_assignment=real_assignment,
        expected_samples=expected_assignment,
        include_self=include_self,
        smoothing=smoothing,
        shrinkage=shrinkage,
    ) # gives n_target x n_target matrix of scores
    st5 = time.time()
    print(f'Estimating propensity scores took {st5-st4:.2f} seconds')

    # every skid of this pre_pair shares the same matrix, so score all of their rows
    # in one gather instead of one .apply per skid
    rows = score_groups[pre_pair]
    scores[rows] = propensity_scores_ct[a_idx[rows], b_idx[rows]]


pair_scores['score'] = scores
print(pair_scores.head())
print(f"Saving pairwise co-targeting scores to 'data/cotargeting_rules/pair_to_pair_cotargeting.csv'")
# pair_scores.to_csv('data/cotargeting_rules/pair_to_pair_cotargeting.csv', index=False)

# %% general neuron identity enrichment average 

con_copy = con.copy()
rng = np.random.default_rng(23)
try:
    perms[0]
except:
    perms = get_perms(con_copy, n_perms=100, rng=rng, connector_col='connector_id', 
                    pre_col='presynaptic_to', post_col='postsynaptic_to', 
                    allow_self_cotarget=True)


# This cell runs over every site at once, so the dense assignment matrix would be
# 139,504 sites x 2,956 targets = 3.3 GB, and 100 permutations of it 0.3 TB. The counts
# it feeds into are only 2,956^2 = 70 MB, so count pairs straight from the hyperedges
# and never build the matrix. Same result as
# pair_count_matrix(get_assignment_matrix(hyperedges, all_neurons)[0]).

def flatten_hyperedges(post_lists, target_to_idx):
    """Hyperedges as (flat target indices, per-site sizes) - no per-site python loop."""
    posts = np.asarray(post_lists, dtype=object)
    sizes = np.fromiter((len(p) for p in posts), dtype=np.int64, count=len(posts))
    parts = [p for p in posts if len(p)]
    flat_raw = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    flat = pd.Series(flat_raw).map(target_to_idx).to_numpy(dtype=np.int64)
    return flat, sizes

def cooccurrence_counts(flat, sizes, n_targets, include_self_repeats=True):
    """Number of sites containing each unordered pair of targets."""
    site_of = np.repeat(np.arange(len(sizes), dtype=np.int64), sizes)

    # deduplicate (site, target): the off-diagonal counts sites, so a target appearing
    # twice at one site must still contribute only one site to its pairs
    key = site_of * n_targets + flat
    uniq, multiplicity = np.unique(key, return_counts=True)
    u_site, u_target = np.divmod(uniq, n_targets)

    # uniq is sorted, so each site's distinct targets form one contiguous block; bucket
    # sites by block length so every bucket's pairs come from one vectorised gather
    deg = np.bincount(u_site, minlength=len(sizes))
    starts = np.concatenate(([0], np.cumsum(deg)[:-1]))
    codes = []
    for d in np.unique(deg[deg > 0]):
        block = u_target[starts[deg == d][:, None] + np.arange(d)]   # (n_sites_with_deg_d, d)
        codes.append((np.repeat(block, d, axis=1) * n_targets + np.tile(block, (1, d))).ravel())
    counts = np.bincount(
        np.concatenate(codes) if codes else np.zeros(0, dtype=np.int64),
        minlength=n_targets * n_targets,
    ).reshape(n_targets, n_targets)

    if include_self_repeats:
        np.fill_diagonal(counts, np.bincount(u_target[multiplicity >= 2], minlength=n_targets))
    else:
        np.fill_diagonal(counts, 0)
    return counts

# just general blank target table
blank_scores = blank_supplied_format(all_neurons)

target_to_idx = {sk: i for i, sk in enumerate(all_neurons)}
n_targets = len(all_neurons)

observed = cooccurrence_counts(*flatten_hyperedges(con_copy[postsyn_col], target_to_idx),
                               n_targets, include_self)
expected = np.zeros((n_targets, n_targets), dtype=np.int64)
for perm_i in perms:   # accumulated in place, only one permutation held at a time
    expected += cooccurrence_counts(*flatten_hyperedges(perm_i[postsyn_col], target_to_idx),
                                    n_targets, include_self)
expected = expected / len(perms)

# same shrunk log enrichment as estimate_individual_propensity_sparse, inlined because
# the counts here never pass through an assignment matrix
raw = np.log((observed + smoothing) / (expected + smoothing))
support = observed + expected
propensity_scores = (support / (support + shrinkage)) * raw   # n_target x n_target
if not include_self:
    np.fill_diagonal(propensity_scores, 0.0)

blank_scores['score'] = propensity_scores[
    blank_scores['target_a'].map(target_to_idx).to_numpy(dtype=np.intp),
    blank_scores['target_b'].map(target_to_idx).to_numpy(dtype=np.intp),
]
blank_scores = blank_scores.loc[blank_scores['score'] != 0.0]
blank_scores.to_csv('data/cotargeting_rules/neuron_id_based_cotargeting.csv', index=False)

# %%
