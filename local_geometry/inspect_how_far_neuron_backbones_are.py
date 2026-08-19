''' This is to inspect for different neuron backbone types, how far away presynaptic sites are from the target backbone. '''

import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

#%% load data

con = pd.read_csv('../data/con_short_2026_correctnodes.csv')

nodes = pd.read_csv('../data/nodes_manual_wdendrites_euclid_strahler.csv')

backbone_s1 = pd.read_csv('../data/neuron_backbone_Sover1.csv')
backbone_top2 = pd.read_csv('../data/neuron_backbone_top2S.csv')
backbone_top3 = pd.read_csv('../data/neuron_backbone_top3S.csv')

#%% get all relevant nodes 

con['postsynaptic_to'] = con['postsynaptic_to'].apply(ast.literal_eval)
con['postsynaptic_to_node'] = con['postsynaptic_to_node'].apply(ast.literal_eval)

cone = con.explode(['postsynaptic_to', 'postsynaptic_to_node'])

# %% get all distances
def build_node_lookup(nodes):
    """Sort node ids and coordinates once so positions can be looked up by binary search."""
    node_ids = nodes['node_id'].to_numpy(dtype=np.int64)
    xyz = np.ascontiguousarray(nodes[['x', 'y', 'z']].to_numpy(dtype=np.float64))
    order = np.argsort(node_ids, kind='stable')
    return node_ids[order], xyz[order]


def lookup_xyz(query_ids, sorted_ids, sorted_xyz):
    """Return coordinates for query node ids, plus a mask of the ids that were found."""
    query_ids = np.asarray(query_ids, dtype=np.int64)
    positions = np.searchsorted(sorted_ids, query_ids)
    clipped = np.minimum(positions, len(sorted_ids) - 1)
    found = sorted_ids[clipped] == query_ids
    xyz = np.full((len(query_ids), 3), np.nan)
    xyz[found] = sorted_xyz[clipped[found]]
    return xyz, found


def as_int_ids(column):
    """Coerce an exploded object column of ids to int64, flagging missing entries."""
    values = pd.to_numeric(pd.Series(column).reset_index(drop=True), errors='coerce').to_numpy()
    present = np.isfinite(values)
    ids = np.zeros(len(values), dtype=np.int64)
    ids[present] = values[present].astype(np.int64)
    return ids, present


def synapse_node_distances(cone, nodes):
    """Euclidean distance between the presynaptic node and the postsynaptic node of each row."""
    sorted_ids, sorted_xyz = build_node_lookup(nodes)

    pre_ids, pre_present = as_int_ids(cone['presynaptic_to_node'])
    post_ids, post_present = as_int_ids(cone['postsynaptic_to_node'])

    pre_xyz, pre_found = lookup_xyz(pre_ids, sorted_ids, sorted_xyz)
    post_xyz, post_found = lookup_xyz(post_ids, sorted_ids, sorted_xyz)

    valid = pre_present & post_present & pre_found & post_found
    distances = np.full(len(cone), np.nan)
    delta = pre_xyz[valid] - post_xyz[valid]
    distances[valid] = np.sqrt(np.einsum('ij,ij->i', delta, delta))
    return distances


def presyn_to_backbone_distances(cone, nodes, backbone, workers=-1, verbose=True):
    """Distance from each presynaptic node to the nearest backbone node of its postsynaptic neuron.

    One KD-tree is built per postsynaptic skeleton, because the nearest node has to lie on that
    particular target neuron. Returns the distances and the id of the matched backbone node.
    """
    sorted_ids, sorted_xyz = build_node_lookup(nodes)

    pre_ids, pre_present = as_int_ids(cone['presynaptic_to_node'])
    target_ids, target_present = as_int_ids(cone['postsynaptic_to'])
    pre_xyz, pre_found = lookup_xyz(pre_ids, sorted_ids, sorted_xyz)

    # backbone sorted by skeleton so each neuron is one contiguous slice
    bb_skeletons = backbone['skeleton_id'].to_numpy(dtype=np.int64)
    bb_order = np.argsort(bb_skeletons, kind='stable')
    bb_skeletons = bb_skeletons[bb_order]
    bb_xyz = np.ascontiguousarray(backbone[['x', 'y', 'z']].to_numpy(dtype=np.float64)[bb_order])
    bb_node_ids = backbone['node_id'].to_numpy(dtype=np.int64)[bb_order]
    bb_unique, bb_starts = np.unique(bb_skeletons, return_index=True)
    bb_ends = np.append(bb_starts[1:], len(bb_skeletons))

    distances = np.full(len(cone), np.nan)
    matched_node = np.full(len(cone), -1, dtype=np.int64)

    valid = pre_present & target_present & pre_found
    valid_rows = np.flatnonzero(valid)
    # group the query rows by target skeleton, again by sorting into contiguous slices
    q_order = valid_rows[np.argsort(target_ids[valid_rows], kind='stable')]
    q_targets = target_ids[q_order]
    q_unique, q_starts = np.unique(q_targets, return_index=True)
    q_ends = np.append(q_starts[1:], len(q_targets))

    # only skeletons that exist in the backbone table can be queried
    tree_slot = np.searchsorted(bb_unique, q_unique)
    tree_slot_clipped = np.minimum(tree_slot, len(bb_unique) - 1)
    in_backbone = bb_unique[tree_slot_clipped] == q_unique

    for i in np.flatnonzero(in_backbone):
        rows = q_order[q_starts[i]:q_ends[i]]
        slot = tree_slot[i]
        tree = cKDTree(bb_xyz[bb_starts[slot]:bb_ends[slot]])

        # the same presynaptic node repeats across polyadic partners, so query it only once
        _, first, inverse = np.unique(pre_ids[rows], return_index=True, return_inverse=True)
        dist, idx = tree.query(pre_xyz[rows[first]], workers=workers)

        distances[rows] = dist[inverse]
        matched_node[rows] = bb_node_ids[bb_starts[slot] + idx[inverse]]

        if verbose and i % 500 == 0:
            print(f'{i}/{len(q_unique)} skeletons done', flush=True)

    return distances, matched_node


#%% run distances

cone['syn_node_dist'] = synapse_node_distances(cone, nodes)

cone['backbone_dist_sover1'], cone['backbone_node'] = presyn_to_backbone_distances(cone, nodes, backbone_s1)
cone['backbone_dist_top2'], cone['backbone_node_top2'] = presyn_to_backbone_distances(cone, nodes, backbone_top2)
cone['backbone_dist_top3'], cone['backbone_node_top3'] = presyn_to_backbone_distances(cone, nodes, backbone_top3)

#%% plot
fig, ax = plt.subplots(2,2, figsize=(8,8))
ax = ax.flatten()
max_dist = max(cone[['syn_node_dist', 'backbone_dist_sover1', 'backbone_dist_top2', 'backbone_dist_top3']].max())
ylog = True
for a, col in zip(ax, ['syn_node_dist', 'backbone_dist_sover1', 'backbone_dist_top2', 'backbone_dist_top3']):
    a.hist(cone[col], bins=100)
    a.set_title(col)
    a.set_xlim(0, max_dist)
    if ylog:
        a.set_yscale('log')
        a.set_ylabel('log(count)')
    else:
        a.set_ylabel('count')
fig.tight_layout()


# %%

cone[['syn_node_dist', 'backbone_dist_sover1', 'backbone_dist_top2', 'backbone_dist_top3']].describe()

# %% inspect by neuron 

max_by_sk = cone.groupby('presynaptic_to')[['syn_node_dist', 'backbone_dist_sover1', 'backbone_dist_top2', 'backbone_dist_top3']].max()
max_by_sk.describe()


fig, ax = plt.subplots(2,2, figsize=(8,8))
ax = ax.flatten()
max_dist = max(max_by_sk[['syn_node_dist', 'backbone_dist_sover1', 'backbone_dist_top2', 'backbone_dist_top3']].max())
ylog = False
for a, col in zip(ax, ['syn_node_dist', 'backbone_dist_sover1', 'backbone_dist_top2', 'backbone_dist_top3']):
    a.hist(max_by_sk[col], bins=100)
    a.set_title(col)
    a.set_xlim(0, max_dist)
    if ylog:
        a.set_yscale('log')
        a.set_ylabel('log(count)')
    else:
        a.set_ylabel('count')
fig.suptitle('Max distance per presynaptic neuron')
fig.tight_layout()

#%% rename to use in scripts 

max_by_sk.rename(columns={'syn_node_dist': 'max_syn_node_dist',
                          'backbone_dist_sover1': 'Sover1',
                          'backbone_dist_top2': 'top2S',
                          'backbone_dist_top3': 'top3S'}, inplace=True)

max_by_sk.reset_index(inplace=True)

# max_by_sk.to_csv('../data/max_presyn_to_backbone_distances_by_skid.csv', index=False)

# %%
