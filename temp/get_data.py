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

import pymaid
from pymaid_creds import url, name, password, token
rm = pymaid.CatmaidInstance(url, token, name, password)
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

nodes = pd.read_csv('data/nodes_manual_wdendrites_euclid.csv')
#%% get lineages 


meta_annotation = 'Ingrid Lineage Brain'

# Find every annotation beneath the meta-annotation.
entities = pymaid.get_annotated(
    meta_annotation,
    include_sub_annotations=True,
    remote_instance=rm,  # omit if using the global CATMAID instance
)

sub_annotations = (
    entities.loc[entities["type"] == "annotation", "name"]
    .drop_duplicates()
    .tolist()
)

# Create one row per skeleton/sub-annotation relationship.
records = [
    {
        "skeleton_id": skid,
        "sub_annotation": annotation,
    }
    for annotation in sub_annotations
    for skid in pymaid.get_skids_by_annotation(
        annotation,
        raise_not_found=False,
        remote_instance=rm,
    )
]

result = (
    pd.DataFrame(records, columns=["skeleton_id", "sub_annotation"])
    .drop_duplicates()
    .sort_values(["sub_annotation", "skeleton_id"])
    .reset_index(drop=True)
)

skid_to_lineage = dict(zip(result['skeleton_id'], result['sub_annotation']))


#%%

meta_annotation = 'Volker'

# Find every annotation beneath the meta-annotation.
entities = pymaid.get_annotated(
    meta_annotation,
    include_sub_annotations=True,
    remote_instance=rm,  # omit if using the global CATMAID instance
)

sub_annotations = (
    entities.loc[entities["type"] == "annotation", "name"]
    .drop_duplicates()
    .tolist()
)

# Create one row per skeleton/sub-annotation relationship.
records = [
    {
        "skeleton_id": skid,
        "sub_annotation": annotation,
    }
    for annotation in sub_annotations
    for skid in pymaid.get_skids_by_annotation(
        annotation,
        raise_not_found=False,
        remote_instance=rm,
    )
]

result_volker = (
    pd.DataFrame(records, columns=["skeleton_id", "sub_annotation"])
    .drop_duplicates()
    .sort_values(["sub_annotation", "skeleton_id"])
    .reset_index(drop=True)
)

skid_to_volkerlineage = dict(zip(result['skeleton_id'], result['sub_annotation']))


#%%
OUT_PATH = 'data/out_for_lidakanari/'

nodecols = nodes.columns
nodes_here = nodes[['node_id', 'parent_node_id', 'skeleton_id', 'compartment', 'x', 'y','z']]

nodes_here.to_csv(OUT_PATH + 'skeleton_nodes.csv', index=False)

### connector details too
con.rename(columns={'postsynaptic_to': 'postsynaptic_to_skeleton_id', 'presynaptic_to': 'presynaptic_to_skeleton_id'}, inplace=True)
con.to_csv(OUT_PATH + 'synapses.csv', index=False)


### pairs 
pairs.to_csv(OUT_PATH + 'bilateral_pairs.csv', index=False)


#%% neuron details 

neus = pd.DataFrame(skid_to_name.items(), columns=['skeleton_id', 'name'])
neus['lineage'] = neus['skeleton_id'].map(skid_to_lineage)
neus['soma_hemisphere'] = neus['skeleton_id'].apply(lambda x: 'left' if x in left_neus else 'right' if x in right_neus else 'center' if x in center_neus else 'unknown')
neus['celltype'] = neus['skeleton_id'].map(skid_to_celltype)
# neus.to_csv(OUT_PATH + 'neuron_details.csv', index=False)
neus['volker_lineage'] = neus['skeleton_id'].map(skid_to_volkerlineage)


# %%
