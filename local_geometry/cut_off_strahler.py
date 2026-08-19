'''' this is to make neuronal backbones based on strahler index for neurons in a couple of different ways.'''

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns


nodes = pd.read_csv('../data/nodes_manual_wdendrites_euclid_strahler.csv')
strahler_summary = nodes.groupby('skeleton_id')['strahler'].agg(['max', 'mean', 'std', 'count']).reset_index()

#%% 
# 
# #only terminal twigs unless it's the only one 
just_terminal = strahler_summary[strahler_summary['max'] == 1]['skeleton_id'].values
nodes_term = nodes.copy()
nodes_term = nodes_term[(nodes_term['skeleton_id'].isin(just_terminal)) | (nodes_term['strahler'] > 1)]

# nodes_term.to_csv('../data/neuron_backbone_Sover1.csv', index=False)

# %%
# only keep the top 2 or top 3 strahler branches, except 1. 

strahler_summary['top_2_above'] = strahler_summary['max'] - 2

# threshold per skeleton, looked up by hash instead of a full scan per row
cut = nodes_term['skeleton_id'].map(strahler_summary.set_index('skeleton_id')['top_2_above'])

nodes_top2 = nodes_term[nodes_term['strahler'] > cut]
nodes_top3 = nodes_term[nodes_term['strahler'] >= cut]
nodes_top2.to_csv('../data/neuron_backbone_top2S.csv', index=False)
nodes_top3.to_csv('../data/neuron_backbone_top3S.csv', index=False)

# %%
