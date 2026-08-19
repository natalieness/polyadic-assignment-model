
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns


nodes = pd.read_csv('../data/nodes_manual_wdendrites_euclid_strahler.csv')

#%%

strahler_summary = nodes.groupby('skeleton_id')['strahler'].agg(['max', 'mean', 'std', 'count']).reset_index()

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
bins = np.arange(0, strahler_summary['max'].max() + 1) - 0.5
sns.histplot(strahler_summary['max'], bins=bins, ax=ax)
ax.set_xlabel('Max Strahler Number')
ax.set_ylabel('Number of Skeletons')

strahler_summary['max'].value_counts()

# some neurons legitamately only have low strahler number - e.g. abdominal sensory neuron max strahler = 1. 
# need to use some relative metric to remove lower branches 

# %% inspect skeleton 
from view_node_skeleton import build_figure

sk = 15620432
cut_off = 0
print(f'{sk} has max strahler {strahler_summary.loc[strahler_summary["skeleton_id"] == sk, "max"].values[0]}')
print(f'Showing strahler above {cut_off} for skeleton {sk}')
subset_nodes = nodes[(nodes['skeleton_id'] == sk) & (nodes['strahler'] > cut_off)]

fig = build_figure(subset_nodes, skeleton_id=sk, marker_size=2.0)
fig.show()
# %%
