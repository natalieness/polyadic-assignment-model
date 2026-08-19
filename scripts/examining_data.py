import pandas as pd 
import numpy as np 
import ast

#%% corrected version of cut out frags that also aligns its to keep nodes etc 
# def cut_out_frags(con, all_neurons):
#     con = con.iloc[:,:5]
#     con = con[con['presynaptic_to'].isin(all_neurons)]
#     for c in con.index:
#         keep_idx = [i for i, p in enumerate(con.at[c, 'postsynaptic_to']) if p in all_neurons]
#         con.at[c, 'postsynaptic_to'] = [con.at[c, 'postsynaptic_to'][i] for i in keep_idx]
#         con.at[c, 'postsynaptic_to_node'] = [con.at[c, 'postsynaptic_to_node'][i] for i in keep_idx]
#     con = con[con['postsynaptic_to'].apply(lambda x: len(x) > 0)]
#     con = con.reset_index(drop=True)
#     return con

# con = pd.read_csv('../data/connector_details2025.csv')
# all_neurons = pd.read_csv('../data/all_neurons.csv').to_numpy(np.int64, copy=False).ravel()
# con['postsynaptic_to'] = con['postsynaptic_to'].apply(ast.literal_eval)
# con['postsynaptic_to_node'] = con['postsynaptic_to_node'].apply(ast.literal_eval)

# con_short = cut_out_frags(con, all_neurons)
# con_short.to_csv('../data/con_short_2026_correctnodes.csv', index=False)

# %%
