import json
from pathlib import Path

import pandas as pd
import os
import sys
import ast

from sympy import false, true

from sympy import true

REPO_ROOT = Path(
    "/Users/nessn/git-repos/polyadic-assignment-model"
).resolve()

# Makes `models.approach1` importable.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensures the relative data/model paths in the config resolve correctly.
os.chdir(REPO_ROOT)

from models.approach1.data import DataRepository
from models.approach1.model import PolyadicAssignmentModel


# MODEL_PATH = Path("models/approach1/fitted/mbons_no_target_availability.pkl")
MODEL_PATH = Path("models/approach1/fitted/mbons_new_ctrl.pkl")
SPLIT_PATH = Path("models/approach1/mbon_splits.json")


#%% Load fitted model, data access, and test neuron IDs.
model = PolyadicAssignmentModel.load(MODEL_PATH)
repository = DataRepository(model.config.paths)

with SPLIT_PATH.open() as stream:
    splits = json.load(stream)

test_ids = [int(skeleton_id) for skeleton_id in splits["test"]]

site_frames = []
assignment_frames = []
diagnostic_rows = []
failure_rows = []

for number, skeleton_id in enumerate(test_ids, start=1):
    print(f"Predicting {number}/{len(test_ids)}: skeleton {skeleton_id}")

    try:
        # These are required model inputs, not predicted labels.
        observed_inputs = repository.connectors.neuron(skeleton_id)

        # to debug self-cotargeting propensity try this 
        # model.config.partner.max_same_target_per_site = 1
        # model.config.features.use_co_targeting = False
        # model.config.features.use_target_celltype = False
        # model.config.features.use_presynaptic_celltype = False
        # model.config.features.use_hemisphere = False

        # model.config.use_target_distance_for_partner = False
        # model.config.features.use_target_availability = False
        # model.config.features.use_target_backbones = False


        # try a completely random assignment:
        # possible_configs = ['use_compartment', 'use_path_distance', 'use_branch_features', 'use_coordinates', 'use_local_cable', 'use_target_backbones', 'use_target_availability', 'use_target_distance_for_partner', 'use_target_celltype', 'use_presynaptic_celltype', 'use_hemisphere', 'use_co_targeting', 'use_target_identity_prior', 'use_target_count_features']
        # for f in possible_configs:
        #     model.config.features[f] = False

        # keep some 
        #model.config.features.use_branch_features = True

        # don't predict synaptic site location 
        dont_predict_site_location = True
        if dont_predict_site_location:
            observed = repository.connectors.neuron(skeleton_id)

            prediction = model.predict(
                repository,
                skeleton_id=skeleton_id,
                n_sites=observed.n_sites,
                target_counts=observed.target_counts,
                exact_site_node_ids=(
                    observed.rows["presynaptic_to_node"]
                    .astype(int)
                    .tolist()
                ),
                degree_mode="exact_histogram",
                degree_histogram=observed.degree_histogram,
            )


        else:

            prediction = model.predict(
                repository=repository,
                skeleton_id=skeleton_id,
                n_sites=observed_inputs.n_sites,
                target_counts=observed_inputs.target_counts,
                degree_mode='learnt',
        
            )

        sites = prediction.sites.copy()
        sites.insert(0, "skeleton_id", skeleton_id)
        site_frames.append(sites)

        assignments = prediction.assignments.copy()
        assignments.insert(0, "skeleton_id", skeleton_id)
        assignment_frames.append(assignments)

        diagnostic_rows.append({
            "skeleton_id": skeleton_id,
            **prediction.diagnostics,
        })

    except Exception as error:
        print(f"  Failed: {error}")
        failure_rows.append({
            "skeleton_id": skeleton_id,
            "error": str(error),
        })


predicted_sites_df = pd.concat(site_frames, ignore_index=True)
predicted_assignments_df = pd.concat(assignment_frames, ignore_index=True)
diagnostics_df = pd.DataFrame(diagnostic_rows)
failures_df = pd.DataFrame(failure_rows)


# Add site coordinates and predicted degree to each site-target assignment.
site_information = predicted_sites_df[[
    "skeleton_id",
    "predicted_site_id",
    "node_id",
    "parent_node_id",
    "compartment",
    "x",
    "y",
    "z",
    "branch_id",
    "cum_dist",
    "site_intensity",
    "predicted_degree_mean",
    "predicted_degree",
]]

preds = predicted_assignments_df.merge(
    site_information,
    on=["skeleton_id", "predicted_site_id"],
    how="left",
    validate="many_to_one",
)

#%% check output 

# main outputs: diagnostics_df as a summary of per neuron key features
# and preds with the actual site predictions

neus = preds.skeleton_id.unique()

# load real data 
con = pd.read_csv('data/con_short_2026_correctnodes.csv')
con_sub = con[con.presynaptic_to.isin(neus)]
con_sub['postsynaptic_to'] = con_sub.postsynaptic_to.apply(ast.literal_eval)
con_sub['postsynaptic_to_node'] = con_sub.postsynaptic_to_node.apply(ast.literal_eval)
con_e = con_sub.explode(['postsynaptic_to','postsynaptic_to_node']).reset_index(drop=True)

#%% 
# for preds, expand on multiplicity 
preds_e = (preds.loc[preds.index.repeat(preds["multiplicity"])].reset_index(drop=True))

#%%
preds_e.groupby(['skeleton_id'])['predicted_site_id'].nunique()
con_e.groupby(['presynaptic_to'])['connector_id'].nunique()
# --> same synapse counts, same unique connector id counts, 

# get aggregate connectors 

preds_agg = preds_e.groupby(['skeleton_id','predicted_site_id']).agg(list).reset_index()

con_sub['n_post'] = con_sub.postsynaptic_to.apply(len)
preds_agg['n_post'] = preds_agg.target_id.apply(len)

# check distribution of polyadic degrees 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
n_neus = len(neus)
bins = np.arange(0, 20, 1)
fig, ax = plt.subplots(2,3,figsize=(15,10))
ax = ax.flatten()
for a, s in zip(ax, neus):
    sns.histplot(data=con_sub.loc[con_sub.presynaptic_to==s], x='n_post', bins=bins, ax=a, color='blue', alpha=0.5)
    sns.histplot(data=preds_agg.loc[preds_agg.skeleton_id==s], x='n_post', bins=bins, ax=a, color='orange', alpha=0.5)


#%% target co-targeting 
from models.validate_predictions_functions import get_target_co_occurrence
from itertools import chain

def get_assignment_matrix(hyperedges):
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



def get_co_targeting_matrix(con_sub, skeleton_id, presyn_col='presynaptic_to', postsyn_col='postsynaptic_to'):
    # assume postsynaptic_to already a list 
    hyperedges = con_sub.loc[con_sub[presyn_col]==skeleton_id][postsyn_col].tolist()
    assignment_matrix, u_targets = get_assignment_matrix(hyperedges)
    co_targeting_matrix = get_target_co_occurrence(assignment_matrix)
    return co_targeting_matrix, u_targets

cos_sim = {}
for s in neus[:]:
    padj_real, u_targets_real = get_co_targeting_matrix(con_sub, s)
    padj_pred, u_targets_pred = get_co_targeting_matrix(preds_agg, s, presyn_col='skeleton_id', postsyn_col='target_id')
    padj_real = pd.DataFrame(padj_real, index=u_targets_real, columns=u_targets_real)
    padj_pred = pd.DataFrame(padj_pred, index=u_targets_pred, columns=u_targets_pred)
    padj_pred = padj_pred.reindex(index=u_targets_real, columns=u_targets_real, fill_value=0)
    # flatten and use only the upper triangle to avoid double counting
    real_flat = padj_real.values[np.triu_indices_from(padj_real, k=1)]
    pred_flat = padj_pred.values[np.triu_indices_from(padj_pred, k=1)]
    # compute cosine similarity
    dot_product = np.dot(real_flat, pred_flat)
    norm_real = np.linalg.norm(real_flat)
    norm_pred = np.linalg.norm(pred_flat)
    if norm_real == 0 or norm_pred == 0:
        cos_sim[s] = np.nan
    else:
        cos_sim[s] = (dot_product / (norm_real * norm_pred))
    max_max = max(padj_real.max().max(), padj_pred.max().max())
    cmap = 'Oranges'
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    sns.heatmap(padj_real, ax=ax[0], cmap=cmap, cbar_kws={'label': 'Co-targeting count'}, vmin=0, vmax=max_max)
    ax[0].set_title(f"Real co-targeting for skeleton {s}")
    sns.heatmap(padj_pred, ax=ax[1], cmap=cmap, cbar_kws={'label': 'Co-targeting count'}, vmin=0, vmax=max_max)
    ax[1].set_title(f"Predicted co-targeting for skeleton {s}")
    sns.heatmap(padj_pred - padj_real, ax=ax[2], cmap='coolwarm', center=0, cbar_kws={'label': 'Difference in co-targeting count'})
    ax[2].set_title(f"Difference (Predicted - Real) for skeleton {s}")
    fig.tight_layout()

cos_sim_df = pd.DataFrame.from_dict(cos_sim, orient='index', columns=['cosine_similarity'])

#%% to examine the self co-targeting propensity, calculate it explicitly here 

self_real = {}
self_pred = {}
for s in neus[:]:
    padj_real, u_targets_real = get_co_targeting_matrix(con_sub, s)
    padj_pred, u_targets_pred = get_co_targeting_matrix(preds_agg, s, presyn_col='skeleton_id', postsyn_col='target_id')
    padj_real_self = np.diag(padj_real)
    padj_pred_self = np.diag(padj_pred)
    self_real[s] = {}
    self_real[s]['mean'] = np.mean(padj_real_self)
    self_real[s]['std'] = np.std(padj_real_self)
    self_real[s]['total'] = np.sum(padj_real_self)
    self_real[s]['max'] = np.max(padj_real_self)
    self_real[s]['min'] = np.min(padj_real_self)
    self_real[s]['n_sites'] = (padj_real_self > 0).sum()
    self_real[s]['frac_sites'] = (padj_real_self > 0).sum() / len(padj_real_self)

    self_pred[s] = {}
    self_pred[s]['mean'] = np.mean(padj_pred_self)
    self_pred[s]['std'] = np.std(padj_pred_self)
    self_pred[s]['total'] = np.sum(padj_pred_self)
    self_pred[s]['max'] = np.max(padj_pred_self)
    self_pred[s]['min'] = np.min(padj_pred_self)
    self_pred[s]['n_sites'] = (padj_pred_self > 0).sum()
    self_pred[s]['frac_sites'] = (padj_pred_self > 0).sum() / len(padj_pred_self)

self_real_df = pd.DataFrame(self_real).T
self_pred_df = pd.DataFrame(self_pred).T
self_real_df['type'] = 'real'
self_pred_df['type'] = 'predicted'
self_df = pd.merge(self_real_df, self_pred_df, left_index=True, right_index=True, suffixes=('_real', '_predicted'))

self_df['propensity_score'] = self_df['mean_predicted'] / (self_df['mean_real'] + 1e-6)  # Avoid division by zero
self_df['fractional_propensity'] = self_df['frac_sites_predicted'] / (self_df['frac_sites_real'] + 1e-6)  # Avoid division by zero

#%% final score 
print(f'Average cosine similarity across all neurons: {cos_sim_df["cosine_similarity"].mean()}')
print("Mean propensity score across all neurons:", self_df['propensity_score'].mean())
print("Mean fractional propensity across all neurons:", self_df['fractional_propensity'].mean())


# %%
