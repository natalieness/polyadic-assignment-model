from pathlib import Path
import os
import sys

import pandas as pd
from itertools import chain
import numpy as np 

REPO_ROOT = Path("/Users/nessn/git-repos/polyadic-assignment-model")
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)
from models.validate_predictions_functions import get_target_co_occurrence

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

def compute_cosine(real_flat, padj_perm):
    # k=0 keeps the diagonal, so same-target repeats count towards the similarity.
    pred_flat = padj_perm[np.triu_indices_from(padj_perm, k=0)]
    dot_product = np.dot(real_flat, pred_flat)
    norm_real = np.linalg.norm(real_flat)
    norm_pred = np.linalg.norm(pred_flat)
    if norm_real == 0 or norm_pred == 0:
        return np.nan
    else:
        return (dot_product / (norm_real * norm_pred))