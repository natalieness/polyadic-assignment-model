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

#%% load geometry samples and real samples

from models.approach2.propensity import pair_count_matrix, mean_pair_count_matrix

# get cotargeting numbers for each neuron 

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

#%% get null cotargeting with columns: null_sample, pre_id, target1, target2, null_count
from models.validate_predictions_functions import get_target_co_occurrence


all_presynaptic = list(geo_samples.keys())
obs = []
null = []
for e, skid in enumerate(all_presynaptic):
    if e % 100 == 0:
        print(f'Processing neuron {e+1}/{len(all_presynaptic)}')
    obs_ass = real_samples[skid]
    null_samples = geo_samples[skid]
    targets = target_skids[skid]

    observed = get_target_co_occurrence(obs_ass)
    expected_samples = np.stack([get_target_co_occurrence(x) for x in null_samples], axis=0)

    target1, target2 = np.meshgrid(targets, targets)
    mask = np.triu(np.ones_like(observed, dtype=bool), k=1)
    # get real 
    local_obs = np.array([target1[mask], target2[mask], observed[mask]]).T
    # get all null 
    local_nulls = []
    for n_idx, n in enumerate(expected_samples):
        local_null = np.array([target1[mask], target2[mask], n[mask]]).T
        local_null = np.hstack((np.full((local_null.shape[0], 1), n_idx), local_null))
        local_nulls.append(local_null)
    big_local_null = np.vstack(local_nulls)

    # add pre to both 
    big_local_null = np.hstack((np.full((big_local_null.shape[0], 1), skid), big_local_null))
    local_obs = np.hstack((np.full((local_obs.shape[0], 1), skid), local_obs))
    obs.append(local_obs)
    null.append(big_local_null)
obs = np.vstack(obs)
null = np.vstack(null)
observed_co = pd.DataFrame(obs, columns=['pre_id', 'target1', 'target2', 'observed_count'])
null_co = pd.DataFrame(null, columns=['pre_id', 'null_sample', 'target1', 'target2', 'null_count'])
observed_co = observed_co[observed_co['observed_count'] > 0]
null_co = null_co[null_co['null_count'] > 0]

#%% get edges and metadata for pre and post neurons

adj = pd.read_csv('data/adj.csv')
adj.set_index('sources', inplace=True)
adj = adj.astype(int)
adj.columns = adj.columns.astype(int)
adj.index = adj.index.astype(int)

source, target = np.meshgrid(adj.index.values, adj.columns.values, indexing='ij')
edges = np.array([source.flatten(), target.flatten(), adj.values.flatten()])
edges = pd.DataFrame(edges.T, columns=['source', 'target', 'weight'])


# presynaptic neuron metadata 
pre = pd.DataFrame({'pre_id': edges['source'].unique()})
pre['pre_celltype'] = pre['pre_id'].map(skid_to_celltype)
pre['pre_ei'] = pre['pre_id'].map(lambda x: 'I' if x in inhibitory_neurons else 'E')
pre['pre_hemisphere'] = pre['pre_id'].map(lambda x: 'L' if x in left_neus else ('R' if x in right_neus else 'C'))
pre['pre_hierarchy'] = pre['pre_id'].map(skid_to_flow)

# postsynaptic neuron metadata
post = pd.DataFrame({'neuron_id': edges['target'].unique()})
post['celltype'] = post['neuron_id'].map(skid_to_celltype)
post['ei'] = post['neuron_id'].map(lambda x: 'I' if x in inhibitory_neurons else 'E')
post['hierarchy'] = post['neuron_id'].map(skid_to_flow)
post['hemisphere'] = post['neuron_id'].map(lambda x: 'L' if x in left_neus else ('R' if x in right_neus else 'C'))


# %% run model 

import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr

MIN_NULL_EXPECTATION = 1e-6
TOP_N_INTERACTIONS = 30

pre["pre_id"] = pre["pre_id"].astype(int)
post["neuron_id"] = post["neuron_id"].astype(int)
observed_co[["pre_id","target1","target2"]] = observed_co[["pre_id","target1","target2"]].astype(int)
null_co[["pre_id","target1","target2"]] = null_co[["pre_id","target1","target2"]].astype(int)
edges[["source","target"]] = edges[["source","target"]].astype(int)

def canonical_pairs(df):
    df = df.copy()
    a = df["target1"].astype(int).values
    b = df["target2"].astype(int).values
    df["target1"] = np.minimum(a,b)
    df["target2"] = np.maximum(a,b)
    return df

obs = canonical_pairs(observed_co)
null = canonical_pairs(null_co)

null_summary = null.groupby(["pre_id","target1","target2"],as_index=False)["null_count"].agg(null_mean="mean",null_sd="std",null_median="median",null_min="min",null_max="max")
df = obs.merge(null_summary,on=["pre_id","target1","target2"],how="left",validate="one_to_one")
df["null_sd"] = df["null_sd"].fillna(0)
df["raw_fold_enrichment"] = np.where(df["null_mean"] > 0,df["observed_count"] / df["null_mean"],np.nan)
df["log2_enrichment"] = np.log2((df["observed_count"] + 0.5) / (df["null_mean"] + 0.5))
df["null_zero"] = df["null_mean"].fillna(0) <= 0

print("\n=== BASIC DATA CHECK ===")
print(df[["observed_count","null_mean","raw_fold_enrichment","log2_enrichment"]].describe())
print("\nPairs with zero expected count under the null:",df["null_zero"].sum())
print("Total target pairs:",len(df))
print("Presynaptic neurons:",df["pre_id"].nunique())
#%%
post1 = post.add_prefix("t1_").rename(columns={"t1_neuron_id":"target1"})
post2 = post.add_prefix("t2_").rename(columns={"t2_neuron_id":"target2"})
df = df.merge(post1,on="target1",how="left",validate="many_to_one")
df = df.merge(post2,on="target2",how="left",validate="many_to_one")
df = df.merge(pre,on="pre_id",how="left",validate="many_to_one")

df["same_celltype"] = (df["t1_celltype"] == df["t2_celltype"]).astype(int)
df["celltype_pair"] = df.apply(lambda x:"|".join(sorted([str(x["t1_celltype"]),str(x["t2_celltype"])])),axis=1)
df["ei_pair"] = df.apply(lambda x:"|".join(sorted([str(x["t1_ei"]),str(x["t2_ei"])])),axis=1)
df["same_ei"] = (df["t1_ei"] == df["t2_ei"]).astype(int)
df["same_hemisphere"] = (df["t1_hemisphere"] == df["t2_hemisphere"]).astype(int)
df["hierarchy_difference"] = (df["t1_hierarchy"] - df["t2_hierarchy"]).abs()
df["hierarchy_mean"] = (df["t1_hierarchy"] + df["t2_hierarchy"]) / 2
df["t1_relative_hierarchy"] = df["t1_hierarchy"] - df["pre_hierarchy"]
df["t2_relative_hierarchy"] = df["t2_hierarchy"] - df["pre_hierarchy"]
df["both_above_pre"] = ((df["t1_relative_hierarchy"] > 0) & (df["t2_relative_hierarchy"] > 0)).astype(int)
df["both_below_pre"] = ((df["t1_relative_hierarchy"] < 0) & (df["t2_relative_hierarchy"] < 0)).astype(int)
df["opposite_sides_pre"] = ((df["t1_relative_hierarchy"] * df["t2_relative_hierarchy"]) < 0).astype(int)

if "t1_modality" in df.columns and "t2_modality" in df.columns:
    df["same_modality"] = np.where(df["t1_modality"].notna() & df["t2_modality"].notna(),(df["t1_modality"] == df["t2_modality"]).astype(float),np.nan)

nodes = pd.Index(pd.unique(pd.concat([edges["source"],edges["target"],post["neuron_id"]],ignore_index=True)))
node_to_index = {n:i for i,n in enumerate(nodes)}
row_out = edges["source"].map(node_to_index).values
col_out = edges["target"].map(node_to_index).values
weights = edges["weight"].astype(float).values
out_matrix = csr_matrix((weights,(row_out,col_out)),shape=(len(nodes),len(nodes)))
in_matrix = out_matrix.T.tocsr()

def sparse_cosine(matrix,a,b):
    if a not in node_to_index or b not in node_to_index:
        return np.nan
    x = matrix.getrow(node_to_index[a])
    y = matrix.getrow(node_to_index[b])
    nx = np.sqrt(x.multiply(x).sum())
    ny = np.sqrt(y.multiply(y).sum())
    return float(x.multiply(y).sum() / (nx * ny)) if nx > 0 and ny > 0 else np.nan

unique_pairs = df[["target1","target2"]].drop_duplicates().copy()
unique_pairs["input_similarity"] = [sparse_cosine(in_matrix,a,b) for a,b in zip(unique_pairs["target1"],unique_pairs["target2"])]
unique_pairs["output_similarity"] = [sparse_cosine(out_matrix,a,b) for a,b in zip(unique_pairs["target1"],unique_pairs["target2"])]
df = df.merge(unique_pairs,on=["target1","target2"],how="left",validate="many_to_one")

#%%
print("\n=== PART 1: OVERALL OBSERVED VS EXPECTED ===")
overall = pd.DataFrame({"observed_total":[df["observed_count"].sum()],"expected_total":[df["null_mean"].sum()]})
overall["fold_enrichment"] = overall["observed_total"] / overall["expected_total"]
print(overall)
#%%
print("\n=== PART 2: TARGET CELL-TYPE PAIR ENRICHMENT ===")
celltype_summary = df.groupby("celltype_pair",as_index=False).agg(observed=("observed_count","sum"),expected=("null_mean","sum"),n_pairs=("observed_count","size"))
celltype_summary["fold_enrichment"] = celltype_summary["observed"] / celltype_summary["expected"]
celltype_summary["log2_enrichment"] = np.log2((celltype_summary["observed"] + 0.5) / (celltype_summary["expected"] + 0.5))
celltype_summary = celltype_summary.sort_values("fold_enrichment",ascending=False)
print(celltype_summary.to_string(index=False))

plot_df = celltype_summary[celltype_summary["expected"] > 0].head(30).sort_values("log2_enrichment")
plt.figure(figsize=(8,max(5,len(plot_df) * 0.25)))
plt.barh(plot_df["celltype_pair"],plot_df["log2_enrichment"])
plt.axvline(0,linewidth=1)
plt.xlabel("Log2 observed / expected co-targeting")
plt.ylabel("Postsynaptic cell-type pair")
plt.title("Most enriched postsynaptic cell-type pairs")
plt.tight_layout()
plt.show()
#%%
print("\n=== PART 3: SAME VS DIFFERENT POSTSYNAPTIC CELL TYPE ===")
same_type_summary = df.groupby("same_celltype",as_index=False).agg(observed=("observed_count","sum"),expected=("null_mean","sum"),n_pairs=("observed_count","size"))
same_type_summary["group"] = same_type_summary["same_celltype"].map({0:"different cell types",1:"same cell type"})
same_type_summary["fold_enrichment"] = same_type_summary["observed"] / same_type_summary["expected"]
same_type_summary["log2_enrichment"] = np.log2((same_type_summary["observed"] + 0.5) / (same_type_summary["expected"] + 0.5))
print(same_type_summary[["group","observed","expected","fold_enrichment","log2_enrichment","n_pairs"]].to_string(index=False))
#%%
print("\n=== PART 4: SAME VS DIFFERENT TARGET TYPE BY PRESYNAPTIC E/I ===")
pre_ei_summary = df.groupby(["pre_ei","same_celltype"],as_index=False).agg(observed=("observed_count","sum"),expected=("null_mean","sum"),n_pairs=("observed_count","size"))
pre_ei_summary["target_relationship"] = pre_ei_summary["same_celltype"].map({0:"different",1:"same"})
pre_ei_summary["fold_enrichment"] = pre_ei_summary["observed"] / pre_ei_summary["expected"]
pre_ei_summary["log2_enrichment"] = np.log2((pre_ei_summary["observed"] + 0.5) / (pre_ei_summary["expected"] + 0.5))
print(pre_ei_summary.to_string(index=False))

for pre_class in sorted(df["pre_ei"].dropna().unique()):
    x = pre_ei_summary[pre_ei_summary["pre_ei"] == pre_class]
    plt.figure(figsize=(5,4))
    plt.bar(x["target_relationship"],x["log2_enrichment"])
    plt.axhline(0,linewidth=1)
    plt.ylabel("Log2 observed / expected")
    plt.title("Presynaptic " + str(pre_class))
    plt.tight_layout()
    plt.show()
#%%
print("\n=== PART 5: PRESYNAPTIC CELL TYPE x POSTSYNAPTIC CELL-TYPE PAIR ===")
motif_summary = df.groupby(["pre_celltype","celltype_pair"],as_index=False).agg(observed=("observed_count","sum"),expected=("null_mean","sum"),n_pairs=("observed_count","size"),n_pre=("pre_id","nunique"))
motif_summary["fold_enrichment"] = motif_summary["observed"] / motif_summary["expected"]
motif_summary["log2_enrichment"] = np.log2((motif_summary["observed"] + 0.5) / (motif_summary["expected"] + 0.5))
motif_summary = motif_summary.sort_values("fold_enrichment",ascending=False)
print(motif_summary.head(100).to_string(index=False))
#%%
print("\n=== PART 6: POSTSYNAPTIC SIMILARITY / HOMOPHILY SUMMARIES ===")
similarity_summary = df.groupby("same_celltype",as_index=False).agg(mean_input_similarity=("input_similarity","mean"),mean_output_similarity=("output_similarity","mean"),mean_hierarchy_difference=("hierarchy_difference","mean"),mean_log2_enrichment=("log2_enrichment","mean"),n_pairs=("observed_count","size"))
similarity_summary["group"] = similarity_summary["same_celltype"].map({0:"different cell types",1:"same cell type"})
print(similarity_summary.to_string(index=False))

plot_similarity = df[["log2_enrichment","input_similarity","output_similarity","hierarchy_difference"]].replace([np.inf,-np.inf],np.nan).dropna()
if len(plot_similarity) > 2:
    print("\nCorrelation between enrichment and input similarity:",pearsonr(plot_similarity["log2_enrichment"],plot_similarity["input_similarity"]))
    print("Correlation between enrichment and output similarity:",pearsonr(plot_similarity["log2_enrichment"],plot_similarity["output_similarity"]))
    print("Correlation between enrichment and hierarchy difference:",pearsonr(plot_similarity["log2_enrichment"],plot_similarity["hierarchy_difference"]))
#%%
print("\n=== PART 7: HIERARCHICAL RELATIONSHIP TO PRESYNAPTIC NEURON ===")
hierarchy_summary = pd.DataFrame({"relationship":["both above presynaptic","both below presynaptic","opposite sides of presynaptic"],"observed":[df.loc[df["both_above_pre"] == 1,"observed_count"].sum(),df.loc[df["both_below_pre"] == 1,"observed_count"].sum(),df.loc[df["opposite_sides_pre"] == 1,"observed_count"].sum()],"expected":[df.loc[df["both_above_pre"] == 1,"null_mean"].sum(),df.loc[df["both_below_pre"] == 1,"null_mean"].sum(),df.loc[df["opposite_sides_pre"] == 1,"null_mean"].sum()]})
hierarchy_summary["fold_enrichment"] = hierarchy_summary["observed"] / hierarchy_summary["expected"]
hierarchy_summary["log2_enrichment"] = np.log2((hierarchy_summary["observed"] + 0.5) / (hierarchy_summary["expected"] + 0.5))
print(hierarchy_summary.to_string(index=False))
#%%
print("\n=== PART 8: NEGATIVE-BINOMIAL ENRICHMENT MODELS ===")
model_df = df[df["null_mean"].notna() & (df["null_mean"] > MIN_NULL_EXPECTATION)].copy()
for numeric_predictor in ["input_similarity","output_similarity","hierarchy_difference","hierarchy_mean"]:
    model_df[numeric_predictor] = model_df[numeric_predictor].fillna(model_df[numeric_predictor].median())
model_df["pre_celltype"] = model_df["pre_celltype"].fillna("unknown")
model_df["offset"] = np.log(model_df["null_mean"])

# Standardised versions so slopes are per standard deviation and comparable across predictors.
for numeric_predictor in ["input_similarity","output_similarity","hierarchy_difference"]:
    values = model_df[numeric_predictor]
    model_df[numeric_predictor + "_z"] = (values - values.mean()) / values.std()

# All models must be fitted on the same rows or their AIC/deviance are not comparable.
MODEL_COLUMNS = ["observed_count","offset","pre_id","pre_ei","pre_celltype","celltype_pair","ei_pair","same_celltype","same_ei","same_hemisphere","hierarchy_difference","hierarchy_mean","input_similarity","output_similarity","input_similarity_z","output_similarity_z","hierarchy_difference_z"]
n_before = len(model_df)
model_df = model_df.dropna(subset=MODEL_COLUMNS)
print("\nRows available for modelling:",len(model_df),"(dropped",n_before - len(model_df),"incomplete rows)")

def fit_model(formula):
    model = smf.glm(formula=formula,data=model_df,family=sm.families.NegativeBinomial(alpha=1.0),offset=model_df["offset"])
    # patsy drops rows with missing predictors, but cov_kwds is applied after that,
    # so the cluster groups have to be subset to the rows the design matrix kept.
    groups = model_df.loc[model.data.row_labels,"pre_id"].values
    return model.fit(cov_type="cluster",cov_kwds={"groups":groups})

m0 = fit_model("observed_count ~ 1")
print("\nNull model AIC:",m0.aic,"deviance:",m0.deviance)
m1 = fit_model("observed_count ~ C(pre_ei)")
print("\nPresynaptic E/I model AIC:",m1.aic,"deviance:",m1.deviance)
m2 = fit_model("observed_count ~ C(pre_ei) + same_celltype + same_ei + same_hemisphere")
print("\nTarget broad features model AIC:",m2.aic,"deviance:",m2.deviance)
m3 = fit_model("observed_count ~ C(pre_ei) * same_celltype + same_ei + same_hemisphere")
print("\nPresynaptic E/I x same target type model AIC:",m3.aic,"deviance:",m3.deviance)
m4 = fit_model("observed_count ~ C(pre_ei) * same_celltype + same_ei + same_hemisphere + hierarchy_difference + hierarchy_mean + input_similarity + output_similarity")
print("\nNetwork features model AIC:",m4.aic,"deviance:",m4.deviance)
# Saturated presynaptic-type x target-pair model, written as one combined factor.
# C(a):C(b) builds the full cross product including the many empty cells, which makes
# the design far wider than the data supports; a single factor only creates the levels
# that actually occur, giving the same fitted values from a full-rank design.


model_df["pre_x_pair"] = model_df["pre_celltype"].astype(str) + "|" + model_df["celltype_pair"].astype(str)
print("\nPopulated presynaptic-type x target-pair cells:",model_df["pre_x_pair"].nunique(),"of",model_df["pre_celltype"].nunique() * model_df["celltype_pair"].nunique(),"possible")
m5 = fit_model("observed_count ~ C(pre_x_pair) + hierarchy_difference + input_similarity + output_similarity")
print("\nPresynaptic-celltype x target-celltype-pair model AIC:",m5.aic,"deviance:",m5.deviance)

# Does the presynaptic cell type change how strongly network features predict co-targeting?
# Target-pair identity, E/I composition and hemisphere are held as additive controls, and each
# standardised feature is allowed its own slope per presynaptic cell type.

SLOPE_PREDICTORS = ["input_similarity_z","output_similarity_z","hierarchy_difference_z"] 
m6 = fit_model("observed_count ~ input_similarity_z + output_similarity_z + hierarchy_difference_z + C(celltype_pair) + C(ei_pair) + same_hemisphere + C(pre_celltype):input_similarity_z + C(pre_celltype):output_similarity_z + C(pre_celltype):hierarchy_difference_z") 
print("\nPresynaptic-celltype x network-feature slope model AIC:",m6.aic,"deviance:",m6.deviance)


model_comparison = pd.DataFrame({"model":["null","pre E/I","target broad features","pre E/I x same target type","network features","pre celltype x target pair","pre celltype x feature slopes"],"AIC":[m0.aic,m1.aic,m2.aic,m3.aic,m4.aic,m5.aic,m6.aic],"deviance":[m0.deviance,m1.deviance,m2.deviance,m3.deviance,m4.deviance,m5.deviance,m6.deviance],"df_model":[m0.df_model,m1.df_model,m2.df_model,m3.df_model,m4.df_model,m5.df_model,m6.df_model]})
print("\nModel comparison:")
print(model_comparison.to_string(index=False))



#%%
def coefficient_table(model):
    out = pd.DataFrame({"term":model.params.index,"log_enrichment":model.params.values,"SE":model.bse.values,"p":model.pvalues.values})
    ci = model.conf_int()
    out["fold_enrichment"] = np.exp(out["log_enrichment"])
    out["fold_CI_low"] = np.exp(ci[0].values)
    out["fold_CI_high"] = np.exp(ci[1].values)
    return out.sort_values("p")

print("\nBroad biological model coefficients:")
broad_coefficients = coefficient_table(m4)
print(broad_coefficients.to_string(index=False))

print("\nPresynaptic-celltype x target-celltype-pair model, strongest terms:")
interaction_coefficients = coefficient_table(m5)
print(interaction_coefficients.head(100).to_string(index=False))
#%% 
ic = interaction_coefficients.copy()
ic["pre_celltype"] = ic["term"].str.extract(r"C\(pre_x_pair\)\[T\.(.*?)\|")[0]
ic["celltype_pair"] = ic["term"].str.extract(r"\|(.*)\]")[0]
ic = ic.merge(motif_summary,on=["pre_celltype","celltype_pair"],how="left",validate="many_to_one")



#%%


plot_coefficients = interaction_coefficients[(interaction_coefficients["term"] != "Intercept") & interaction_coefficients["term"].str.contains("pre_x_pair")].sort_values("p").head(TOP_N_INTERACTIONS).sort_values("fold_enrichment")
if len(plot_coefficients) > 0:
    plt.figure(figsize=(9,max(5,len(plot_coefficients) * 0.28)))
    plt.barh(plot_coefficients["term"],np.log2(plot_coefficients["fold_enrichment"]))
    plt.axvline(0,linewidth=1)
    plt.xlabel("Log2 fold enrichment relative to geometry null")
    plt.ylabel("Interaction")
    plt.title("Strongest presynaptic-type x target-pair effects")
    plt.tight_layout()
    plt.show()

print("\n=== PART 8B: FEATURE SLOPES BY PRESYNAPTIC CELL TYPE ===")
from scipy.stats import norm

def celltype_feature_slopes(model,predictor):
    """Total slope of `predictor` within each presynaptic cell type.

    Patsy folds the reference cell type into the main effect and codes every other level as a
    difference from it, so the slope for a given type is main + interaction. The standard error
    of that sum needs the covariance between the two terms, not just their individual variances.
    """
    params = model.params
    cov = model.cov_params()
    rows = []
    for celltype in sorted(model_df["pre_celltype"].astype(str).unique()):
        term = "C(pre_celltype)[T." + celltype + "]:" + predictor
        if term in params.index:
            slope = params[predictor] + params[term]
            variance = cov.loc[predictor,predictor] + cov.loc[term,term] + 2 * cov.loc[predictor,term]
            difference = params[term]
            difference_se = np.sqrt(cov.loc[term,term])
        else:
            slope = params[predictor]
            variance = cov.loc[predictor,predictor]
            difference = 0.0
            difference_se = np.nan
        se = np.sqrt(variance)
        rows.append({"feature":predictor,"pre_celltype":celltype,"is_reference":term not in params.index,"slope":slope,"slope_SE":se,"fold_per_SD":np.exp(slope),"fold_CI_low":np.exp(slope - 1.96 * se),"fold_CI_high":np.exp(slope + 1.96 * se),"p_slope":2 * norm.sf(np.abs(slope / se)),"difference_vs_reference":difference,"p_difference":2 * norm.sf(np.abs(difference / difference_se)) if difference_se == difference_se else np.nan})
    return pd.DataFrame(rows)

celltype_support = model_df.groupby("pre_celltype",as_index=False).agg(n_pairs=("observed_count","size"),n_pre=("pre_id","nunique"))
slope_table = pd.concat([celltype_feature_slopes(m6,predictor) for predictor in SLOPE_PREDICTORS],ignore_index=True)
slope_table = slope_table.merge(celltype_support,on="pre_celltype",how="left")
print("\nPer-standard-deviation fold change in co-targeting, by presynaptic cell type:")
print(slope_table.sort_values(["feature","slope"],ascending=[True,False]).to_string(index=False))

print("\nGlobal feature slopes (reference cell type) and their interaction significance:")
feature_coefficients = coefficient_table(m6)
print(feature_coefficients[feature_coefficients["term"].str.contains("similarity_z|hierarchy_difference_z")].to_string(index=False))

for predictor in SLOPE_PREDICTORS:
    subset = slope_table[(slope_table["feature"] == predictor) & (slope_table["n_pre"] >= 2)].sort_values("slope")
    if len(subset) == 0:
        continue
    plt.figure(figsize=(7,max(4,len(subset) * 0.3)))
    plt.errorbar(subset["slope"],range(len(subset)),xerr=1.96 * subset["slope_SE"],fmt="o",capsize=3)
    plt.yticks(range(len(subset)),subset["pre_celltype"])
    plt.axvline(0,linewidth=1,color="grey")
    plt.xlabel("Log fold change in co-targeting per SD of " + predictor.replace("_z",""))
    plt.ylabel("Presynaptic cell type")
    plt.title("Feature sensitivity by presynaptic cell type")
    plt.tight_layout()
    plt.show()

print("\n=== PART 9: SPECIFIC CELL-TYPE MOTIFS ===")
specific_motifs = motif_summary[(motif_summary["expected"] >= 5) & (motif_summary["n_pre"] >= 2)].copy()
specific_motifs = specific_motifs.sort_values("log2_enrichment",ascending=False)
print("\nMost enriched motifs:")
print(specific_motifs.head(30).to_string(index=False))
print("\nMost depleted motifs:")
print(specific_motifs.tail(30).sort_values("log2_enrichment").to_string(index=False))

print("\n=== PART 10: EMPIRICAL NULL TESTS FOR CELL-TYPE MOTIFS ===")
observed_motif = df.groupby(["pre_celltype","celltype_pair"],as_index=False)["observed_count"].sum().rename(columns={"observed_count":"observed"})
null_with_pre = null.merge(pre[["pre_id","pre_celltype"]],on="pre_id",how="left")
null_post1 = post[["neuron_id","celltype"]].rename(columns={"neuron_id":"target1","celltype":"t1_celltype"})
null_post2 = post[["neuron_id","celltype"]].rename(columns={"neuron_id":"target2","celltype":"t2_celltype"})
null_with_pre = null_with_pre.merge(null_post1,on="target1",how="left").merge(null_post2,on="target2",how="left")
null_with_pre["celltype_pair"] = null_with_pre.apply(lambda x:"|".join(sorted([str(x["t1_celltype"]),str(x["t2_celltype"])])),axis=1)
null_motif_samples = null_with_pre.groupby(["null_sample","pre_celltype","celltype_pair"],as_index=False)["null_count"].sum()
null_motif_stats = null_motif_samples.groupby(["pre_celltype","celltype_pair"],as_index=False)["null_count"].agg(null_mean="mean",null_sd="std")
empirical_motifs = observed_motif.merge(null_motif_stats,on=["pre_celltype","celltype_pair"],how="left")
empirical_motifs["fold_enrichment"] = empirical_motifs["observed"] / empirical_motifs["null_mean"]
empirical_motifs["z_vs_null"] = (empirical_motifs["observed"] - empirical_motifs["null_mean"]) / empirical_motifs["null_sd"]
empirical_motifs = empirical_motifs.sort_values("z_vs_null",ascending=False)
print(empirical_motifs.head(50).to_string(index=False))

# %%
print("\n=== PART 11: BROAD FEATURE MODELS PER PRESYNAPTIC NEURON ===")

feature_columns = ["input_similarity","output_similarity","hierarchy_difference","hierarchy_mean","same_celltype","same_ei","same_hemisphere"]
if "same_modality" in model_df.columns:
    feature_columns.append("same_modality")

broad_df = model_df.copy()
broad_df = broad_df.replace([np.inf,-np.inf],np.nan)
broad_df = broad_df.dropna(subset=["observed_count","null_mean","pre_id"])
broad_df = broad_df[broad_df["null_mean"] > MIN_NULL_EXPECTATION].copy()

continuous_features = [x for x in ["input_similarity","output_similarity","hierarchy_difference","hierarchy_mean"] if x in feature_columns]
binary_features = [x for x in feature_columns if x not in continuous_features]

for feature in continuous_features:
    broad_df[feature] = broad_df[feature].fillna(broad_df[feature].median())

for feature in binary_features:
    broad_df[feature] = broad_df[feature].fillna(0)

for feature in continuous_features:
    broad_df[feature + "_z"] = (broad_df[feature] - broad_df[feature].mean()) / broad_df[feature].std()

model_features = [x + "_z" if x in continuous_features else x for x in feature_columns]

print("Features used:",model_features)

def fit_simple_nb(data,features):
    formula = "observed_count ~ " + " + ".join(features)
    return smf.glm(formula=formula,data=data,family=sm.families.NegativeBinomial(alpha=1.0),offset=np.log(data["null_mean"])).fit()

def deviance_importance(data,features):
    full = fit_simple_nb(data,features)
    rows = []
    for feature in features:
        reduced_features = [x for x in features if x != feature]
        if len(reduced_features) == 0:
            continue
        reduced = fit_simple_nb(data,reduced_features)
        rows.append({"feature":feature,"full_deviance":full.deviance,"reduced_deviance":reduced.deviance,"delta_deviance":reduced.deviance - full.deviance})
    return full,pd.DataFrame(rows)

per_neuron_coefficients = []
per_neuron_importance = []

for pre_id,sub in broad_df.groupby("pre_id"):
    if len(sub) < 50:
        continue
    if sub["observed_count"].sum() < 5:
        continue
    usable_features = [x for x in model_features if sub[x].nunique(dropna=True) > 1]
    if len(usable_features) < 2:
        continue
    try:
        full,importance = deviance_importance(sub,usable_features)
        for feature in usable_features:
            if feature in full.params.index:
                per_neuron_coefficients.append({"pre_id":pre_id,"feature":feature,"coefficient":full.params[feature],"fold_effect":np.exp(full.params[feature]),"p":full.pvalues[feature],"n_pairs":len(sub),"observed_total":sub["observed_count"].sum(),"expected_total":sub["null_mean"].sum()})
        if len(importance) > 0:
            importance["pre_id"] = pre_id
            importance["n_pairs"] = len(sub)
            per_neuron_importance.append(importance)
    except Exception as e:
        print("Skipped neuron",pre_id,"because:",str(e))

per_neuron_coefficients = pd.DataFrame(per_neuron_coefficients)
per_neuron_importance = pd.concat(per_neuron_importance,ignore_index=True) if len(per_neuron_importance) > 0 else pd.DataFrame()

print("\nPer-neuron coefficient summary:")
if len(per_neuron_coefficients) > 0:
    print(per_neuron_coefficients.groupby("feature").agg(n_neurons=("pre_id","nunique"),median_coefficient=("coefficient","median"),mean_coefficient=("coefficient","mean"),median_fold_effect=("fold_effect","median"),fraction_positive=("coefficient",lambda x:(x > 0).mean())).reset_index().sort_values("median_coefficient",ascending=False).to_string(index=False))

print("\nPer-neuron feature importance summary:")
if len(per_neuron_importance) > 0:
    neuron_importance_summary = per_neuron_importance.groupby("feature").agg(n_neurons=("pre_id","nunique"),median_delta_deviance=("delta_deviance","median"),mean_delta_deviance=("delta_deviance","mean")).reset_index().sort_values("median_delta_deviance",ascending=False)
    print(neuron_importance_summary.to_string(index=False))

print("\n=== PART 12: PER-NEURON FEATURE COEFFICIENT HEATMAP ===")

if len(per_neuron_coefficients) > 0:
    coefficient_matrix = per_neuron_coefficients.pivot(index="pre_id",columns="feature",values="coefficient")
    coefficient_matrix = coefficient_matrix.loc[coefficient_matrix.notna().sum(axis=1).sort_values(ascending=False).index]
    plot_matrix = coefficient_matrix.head(100)
    plt.figure(figsize=(10,max(6,len(plot_matrix) * 0.12)))
    plt.imshow(plot_matrix,aspect="auto")
    plt.colorbar(label="Coefficient")
    plt.xticks(range(len(plot_matrix.columns)),plot_matrix.columns,rotation=45,ha="right")
    plt.yticks(range(len(plot_matrix.index)),plot_matrix.index)
    plt.xlabel("Feature")
    plt.ylabel("Presynaptic neuron")
    plt.title("Broad co-targeting feature coefficients per presynaptic neuron")
    plt.tight_layout()
    plt.show()

print("\n=== PART 13: BROAD FEATURE MODELS PER PRESYNAPTIC CELL TYPE ===")

group_coefficients = []
group_importance = []

for pre_celltype,sub in broad_df.groupby("pre_celltype"):
    if len(sub) < 100:
        continue
    if sub["pre_id"].nunique() < 3:
        continue
    usable_features = [x for x in model_features if sub[x].nunique(dropna=True) > 1]
    if len(usable_features) < 2:
        continue
    try:
        formula = "observed_count ~ " + " + ".join(usable_features)
        full = smf.glm(formula=formula,data=sub,family=sm.families.NegativeBinomial(alpha=1.0),offset=np.log(sub["null_mean"])).fit(cov_type="cluster",cov_kwds={"groups":sub["pre_id"]})
        for feature in usable_features:
            if feature in full.params.index:
                group_coefficients.append({"pre_celltype":pre_celltype,"feature":feature,"coefficient":full.params[feature],"fold_effect":np.exp(full.params[feature]),"SE":full.bse[feature],"p":full.pvalues[feature],"n_pairs":len(sub),"n_pre":sub["pre_id"].nunique()})
        for feature in usable_features:
            reduced_features = [x for x in usable_features if x != feature]
            reduced_formula = "observed_count ~ " + " + ".join(reduced_features)
            reduced = smf.glm(formula=reduced_formula,data=sub,family=sm.families.NegativeBinomial(alpha=1.0),offset=np.log(sub["null_mean"])).fit()
            group_importance.append({"pre_celltype":pre_celltype,"feature":feature,"delta_deviance":reduced.deviance - full.deviance,"n_pairs":len(sub),"n_pre":sub["pre_id"].nunique()})
    except Exception as e:
        print("Skipped presynaptic cell type",pre_celltype,"because:",str(e))

group_coefficients = pd.DataFrame(group_coefficients)
group_importance = pd.DataFrame(group_importance)

print("\nPer-presynaptic-celltype coefficients:")
if len(group_coefficients) > 0:
    print(group_coefficients.sort_values(["pre_celltype","p"]).to_string(index=False))

print("\nPer-presynaptic-celltype feature importance:")
if len(group_importance) > 0:
    print(group_importance.sort_values(["pre_celltype","delta_deviance"],ascending=[True,False]).to_string(index=False))

print("\n=== PART 14: FEATURE IMPORTANCE HEATMAP BY PRESYNAPTIC CELL TYPE ===")

if len(group_importance) > 0:
    importance_matrix = group_importance.pivot(index="pre_celltype",columns="feature",values="delta_deviance")
    plt.figure(figsize=(10,max(5,len(importance_matrix) * 0.4)))
    plt.imshow(importance_matrix,aspect="auto")
    plt.colorbar(label="Increase in deviance when feature removed")
    plt.xticks(range(len(importance_matrix.columns)),importance_matrix.columns,rotation=45,ha="right")
    plt.yticks(range(len(importance_matrix.index)),importance_matrix.index)
    plt.xlabel("Feature")
    plt.ylabel("Presynaptic cell type")
    plt.title("Predictive importance of co-targeting features")
    plt.tight_layout()
    plt.show()

print("\n=== PART 15: COMBINE COEFFICIENT DIRECTION AND FEATURE IMPORTANCE ===")

if len(group_coefficients) > 0 and len(group_importance) > 0:
    group_feature_summary = group_coefficients.merge(group_importance[["pre_celltype","feature","delta_deviance"]],on=["pre_celltype","feature"],how="left")
    group_feature_summary["direction"] = np.where(group_feature_summary["coefficient"] > 0,"positive","negative")
    group_feature_summary = group_feature_summary.sort_values(["pre_celltype","delta_deviance"],ascending=[True,False])
    print(group_feature_summary.to_string(index=False))

print("\n=== PART 16: WHICH FEATURES DOMINATE FOR EACH PRESYNAPTIC NEURON? ===")

if len(per_neuron_importance) > 0:
    strongest_feature_per_neuron = per_neuron_importance.sort_values("delta_deviance",ascending=False).groupby("pre_id",as_index=False).first()
    strongest_feature_per_neuron = strongest_feature_per_neuron.rename(columns={"feature":"strongest_feature","delta_deviance":"strongest_delta_deviance"})
    if len(per_neuron_coefficients) > 0:
        strongest_feature_per_neuron = strongest_feature_per_neuron.merge(per_neuron_coefficients[["pre_id","feature","coefficient","fold_effect"]].rename(columns={"feature":"strongest_feature"}),on=["pre_id","strongest_feature"],how="left")
    strongest_feature_per_neuron = strongest_feature_per_neuron.merge(pre,on="pre_id",how="left")
    print(strongest_feature_per_neuron.head(100).to_string(index=False))

print("\n=== PART 17: WHAT PRESYNAPTIC CELL TYPES USE EACH RULE? ===")

if len(per_neuron_importance) > 0:
    neuron_feature_with_metadata = per_neuron_importance.merge(pre,on="pre_id",how="left")
    feature_by_pretype = neuron_feature_with_metadata.groupby(["pre_celltype","feature"],as_index=False).agg(median_importance=("delta_deviance","median"),mean_importance=("delta_deviance","mean"),n_neurons=("pre_id","nunique"))
    feature_by_pretype = feature_by_pretype.sort_values(["feature","median_importance"],ascending=[True,False])
    print(feature_by_pretype.to_string(index=False))

print("\n=== BROAD FEATURE ANALYSIS FINISHED ===")
# %% plot presynaptic rule comparison 

feature_by_pretype_melt = feature_by_pretype.melt(id_vars=["pre_celltype","feature", "n_neurons"],value_vars=["mean_importance","median_importance"],var_name="statistic",value_name="importance")

fig, ax = plt.subplots(1,2,figsize=(11,6))

sns.barplot(data=feature_by_pretype_melt[feature_by_pretype_melt["statistic"]=="mean_importance"],x="importance",y="pre_celltype",hue="feature",ax=ax[0], palette="Set2")
ax[0].set_title("Mean importance by presynaptic cell type")
sns.barplot(data=feature_by_pretype_melt[feature_by_pretype_melt["statistic"]=="median_importance"],x="importance",y="pre_celltype",hue="feature",ax=ax[1], palette="Set2")
ax[1].set_title("Median importance by presynaptic cell type")
ax[0].legend_.remove()
ax[1].legend(title="Feature", bbox_to_anchor=(1, 1), loc='upper left')
plt.tight_layout()

# %%
print("\n=== PART 18: DEVIANCE EXPLAINED AND FEATURE COMPOSITION BY PRESYNAPTIC CELL TYPE ===")

deviance_summary = []
deviance_composition = []

for pre_celltype,sub in broad_df.groupby("pre_celltype"):
    if len(sub) < 100:
        continue
    if sub["pre_id"].nunique() < 3:
        continue
    usable_features = [x for x in model_features if sub[x].nunique(dropna=True) > 1]
    if len(usable_features) < 2:
        continue
    try:
        null_model = smf.glm(formula="observed_count ~ 1",data=sub,family=sm.families.NegativeBinomial(alpha=1.0),offset=np.log(sub["null_mean"])).fit()
        full_formula = "observed_count ~ " + " + ".join(usable_features)
        full_model = smf.glm(formula=full_formula,data=sub,family=sm.families.NegativeBinomial(alpha=1.0),offset=np.log(sub["null_mean"])).fit()
        null_deviance = null_model.deviance
        full_deviance = full_model.deviance
        explained_deviance = null_deviance - full_deviance
        pseudo_r2_deviance = 1 - (full_deviance / null_deviance)
        deviance_summary.append({"pre_celltype":pre_celltype,"n_pre":sub["pre_id"].nunique(),"n_pairs":len(sub),"null_deviance":null_deviance,"full_deviance":full_deviance,"explained_deviance":explained_deviance,"pseudo_r2_deviance":pseudo_r2_deviance})
        for feature in usable_features:
            reduced_features = [x for x in usable_features if x != feature]
            reduced_formula = "observed_count ~ " + " + ".join(reduced_features)
            reduced_model = smf.glm(formula=reduced_formula,data=sub,family=sm.families.NegativeBinomial(alpha=1.0),offset=np.log(sub["null_mean"])).fit()
            delta_deviance = reduced_model.deviance - full_deviance
            percent_explained_deviance = 100 * delta_deviance / explained_deviance if explained_deviance > 0 else np.nan
            deviance_composition.append({"pre_celltype":pre_celltype,"feature":feature,"delta_deviance":delta_deviance,"percent_of_explained_deviance":percent_explained_deviance})
    except Exception as e:
        print("Skipped",pre_celltype,"because:",str(e))

deviance_summary = pd.DataFrame(deviance_summary)
deviance_composition = pd.DataFrame(deviance_composition)

print("\nOverall deviance explained by presynaptic cell type:")
print(deviance_summary.sort_values("pseudo_r2_deviance",ascending=False).to_string(index=False))

print("\nFeature composition of explained deviance:")
print(deviance_composition.sort_values(["pre_celltype","percent_of_explained_deviance"],ascending=[True,False]).to_string(index=False))

print("\n=== PART 19: PLOT TOTAL DEVIANCE EXPLAINED ===")

if len(deviance_summary) > 0:
    plot_total = deviance_summary.sort_values("pseudo_r2_deviance")
    plt.figure(figsize=(8,max(5,len(plot_total) * 0.35)))
    plt.barh(plot_total["pre_celltype"],100 * plot_total["pseudo_r2_deviance"])
    plt.xlabel("Deviance explained (%)")
    plt.ylabel("Presynaptic cell type")
    plt.title("Total co-targeting deviance explained by biological features")
    plt.tight_layout()
    plt.show()

print("\n=== PART 20: PLOT FEATURE COMPOSITION ===")

if len(deviance_composition) > 0:
    composition_matrix = deviance_composition.pivot(index="pre_celltype",columns="feature",values="percent_of_explained_deviance").fillna(0)
    composition_matrix.plot(kind="bar",stacked=True,figsize=(12,6))
    plt.ylabel("% of explained deviance")
    plt.xlabel("Presynaptic cell type")
    plt.title("Composition of explained co-targeting deviance")
    plt.legend(title="Feature",bbox_to_anchor=(1.02,1),loc="upper left")
    plt.tight_layout()
    plt.show()

# %%
