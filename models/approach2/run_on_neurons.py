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
from models.validate_predictions_functions import get_target_co_occurrence
from models.padj_functions import get_assignment_matrix, get_co_targeting_matrix, compute_cosine
from models.approach1.data import DataRepository
from models.approach2 import Approach2Config, run_attribution_experiment


# get config and repository for sampler
config = Approach2Config.load(
    REPO_ROOT / "models/approach2/sover1_config.json"
)
repository = DataRepository(config.paths)

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

# decide which neurons to run on
skids = all_neurons
# skids = list(chain.from_iterable([celltype_df[celltype_df.name == ct_nam]['skids'].values[0] for ct_nam in ['LHNs','PNs',  'MBINs']]))
# skids = celltype_df[celltype_df.name == 'MBONs']['skids'].values[0]
# for sensories try just orns 
# skids = [sk for sk, nam in skid_to_name.items() if ' ORN' in nam]
print(f'Running on {len(skids)} neurons')

#%% Optional: same run, spread over worker processes
# run parallel or serial.

# n_jobs=1 stays in this process (identical to the loop); > 1 uses that many
# workers; -1 uses every core. The sampler seeds each draw from the skeleton id,
# so parallel results are identical to serial ones regardless of finish order.
RUN_PARALLEL = True
N_JOBS = -1

# local params 
config.cotarget.source = "supplied"
config.cotarget.supplied_scores = Path("data/cotargeting_rules/neuron_id_based_cotargeting.csv")

if RUN_PARALLEL:
    from models.approach2.parallel import run_experiments
    start_all = time.time()
    results = run_experiments(
        skids[:],
        config=config,
        repository=repository,
        n_jobs=N_JOBS,
    )
    print(f'Finished {len(results.conditions)} neurons in {time.time() - start_all:.2f} s')

    observed_assignments = results.observed_assignments
    distances_um = results.distances_um
    conditions = results.conditions
    targets = results.targets
    null_samples = results.null_samples
    geometry_samples = results.geometry_samples
    cotargeting_samples = results.cotargeting_samples
    combined_samples = results.combined_samples

else:
    conditions = {}
    null_samples = {}
    geometry_samples = {}
    cotargeting_samples = {}
    combined_samples = {}
    targets = {}
    observed_assignments = {}
    distances_um = {}


    for e, sk in enumerate(skids[:]):
        stime = time.time()
        try:
            result = run_attribution_experiment(
                repository,
                skeleton_id=sk,
                config=config,
            )
        except Exception as ex:
            print(f"Error processing skeleton {sk}: {ex}")
            continue
        observed_assignments[sk] = result.inputs.observed_assignment
        distances_um[sk] = result.geometry.distances_um
        # Four-row summary: null, geometry, co-targeting and combined.
        conditions[sk] = result.condition_summary

        # One row per target pair with the attribution results.
        # pair_df = result.pair_attribution
        targets[sk] = result.inputs.target_ids

        # Exact integer samples, each shaped: samples × sites × targets.
        null_samples[sk] = result.samples["null"]
        geometry_samples[sk] = result.samples["geometry"]
        cotargeting_samples[sk] = result.samples["cotargeting"]
        combined_samples[sk] = result.samples["combined"]
        print(f'Finished neuron # {e+1} in {time.time() - stime:.2f} s')

#%%
cdf = pd.concat(conditions, names=["skeleton_id"]).reset_index(level='skeleton_id').reset_index(drop=True)


# %%
cos_sims = {}
# padj_perms = {}
for e, sk in enumerate(skids[:]):
    if sk not in targets:
        print(f"Skipping skeleton {sk} because targets are missing.")
        continue
    padj_real, u_targets = get_co_targeting_matrix(con, sk)
    padj_real = pd.DataFrame(padj_real, index=u_targets, columns=u_targets)

    ordered_targets = targets[sk]
    padj_real_reindexed = padj_real.reindex(index=ordered_targets, columns=ordered_targets)
    real_flat = padj_real_reindexed.values[np.triu_indices_from(padj_real_reindexed, k=0)]

    cos_sims[sk]  = {}
    cos_sims[sk]['null'] = []
    cos_sims[sk]['geometry'] = []
    cos_sims[sk]['cotargeting'] = []
    cos_sims[sk]['combined'] = []
    padj_perms_null = []
    padj_perms_combined = []
    padj_perms_geometry = []
    padj_perms_cotargeting = []
    for nsamp, gsamp, cosamp, combsamp in zip(null_samples[sk], geometry_samples[sk], cotargeting_samples[sk], combined_samples[sk]):
        padj_perm_nsamp = get_target_co_occurrence(nsamp)
        padj_perm_gsamp = get_target_co_occurrence(gsamp)
        padj_perm_cosamp = get_target_co_occurrence(cosamp)
        padj_perm_combsamp = get_target_co_occurrence(combsamp)
        padj_perms_null.append(padj_perm_nsamp)
        padj_perms_combined.append(padj_perm_combsamp)
        padj_perms_geometry.append(padj_perm_gsamp)
        padj_perms_cotargeting.append(padj_perm_cosamp)
        # compute cosine similarity
        cos_sims[sk]['null'].append(compute_cosine(real_flat, padj_perm_nsamp))
        cos_sims[sk]['geometry'].append(compute_cosine(real_flat, padj_perm_gsamp))
        cos_sims[sk]['cotargeting'].append(compute_cosine(real_flat, padj_perm_cosamp))
        cos_sims[sk]['combined'].append(compute_cosine(real_flat, padj_perm_combsamp))
    mean_null = np.mean(np.stack(padj_perms_null), axis=0)
    mean_combined = np.mean(np.stack(padj_perms_combined), axis=0)
    mean_geometry = np.mean(np.stack(padj_perms_geometry), axis=0)
    mean_cotargeting = np.mean(np.stack(padj_perms_cotargeting), axis=0)


    # plot the padj matrices
    if e < 5: # if too many dont plot
        fig, ax = plt.subplots(1,5, figsize=(19, 4))
        sns.heatmap(padj_real_reindexed, ax=ax[0])
        sns.heatmap(mean_null, ax=ax[1])
        sns.heatmap(mean_geometry, ax=ax[2])
        sns.heatmap(mean_cotargeting, ax=ax[3])
        sns.heatmap(mean_combined, ax=ax[4])
        ax[0].set_title('Observed')
        ax[1].set_title('Null')
        ax[2].set_title('Geometry')
        ax[3].set_title('Co-targeting')
        ax[4].set_title('Combined')

        
cs = (
    pd.concat(
        {
            sk: pd.DataFrame(similarities)
            for sk, similarities in cos_sims.items()
        },
        names=["sk", "sample"]
    )
    .reset_index()
)

#%% plot the cosine similarity for each neuron and condition
if cs['sk'].nunique() <= 10:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    cs_long = cs.melt(id_vars=['sk'], value_vars=['null', 'geometry', 'cotargeting', 'combined'], var_name='condition', value_name='cosine_similarity')
    sns.barplot(data=cs_long, x='sk', y='cosine_similarity', hue='condition', palette='Set2', ax=ax)
else:
    cs_grouped = cs.groupby('sk').mean().reset_index()
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    cs_long = cs_grouped.melt(id_vars=['sk'], value_vars=['null', 'geometry', 'cotargeting', 'combined'], var_name='condition', value_name='cosine_similarity')
    sns.stripplot(data=cs_long, x='condition', y='cosine_similarity', palette='Set2', ax=ax)

cs.groupby('sk').mean()

#%% examine by hemisphere and other features 
def plot_cos_sim_grouped(cs_mean, id_vars=['sk', 'hemi', 'name', 'celltype'], value_vars=['null', 'geometry', 'cotargeting', 'combined'], hue='hemi', title=None):
    cs_mean_long = cs_mean.melt(id_vars=id_vars, value_vars=value_vars, var_name='condition', value_name='cosine_similarity')
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    sns.stripplot(data=cs_mean_long, x='condition', y='cosine_similarity', hue=hue, palette='Set2', ax=ax, zorder=1, alpha=0.3, dodge=True)
    sns.barplot(data=cs_mean_long, x='condition', y='cosine_similarity', hue=hue, palette='Set2', ax=ax, estimator=np.median, alpha=0.5, zorder=2)
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    if title is not None:
        ax.set_title(title)

cs_mean = cs.groupby('sk').mean().reset_index()
cs_mean['hemi'] = cs_mean['sk'].apply(lambda x: 'left' if x in left_neus else ('right' if x in right_neus else 'center'))
cs_mean['name'] = cs_mean['sk'].apply(lambda x: skid_to_name.get(x, 'Unknown'))
cs_mean['celltype'] = cs_mean['sk'].apply(lambda x: skid_to_celltype.get(x, 'Unknown'))

for hue in ['hemi', 'celltype']:
    plot_cos_sim_grouped(cs_mean, hue=hue)


#%% an attempt to categorise neurons by improvement per sampling method 

def get_improvements(cs_per_skid_df, baseline_cond='null', compare_conds=['geometry', 'cotargeting', 'combined']):
    for cond in compare_conds:
        cs_per_skid_df[f'delta_{cond}'] = cs_per_skid_df[cond] - cs_per_skid_df[baseline_cond]
        # fraction of remaining recovery to full cosine similarity of 1.0
        cs_per_skid_df[f'fractional_gain_{cond}'] = cs_per_skid_df[f'delta_{cond}'] / (1 - cs_per_skid_df[baseline_cond])
    return cs_per_skid_df

baseline_cond = 'null'
compare_conds = ['geometry', 'cotargeting', 'combined']
cs_mean = get_improvements(cs_mean, baseline_cond=baseline_cond, compare_conds=compare_conds)

for hue in ['hemi', 'celltype']:
    plot_cos_sim_grouped(cs_mean, hue=hue, value_vars=[f'delta_{cond}' for cond in compare_conds], title=f'improvement relative to null')
    plot_cos_sim_grouped(cs_mean, hue=hue, value_vars=[f'fractional_gain_{cond}' for cond in compare_conds], title=f'Fractional Gain relative to null')
#%%
def lowess(x, y, frac=0.1):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x); k = max(2, int(np.ceil(frac*n))); yhat = np.empty(n)
    for i, xi in enumerate(x):
        distances = np.abs(x-xi); idx = np.argpartition(distances, k-1)[:k]; d = distances[idx]; dmax = d.max()
        if dmax == 0: yhat[i] = np.mean(y[idx]); continue
        u = d/dmax; w = (1-u**3)**3; X = np.column_stack((np.ones(k), x[idx]-xi))
        WX = X*w[:, None]; beta = np.linalg.lstsq(X.T@WX, X.T@(w*y[idx]), rcond=None)[0]
        yhat[i] = beta[0]
    return yhat

plot_lowess = True
for prefix in ['delta_', 'fractional_gain_']:
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a, cond in zip(ax, compare_conds):
        a.scatter(cs_mean['null'], cs_mean[f'{prefix}{cond}'], alpha=0.6, s=20)
        a.axhline(0, color='black', linestyle='--')
        if plot_lowess:
            data = cs_mean[[f'{prefix}{cond}', 'null']].dropna()
            data = data.sort_values('null')
            x = data['null'].values
            y = data[f'{prefix}{cond}'].values
            yhat = lowess(x, y, frac=0.3)
            a.plot(x, yhat, color='purple', linewidth=2, label='Lowess')
        a.set_title(cond)
        a.set_xlabel('Null cosine similarity')
        if prefix == 'delta_':
            a.set_ylabel(f'Improvement over null')
        elif prefix == 'fractional_gain_':
            a.set_ylabel(f'Fractional gain over null')
        a.legend()
    fig.tight_layout()

# 3D plot of null, delta_geometry, delta_cotargeting, delta_combined

# import plotly.express as px

# fig = px.scatter_3d(
#     cs_mean,
#     x='null',
#     y='delta_geometry',
#     z='delta_cotargeting',
#     color='celltype',
#     hover_name='name',
#     hover_data=['sk', 'hemi', 'delta_combined'],
#     labels={
#         'null': 'Null cosine similarity',
#         'delta_geometry': 'Improvement over null (geometry)',
#         'delta_cotargeting': 'Improvement over null (cotargeting)',
#         'delta_combined': 'Improvement over null (combined)',
#     },
#     opacity=0.6,
# )
# fig.update_traces(marker=dict(size=4))
# fig.update_layout(width=1000, height=800, legend=dict(itemsizing='constant'))
# fig.show()
#%% heatmap improvements 

cs_mean_sorted = cs_mean.copy()
# do a hierarchical sorting 
cs_mean_sorted = cs_mean_sorted.sort_values(by=['null'] + [f'delta_{cond}' for cond in compare_conds], ascending=[True] + [False]*len(compare_conds))
H = cs_mean_sorted[['null'] + [f'delta_{cond}' for cond in compare_conds]].values
fig, ax = plt.subplots(1, 1, figsize=(4, 8))
sns.heatmap(H, ax=ax, cmap='coolwarm', center=0)
ax.set_xticklabels(['null'] + [f'delta_{cond}' for cond in compare_conds], rotation=90, ha='right')

#%% attempt to categorise 

# almost fully recovered by null 

cat = cs_mean[['sk', 'null', 'geometry', 'cotargeting', 'combined', 'hemi', 'name', 'celltype', 'delta_geometry','delta_cotargeting','delta_combined']].copy()
# drop those without data 
cat = cat.dropna(subset=['null', 'geometry', 'cotargeting', 'combined'])

cat['recovery_under_null'] = ['full' if x > 0.75 else 'good' if x > 0.5 else 'ok' if x > 0.25 else 'poor' for x in cat['null']]
cat['best_recovery'] = cat[['null', 'geometry', 'cotargeting', 'combined']].max(axis=1)
cat['recoverable'] = ['full' if x > 0.75 else 'good' if x > 0.5 else 'ok' if x > 0.25 else 'poor' for x in cat['best_recovery']]
cat['best_method'] = cat[['null', 'geometry', 'cotargeting', 'combined']].idxmax(axis=1)
cat['relative_gain_geometry'] = cat['delta_geometry'] / (cat['delta_geometry'] + cat['delta_cotargeting'])
cat['relative_gain_cotargeting'] = cat['delta_cotargeting'] / (cat['delta_geometry'] + cat['delta_cotargeting'])
cat['relative_additional_gain_combined'] = cat.apply(lambda row: (row['delta_combined']) / (row['delta_combined']+ max(row[['delta_geometry', 'delta_cotargeting']])) if (row['delta_combined']+ max(row[['delta_geometry', 'delta_cotargeting']])) != 0 else 0.5, axis=1)
cat['best_gain_over_null'] = cat['best_recovery'] - cat['null']
cat['substantial_gain_over_null'] = cat['best_gain_over_null'] > 0.1
cat['substantial_gain_method'] = cat.apply(lambda row: 'cotargeting' if row['relative_gain_cotargeting'] > 0.6 else 'geometry' if row['relative_gain_cotargeting'] < 0.4 else 'equal', axis=1)

cat['combined_substantially_better'] = cat['relative_additional_gain_combined'] > 0.6 

no_gain = ~cat['substantial_gain_over_null']
cat.loc[no_gain, 'substantial_gain_method'] = 'neither'
cat.loc[no_gain, 'combined_substantially_better'] = False

# can now think of different categories: 
cols_OI = ['recovery_under_null', 'substantial_gain_over_null', 'substantial_gain_method', 'combined_substantially_better']

grouped_counts = cat.groupby(cols_OI).size().reset_index(name='count')

import pandas as pd
import plotly.graph_objects as go

def plot_recovery_sankey(df, count_col="count"):
    """
    Expected columns:
      recovery_under_null
      substantial_gain_over_null
      substantial_gain_method
      combined_substantially_better
      count_col

    If substantial_gain_over_null == False:
      - substantial_gain_method is omitted from the path
      - combined_substantially_better is omitted from the path
    """

    df = df.copy()
    flows = {}

    def add_flow(a, b, n):
        flows[(a, b)] = flows.get((a, b), 0) + n

    for _, row in df.iterrows():
        recovery = f"Recovery: {row['recovery_under_null']}"
        gain = f"Gain: {row['substantial_gain_over_null']}"
        n = row[count_col]

        add_flow(recovery, gain, n)

        if bool(row["substantial_gain_over_null"]):
            method = f"Method: {row['substantial_gain_method']}"
            combined = f"Combined better: {row['combined_substantially_better']}"
            add_flow(gain, method, n)
            add_flow(method, combined, n)

    nodes = list(dict.fromkeys(x for edge in flows for x in edge))
    node_idx = {node: i for i, node in enumerate(nodes)}

    source = [node_idx[a] for a, _ in flows]
    target = [node_idx[b] for _, b in flows]
    values = list(flows.values())
    # Indexed by node position, so `source`/`target` (which hold indices, not
    # labels) can be used to look a colour up directly.
    node_colors_dict = {
        'Recovery: full': "rgba(70, 70, 70, 0.8)",
        'Recovery: good': "rgba(110, 110, 110, 0.8)",
        'Recovery: ok': "rgba(150, 150, 150, 0.8)",
        'Recovery: poor': "rgba(190, 190, 190, 0.8)",
        'Gain: True': "rgba(10, 133, 252, 0.8)",
        'Gain: False': "rgba(0, 200, 255, 0.8)",
        'Method: equal': "rgba(255, 140, 0, 0.8)",
        'Method: cotargeting': "rgba(255, 90, 0, 0.8)",
        'Method: geometry': "rgba(255, 190, 60, 0.8)",
        'Combined better: True': "rgba(120, 230, 100, 0.8)",
        'Combined better: False': "rgba(50, 205, 90, 0.8)",
    }
    node_colors = [node_colors_dict.get(node, "rgba(200, 200, 200, 0.8)") for node in nodes]
    link_colors = [node_colors[s].replace(", 0.8)", ", 0.35)") for s in source]

    fig = go.Figure(go.Sankey(
        node=dict(label=nodes, pad=18, thickness=18, align='left', color=node_colors),
        link=dict(source=source, target=target, value=values, color=link_colors),
        arrangement='fixed',
    ))

    fig.update_layout(title="Recovery outcomes and gains over null", font_size=16, width=900, height=600, margin=dict(l=50, r=300, t=50, b=50))
    fig.show()

    return fig, nodes, node_idx

fig, nodes, node_idx = plot_recovery_sankey(grouped_counts, count_col="count")


# %% check quality of sampler results in terms of geometric adherence  

# def get_observed_distances(obs_ass, distances_um_i):
#     # get any double instances twice 
#     obs_distances = (obs_ass > 0) * distances_um_i
#     obs_distances_chain = obs_distances[obs_distances > 0].flatten()
#     n_obs = obs_ass[obs_ass > 0].flatten()
#     obs_distances_repeated = np.repeat(obs_distances_chain, n_obs)
#     return obs_distances_repeated

def get_observed_distances(obs_ass, distances_um):
    """Distances of assigned contacts, each repeated by its multiplicity."""
    # One mask for both arrays. A distance of exactly 0.0 is real -- a site sitting
    # on its target's backbone, as an autapse does -- so it must not be filtered
    # out as if it meant "unassigned". Only unmeasurable distances are dropped.
    assigned = (obs_ass > 0) & np.isfinite(distances_um)
    return np.repeat(distances_um[assigned], obs_ass[assigned])

def evaluate_geometry_adherence(observed_assignments_i, distances_um_i, sampled_i):
    # get real 
    dist_real = get_observed_distances(observed_assignments_i, distances_um_i)
    # get distances from each geometry sample 
    dist_geo_samples = [get_observed_distances(geo_ass, distances_um_i) for geo_ass in sampled_i]
    frac_within_obs = [len(d[d <= np.max(dist_real)]) / len(d) for d in dist_geo_samples]
    out = {'mean_real': np.mean(dist_real), 'median_real': np.median(dist_real), 'max_real': np.max(dist_real),
           'mean_sampled': np.mean([np.mean(d) for d in dist_geo_samples]), 'median_sampled': np.median([np.median(d) for d in dist_geo_samples]),
           'max_sampled': np.max([np.max(d) for d in dist_geo_samples]),
           'frac_within_obs': frac_within_obs,
           'frac_within_median': np.median(frac_within_obs), 'frac_within_mean': np.mean(frac_within_obs),
           'frac_within_min': np.min(frac_within_obs)}
    return out

def evaluate_geometry_adherence_all(observed_assignments, distances_um, samples):
    results = {}
    for sk in observed_assignments.keys():
        if sk not in samples:
            print(f"Skipping skeleton {sk} because geometry samples are missing.")
            continue
        results[sk] = evaluate_geometry_adherence(observed_assignments[sk], distances_um[sk], samples[sk])
    return pd.DataFrame(results).T

geo_adherence_geo = evaluate_geometry_adherence_all(observed_assignments, distances_um, geometry_samples)
geo_adherence_combined = evaluate_geometry_adherence_all(observed_assignments, distances_um, combined_samples)
geo_adherence_cotargeting = evaluate_geometry_adherence_all(observed_assignments, distances_um, cotargeting_samples)
geo_adherence_null = evaluate_geometry_adherence_all(observed_assignments, distances_um, null_samples)

#%%
fig, ax = plt.subplots(1, 2, figsize=(8, 5))
size =15
alpha=0.7

ax[0].scatter(geo_adherence_geo['median_real'], geo_adherence_geo['median_sampled'], label='geometry', alpha=alpha, s=size)
ax[1].scatter(geo_adherence_geo['max_real'], geo_adherence_geo['max_sampled'], label='geometry', alpha=alpha, s=size)
ax[0].set_xlabel('Median observed distance (um)')
ax[0].set_ylabel('Median sampled distance (um)')
ax[1].set_xlabel('Max observed distance (um)')
ax[1].set_ylabel('Max sampled distance (um)')
ax0_min, ax0_max = min(geo_adherence_geo['median_real'].min(), geo_adherence_geo['median_sampled'].min()), max(geo_adherence_geo['median_real'].max(), geo_adherence_geo['median_sampled'].max())
ax1_min, ax1_max = min(geo_adherence_geo['max_real'].min(), geo_adherence_geo['max_sampled'].min()), max(geo_adherence_geo['max_real'].max(), geo_adherence_geo['max_sampled'].max())
ax[0].set_xlim(ax0_min, ax0_max)
ax[0].set_ylim(ax0_min, ax0_max)
ax[1].set_xlim(ax1_min, ax1_max)
ax[1].set_ylim(ax1_min, ax1_max)

# plot histograms instead 

fig, ax = plt.subplots(1, 2, figsize=(8, 5))
alpha = 0.5
color_real = 'blue'
color_sampled = 'orange'
bins_median = np.linspace(min(geo_adherence_geo['median_real'].min(), geo_adherence_geo['median_sampled'].min()), max(geo_adherence_geo['median_real'].max(), geo_adherence_geo['median_sampled'].max()), 40)
bins_max = np.linspace(min(geo_adherence_geo['max_real'].min(), geo_adherence_geo['max_sampled'].min()), max(geo_adherence_geo['max_real'].max(), geo_adherence_geo['max_sampled'].max()), 40)
ax[0].hist(geo_adherence_geo['median_real'], label='observed', color=color_real, alpha=alpha, bins=bins_median)
ax[0].hist(geo_adherence_geo['median_sampled'], label='sampled', color= color_sampled, alpha=alpha, bins=bins_median)
ax[1].hist(geo_adherence_geo['max_real'], label='observed', color=color_real, alpha=alpha, bins=bins_max)
ax[1].hist(geo_adherence_geo['max_sampled'], label='sampled', color=color_sampled, alpha=alpha, bins=bins_max)

ax[0].set_xlabel('Median distance (um)')
ax[0].set_ylabel('Number of neurons')
ax[1].set_xlabel('Max distance (um)')
ax[1].set_ylabel('Number of neurons') 
ax[0].legend()
ax[1].legend()
fig.tight_layout()

#%% 
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
col = 'frac_within_median'
data_melted = pd.concat([geo_adherence_geo[col], geo_adherence_combined[col], geo_adherence_cotargeting[col], geo_adherence_null[col]], axis=1)
data_melted.columns = ['geometry', 'combined', 'cotargeting', 'null']
data_melted = data_melted.melt(var_name='condition', value_name=col)
sns.violinplot(data=data_melted, x='condition', y=col, ax=ax)

# %%
