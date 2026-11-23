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
from models.approach2 import Approach2Config, run_attribution_experiment, run_attribution_experiment_shapley


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

# One JSON per neuron here, e.g. out/.../29.json, holding the whole
# AttributionResult. Reload with models.approach2.load_result(path) and access it
# exactly as in memory: result.inputs.observed_assignment, result.samples["null"],
# result.condition_summary, ... Set to None to keep everything in memory only.
OUTPUT_DIR = REPO_ROOT / "out/approach2/shapley_run_with_individual_prop_no_multiples_temp1_weights2"

# local params
#config.cotarget.source = "supplied"
#config.cotarget.supplied_scores = Path("data/cotargeting_rules/neuronwise_cotargeting.csv")
config.cotarget.count_multiples = False
config.sampler.prediction_samples = 200
config.sampler.temperature = 1.0
config.sampler.geometry_weight = 2.0
config.sampler.cotargeting_weight = 2.0

if RUN_PARALLEL:
    from models.approach2.parallel import run_experiments
    start_all = time.time()
    results = run_experiments(
        skids[:],
        config=config,
        repository=repository,
        n_jobs=N_JOBS,
        output_dir=OUTPUT_DIR,
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
            result = run_attribution_experiment_shapley(
                repository,
                skeleton_id=sk,
                config=config,
            )
        except Exception as ex:
            print(f"Error processing skeleton {sk}: {ex}")
            continue
        if OUTPUT_DIR is not None:
            result.to_json(
                OUTPUT_DIR / f"{sk}.json",
                include_samples=config.output.save_raw_samples,
            )
        print(f'Finished neuron # {e+1} in {time.time() - stime:.2f} s')

        