# Configurable staged polyadic-assignment model

This package implements the proposed Approach 1:

1. Learn a presynaptic-site intensity over candidate locations on the detailed
   presynaptic skeleton, then select exactly the supplied number of sites.
2. Predict a Poisson mean for the polyadic degree of every site and decode
   integer degrees whose sum exactly equals the supplied number of target
   contacts.
3. Learn site-target contact scores, remove row/column effects already fixed by
   the supplied margins, and use the remaining interaction score to tilt a
   random target-token assignment. Row sums remain the site degrees and column
   sums remain the exact requested target counts.
4. Optionally add a small, clipped co-targeting nudge while sampling. By default
   it affects only the first copy of a target at a site and is averaged rather
   than summed over targets already present.

Repeated contacts from one site to the same target are supported. The corrected
connector table must contain positionally aligned `postsynaptic_to` and
`postsynaptic_to_node` lists.

The recommended settings are in `models/approach1/config2.json`. With no useful
site-target interaction information, the decoder reduces to a random shuffle
of target-contact tokens rather than arbitrary maximum-score packing.
`information_mix=0.25` makes the learned tilt deliberately weak.
`repeat_information_discount=0` prevents evidence for a target's first contact
from automatically encouraging its second or third copy.
`preserve_null_repeat_mass=true` samples each site's complete multiplicity
profile from the random null and then changes only target labels through
profile-preserving swaps. Geometry and co-targeting can therefore choose target
identities but cannot inflate self-co-targeting. The random null itself still
permits multiplicity.

## Required and optional inputs

### SUMMARY:

###### GEOMETRY 
| Backbones | Availability | Partner distance | Behaviour |
|---:|---:|---:|---|
| false | false | false | Complete no-geometry model |
| true | true | false | Geometry affects site placement and degree, but not partner identity |
| true | false | true | Sites are placed without target occupancy, but partners use local geometry |
| true | true | true | Full geometry model; recommended primary model |
| false | true | true | Both derived features become uniform/constant fallbacks, so effectively no spatial information |

##### presyn skeleton features
| Switch | Features created | Main role |
|---|---|---|
| `use_compartment` | Axon, dendrite, etc. | Site placement and degree; also interacts with target cell type |
| `use_path_distance` | Relative path position and its square | Proximal/distal site and degree preferences |
| `use_branch_features` | Child count, leaf status and branchpoint status | Topological site and degree preferences |
| `use_coordinates` | Relative x, y and z within the neuron | Absolute-orientation-like site preferences |
| `use_local_cable` | Distance from the preceding skeleton node | Local morphological sampling/cable information |


##### cell type and co targeting features 
| Switch | Meaning | Overlap |
|---|---|---|
| `use_target_celltype` | Learns where different target cell types tend to be contacted, using interactions with compartment, path position, leaf/branchpoint status and local cable | Can overlap with geometry if cell types occupy characteristic regions |
| `use_presynaptic_celltype` | Allows site-location and degree relationships to differ across presynaptic classes | Also enters a presynaptic type × compartment × target type interaction |
| `use_co_targeting` | Learns whether two distinct target identities tend to occur together at the same synapse | Uses target-cell-type pair scores as a fallback when an identity-pair score is unavailable |
| `use_hemisphere` | Allows site and degree models to learn hemisphere differences | Mostly a broad neuron-level covariate |

###

The detailed presynaptic skeleton, number of outgoing sites, target identities,
and count for each target are always required. Every other feature is controlled
by a boolean in `FeatureFlags`/`config2.json`.

If `use_target_backbones=false`, or a requested target has no cached backbone
and `missing_backbone_policy="uniform"`, that target contributes a constant
availability field everywhere. Geometry is therefore neutral rather than
silently imputed from another target. The exact target totals remain enforced.
Disabling only `use_target_backbones` tests this uniform-availability fallback;
also disabling the two availability/distance flags removes those constant fields
entirely for a strict no-geometry feature ablation.

Important feature switches include:

- `use_target_backbones`
- `use_target_availability`
- `use_target_distance_for_partner`
- `use_target_celltype`
- `use_presynaptic_celltype`
- `use_co_targeting`
- `use_target_identity_prior`
- `use_target_count_features`
- compartment, path-distance, topology, coordinate, and local-cable features

A feature set is part of the trained model. For a valid ablation comparison,
train a separate model for each feature configuration rather than disabling a
feature only after training.

## Partner-assignment controls

- `assignment_mode="stochastic"`: recommended null-anchored model.
- `assignment_mode="random_null"`: ignores all pair and co-targeting evidence;
  useful as the baseline.
- `assignment_mode="deterministic"`: retains the old maximum-score transport
  decoder for comparison and can still pack repeats aggressively.
- `information_mix`: strength of the learned score tilt; zero is the null.
- `assignment_temperature`: larger values weaken the learned tilt.
- `residualize_scores`: removes site-only and target-only effects that cannot be
  identified once exact row and column totals are fixed.
- `score_clip`: bounds standardized site-target interaction scores.
- `preserve_null_repeat_mass`: locks every site's null-sampled multiplicity
  profile. This is the strongest protection against information-driven repeat
  inflation and is recommended.
- `mcmc_steps_per_contact`: target-label swap attempts per contact after the
  null assignment. More steps mix identities more thoroughly but cost time.
- `repeat_information_discount`: controls how much first-contact placement
  evidence also rewards later copies of that identity. It does not change the
  repeat profile while the profile lock is enabled.
- `self_repeat_penalty`: optional sequential-decoder penalty; to use it, set
  `preserve_null_repeat_mass=false`.
- `co_targeting_weight`, `co_targeting_score_clip`, and
  `co_targeting_aggregation`: keep co-targeting bounded and prevent a long list
  of present targets from creating an unbounded summed bonus.
- `prediction_samples`: number of exact integer realizations returned.

The initial assignment is always made by randomly shuffling the requested
target-contact tokens into the slots defined by the predicted site degrees.
With the recommended profile lock, subsequent learned moves only exchange
target labels between sites when both sites retain the same multiplicity
pattern (for example, `[1, 1, 2]` remains `[1, 1, 2]`).



## Setup

Run commands from the repository root. SciPy is optional: without it the package
uses a chunked NumPy nearest-neighbour calculation and greedy evaluation
matching. The learned models and assignment decoders use NumPy directly.

```bash
python -m pip install -r models/approach1/requirements.txt
```

## 1. Build morphology caches

The raw morphology tables are too large to rescan once per neuron. Build a
compressed file for every skeleton in one streaming pass:

```bash
python -m models.approach1 build-cache \
  --config models/approach1/config2.json \
  --kind all
```

The builder checks that each skeleton is a contiguous block in the source CSV.
Existing cache files are retained unless `--overwrite` is passed.

## 2. Make grouped train/validation/test splits

Bilateral partners are kept in the same split:

```bash
python -m models.approach1 split \
  --config models/approach1/config2.json \
  --output models/approach1/splits.json \
  --train-fraction 0.70 \
  --validation-fraction 0.15 \
  --seed 42
```

OR:
python -m models.approach1.manually_generate_subsplits 
for subset 

## 3. Train

```bash
python -m models.approach1 train \
  --config models/approach1/config2.json \
  --splits models/approach1/splits.json \
  --output models/approach1/fitted/full.pkl
```

The CLI permits quick feature overrides. This trains a matched no-backbone
ablation while leaving every count constraint intact:

```bash
python -m models.approach1 train \
  --config models/approach1/config2.json \
  --splits models/approach1/splits.json \
  --output models/approach1/fitted/no_backbones.pkl \
  --disable use_target_backbones \
  --disable use_target_availability \
  --disable use_target_distance_for_partner
```

Other useful ablations are `--disable use_target_celltype` and
`--disable use_co_targeting`. Repeating `--enable`/`--disable` is supported.

For an initial smoke run, set `training.max_train_neurons` to a small value in a
copy of the config. This limit takes the first IDs in the training split and is
not suitable for final comparisons.

## 4. Predict skeleton 29

If `n_sites` and target counts are omitted, the command reads those required
inputs from the corrected connector table. It does not read the observed site
locations or degree histogram:

```bash
python -m models.approach1 predict \
  --model models/approach1/fitted/full.pkl \
  --skeleton-id 29 \
  --output-dir models/approach1/predictions/29
```

Outputs are `predicted_sites.csv`, `predicted_assignments.csv`,
`predicted_assignment_summary.csv`, and `diagnostics.json`. The assignments
file is sample 0, an exact valid realization. The summary reports presence
probability, mean multiplicity, and multiplicity standard deviation over
`partner.prediction_samples` realizations. Use `--prediction-samples 1` for a
quick single realization or override it with any positive number.

Use observed presynaptic locations as an oracle-site upper bound:

```bash
python -m models.approach1 predict \
  --model models/approach1/fitted/full.pkl \
  --skeleton-id 29 \
  --oracle-sites \
  --output-dir models/approach1/predictions/29_oracle_sites
```

Use the exact observed degree histogram as another upper-bound experiment:

```bash
python -m models.approach1 predict \
  --model models/approach1/fitted/full.pkl \
  --skeleton-id 29 \
  --degree-mode exact_histogram \
  --observed-degree-histogram \
  --output-dir models/approach1/predictions/29_exact_histogram
```

Or supply only a degree summary:

```bash
python -m models.approach1 predict \
  --model models/approach1/fitted/full.pkl \
  --skeleton-id 29 \
  --degree-mode supplied_summary \
  --degree-summary '{"minimum":1,"q1":2,"median":2,"q3":3,"maximum":6}' \
  --output-dir models/approach1/predictions/29_summary
```

Custom required inputs can be supplied with `--n-sites` and either a JSON
target-count object or a path to one:

```bash
--n-sites 89 --target-counts '{"8419602":9,"8509987":8}'
```

The target counts must sum to at least the number of sites.

## 5. Evaluate learning

```bash
python -m models.approach1 evaluate \
  --model models/approach1/fitted/full.pkl \
  --splits models/approach1/splits.json \
  --split validation \
  --output models/approach1/results/full_validation.json
```

Metrics include matched site distance, fractions within 1 and 2 µm, degree MAE,
degree exact-match fraction, partner-contact accuracy, and assignment cosine
similarity. Add `--oracle-sites` to separate site-generation error from degree
and partner-assignment error.

## Python API

```python
from models.approach1 import ModelConfig, PolyadicAssignmentModel
from models.approach1.data import DataRepository

config = ModelConfig.load("models/approach1/config2.json")
repository = DataRepository(config.paths)
model = PolyadicAssignmentModel(config)
model.fit(repository, train_ids, validation_ids)

prediction = model.predict(
    repository,
    skeleton_id=29,
    n_sites=89,
    target_counts={8419602: 9, 8509987: 8},
)

# Exact integer samples: shape (samples, sites, targets).
samples = prediction.assignment_samples

# Long-form uncertainty summary, convenient for concatenating in a DataFrame.
summary = prediction.assignment_summary
```

`PredictionResult.validate` checks exact site, degree, and target-count
invariants before any prediction is returned.

Saved version-1 models used the old deterministic decoder. Retrain them with
`config2.json`; version 2 refuses to load an old pickle so the two assignment
semantics cannot be confused.
