# Approach 2: fixed-site, fixed-degree assignment attribution

This package asks a narrower question than Approach 1: when presynaptic sites,
the degree of every site, target identities, and target totals are all fixed,
how much of the target-pair arrangement is explained by local target geometry
versus an individual neuron's co-targeting preferences?

It is a constrained simulation, not a trained site or degree model. For each
random replicate it constructs one matched starting assignment and uses exact
conditional two-site Gibbs updates to run four conditions:

| condition | target geometry | individual co-targeting |
|---|---:|---:|
| `null` | off | off |
| `geometry` | on | off |
| `cotargeting` | off | on |
| `combined` | on | on |

Every assignment has exactly the observed row sums (site degrees) and column
sums (target totals). By default it also keeps each site's observed multiplicity
profile while randomizing identities. A site with multiplicities `[1, 1, 2]`
therefore remains `[1, 1, 2]` in every condition. Geometry or co-targeting cannot
create additional self-repeats; they only change which identities occupy that
profile.

## Run one neuron

From the repository root:

```bash
python -m models.approach2 run \
  --config models/approach2/example_config.json \
  --skeleton-id 29 \
  --output-dir models/approach2/results/29
```

Useful temporary overrides are:

```bash
--samples 20
--geometry-weight 1.0
--cotargeting-weight 1.0
--no-geometry
--no-cotargeting
--cotarget-scores path/to/scores.csv
```

A supplied score file needs `target_a,target_b,score` columns. Pairs absent from
the file receive zero score.

The default standardized-score weights are intentionally one rather than a
small perturbation around the null. A value of zero recovers the null; values
such as `0.5`, `1`, `2`, and `5` are useful sensitivity settings. Every Gibbs
update still preserves the exact site degrees, target totals, and (when enabled)
multiplicity profiles.

### Two implementations of the same chain

`gibbs_target_swaps` runs the chain through a compiled kernel in
`sampler_numba.py`; `gibbs_target_swaps_nonumba` is the interpreted reference.
They take the same arguments and return the same assignment for the same seed,
so either can be substituted for the other — the reference is the one to read
and to debug against, and `test_numba_matches_the_python_reference` pins them
together across the option grid. The compiled path is roughly forty times faster
per Gibbs call, and falls back to the reference automatically when numba is not
installed. Setting `NUMBA_DISABLE_JIT=1` runs the kernel uncompiled, which is
slow but steps through in a debugger.

## Individual co-targeting propensity

The default `source="observed_neuron"` computes the number of observed sites
containing every distinct target pair and compares it with the mean over
`cotarget.n_perm_expectation` independent margin-preserving permutations:

```text
log((observed + smoothing) / (permutation expectation + smoothing))
```

Each permutation shuffles labelled target tokens into the fixed site slots, so
the site degrees and target totals are exact and only the pairing is random.
The draws come from their own RNG stream and never read the simulated
conditions, so the expectation carries no MCMC autocorrelation and its count is
tuned separately from `sampler.prediction_samples` — a low-variance denominator
wants many cheap draws, the conditions want few expensive ones.

Scores are shrunk towards zero when the observed and expected support is small,
then standardized and clipped. Because the expectation is geometry-free, spatial
co-localization is credited to the preference term here; the geometry condition
in the factorial design is what separates the two.

This same-neuron option is descriptive: it uses the observed assignment to
estimate the preference being tested. For predictive evaluation use supplied
scores estimated from held-out sites, a contralateral partner, or other neurons
of the same class.

## Outputs

- `fixed_sites.csv`: exact sites and degrees.
- `targets.csv`: exact target identities/totals and backbone availability.
- `observed_assignments.csv`: reference assignment.
- `assignments_<condition>.csv`: representative exact assignment.
- `assignment_summary_<condition>.csv`: presence probabilities and multiplicity
  summaries over simulations.
- `geometry_scores.csv` and `cotarget_scores.csv`: model inputs.
- `condition_summary.csv`: agreement with observed assignments, pair similarity,
  repeat statistics, and assigned geometry.
- `pair_attribution.csv`: pair-level null, geometry, preference, and combined
  means plus factorial and Shapley contributions.
- `assignment_samples.npz`: all exact integer samples when enabled.

For each target pair, the Shapley contributions are:

```text
geometry = 0.5 * ((geometry - null) + (combined - cotargeting))
preference = 0.5 * ((cotargeting - null) + (combined - geometry))
```

They add exactly to `combined - null`; the nonlinear interaction is shared
equally between geometry and preference.

## Python API

```python
from models.approach1.data import DataRepository
from models.approach2 import Approach2Config, run_attribution_experiment

config = Approach2Config.load("models/approach2/example_config.json")
repository = DataRepository(config.paths)

result = run_attribution_experiment(repository, skeleton_id=29, config=config)

# One row per condition, convenient for combining across neurons.
condition_df = result.condition_summary

# One row per target pair with geometry/preference attribution.
pair_df = result.pair_attribution

# Exact arrays with shape (samples, sites, targets).
combined_samples = result.samples["combined"]
```

Looping over a test set only requires concatenating `condition_summary` and
`pair_attribution` after adding the corresponding skeleton ID.

### Saving results

Two formats, for two different jobs:

```python
# One JSON file holding the whole result, for reloading into Python later.
result.to_json("out/29.json")            # "out/29.json.gz" compresses on the way out

from models.approach2 import load_result
result = load_result("out/29.json")      # same object, same attribute access
result.inputs.observed_assignment, result.samples["null"], result.condition_summary

# A directory of CSVs plus diagnostics.json and assignment_samples.npz,
# for inspecting or loading somewhere other than Python.
result.save("out/29/")
```

`to_json` writes every field, so `load_result` returns a real `AttributionResult`
with `FixedNeuronInputs` and `GeometryResult` inside it — methods such as
`validate()` and `save()` work on the reloaded object. Pass
`include_samples=False` to omit the raw draws, which otherwise dominate the file
size; each condition then reads back as `None`.

`run_experiments(..., output_dir="out")` writes one such JSON per neuron,
`out/<skeleton_id>.json`, from inside the workers.

## Multiplicity choices

Recommended identity-only attribution:

```json
"multiplicity_profile_source": "observed",
"preserve_multiplicity_profile": true
```

To avoid using the observed repeat profile, use `"random_null"`. Each replicate
then draws a random token-shuffle profile, still shared across all four matched
conditions. Setting `preserve_multiplicity_profile=false` allows the conditions
to alter repeats, but this makes geometry/preference attribution and
self-co-targeting effects inseparable and should be treated as a separate
sensitivity analysis.
