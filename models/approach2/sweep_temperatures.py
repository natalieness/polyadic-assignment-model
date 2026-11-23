"""Sweep sampler temperature through selected effective weight/temperature ratios.

The experiment keeps the settings currently used by ``run_and_save.py`` fixed except
``sampler.temperature``. Because geometry and co-targeting have the same weight, one
requested ratio ``r`` is implemented as::

    temperature = geometry_weight / r = cotargeting_weight / r

The geometry-conditioned experiment is used: co-targeting propensity is estimated
relative to the geometry draws at each temperature.  The co-targeting-only condition
is retained in the diagnostic DataFrames, but is omitted from plots unless explicitly
requested because those residual scores are only meaningful together with geometry.

Edit the variables in the ``Interactive settings`` cell and run the file. The
defaults mirror ``run_and_save.py``: all named neurons, 100 prediction samples,
all available CPUs, and ``count_multiples=False``. The underlying function can
also be called directly in a notebook::

    from models.approach2.sweep_temperatures import run_temperature_sweep

    sweep_results = run_temperature_sweep(
        [7092417, 11821320, 3774568, 4154421],
        samples=100,
        count_multiples=False,
        n_jobs=-1,
    )
    sample_diagnostics = sweep_results["sample_diagnostics"]
    comparison_diagnostics = sweep_results["comparison_diagnostics"]

No files are written. When run with ``%run -i`` or from an IDE, the following
objects remain available for interactive analysis:

* ``sample_diagnostics``: one row per sampled assignment, plus observed rows.
* ``summary_diagnostics``: means, SDs, and cosine to the mean pair pattern.
* ``comparison_diagnostics``: explicit combined-minus-geometry reversals.
* ``aggregate_diagnostics``: across-neuron medians and interquartile ranges.
* ``error_diagnostics``, ``run_metadata``, and ``diagnostic_figures``.

The aggregate and per-neuron figures are displayed with ``plt.show()``.
"""

from __future__ import annotations

import copy
import multiprocessing
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

_RUNTIME_CACHE = Path(tempfile.gettempdir()) / "polyadic_assignment_model"
os.environ.setdefault("MPLCONFIGDIR", str(_RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_RUNTIME_CACHE / "cache"))
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.approach1.data import DataRepository
from models.approach2 import Approach2Config, run_attribution_experiment_shapley
from models.approach2.propensity import pair_count_matrix
from models.approach2.sampler import _row_objective


DEFAULT_RATIOS = (0.25, 0.5, 1.0, 2.0, 5.0, 20.0)
PLOTTED_CONDITIONS = ("null", "geometry", "combined")
CONDITION_LABELS = {
    "observed": "observed",
    "null": "random",
    "geometry": "geometry",
    "cotargeting": "conditional propensity only (diagnostic)",
    "combined": "geometry + conditional propensity",
}
CONDITION_COLORS = {
    "observed": "#111111",
    "null": "#C5A000",
    "geometry": "#2E9B41",
    "cotargeting": "#8B6BB1",
    "combined": "#D84A32",
}

#%% Interactive settings
# These defaults match the current local settings in run_and_save.py.
CONFIG_PATH = REPO_ROOT / "models/approach2/sover1_config.json"

neuron_names = pd.read_csv(REPO_ROOT / "data/skid_to_name_wMB.csv")
neuron_names["skid"] = neuron_names["skid"].astype(int)
all_neurons = neuron_names["skid"].to_numpy()

SKIDS = all_neurons
# For a quicker targeted run, for example:
# SKIDS = [7092417, 11821320, 3774568, 4154421]

EFFECTIVE_RATIOS = DEFAULT_RATIOS
PREDICTION_SAMPLES = 10
COUNT_MULTIPLES = False
N_JOBS = -1

INCLUDE_COTARGETING_ONLY = False
# Keep this False for the all-neuron run; set True for a small selected SKIDS list.
PER_NEURON_PLOTS = False
SHOW_PLOTS = True


_WORKER_BASE_CONFIG: Approach2Config | None = None
_WORKER_REPOSITORY: DataRepository | None = None
_WORKER_SAMPLES: int | None = None


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _validate_ratios(ratios: Iterable[float]) -> list[float]:
    values = sorted(set(float(value) for value in ratios))
    if not values or any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Every effective ratio must be finite and positive")
    return values


def _temperature_for_ratio(config: Approach2Config, ratio: float) -> float:
    geometry_weight = float(config.sampler.geometry_weight)
    cotargeting_weight = float(config.sampler.cotargeting_weight)
    if not np.isclose(geometry_weight, cotargeting_weight):
        raise ValueError(
            "A single common effective ratio requires equal geometry and co-targeting "
            f"weights; the config has {geometry_weight} and {cotargeting_weight}."
        )
    if geometry_weight <= 0:
        raise ValueError("The common geometry/co-targeting weight must be positive")
    return geometry_weight / float(ratio)


def _load_neuron_names() -> dict[int, str]:
    table = pd.read_csv(REPO_ROOT / "data/skid_to_name_wMB.csv")
    return dict(zip(table["skid"].astype(int), table["name"].astype(str)))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float).ravel()
    right = np.asarray(right, dtype=float).ravel()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator else float("nan")


def _pair_vector(
    assignment: np.ndarray,
    include_self_repeats: bool,
    count_multiples: bool,
) -> np.ndarray:
    matrix = pair_count_matrix(
        assignment,
        include_self_repeats=include_self_repeats,
        count_multiples=count_multiples,
    )
    triangle = np.triu_indices_from(matrix, k=0)
    return np.asarray(matrix[triangle], dtype=float)


def _energy_components(
    assignment: np.ndarray,
    geometry_scores: np.ndarray,
    cotarget_scores: np.ndarray,
    config: Approach2Config,
) -> tuple[float, float]:
    """Return conventional energies, ``E = -objective``, for both score terms."""
    common = dict(
        geometry_repeat_discount=config.geometry.repeat_discount,
        cotarget_aggregation=config.cotarget.aggregation,
        cotarget_site_clip=config.cotarget.site_score_clip,
        self_repeat_penalty=0.0,
        cotarget_include_self=config.cotarget.include_self_repeats,
    )
    geometry_energy = 0.0
    cotarget_energy = 0.0
    for row, geometry_row in zip(assignment, geometry_scores):
        geometry_energy -= _row_objective(
            row,
            geometry_row,
            cotarget_scores,
            geometry_weight=1.0,
            cotargeting_weight=0.0,
            **common,
        )
        cotarget_energy -= _row_objective(
            row,
            geometry_row,
            cotarget_scores,
            geometry_weight=0.0,
            cotargeting_weight=1.0,
            **common,
        )
    return float(geometry_energy), float(cotarget_energy)


def _diagnose_result(
    result,
    config: Approach2Config,
    ratio: float,
    neuron_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observed = result.inputs.observed_assignment
    geometry_scores = result.geometry.scores
    cotarget_scores = result.cotarget_scores
    include_self = bool(config.cotarget.include_self_repeats)
    count_multiples = bool(config.cotarget.count_multiples)
    observed_pair_vector = _pair_vector(observed, include_self, count_multiples)
    observed_geometry, observed_cotarget = _energy_components(
        observed, geometry_scores, cotarget_scores, config
    )
    n_sites = int(len(result.inputs.site_degrees))
    n_contacts = int(result.inputs.site_degrees.sum())
    n_targets = int(len(result.inputs.target_ids))
    temperature = float(config.sampler.temperature)
    geometry_weight = float(config.sampler.geometry_weight)
    cotargeting_weight = float(config.sampler.cotargeting_weight)

    common = {
        "skeleton_id": int(result.inputs.skeleton_id),
        "neuron_name": neuron_name,
        "effective_ratio": float(ratio),
        "temperature": temperature,
        "geometry_weight": geometry_weight,
        "cotargeting_weight": cotargeting_weight,
        "geometry_beta": geometry_weight / temperature,
        "cotargeting_beta": cotargeting_weight / temperature,
        "n_sites": n_sites,
        "n_contacts": n_contacts,
        "n_targets": n_targets,
        "count_multiples": count_multiples,
        "observed_geometry_energy": observed_geometry,
        "observed_cotargeting_energy": observed_cotarget,
    }

    sample_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    assignments_by_condition: dict[str, np.ndarray] = {
        "observed": observed[None, :, :],
        **result.samples,
    }
    for condition, assignments in assignments_by_condition.items():
        pair_vectors = []
        for sample_index, assignment in enumerate(assignments):
            geometry_energy, cotarget_energy = _energy_components(
                assignment, geometry_scores, cotarget_scores, config
            )
            weighted_energy = (
                geometry_weight * geometry_energy
                + cotargeting_weight * cotarget_energy
            )
            effective_energy = weighted_energy / temperature
            pair_vector = _pair_vector(assignment, include_self, count_multiples)
            pair_vectors.append(pair_vector)
            sample_rows.append({
                **common,
                "condition": condition,
                "condition_role": (
                    "diagnostic_only" if condition == "cotargeting" else "model"
                ),
                "sample_index": -1 if condition == "observed" else sample_index,
                "geometry_energy": geometry_energy,
                "cotargeting_energy": cotarget_energy,
                "weighted_joint_energy": weighted_energy,
                "effective_joint_energy": effective_energy,
                "geometry_energy_per_site": geometry_energy / n_sites,
                "cotargeting_energy_per_site": cotarget_energy / n_sites,
                "effective_joint_energy_per_site": effective_energy / n_sites,
                "geometry_energy_gap_per_site": (
                    geometry_energy - observed_geometry
                ) / n_sites,
                "cotargeting_energy_gap_per_site": (
                    cotarget_energy - observed_cotarget
                ) / n_sites,
                "effective_joint_energy_gap_per_site": (
                    weighted_energy
                    - geometry_weight * observed_geometry
                    - cotargeting_weight * observed_cotarget
                ) / (temperature * n_sites),
                "pair_cosine": _cosine(pair_vector, observed_pair_vector),
            })

        pair_vectors_array = np.asarray(pair_vectors, dtype=float)
        mean_pair_vector = np.mean(pair_vectors_array, axis=0)
        summary_rows.append({
            **common,
            "condition": condition,
            "condition_role": (
                "diagnostic_only" if condition == "cotargeting" else "model"
            ),
            "mean_pattern_pair_cosine": _cosine(
                mean_pair_vector, observed_pair_vector
            ),
        })
    return sample_rows, summary_rows


def _init_worker(
    config_path: str,
    samples: int | None,
    count_multiples: bool | None,
) -> None:
    global _WORKER_BASE_CONFIG, _WORKER_REPOSITORY, _WORKER_SAMPLES
    os.chdir(REPO_ROOT)
    _WORKER_BASE_CONFIG = Approach2Config.load(config_path)
    if count_multiples is not None:
        _WORKER_BASE_CONFIG.cotarget.count_multiples = bool(count_multiples)
    _WORKER_REPOSITORY = DataRepository(_WORKER_BASE_CONFIG.paths)
    _WORKER_SAMPLES = samples


def _run_one(task: tuple[float, int, str]) -> dict[str, Any]:
    ratio, skeleton_id, neuron_name = task
    if _WORKER_BASE_CONFIG is None or _WORKER_REPOSITORY is None:
        raise RuntimeError("Temperature-sweep worker was not initialized")
    config = copy.deepcopy(_WORKER_BASE_CONFIG)
    config.sampler.temperature = _temperature_for_ratio(config, ratio)
    if _WORKER_SAMPLES is not None:
        config.sampler.prediction_samples = int(_WORKER_SAMPLES)
    try:
        result = run_attribution_experiment_shapley(
            _WORKER_REPOSITORY,
            skeleton_id=int(skeleton_id),
            config=config,
        )
        samples, summaries = _diagnose_result(
            result, config, ratio, neuron_name
        )
        return {
            "ratio": ratio,
            "skeleton_id": skeleton_id,
            "sample_rows": samples,
            "summary_rows": summaries,
            "error": None,
        }
    except Exception as error:  # noqa: BLE001 - preserve the rest of the sweep
        return {
            "ratio": ratio,
            "skeleton_id": skeleton_id,
            "sample_rows": [],
            "summary_rows": [],
            "error": f"{type(error).__name__}: {error}",
        }


def _resolve_workers(n_jobs: int) -> int:
    if n_jobs < 0:
        return max(1, (os.cpu_count() or 1) + 1 + n_jobs)
    return max(1, n_jobs)


def _run_tasks(
    tasks: list[tuple[float, int, str]],
    config_path: Path,
    samples: int | None,
    count_multiples: bool | None,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    started = time.time()
    workers = _resolve_workers(n_jobs)

    if workers == 1:
        _init_worker(str(config_path), samples, count_multiples)
        results = (_run_one(task) for task in tasks)
        for done, result in enumerate(results, start=1):
            sample_rows.extend(result["sample_rows"])
            summary_rows.extend(result["summary_rows"])
            if result["error"]:
                errors.append({
                    "skeleton_id": result["skeleton_id"],
                    "effective_ratio": result["ratio"],
                    "error": result["error"],
                })
            print(
                f"Finished {done}/{len(tasks)}: skeleton {result['skeleton_id']}, "
                f"ratio {result['ratio']:g}, {time.time() - started:.1f} s elapsed",
                flush=True,
            )
    else:
        methods = multiprocessing.get_all_start_methods()
        context = multiprocessing.get_context("fork" if "fork" in methods else "spawn")
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=_init_worker,
                initargs=(str(config_path), samples, count_multiples),
            ) as executor:
                future_to_task = {
                    executor.submit(_run_one, task): task for task in tasks
                }
                for done, future in enumerate(as_completed(future_to_task), start=1):
                    result = future.result()
                    sample_rows.extend(result["sample_rows"])
                    summary_rows.extend(result["summary_rows"])
                    if result["error"]:
                        errors.append({
                            "skeleton_id": result["skeleton_id"],
                            "effective_ratio": result["ratio"],
                            "error": result["error"],
                        })
                    print(
                        f"Finished {done}/{len(tasks)}: skeleton "
                        f"{result['skeleton_id']}, ratio {result['ratio']:g}, "
                        f"{time.time() - started:.1f} s elapsed",
                        flush=True,
                    )
        except OSError as error:
            print(
                f"Parallel workers unavailable ({error}); falling back to serial.",
                flush=True,
            )
            return _run_tasks(
                tasks,
                config_path,
                samples,
                count_multiples,
                n_jobs=1,
            )
    return (
        pd.DataFrame.from_records(sample_rows),
        pd.DataFrame.from_records(summary_rows),
        pd.DataFrame.from_records(errors),
    )


def _summarize_samples(
    samples: pd.DataFrame,
    pattern_summaries: pd.DataFrame,
) -> pd.DataFrame:
    group_columns = [
        "skeleton_id",
        "neuron_name",
        "effective_ratio",
        "temperature",
        "geometry_beta",
        "cotargeting_beta",
        "n_sites",
        "n_contacts",
        "n_targets",
        "condition",
        "condition_role",
    ]
    value_columns = [
        "geometry_energy_per_site",
        "cotargeting_energy_per_site",
        "effective_joint_energy_per_site",
        "geometry_energy_gap_per_site",
        "cotargeting_energy_gap_per_site",
        "effective_joint_energy_gap_per_site",
        "pair_cosine",
    ]
    summary = samples.groupby(group_columns, as_index=False)[value_columns].agg(
        ["mean", "std"]
    )
    summary.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary = summary.rename(columns={
        column: column.removesuffix("_mean")
        for column in group_columns
        if f"{column}_mean" in summary.columns
    })
    summary = summary.merge(
        pattern_summaries[
            ["skeleton_id", "effective_ratio", "condition", "mean_pattern_pair_cosine"]
        ],
        on=["skeleton_id", "effective_ratio", "condition"],
        how="left",
        validate="one_to_one",
    )
    return summary.sort_values(
        ["skeleton_id", "effective_ratio", "condition"]
    ).reset_index(drop=True)


def _comparison_table(
    samples: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Put the hierarchy and its reversal diagnostics on one row per run."""
    keys = [
        "skeleton_id",
        "neuron_name",
        "effective_ratio",
        "temperature",
        "geometry_beta",
        "cotargeting_beta",
        "n_sites",
        "n_contacts",
        "n_targets",
    ]
    result = summary[keys].drop_duplicates().copy()
    metrics = [
        "mean_pattern_pair_cosine",
        "geometry_energy_gap_per_site_mean",
        "cotargeting_energy_gap_per_site_mean",
        "effective_joint_energy_gap_per_site_mean",
    ]
    for condition in ("null", "geometry", "cotargeting", "combined"):
        condition_table = summary.loc[
            summary["condition"] == condition,
            ["skeleton_id", "effective_ratio", *metrics],
        ].rename(columns={metric: f"{condition}_{metric}" for metric in metrics})
        result = result.merge(
            condition_table,
            on=["skeleton_id", "effective_ratio"],
            how="left",
            validate="one_to_one",
        )

    result["combined_minus_geometry_mean_pattern_cosine"] = (
        result["combined_mean_pattern_pair_cosine"]
        - result["geometry_mean_pattern_pair_cosine"]
    )
    result["combined_minus_cotargeting_mean_pattern_cosine"] = (
        result["combined_mean_pattern_pair_cosine"]
        - result["cotargeting_mean_pattern_pair_cosine"]
    )

    paired = samples.loc[
        samples["sample_index"] >= 0,
        [
            "skeleton_id",
            "effective_ratio",
            "sample_index",
            "condition",
            "pair_cosine",
        ],
    ].pivot(
        index=["skeleton_id", "effective_ratio", "sample_index"],
        columns="condition",
        values="pair_cosine",
    )
    paired["combined_minus_geometry"] = paired["combined"] - paired["geometry"]
    paired["combined_minus_cotargeting"] = (
        paired["combined"] - paired["cotargeting"]
    )
    paired = paired.reset_index()
    paired_summary = paired.groupby(
        ["skeleton_id", "effective_ratio"], as_index=False
    ).agg(
        combined_minus_geometry_paired_mean=("combined_minus_geometry", "mean"),
        combined_minus_geometry_paired_sd=("combined_minus_geometry", "std"),
        combined_minus_geometry_negative_fraction=(
            "combined_minus_geometry",
            lambda values: float(np.mean(values < 0)),
        ),
        combined_minus_cotargeting_paired_mean=(
            "combined_minus_cotargeting",
            "mean",
        ),
        combined_minus_cotargeting_paired_sd=(
            "combined_minus_cotargeting",
            "std",
        ),
        combined_minus_cotargeting_negative_fraction=(
            "combined_minus_cotargeting",
            lambda values: float(np.mean(values < 0)),
        ),
    )
    result = result.merge(
        paired_summary,
        on=["skeleton_id", "effective_ratio"],
        how="left",
        validate="one_to_one",
    )
    result["combined_worse_than_geometry"] = (
        result["combined_minus_geometry_mean_pattern_cosine"] < 0
    )
    result["combined_worse_than_cotargeting_diagnostic"] = (
        result["combined_minus_cotargeting_mean_pattern_cosine"] < 0
    )
    return result.sort_values(["skeleton_id", "effective_ratio"]).reset_index(
        drop=True
    )


def _plot_lines(
    axis,
    table: pd.DataFrame,
    y_column: str,
    y_sd_column: str | None,
    conditions: tuple[str, ...],
) -> None:
    for condition in conditions:
        condition_table = table[table["condition"] == condition].sort_values(
            "effective_ratio"
        )
        if condition_table.empty:
            continue
        x = condition_table["effective_ratio"].to_numpy(float)
        y = condition_table[y_column].to_numpy(float)
        axis.plot(
            x,
            y,
            marker="o",
            linewidth=1.7,
            markersize=4,
            color=CONDITION_COLORS[condition],
            label=CONDITION_LABELS[condition],
        )
        if y_sd_column and y_sd_column in condition_table:
            sd = condition_table[y_sd_column].fillna(0).to_numpy(float)
            axis.fill_between(
                x,
                y - sd,
                y + sd,
                color=CONDITION_COLORS[condition],
                alpha=0.12,
                linewidth=0,
            )
    axis.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
    axis.set_xscale("log")
    axis.set_xlabel("effective weight / temperature")


def _plot_one_neuron(
    table: pd.DataFrame,
    include_cotargeting_only: bool,
) -> plt.Figure:
    conditions = PLOTTED_CONDITIONS + (
        ("cotargeting",) if include_cotargeting_only else ()
    )
    skeleton_id = int(table["skeleton_id"].iloc[0])
    neuron_name = str(table["neuron_name"].iloc[0])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.8))
    panels = [
        (
            "geometry_energy_gap_per_site_mean",
            "geometry_energy_gap_per_site_std",
            "Geometry energy minus observed, per site",
        ),
        (
            "cotargeting_energy_gap_per_site_mean",
            "cotargeting_energy_gap_per_site_std",
            "Conditional-propensity energy minus observed, per site",
        ),
        (
            "effective_joint_energy_gap_per_site_mean",
            "effective_joint_energy_gap_per_site_std",
            "Effective joint energy minus observed, per site",
        ),
        (
            "mean_pattern_pair_cosine",
            None,
            "Cosine to observed mean pair pattern",
        ),
    ]
    for axis, (column, sd_column, ylabel) in zip(axes.flat, panels):
        _plot_lines(axis, table, column, sd_column, conditions)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes.flat[-1].set_ylim(-0.05, 1.05)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(f"Temperature sweep: {neuron_name} ({skeleton_id})", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def _aggregate_for_plot(summary: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        "geometry_energy_gap_per_site_mean",
        "cotargeting_energy_gap_per_site_mean",
        "effective_joint_energy_gap_per_site_mean",
        "mean_pattern_pair_cosine",
    ]
    rows = []
    for keys, group in summary.groupby(["effective_ratio", "condition"]):
        ratio, condition = keys
        row: dict[str, Any] = {
            "effective_ratio": ratio,
            "condition": condition,
            "n_neurons": int(group["skeleton_id"].nunique()),
        }
        for column in value_columns:
            values = group[column].dropna().to_numpy(float)
            row[f"{column}_median"] = float(np.median(values)) if len(values) else np.nan
            row[f"{column}_q25"] = float(np.quantile(values, 0.25)) if len(values) else np.nan
            row[f"{column}_q75"] = float(np.quantile(values, 0.75)) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _plot_aggregate(
    aggregate: pd.DataFrame,
    include_cotargeting_only: bool,
) -> plt.Figure:
    conditions = PLOTTED_CONDITIONS + (
        ("cotargeting",) if include_cotargeting_only else ()
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.8))
    panels = [
        ("geometry_energy_gap_per_site_mean", "Geometry energy minus observed, per site"),
        (
            "cotargeting_energy_gap_per_site_mean",
            "Conditional-propensity energy minus observed, per site",
        ),
        (
            "effective_joint_energy_gap_per_site_mean",
            "Effective joint energy minus observed, per site",
        ),
        ("mean_pattern_pair_cosine", "Cosine to observed mean pair pattern"),
    ]
    for axis, (stem, ylabel) in zip(axes.flat, panels):
        for condition in conditions:
            table = aggregate[aggregate["condition"] == condition].sort_values(
                "effective_ratio"
            )
            if table.empty:
                continue
            x = table["effective_ratio"].to_numpy(float)
            median = table[f"{stem}_median"].to_numpy(float)
            q25 = table[f"{stem}_q25"].to_numpy(float)
            q75 = table[f"{stem}_q75"].to_numpy(float)
            axis.plot(
                x,
                median,
                marker="o",
                linewidth=1.7,
                markersize=4,
                color=CONDITION_COLORS[condition],
                label=CONDITION_LABELS[condition],
            )
            axis.fill_between(
                x,
                q25,
                q75,
                color=CONDITION_COLORS[condition],
                alpha=0.14,
                linewidth=0,
            )
        axis.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
        axis.set_xscale("log")
        axis.set_xlabel("effective weight / temperature")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes.flat[-1].set_ylim(-0.05, 1.05)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Temperature sweep across neurons", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def run_temperature_sweep(
    skeleton_ids: Iterable[int] | None = None,
    *,
    all_neurons: bool = False,
    ratios: Iterable[float] = DEFAULT_RATIOS,
    config_path: str | Path = REPO_ROOT / "models/approach2/sover1_config.json",
    samples: int | None = None,
    count_multiples: bool | None = None,
    n_jobs: int = 1,
    include_cotargeting_only: bool = False,
    per_neuron_plots: bool = True,
    show_plots: bool = True,
) -> dict[str, Any]:
    """Run the sweep and return its DataFrames and live Matplotlib figures."""
    os.chdir(REPO_ROOT)
    config_path = _resolve_path(Path(config_path)).resolve()
    ratios = _validate_ratios(ratios)
    base_config = Approach2Config.load(config_path)
    if count_multiples is not None:
        base_config.cotarget.count_multiples = bool(count_multiples)
    if samples is not None and samples < 1:
        raise ValueError("samples must be at least one")
    requested_samples = samples
    for ratio in ratios:
        _temperature_for_ratio(base_config, ratio)

    neuron_names = _load_neuron_names()
    if all_neurons:
        if skeleton_ids is not None:
            raise ValueError("Choose either skeleton_ids or all_neurons, not both")
        selected_skeleton_ids = sorted(neuron_names)
    else:
        if skeleton_ids is None:
            raise ValueError("Supply skeleton_ids or set all_neurons=True")
        selected_skeleton_ids = list(
            dict.fromkeys(int(value) for value in skeleton_ids)
        )
    tasks = [
        (ratio, skeleton_id, neuron_names.get(skeleton_id, "unknown"))
        for ratio in ratios
        for skeleton_id in selected_skeleton_ids
    ]
    print(
        f"Running {len(selected_skeleton_ids)} neurons x {len(ratios)} ratios = "
        f"{len(tasks)} experiments",
        flush=True,
    )

    sample_table, pattern_summaries, errors = _run_tasks(
        tasks,
        config_path=config_path,
        samples=requested_samples,
        count_multiples=count_multiples,
        n_jobs=n_jobs,
    )
    if sample_table.empty:
        raise RuntimeError("No sweep experiments completed successfully")

    summary = _summarize_samples(sample_table, pattern_summaries)
    comparisons = _comparison_table(sample_table, summary)
    sample_table = sample_table.sort_values(
        ["skeleton_id", "effective_ratio", "condition", "sample_index"]
    ).reset_index(drop=True)

    aggregate = _aggregate_for_plot(summary)
    figures: dict[str, Any] = {
        "aggregate": _plot_aggregate(
            aggregate,
            include_cotargeting_only,
        ),
        "per_neuron": {},
    }
    if per_neuron_plots:
        figures["per_neuron"] = {
            int(skeleton_id): _plot_one_neuron(
                table,
                include_cotargeting_only,
            )
            for skeleton_id, table in summary.groupby("skeleton_id")
        }
    if show_plots:
        plt.show()

    metadata = {
        "config": str(config_path),
        "base_config": base_config.to_dict(),
        "effective_ratios": ratios,
        "derived_temperatures": {
            str(ratio): _temperature_for_ratio(base_config, ratio)
            for ratio in ratios
        },
        "skeleton_ids": selected_skeleton_ids,
        "prediction_samples": (
            requested_samples
            if requested_samples is not None
            else base_config.sampler.prediction_samples
        ),
        "count_multiples": base_config.cotarget.count_multiples,
        "n_jobs": n_jobs,
        "completed_experiments": int(
            summary[["skeleton_id", "effective_ratio"]].drop_duplicates().shape[0]
        ),
        "failed_experiments": int(len(errors)),
        "notes": [
            "The interactive defaults mirror the local overrides in run_and_save.py.",
            "Propensity is re-estimated relative to geometry at every effective ratio.",
            "The cotargeting-only condition is diagnostic and is not a standalone biological model.",
            "Energy is minus the sampler objective; a positive sample-minus-observed gap indicates worse fit than observed.",
        ],
    }
    return {
        "sample_diagnostics": sample_table,
        "pattern_diagnostics": pattern_summaries,
        "summary_diagnostics": summary,
        "comparison_diagnostics": comparisons,
        "aggregate_diagnostics": aggregate,
        "error_diagnostics": errors,
        "run_metadata": metadata,
        "diagnostic_figures": figures,
    }


#%% Run the sweep and keep every result available in the interactive namespace.
if __name__ == "__main__":
    sweep_results = run_temperature_sweep(
        skeleton_ids=SKIDS,
        ratios=EFFECTIVE_RATIOS,
        config_path=CONFIG_PATH,
        samples=PREDICTION_SAMPLES,
        count_multiples=COUNT_MULTIPLES,
        n_jobs=N_JOBS,
        include_cotargeting_only=INCLUDE_COTARGETING_ONLY,
        per_neuron_plots=PER_NEURON_PLOTS,
        show_plots=SHOW_PLOTS,
    )
    sample_diagnostics = sweep_results["sample_diagnostics"]
    pattern_diagnostics = sweep_results["pattern_diagnostics"]
    summary_diagnostics = sweep_results["summary_diagnostics"]
    comparison_diagnostics = sweep_results["comparison_diagnostics"]
    aggregate_diagnostics = sweep_results["aggregate_diagnostics"]
    error_diagnostics = sweep_results["error_diagnostics"]
    run_metadata = sweep_results["run_metadata"]
    diagnostic_figures = sweep_results["diagnostic_figures"]
