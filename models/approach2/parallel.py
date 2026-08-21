"""Optional process-parallel driver for per-neuron attribution experiments.

Each neuron is an independent experiment, and the sampler seeds its draws from
``(config, skeleton_id, sample, stream)`` rather than from a shared stream, so
results do not depend on how many workers run or in what order they finish:
``n_jobs=8`` reproduces ``n_jobs=1`` exactly.

Only the fields the analysis actually consumes are shipped back from workers.
The full ``AttributionResult`` also carries per-condition diagnostics and the
propensity matrices, and pickling those through a pipe for every neuron costs
more than recomputing them would.
"""

from __future__ import annotations

import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from models.approach1.data import DataRepository

from .config import Approach2Config
from .simulator import run_attribution_experiment

CONDITIONS = ("null", "geometry", "cotargeting", "combined")

# Set in each worker by ``_init_worker`` so the connector table is parsed once
# per process instead of once per neuron.
_WORKER_CONFIG: Approach2Config | None = None
_WORKER_REPOSITORY: DataRepository | None = None

# Set in the parent immediately before a forked pool starts. Forked children
# inherit module globals, so they can read the parent's repository straight out
# of copy-on-write memory rather than re-parsing the connector table.
_PARENT_REPOSITORY: DataRepository | None = None


@dataclass
class ExperimentResults:
    """Per-skeleton outputs, keyed by skeleton ID, mirroring the serial loop."""

    conditions: dict[int, pd.DataFrame] = field(default_factory=dict)
    targets: dict[int, Any] = field(default_factory=dict)
    observed_assignments: dict[int, np.ndarray] = field(default_factory=dict)
    distances_um: dict[int, np.ndarray] = field(default_factory=dict)
    samples: dict[str, dict[int, np.ndarray]] = field(
        default_factory=lambda: {condition: {} for condition in CONDITIONS}
    )
    errors: dict[int, str] = field(default_factory=dict)

    # Named accessors so callers can keep the flat variable names they already use.
    @property
    def null_samples(self) -> dict[int, np.ndarray]:
        return self.samples["null"]

    @property
    def geometry_samples(self) -> dict[int, np.ndarray]:
        return self.samples["geometry"]

    @property
    def cotargeting_samples(self) -> dict[int, np.ndarray]:
        return self.samples["cotargeting"]

    @property
    def combined_samples(self) -> dict[int, np.ndarray]:
        return self.samples["combined"]

    def add(self, skeleton_id: int, payload: dict[str, Any]) -> None:
        self.observed_assignments[skeleton_id] = payload["observed_assignment"]
        self.distances_um[skeleton_id] = payload["distances_um"]
        self.conditions[skeleton_id] = payload["condition_summary"]
        self.targets[skeleton_id] = payload["target_ids"]
        for condition in CONDITIONS:
            self.samples[condition][skeleton_id] = payload["samples"][condition]


def _extract_payload(result, keep_samples: bool) -> dict[str, Any]:
    """Reduce an ``AttributionResult`` to the fields the analysis needs."""
    samples = (
        {condition: result.samples[condition] for condition in CONDITIONS}
        if keep_samples
        else {condition: None for condition in CONDITIONS}
    )
    return {
        "observed_assignment": result.inputs.observed_assignment,
        "distances_um": result.geometry.distances_um,
        "condition_summary": result.condition_summary,
        "target_ids": result.inputs.target_ids,
        "samples": samples,
    }


def _init_worker(
    config: Approach2Config,
    threads_per_worker: int | None,
    inherit_repository: bool,
) -> None:
    """Give the worker process a repository and pin its BLAS thread count.

    ``inherit_repository`` is set only for forked pools, where the parent's
    repository has already been inherited and rebuilding it would repeat a
    multi-second CSV parse in every worker for no gain.
    """
    global _WORKER_CONFIG, _WORKER_REPOSITORY
    if threads_per_worker is not None:
        # Without this every worker starts a full-width BLAS pool and the
        # processes fight over the same cores.
        for variable in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            os.environ[variable] = str(threads_per_worker)
    _WORKER_CONFIG = config
    if inherit_repository and _PARENT_REPOSITORY is not None:
        _WORKER_REPOSITORY = _PARENT_REPOSITORY
    else:
        _WORKER_REPOSITORY = DataRepository(config.paths)


def _run_one(skeleton_id: int, keep_samples: bool) -> tuple[int, dict[str, Any] | None, str | None]:
    """Worker entry point. Errors come back as values so one bad neuron cannot
    tear down the pool, matching the serial loop's try/except behaviour."""
    if _WORKER_REPOSITORY is None or _WORKER_CONFIG is None:
        raise RuntimeError("Worker was not initialised via _init_worker")
    try:
        result = run_attribution_experiment(
            _WORKER_REPOSITORY,
            skeleton_id=int(skeleton_id),
            config=_WORKER_CONFIG,
        )
    except Exception as error:  # noqa: BLE001 - reported back to the parent
        return int(skeleton_id), None, f"{type(error).__name__}: {error}"
    return int(skeleton_id), _extract_payload(result, keep_samples), None


def _resolve_workers(n_jobs: int) -> int:
    if n_jobs < 0:
        return max(1, (os.cpu_count() or 1) + 1 + n_jobs)
    return max(1, n_jobs)


def _resolve_start_method(start_method: str | None) -> str:
    if start_method is not None:
        return start_method
    available = multiprocessing.get_all_start_methods()
    # "fork" is the default on POSIX because it needs no ``if __name__ ==
    # "__main__"`` guard, which a cell-style script such as run_on_neurons.py
    # does not have: "spawn" and "forkserver" re-import the main module in each
    # child and would re-run the whole script. Fork also inherits the parent's
    # already-imported numpy/numba state.
    return "fork" if "fork" in available else "spawn"


def run_experiments(
    skids: Sequence[int] | Iterable[int],
    config: Approach2Config,
    repository: DataRepository | None = None,
    n_jobs: int = 1,
    keep_samples: bool = True,
    start_method: str | None = None,
    threads_per_worker: int | None = 1,
    progress: bool = True,
) -> ExperimentResults:
    """Run ``run_attribution_experiment`` for every skeleton, optionally in parallel.

    Parameters
    ----------
    skids:
        Skeleton IDs to run.
    config:
        Shared configuration. Sent to each worker once, at pool start-up.
    repository:
        Used only when ``n_jobs == 1``; workers build their own. Required for
        the serial path.
    n_jobs:
        ``1`` runs in-process (no pickling, easiest to debug and profile).
        ``> 1`` uses that many worker processes; ``-1`` uses every core.
    keep_samples:
        Set ``False`` to drop the sample arrays. They dominate the amount of
        data pushed back through the pipe, so this is much faster when only the
        condition summaries are wanted.
    start_method:
        ``"fork"`` (POSIX default here), ``"spawn"`` or ``"forkserver"``.
        Non-fork methods re-import ``__main__`` in each child, so they are only
        safe from a script that guards its entry point with
        ``if __name__ == "__main__":``.
    threads_per_worker:
        Value forced into the BLAS/OpenMP thread-count environment variables
        inside each worker; ``None`` leaves them alone. Only takes effect under
        start methods that import numpy afresh (``spawn``/``forkserver``).
    progress:
        Print one line per finished neuron, as the serial loop does.
    """
    skids = [int(value) for value in skids]
    results = ExperimentResults()
    if not skids:
        return results

    workers = _resolve_workers(n_jobs)
    if workers == 1:
        if repository is None:
            raise ValueError("Serial mode (n_jobs=1) needs a repository")
        for index, skeleton_id in enumerate(skids):
            started = time.time()
            try:
                result = run_attribution_experiment(
                    repository, skeleton_id=skeleton_id, config=config
                )
            except Exception as error:  # noqa: BLE001 - same policy as the pool
                message = f"{type(error).__name__}: {error}"
                results.errors[skeleton_id] = message
                print(f"Error processing skeleton {skeleton_id}: {message}")
                continue
            results.add(skeleton_id, _extract_payload(result, keep_samples))
            if progress:
                print(
                    f"Finished neuron # {index + 1}/{len(skids)} "
                    f"in {time.time() - started:.2f} s"
                )
        return results

    global _PARENT_REPOSITORY
    method = _resolve_start_method(start_method)
    context = multiprocessing.get_context(method)
    # Publish the repository before the first fork so children inherit it.
    # Under spawn/forkserver nothing is inherited, so workers must rebuild.
    inherit_repository = method == "fork" and repository is not None
    if inherit_repository:
        _PARENT_REPOSITORY = repository
    started_all = time.time()
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_init_worker,
            initargs=(config, threads_per_worker, inherit_repository),
        ) as executor:
            futures = {
                executor.submit(_run_one, skeleton_id, keep_samples): skeleton_id
                for skeleton_id in skids
            }
            for done, future in enumerate(as_completed(futures), start=1):
                skeleton_id, payload, error = future.result()
                if error is not None:
                    results.errors[skeleton_id] = error
                    print(f"Error processing skeleton {skeleton_id}: {error}")
                    continue
                results.add(skeleton_id, payload)
                if progress:
                    elapsed = time.time() - started_all
                    rate = done / elapsed if elapsed > 0 else float("nan")
                    print(
                        f"Finished neuron # {done}/{len(skids)} (skid {skeleton_id}) "
                        f"- {elapsed:.1f} s elapsed, {rate:.2f} neurons/s"
                    )
    finally:
        _PARENT_REPOSITORY = None
    return results
