"""Validated connector data and disk-indexed morphology access."""

from __future__ import annotations

import ast
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import numpy as np
import pandas as pd

from .config import DataPaths


MORPH_COLUMNS = [
    "node_id", "parent_node_id", "skeleton_id", "compartment", "x", "y", "z",
    "rank", "branch_id", "step_dist", "cum_dist",
]


def parse_id_list(value: object) -> list[int]:
    """Parse a serialized list while rejecting malformed/non-integer IDs."""
    if isinstance(value, list):
        parsed = value
    elif pd.isna(value):
        parsed = []
    else:
        parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a list, got {type(parsed).__name__}")
    return [int(item) for item in parsed]


@dataclass
class ConnectorNeuron:
    skeleton_id: int
    rows: pd.DataFrame

    @property
    def n_sites(self) -> int:
        return len(self.rows)

    @property
    def target_counts(self) -> dict[int, int]:
        counter: Counter[int] = Counter()
        for targets in self.rows["postsynaptic_to"]:
            counter.update(targets)
        return dict(counter)

    @property
    def degree_histogram(self) -> dict[int, int]:
        return dict(Counter(len(targets) for targets in self.rows["postsynaptic_to"]))


class ConnectorTable:
    """One validated row per presynaptic connector."""

    REQUIRED = {
        "connector_id", "presynaptic_to", "postsynaptic_to",
        "presynaptic_to_node", "postsynaptic_to_node",
    }

    def __init__(self, path: str | Path):
        self.path = Path(path)
        table = pd.read_csv(self.path)
        missing = self.REQUIRED - set(table.columns)
        if missing:
            raise ValueError(f"Connector table is missing columns: {sorted(missing)}")
        table = table[list(self.REQUIRED)].copy()
        table["presynaptic_to"] = table["presynaptic_to"].astype("int64")
        table["presynaptic_to_node"] = table["presynaptic_to_node"].astype("int64")
        table["connector_id"] = table["connector_id"].astype("int64")
        table["postsynaptic_to"] = table["postsynaptic_to"].map(parse_id_list)
        table["postsynaptic_to_node"] = table["postsynaptic_to_node"].map(parse_id_list)
        bad = table["postsynaptic_to"].map(len) != table["postsynaptic_to_node"].map(len)
        if bad.any():
            examples = table.loc[bad, "connector_id"].head(10).tolist()
            raise ValueError(
                f"{int(bad.sum())} connectors have unaligned target/node lists; "
                f"first connector IDs: {examples}"
            )
        if table["connector_id"].duplicated().any():
            raise ValueError("connector_id must be unique")
        self.table = table
        self._groups = table.groupby("presynaptic_to", sort=False).indices

    @property
    def neuron_ids(self) -> np.ndarray:
        return np.asarray(list(self._groups), dtype=np.int64)

    def neuron(self, skeleton_id: int) -> ConnectorNeuron:
        skeleton_id = int(skeleton_id)
        positions = self._groups.get(skeleton_id)
        if positions is None:
            raise KeyError(f"No outgoing connectors for skeleton {skeleton_id}")
        return ConnectorNeuron(skeleton_id, self.table.iloc[positions].reset_index(drop=True))

    def subset(self, skeleton_ids: Iterable[int]) -> pd.DataFrame:
        wanted = {int(value) for value in skeleton_ids}
        return self.table.loc[self.table["presynaptic_to"].isin(wanted)].copy()


@dataclass
class Morphology:
    skeleton_id: int
    nodes: pd.DataFrame

    def validate(self) -> None:
        if self.nodes.empty:
            raise ValueError(f"Skeleton {self.skeleton_id} has no morphology nodes")
        if self.nodes["node_id"].duplicated().any():
            raise ValueError(f"Skeleton {self.skeleton_id} has duplicate node IDs")


def _safe_npz_string_array(values: pd.Series) -> np.ndarray:
    width = max(8, int(values.astype(str).map(len).max()))
    return values.fillna("other").astype(str).to_numpy(dtype=f"U{width}")


class MorphologyCache:
    """One compressed NPZ per skeleton, produced in one streaming pass."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def path_for(self, skeleton_id: int) -> Path:
        return self.directory / f"{int(skeleton_id)}.npz"

    def exists(self, skeleton_id: int) -> bool:
        return self.path_for(skeleton_id).exists()

    def load(self, skeleton_id: int) -> Morphology:
        path = self.path_for(skeleton_id)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing morphology cache for skeleton {skeleton_id}: {path}. "
                "Run the build-cache command first."
            )
        with np.load(path, allow_pickle=False) as values:
            frame = pd.DataFrame({key: values[key] for key in values.files})
        morphology = Morphology(int(skeleton_id), frame)
        morphology.validate()
        return morphology

    def available_ids(self) -> set[int]:
        if not self.directory.exists():
            return set()
        return {int(path.stem) for path in self.directory.glob("*.npz") if path.stem.isdigit()}

    def build(self, source_csv: str | Path, overwrite: bool = False) -> dict[str, int]:
        """Build cache assuming each skeleton occupies one contiguous CSV block."""
        source_csv = Path(source_csv)
        self.directory.mkdir(parents=True, exist_ok=True)
        seen: set[int] = set()
        current_id: int | None = None
        rows: list[dict[str, str]] = []
        written = skipped = total_rows = 0

        def flush() -> None:
            nonlocal rows, written, skipped, current_id
            if current_id is None or not rows:
                return
            if current_id in seen:
                raise ValueError(
                    f"Skeleton {current_id} is not contiguous in {source_csv}; "
                    "sort the file by skeleton_id before building the cache."
                )
            seen.add(current_id)
            output = self.path_for(current_id)
            if output.exists() and not overwrite:
                skipped += 1
                rows = []
                return
            frame = pd.DataFrame.from_records(rows)
            numeric_int = ["node_id", "parent_node_id", "skeleton_id", "rank", "branch_id"]
            numeric_float = ["x", "y", "z", "step_dist", "cum_dist"]
            arrays: dict[str, np.ndarray] = {}
            for column in numeric_int:
                arrays[column] = pd.to_numeric(frame[column], errors="raise").to_numpy(np.int64)
            for column in numeric_float:
                arrays[column] = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
            arrays["compartment"] = _safe_npz_string_array(frame["compartment"])
            np.savez_compressed(output, **arrays)
            written += 1
            rows = []

        with source_csv.open(newline="") as stream:
            reader = csv.DictReader(stream)
            missing = set(MORPH_COLUMNS) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{source_csv} is missing morphology columns: {sorted(missing)}")
            for raw in reader:
                total_rows += 1
                skeleton_id = int(raw["skeleton_id"])
                if current_id is None:
                    current_id = skeleton_id
                elif skeleton_id != current_id:
                    flush()
                    current_id = skeleton_id
                rows.append({column: raw[column] for column in MORPH_COLUMNS})
        flush()
        manifest = {"rows": total_rows, "skeletons": len(seen), "written": written, "skipped": skipped}
        (self.directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest


def load_celltype_mapping(path: str | Path | None) -> dict[int, str]:
    if path is None or not Path(path).exists():
        return {}
    table = pd.read_csv(path)
    if {"name", "skids"} <= set(table.columns):
        result: dict[int, str] = {}
        for _, row in table.iterrows():
            for skeleton_id in parse_id_list(row["skids"]):
                result.setdefault(skeleton_id, str(row["name"]))
        return result
    for id_column in ["skeleton_id", "skid"]:
        if id_column in table and "celltype" in table:
            return dict(zip(table[id_column].astype(int), table["celltype"].astype(str)))
    raise ValueError(f"Cannot infer cell-type mapping columns from {path}")


def load_hemisphere_mapping(path: str | Path | None) -> dict[int, str]:
    if path is None or not Path(path).exists():
        return {}
    table = pd.read_csv(path)
    if not {"skid", "hemisphere"} <= set(table.columns):
        raise ValueError(f"Hemisphere file {path} needs skid and hemisphere columns")
    return dict(zip(table["skid"].astype(int), table["hemisphere"].astype(str)))


class DataRepository:
    """All model inputs with morphology loaded lazily from per-skeleton caches."""

    def __init__(self, paths: DataPaths):
        self.paths = paths
        self.connectors = ConnectorTable(paths.connectors)
        self.morphologies = MorphologyCache(paths.morphology_cache)
        self.backbones = MorphologyCache(paths.backbone_cache)
        self.celltypes = load_celltype_mapping(paths.celltypes)
        self.hemispheres = load_hemisphere_mapping(paths.hemispheres)

    def load_backbones(self, target_ids: Iterable[int], missing_policy: str = "uniform") -> dict[int, Morphology]:
        result: dict[int, Morphology] = {}
        missing = []
        for target_id in {int(value) for value in target_ids}:
            if self.backbones.exists(target_id):
                result[target_id] = self.backbones.load(target_id)
            else:
                missing.append(target_id)
        if missing and missing_policy == "error":
            raise FileNotFoundError(f"Missing target backbones for {missing[:20]}")
        return result
