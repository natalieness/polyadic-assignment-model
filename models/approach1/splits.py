"""Leakage-aware train/validation/test neuron splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def grouped_neuron_split(
    neuron_ids: Iterable[int],
    pair_file: str | Path | None,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, list[int]]:
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("Invalid split fractions")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than one")
    ids = sorted({int(value) for value in neuron_ids})
    id_set = set(ids)
    group_for = {skeleton_id: skeleton_id for skeleton_id in ids}
    if pair_file is not None and Path(pair_file).exists():
        pairs = pd.read_csv(pair_file)
        if not {"leftid", "rightid"} <= set(pairs.columns):
            raise ValueError("Pair file requires leftid and rightid columns")
        for group_number, row in pairs.iterrows():
            left, right = int(row["leftid"]), int(row["rightid"])
            if left in id_set or right in id_set:
                group_id = -(int(group_number) + 1)
                if left in id_set:
                    group_for[left] = group_id
                if right in id_set:
                    group_for[right] = group_id
    groups: dict[int, list[int]] = {}
    for skeleton_id, group_id in group_for.items():
        groups.setdefault(group_id, []).append(skeleton_id)
    group_ids = np.asarray(list(groups), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(group_ids)
    target_train = train_fraction * len(ids)
    target_validation = validation_fraction * len(ids)
    result = {"train": [], "validation": [], "test": []}
    for group_id in group_ids:
        members = groups[int(group_id)]
        if len(result["train"]) < target_train:
            split = "train"
        elif len(result["validation"]) < target_validation:
            split = "validation"
        else:
            split = "test"
        result[split].extend(members)
    for split in result:
        result[split] = sorted(result[split])
    return result


def save_splits(splits: dict[str, list[int]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2) + "\n")
