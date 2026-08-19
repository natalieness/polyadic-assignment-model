#!/usr/bin/env python3
"""Annotate every node of every skeleton with its Strahler number.

The ordering required to number a tree is constructed from ``node_id`` and
``parent_node_id`` rather than taken from the CSV's ``rank`` column, which is
not guaranteed to place a parent before its children.

Strahler rules applied to each skeleton's rooted tree:
  * a terminal node (no children) has k = 1;
  * where two or more branches of equal k meet, the parent takes k + 1;
  * where branches of unequal k meet, the parent inherits the larger k
    (a node with a single child therefore just inherits that child's k).

Because ``node_id`` is unique across the whole table and no edge crosses a
skeleton boundary, all skeletons are numbered together as one forest in a
single pass, which avoids per-skeleton Python overhead.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        """Use the same implementation without JIT compilation if numba is unavailable."""
        if args and callable(args[0]):
            return args[0]
        return lambda function: function


REQUIRED_COLUMNS = {"node_id", "parent_node_id", "skeleton_id"}
ID_DTYPES = {"node_id": "int64", "parent_node_id": "int64", "skeleton_id": "int64"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add a 'strahler' column to a node table, numbering each skeleton independently."
    )
    parser.add_argument(
        "--input_csv", type=Path, default="../data/nodes_manual_wdendrites_euclid.csv",
        help="Path to the node table",
    )
    parser.add_argument(
        "--output_csv", type=Path, default="../data/nodes_manual_wdendrites_euclid_strahler.csv",
        help="Path for the annotated CSV",
    )
    parser.add_argument(
        "--summary_csv", type=Path, default=None,
        help="Optional path for the per-skeleton summary table",
    )
    return parser.parse_args()


def map_parent_indices(node_ids: np.ndarray, parent_ids: np.ndarray) -> np.ndarray:
    """Map parent node IDs to row indices without constructing a Python dictionary."""
    order = np.argsort(node_ids)
    sorted_ids = node_ids[order]
    locations = np.searchsorted(sorted_ids, parent_ids)
    candidates = np.minimum(locations, len(sorted_ids) - 1)
    found = (locations < len(sorted_ids)) & (sorted_ids[candidates] == parent_ids)
    parent = np.full(len(node_ids), -1, dtype=np.int64)
    parent[found] = order[candidates[found]]
    return parent


def sever_cross_skeleton_edges(parent: np.ndarray, skeleton_ids: np.ndarray) -> int:
    """Detach any edge whose endpoints belong to different skeletons.

    Each skeleton has to be numbered as its own tree, so an edge leaking across
    a boundary would merge two trees and inflate the Strahler numbers of both.
    """
    linked = parent >= 0
    crosses = np.zeros(len(parent), dtype=bool)
    crosses[linked] = skeleton_ids[parent[linked]] != skeleton_ids[linked]
    parent[crosses] = -1
    return int(crosses.sum())


@njit(cache=True)
def strahler_numbers(parent: np.ndarray, child_count: np.ndarray) -> np.ndarray:
    """Number a forest of rooted trees by peeling leaves towards the roots.

    Every node is pushed exactly once, when the last of its children has been
    numbered, so the whole forest is resolved in a single O(n) sweep without
    recursion and without sorting the nodes into a traversal order.
    """
    n_nodes = len(parent)
    strahler = np.zeros(n_nodes, dtype=np.int32)
    best = np.zeros(n_nodes, dtype=np.int32)        # largest child Strahler number so far
    best_count = np.zeros(n_nodes, dtype=np.int32)  # how many children reached that number
    pending = child_count.copy()

    stack = np.empty(n_nodes, dtype=np.int64)
    top = 0
    for node in range(n_nodes):
        if pending[node] == 0:
            stack[top] = node
            top += 1

    while top > 0:
        top -= 1
        node = stack[top]
        if best_count[node] == 0:
            number = 1                    # terminal node
        elif best_count[node] >= 2:
            number = best[node] + 1       # branches of equal order meet
        else:
            number = best[node]           # inherit the larger order
        strahler[node] = number

        parent_index = parent[node]
        if parent_index >= 0:
            if number > best[parent_index]:
                best[parent_index] = number
                best_count[parent_index] = 1
            elif number == best[parent_index]:
                best_count[parent_index] += 1
            pending[parent_index] -= 1
            if pending[parent_index] == 0:
                stack[top] = parent_index
                top += 1
    return strahler


def annotate_strahler(nodes: pd.DataFrame) -> np.ndarray:
    """Calculate the Strahler number of every row of the node table."""
    node_ids = nodes["node_id"].to_numpy(np.int64, copy=False)
    parent_ids = nodes["parent_node_id"].to_numpy(np.int64, copy=False)
    skeleton_ids = nodes["skeleton_id"].to_numpy(np.int64, copy=False)

    if len(np.unique(node_ids)) != len(node_ids):
        raise ValueError("node_id is not unique, so parents cannot be resolved unambiguously")

    parent = map_parent_indices(node_ids, parent_ids)
    dangling = int(((parent_ids != -1) & (parent < 0)).sum())
    if dangling:
        print(f"Warning: {dangling:,} nodes name a parent that is absent; treated as roots")
    crossed = sever_cross_skeleton_edges(parent, skeleton_ids)
    if crossed:
        print(f"Warning: severed {crossed:,} edges that crossed a skeleton boundary")

    child_count = np.bincount(parent[parent >= 0], minlength=len(parent)).astype(np.int64)
    strahler = strahler_numbers(parent, child_count)

    unresolved = int((strahler == 0).sum())
    if unresolved:
        raise ValueError(
            f"{unresolved:,} nodes could not be numbered, which means the parent links "
            "contain a cycle rather than a tree"
        )
    return strahler


def summarise(nodes: pd.DataFrame) -> pd.DataFrame:
    """Report the order of each skeleton and how its nodes are distributed across orders."""
    grouped = nodes.groupby("skeleton_id", sort=True)["strahler"]
    summary = pd.DataFrame({
        "skeleton_id": grouped.size().index,
        "nodes": grouped.size().to_numpy(),
        "max_strahler": grouped.max().to_numpy(),
        "terminal_nodes": grouped.apply(lambda order: int((order == 1).sum())).to_numpy(),
    })
    return summary


def main() -> None:
    args = parse_args()
    start = time.time()

    columns = pd.read_csv(args.input_csv, nrows=0).columns.tolist()
    missing = REQUIRED_COLUMNS.difference(columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")
    if "strahler" in columns:
        raise ValueError("Input CSV already contains a 'strahler' column")

    print(f"Reading {args.input_csv}")
    nodes = pd.read_csv(args.input_csv, dtype=ID_DTYPES)
    print(f"  loaded {len(nodes):,} nodes from {nodes['skeleton_id'].nunique():,} skeletons "
          f"in {time.time() - start:.1f} seconds")

    numbering_start = time.time()
    nodes["strahler"] = annotate_strahler(nodes)
    print(f"Numbered all nodes in {time.time() - numbering_start:.1f} seconds")

    summary = summarise(nodes)
    distribution = nodes["strahler"].value_counts().sort_index()
    print("\nNodes per Strahler order:")
    for order, count in distribution.items():
        print(f"  k = {order:>2}: {count:>12,} nodes")
    print(f"\nSkeleton order: min {summary['max_strahler'].min()}, "
          f"median {summary['max_strahler'].median():.0f}, max {summary['max_strahler'].max()}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(args.output_csv, columns=columns + ["strahler"], index=False)
    if args.summary_csv is not None:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_csv, index=False)
        print(f"Saved per-skeleton summary to {args.summary_csv}")
    print(f"Saved {len(nodes):,} annotated nodes to {args.output_csv} "
          f"in {time.time() - start:.1f} seconds total")


if __name__ == "__main__":
    main()
