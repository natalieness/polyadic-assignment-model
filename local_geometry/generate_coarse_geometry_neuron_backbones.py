#!/usr/bin/env python3
"""Prune short terminal twigs from selected neuron morphologies.

The input is streamed in chunks so the complete morphology table does not need
to fit in memory. Axon and dendrite are pruned independently, matching the
coarse-geometry model. Twig length is the sum of Euclidean parent-child edge
lengths along the actual ``parent_node_id`` tree.
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


REQUIRED_COLUMNS = {"node_id", "parent_node_id", "skeleton_id", "compartment", "x", "y", "z"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove terminal twigs shorter than a threshold from selected skeletons."
    )
    parser.add_argument("--input_csv", type=Path, default='../data/nodes_manual_wdendrites_euclid.csv', help="Path to nodes_manual_wdendrites_euclid.csv")
    parser.add_argument("--output_csv", type=Path, default='../data/neuron_backbones.csv', help="Path for the pruned CSV")
    parser.add_argument(
        "--pruning-threshold", type=float, required=False, default=10000.0,
        help="Twig-length threshold in the CSV's spatial units (nanometres in the supplied data)",
    )
    parser.add_argument(
        "--skeleton_ids", type=Path, required=False, default='../data/all_neurons.csv',
        help="Text file containing skeleton IDs separated by whitespace or commas",
    )
    parser.add_argument("--chunksize", type=int, default=500_000, help=argparse.SUPPRESS)
    return parser.parse_args()


def map_parent_indices(node_ids: np.ndarray, parent_ids: np.ndarray) -> np.ndarray:
    """Map parent node IDs to local row indices without constructing a Python dictionary."""
    order = np.argsort(node_ids)
    sorted_ids = node_ids[order]
    locations = np.searchsorted(sorted_ids, parent_ids)
    candidates = np.minimum(locations, len(sorted_ids) - 1)
    found = (locations < len(sorted_ids)) & (sorted_ids[candidates] == parent_ids)
    parent = np.full(len(node_ids), -1, dtype=np.int64)
    parent[found] = order[candidates[found]]
    return parent


def parent_child_edge_lengths(xyz: np.ndarray, parent: np.ndarray) -> np.ndarray:
    """Calculate Euclidean length for every actual parent-child edge."""
    edge_length = np.zeros(len(parent), dtype=np.float64)
    children = np.flatnonzero(parent >= 0)
    if len(children):
        edge_length[children] = np.linalg.norm(xyz[children] - xyz[parent[children]], axis=1)
    edge_length[~np.isfinite(edge_length)] = 0.0
    return edge_length


def flag_compartment_junctions(nodes: pd.DataFrame) -> np.ndarray:
    """Flag both endpoints of every edge that crosses a compartment boundary.

    Pruning each compartment as a separate tree turns the dendrite node at the
    axon-dendrite split into a terminal leaf, so the branch reaching the split
    becomes an ordinary pruning candidate. Flagged nodes are exempt from pruning,
    which also retains everything between the split and the compartment root.
    """
    compartment_of = pd.Series(nodes["compartment"].to_numpy(), index=nodes["node_id"].to_numpy())
    parent_compartment = nodes["parent_node_id"].map(compartment_of)
    crosses = (parent_compartment.notna() & (parent_compartment != nodes["compartment"])).to_numpy()
    junction_ids = nodes["parent_node_id"].to_numpy()[crosses]
    return crosses | nodes["node_id"].isin(junction_ids).to_numpy()


@njit(cache=True)
def prune_short_terminal_twigs(parent: np.ndarray, edge_length: np.ndarray,
                               protected: np.ndarray, threshold: float) -> np.ndarray:
    """Return a mask retaining terminal paths whose cable length reaches the threshold.

    Paths containing a protected node are retained regardless of their length.
    """
    n_nodes = len(parent)
    active = np.ones(n_nodes, dtype=np.bool_)
    if n_nodes <= 3 or threshold <= 0:
        return active

    child_count = np.empty(n_nodes, dtype=np.int64)
    remove = np.empty(n_nodes, dtype=np.bool_)
    while True:
        child_count[:] = 0
        remove[:] = False
        for child in range(n_nodes):
            parent_index = parent[child]
            if active[child] and parent_index >= 0 and active[parent_index]:
                child_count[parent_index] += 1

        removed_this_pass = 0
        for leaf in range(n_nodes):
            if not active[leaf] or child_count[leaf] != 0:
                continue
            current = leaf
            length = 0.0
            while active[current]:
                length += edge_length[current]
                parent_index = parent[current]
                if parent_index < 0 or not active[parent_index]:
                    break
                if child_count[parent_index] != 1:
                    if child_count[parent_index] > 1 and length < threshold:
                        path_node = leaf
                        blocked = False
                        while path_node != parent_index:
                            if protected[path_node]:
                                blocked = True
                                break
                            path_node = parent[path_node]
                        if not blocked:
                            path_node = leaf
                            while path_node != parent_index:
                                if not remove[path_node]:
                                    remove[path_node] = True
                                    removed_this_pass += 1
                                path_node = parent[path_node]
                    break
                current = parent_index

        if removed_this_pass == 0:
            break
        for node in range(n_nodes):
            if remove[node]:
                active[node] = False
    return active


def load_selected_nodes(input_csv: Path, skeleton_ids: np.ndarray, chunksize: int) -> tuple[pd.DataFrame, list[str]]:
    columns = pd.read_csv(input_csv, nrows=0).columns.tolist()
    missing = REQUIRED_COLUMNS.difference(columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    selected_parts = []
    rows_read = 0
    wanted = set(skeleton_ids.tolist())
    for chunk in pd.read_csv(input_csv, chunksize=chunksize, dtype={"skeleton_id": "int64"}):
        rows_read += len(chunk)
        selected = chunk.loc[chunk["skeleton_id"].isin(wanted)]
        if not selected.empty:
            selected_parts.append(selected.copy())
        if rows_read % 2_000_000 < chunksize:
            print(f"  scanned {rows_read:,} rows; selected {sum(len(part) for part in selected_parts):,}")
    if not selected_parts:
        raise ValueError("None of the requested skeleton IDs occur in the input CSV")
    return pd.concat(selected_parts, ignore_index=True), columns


def prune_selected_nodes(nodes: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep = np.ones(len(nodes), dtype=bool)
    audit_rows = []
    junction = flag_compartment_junctions(nodes)
    groups = nodes.groupby(["skeleton_id", "compartment"], sort=False, dropna=False).indices
    for group_number, ((skeleton_id, compartment), positions) in enumerate(groups.items(), start=1):
        positions = np.asarray(positions, dtype=np.int64)
        group = nodes.iloc[positions]
        node_ids = group["node_id"].to_numpy(np.int64, copy=False)
        parent_ids = group["parent_node_id"].to_numpy(np.int64, copy=False)
        xyz = group[["x", "y", "z"]].to_numpy(np.float64, copy=False)
        group_junction = junction[positions]
        parent = map_parent_indices(node_ids, parent_ids)
        edge_length = parent_child_edge_lengths(xyz, parent)
        group_keep = prune_short_terminal_twigs(parent, edge_length, group_junction, threshold)
        keep[positions] = group_keep
        audit_rows.append({
            "skeleton_id": int(skeleton_id), "compartment": str(compartment),
            "input_nodes": len(group_keep), "retained_nodes": int(group_keep.sum()),
            "pruned_nodes": int((~group_keep).sum()),
            "junction_nodes": int(group_junction.sum()),
            "junction_nodes_retained": int((group_junction & group_keep).sum()),
        })
        if group_number % 250 == 0:
            print(f"  pruned {group_number:,}/{len(groups):,} skeleton-compartment groups")
    return nodes.loc[keep].copy(), pd.DataFrame(audit_rows)


def main() -> None:
    args = parse_args()
    if args.pruning_threshold < 0:
        raise ValueError("--pruning-threshold must be non-negative")
    
    skeleton_ids = pd.read_csv(args.skeleton_ids).to_numpy(np.int64, copy=False).ravel()
    start = time.time()
    print(f"Reading {args.input_csv} for {len(skeleton_ids):,} requested skeleton IDs")
    nodes, input_columns = load_selected_nodes(args.input_csv, skeleton_ids, args.chunksize)
    found_ids = np.sort(nodes["skeleton_id"].unique())
    missing_ids = np.setdiff1d(skeleton_ids, found_ids)
    if len(missing_ids):
        preview = ", ".join(map(str, missing_ids[:20]))
        suffix = " ..." if len(missing_ids) > 20 else ""
        print(f"Warning: {len(missing_ids)} requested skeleton IDs were absent: {preview}{suffix}")

    print(f"Pruning terminal twigs shorter than {args.pruning_threshold:g} distance units")
    pruned, audit = prune_selected_nodes(nodes, args.pruning_threshold)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pruned.to_csv(args.output_csv, columns=input_columns, index=False)
    removed = len(nodes) - len(pruned)
    print(audit.to_string(index=False))
    junctions = int(audit["junction_nodes"].sum())
    if junctions:
        retained = int(audit["junction_nodes_retained"].sum())
        print(f"Retained {retained:,}/{junctions:,} compartment-junction nodes")
    print(
        f"Saved {len(pruned):,} nodes to {args.output_csv}; removed {removed:,}/{len(nodes):,} "
        f"nodes ({removed / len(nodes):.1%}) in {time.time() - start:.1f} seconds"
    )


if __name__ == "__main__":
    main()
