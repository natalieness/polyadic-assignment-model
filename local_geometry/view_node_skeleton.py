#!/usr/bin/env python3
"""Plot one skeleton's nodes as an interactive 3D plotly graph.

The nodes CSV is streamed in chunks so the complete morphology table does not
need to fit in memory. Edges are drawn along the actual ``parent_node_id`` tree,
with axon and dendrite coloured separately. All parent-child segments of a
compartment are packed into a single trace separated by ``None`` breaks, which
keeps thousands of edges to two browser-side objects instead of thousands.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

REQUIRED_COLUMNS = {"node_id", "parent_node_id", "skeleton_id", "compartment", "x", "y", "z"}

# Categorical slots 1 and 2 of the reference palette, on the light chart surface.
COMPARTMENT_COLOURS = {"dendrite": "#2a78d6", "axon": "#eb6834"}
FALLBACK_COLOUR = "#1baf7a"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
ROOT_COLOUR = "#0b0b0b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View a single skeleton as an interactive 3D plotly graph."
    )
    parser.add_argument(
        "--input_csv", type=Path, default="data/nodes_manual_wdendrites_euclid.csv",
        help="Path to a nodes CSV such as nodes_manual_wdendrites_euclid.csv",
    )
    parser.add_argument("--skeleton_id", type=int, default=29, help="Skeleton ID to plot")
    parser.add_argument(
        "--output_html", type=Path, default=None,
        help="Write the figure to this HTML file instead of opening a browser window",
    )
    parser.add_argument(
        "--marker_size", type=float, default=2.0,
        help="Node marker size in points; set to 0 to draw edges only",
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


def load_skeleton(input_csv: Path, skeleton_id: int, chunksize: int) -> pd.DataFrame:
    """Read only the rows belonging to one skeleton, scanning the CSV in chunks."""
    columns = pd.read_csv(input_csv, nrows=0).columns.tolist()
    missing = REQUIRED_COLUMNS.difference(columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    parts = []
    for chunk in pd.read_csv(input_csv, chunksize=chunksize, dtype={"skeleton_id": "int64"}):
        selected = chunk.loc[chunk["skeleton_id"] == skeleton_id]
        if not selected.empty:
            parts.append(selected.copy())
    if not parts:
        raise ValueError(f"Skeleton ID {skeleton_id} does not occur in {input_csv}")
    return pd.concat(parts, ignore_index=True)


def edge_segments(xyz: np.ndarray, parent: np.ndarray, selection: np.ndarray) -> tuple[list, list, list]:
    """Build None-separated x, y, z segment lists for the selected nodes' parent edges.

    Parents are resolved across the whole skeleton so the edge joining the axon to
    its dendritic parent is drawn, coloured by the child node's compartment.
    """
    children = np.flatnonzero(selection & (parent >= 0))
    if not len(children):
        return [], [], []

    starts = xyz[children]
    ends = xyz[parent[children]]
    # Interleave start, end, None per edge so plotly lifts the pen between segments.
    segments = np.full((len(children), 3, 3), np.nan)
    segments[:, 0, :] = starts
    segments[:, 1, :] = ends
    flat = segments.reshape(-1, 3)
    return flat[:, 0].tolist(), flat[:, 1].tolist(), flat[:, 2].tolist()


def hover_text(group: pd.DataFrame) -> list[str]:
    """Describe each node with the identifiers most useful when inspecting a morphology."""
    lines = []
    for row in group.itertuples(index=False):
        parts = [f"node {row.node_id}", f"parent {row.parent_node_id}", f"{row.compartment}"]
        if hasattr(row, "rank"):
            parts.append(f"rank {row.rank}")
        if hasattr(row, "cum_dist") and pd.notna(row.cum_dist):
            parts.append(f"cum_dist {row.cum_dist:,.0f}")
        lines.append("<br>".join(parts))
    return lines


def build_figure(nodes: pd.DataFrame, skeleton_id: int, marker_size: float) -> go.Figure:
    figure = go.Figure()
    compartments = [c for c in COMPARTMENT_COLOURS if c in set(nodes["compartment"])]
    compartments += [c for c in sorted(set(nodes["compartment"])) if c not in COMPARTMENT_COLOURS]

    xyz = nodes[["x", "y", "z"]].to_numpy(np.float64, copy=False)
    parent = map_parent_indices(
        nodes["node_id"].to_numpy(np.int64, copy=False),
        nodes["parent_node_id"].to_numpy(np.int64, copy=False),
    )

    for compartment in compartments:
        selection = (nodes["compartment"] == compartment).to_numpy()
        group = nodes.loc[selection]
        colour = COMPARTMENT_COLOURS.get(compartment, FALLBACK_COLOUR)
        edge_x, edge_y, edge_z = edge_segments(xyz, parent, selection)
        figure.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z, mode="lines", name=f"{compartment} ({len(group):,} nodes)",
            line=dict(color=colour, width=2), hoverinfo="skip", legendgroup=compartment,
        ))
        if marker_size > 0:
            figure.add_trace(go.Scatter3d(
                x=group["x"], y=group["y"], z=group["z"], mode="markers", name=compartment,
                marker=dict(size=marker_size, color=colour), text=hover_text(group),
                hovertemplate="%{text}<extra></extra>", legendgroup=compartment, showlegend=False,
            ))

    roots = nodes.loc[nodes["parent_node_id"] < 0]
    if not roots.empty:
        figure.add_trace(go.Scatter3d(
            x=roots["x"], y=roots["y"], z=roots["z"], mode="markers", name="root node",
            marker=dict(size=max(marker_size, 2.0) * 3, color=ROOT_COLOUR, symbol="diamond"),
            text=hover_text(roots), hovertemplate="%{text}<extra></extra>",
        ))

    axis = dict(
        showbackground=False, gridcolor="#e6e5e1", zeroline=False,
        title=dict(font=dict(color=TEXT_SECONDARY)), tickfont=dict(color=TEXT_SECONDARY),
    )
    figure.update_layout(
        title=dict(text=f"Skeleton {skeleton_id} — {len(nodes):,} nodes", font=dict(color=TEXT_PRIMARY)),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(color=TEXT_PRIMARY),
        scene=dict(
            xaxis=dict(axis, title_text="x"), yaxis=dict(axis, title_text="y"),
            zaxis=dict(axis, title_text="z"), bgcolor=SURFACE,
            aspectmode="data",  # keep one spatial scale across all three axes
        ),
        legend=dict(font=dict(color=TEXT_PRIMARY)), margin=dict(l=0, r=0, t=48, b=0),
    )
    return figure


def main() -> None:
    args = parse_args()
    start = time.time()
    print(f"Reading {args.input_csv} for skeleton {args.skeleton_id}")
    nodes = load_skeleton(args.input_csv, args.skeleton_id, args.chunksize)
    counts = nodes["compartment"].value_counts().to_dict()
    print(f"  loaded {len(nodes):,} nodes in {time.time() - start:.1f} seconds: {counts}")

    figure = build_figure(nodes, args.skeleton_id, args.marker_size)
    if args.output_html is None:
        figure.show()
    else:
        args.output_html.parent.mkdir(parents=True, exist_ok=True)
        figure.write_html(args.output_html)
        print(f"Saved figure to {args.output_html}")


if __name__ == "__main__":
    main()
