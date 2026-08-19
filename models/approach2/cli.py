"""Command-line interface for the fixed-site/fixed-degree attribution experiment."""

from __future__ import annotations

import argparse
import json

from models.approach1.data import DataRepository, MorphologyCache

from .config import Approach2Config
from .simulator import run_attribution_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init-config", help="Write a default configuration")
    init.add_argument("output")

    cache = subparsers.add_parser(
        "build-cache", help="Build per-skeleton morphology caches from this config's paths"
    )
    cache.add_argument("--config", required=True)
    cache.add_argument("--kind", choices=["detailed", "backbone", "all"], default="all")
    cache.add_argument("--overwrite", action="store_true")

    run = subparsers.add_parser("run", help="Run all four matched assignment conditions")
    run.add_argument("--config", required=True)
    run.add_argument("--skeleton-id", type=int, required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--samples", type=int)
    run.add_argument("--geometry-weight", type=float)
    run.add_argument("--cotargeting-weight", type=float)
    run.add_argument("--no-geometry", action="store_true")
    run.add_argument("--no-cotargeting", action="store_true")
    run.add_argument("--cotarget-scores", help="CSV with target_a,target_b,score")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "init-config":
        Approach2Config().save(args.output)
        print(f"Wrote {args.output}")
        return

    config = Approach2Config.load(args.config)

    if args.command == "build-cache":
        if args.kind in {"detailed", "all"}:
            print(MorphologyCache(config.paths.morphology_cache).build(
                config.paths.detailed_nodes, overwrite=args.overwrite
            ))
        if args.kind in {"backbone", "all"}:
            if config.paths.target_backbones is None:
                raise ValueError("No target_backbones path is configured")
            print(MorphologyCache(config.paths.backbone_cache).build(
                config.paths.target_backbones, overwrite=args.overwrite
            ))
        return

    if args.samples is not None:
        config.sampler.prediction_samples = args.samples
    if args.geometry_weight is not None:
        config.sampler.geometry_weight = args.geometry_weight
    if args.cotargeting_weight is not None:
        config.sampler.cotargeting_weight = args.cotargeting_weight
    if args.no_geometry:
        config.geometry.enabled = False
    if args.no_cotargeting:
        config.cotarget.enabled = False
        config.cotarget.source = "none"
    if args.cotarget_scores:
        config.cotarget.enabled = True
        config.cotarget.source = "supplied"
        config.cotarget.supplied_scores = args.cotarget_scores
    config.validate()
    repository = DataRepository(config.paths)
    result = run_attribution_experiment(repository, args.skeleton_id, config)
    result.save(args.output_dir, save_raw_samples=config.output.save_raw_samples)
    print(json.dumps(result.diagnostics, indent=2))
    print(result.condition_summary.to_string(index=False))


if __name__ == "__main__":
    main()
