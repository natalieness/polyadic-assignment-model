"""Command-line interface for cache building, training, prediction, and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import DegreeSummary, ModelConfig
from .data import DataRepository, MorphologyCache
from .evaluation import evaluate_split
from .model import PolyadicAssignmentModel
from .splits import grouped_neuron_split, save_splits


def _load_config(path: str | None) -> ModelConfig:
    return ModelConfig() if path is None else ModelConfig.load(path)


def _apply_feature_overrides(config: ModelConfig, enable: list[str], disable: list[str]) -> None:
    valid = set(vars(config.features))
    unknown = (set(enable) | set(disable)) - valid
    if unknown:
        raise ValueError(f"Unknown feature flags: {sorted(unknown)}")
    for name in enable:
        setattr(config.features, name, True)
    for name in disable:
        setattr(config.features, name, False)


def _read_splits(path: str | Path) -> dict[str, list[int]]:
    values = json.loads(Path(path).read_text())
    for required in ["train", "validation", "test"]:
        if required not in values:
            raise ValueError(f"Split file is missing {required}")
        values[required] = [int(value) for value in values[required]]
    return values


def _write_prediction(result, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.sites.to_csv(output_dir / "predicted_sites.csv", index=False)
    result.assignments.to_csv(output_dir / "predicted_assignments.csv", index=False)
    if not result.assignment_summary.empty:
        result.assignment_summary.to_csv(
            output_dir / "predicted_assignment_summary.csv", index=False
        )
    (output_dir / "diagnostics.json").write_text(json.dumps(result.diagnostics, indent=2) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init-config", help="Write a fully documented default JSON config")
    init.add_argument("output")

    cache = subparsers.add_parser("build-cache", help="Build per-skeleton morphology caches")
    cache.add_argument("--config")
    cache.add_argument("--kind", choices=["detailed", "backbone", "all"], default="all")
    cache.add_argument("--overwrite", action="store_true")

    split = subparsers.add_parser("split", help="Create a bilateral-pair-grouped neuron split")
    split.add_argument("--config")
    split.add_argument("--output", required=True)
    split.add_argument("--train-fraction", type=float, default=0.70)
    split.add_argument("--validation-fraction", type=float, default=0.15)
    split.add_argument("--seed", type=int, default=42)

    train = subparsers.add_parser("train", help="Train all three staged components")
    train.add_argument("--config", required=True)
    train.add_argument("--splits", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--validate-during-fit", action="store_true")
    train.add_argument("--enable", action="append", default=[])
    train.add_argument("--disable", action="append", default=[])

    predict = subparsers.add_parser("predict", help="Predict one neuron using known N and target totals")
    predict.add_argument("--model", required=True)
    predict.add_argument("--skeleton-id", type=int, required=True)
    predict.add_argument("--output-dir", required=True)
    predict.add_argument("--n-sites", type=int)
    predict.add_argument("--target-counts", help="JSON object or path to a JSON object")
    predict.add_argument("--oracle-sites", action="store_true")
    predict.add_argument("--degree-mode", choices=["learned", "exact_histogram", "supplied_summary"])
    predict.add_argument("--observed-degree-histogram", action="store_true")
    predict.add_argument("--degree-summary", help="JSON object/path with minimum,q1,median,q3,maximum")
    predict.add_argument(
        "--prediction-samples", type=int,
        help="Override partner.prediction_samples for this prediction",
    )

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a saved model on one split")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--splits", required=True)
    evaluate.add_argument("--split", choices=["train", "validation", "test"], default="test")
    evaluate.add_argument("--oracle-sites", action="store_true")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--limit", type=int)
    return parser


def _json_object(value: str) -> dict:
    path = Path(value)
    raw = path.read_text() if path.exists() else value
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "init-config":
        ModelConfig().save(args.output)
        print(f"Wrote {args.output}")
        return

    if args.command == "build-cache":
        config = _load_config(args.config)
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

    if args.command == "split":
        config = _load_config(args.config)
        repository = DataRepository(config.paths)
        splits = grouped_neuron_split(
            repository.connectors.neuron_ids,
            config.paths.bilateral_pairs,
            args.train_fraction,
            args.validation_fraction,
            args.seed,
        )
        save_splits(splits, args.output)
        print({name: len(values) for name, values in splits.items()})
        return

    if args.command == "train":
        config = ModelConfig.load(args.config)
        _apply_feature_overrides(config, args.enable, args.disable)
        repository = DataRepository(config.paths)
        splits = _read_splits(args.splits)
        model = PolyadicAssignmentModel(config)
        summary = model.fit(
            repository,
            splits["train"],
            splits["validation"] if args.validate_during_fit else None,
        )
        model.save(args.output)
        print(json.dumps(summary, indent=2))
        return

    if args.command == "predict":
        model = PolyadicAssignmentModel.load(args.model)
        repository = DataRepository(model.config.paths)
        observed = repository.connectors.neuron(args.skeleton_id)
        n_sites = observed.n_sites if args.n_sites is None else args.n_sites
        target_counts = observed.target_counts if args.target_counts is None else {
            int(key): int(value) for key, value in _json_object(args.target_counts).items()
        }
        exact_sites = (
            observed.rows["presynaptic_to_node"].astype(int).tolist() if args.oracle_sites else None
        )
        histogram = observed.degree_histogram if args.observed_degree_histogram else None
        summary = None
        if args.degree_summary:
            summary = DegreeSummary(**_json_object(args.degree_summary))
        result = model.predict(
            repository,
            args.skeleton_id,
            n_sites,
            target_counts,
            exact_site_node_ids=exact_sites,
            degree_mode=args.degree_mode,
            degree_histogram=histogram,
            degree_summary=summary,
            prediction_samples=args.prediction_samples,
        )
        _write_prediction(result, args.output_dir)
        print(json.dumps(result.diagnostics, indent=2))
        return

    if args.command == "evaluate":
        model = PolyadicAssignmentModel.load(args.model)
        repository = DataRepository(model.config.paths)
        ids = _read_splits(args.splits)[args.split]
        if args.limit is not None:
            ids = ids[: args.limit]
        result = evaluate_split(model, repository, ids, oracle_sites=args.oracle_sites)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["mean_metrics"], indent=2))


if __name__ == "__main__":
    main()
