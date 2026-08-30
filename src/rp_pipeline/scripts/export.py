#!/usr/bin/env python3
"""
CLI script for exporting pipeline output to training-ready dataset files.

Runs after rewrite (and optionally after judge): validates scenes, drops
duplicates and quality failures, builds full-card system prompts, splits
train/val by card, and writes a manifest + metadata sidecar.
"""

import argparse
import sys
from pathlib import Path
import json
from typing import Any, Dict, Optional

from rp_pipeline.config.settings import get_settings
from rp_pipeline.core.export import export_scenes
from rp_pipeline.utils.logging import StructuredLogger


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Export final scenes to a training-ready dataset"
    )

    # Input/Output
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Input directory with final scenes (JSONL format; default: config final dir)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for dataset files (default: data/output/export)"
    )
    parser.add_argument(
        "--cards",
        type=str,
        default=None,
        help="Cards JSONL file or directory (for full-card system prompts)"
    )
    parser.add_argument(
        "--judge-dir",
        type=str,
        default=None,
        help="Judge output directory (verdicts merged into the metadata sidecar)"
    )

    # Format
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["messages", "sharegpt", "text"],
        default=None,
        help="Dataset format (default: messages = {role, content} chat format)"
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=None,
        help="Fraction of scenes held out for validation (stratified by card)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for the shuffle before splitting (default: 42)"
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable exact + near duplicate removal"
    )
    parser.add_argument(
        "--similarity",
        type=float,
        default=None,
        help="Jaccard near-duplicate threshold (default: from config)"
    )

    # Quality gate overrides
    parser.add_argument("--min-turns", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--min-tokens", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)

    # Config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a settings YAML (overrides default)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Log level (default: from config)"
    )
    parser.add_argument(
        "--log-format",
        type=str,
        default=None,
        help="Log format: json or text (default: from config)"
    )

    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Merge config defaults with CLI overrides."""
    settings = get_settings()
    export_cfg = settings.get("export", {})

    fmt = args.format or export_cfg.get("format", "messages")
    val_frac = args.val_frac if args.val_frac is not None else float(export_cfg.get("val_frac", 0.02))
    seed = args.seed if args.seed is not None else int(export_cfg.get("seed", 42))
    dedup = not args.no_dedup and bool(export_cfg.get("dedup", True))
    similarity = args.similarity or float(
        export_cfg.get("similarity_threshold", settings.quality.get("similarity_threshold", 0.85))
    )

    quality = dict(settings.quality)
    if args.min_turns is not None:
        quality["min_turn_count"] = args.min_turns
    if args.max_turns is not None:
        quality["max_turn_count"] = args.max_turns
    if args.min_tokens is not None:
        quality["min_token_count"] = args.min_tokens
    if args.max_tokens is not None:
        quality["max_token_count"] = args.max_tokens

    return {
        "fmt": fmt,
        "val_frac": val_frac,
        "seed": seed,
        "dedup": dedup,
        "similarity": similarity,
        "quality": quality,
    }


def main():
    """Main entry point."""
    args = parse_args()
    config = load_config(args)

    logger = StructuredLogger()
    logger.log("info", "Starting export", **{k: v for k, v in config.items()})

    # Resolve paths
    paths = get_settings().paths
    input_dir = args.input or paths.get("output", {}).get("final", "data/output/final")
    output_dir = args.output or "data/output/export"
    cards = args.cards or paths.get("input", {}).get("cards_dir")

    system_header = (
        "You are entering a roleplay scene. Stay fully in character as the "
        "assistant character across the whole scene."
    )

    manifest = export_scenes(
        Path(input_dir),
        Path(output_dir),
        cards_dir=Path(cards) if cards else None,
        judge_dir=Path(args.judge_dir) if args.judge_dir else None,
        fmt=config["fmt"],
        system_header=system_header,
        val_frac=config["val_frac"],
        seed=config["seed"],
        dedup=config["dedup"],
        similarity_threshold=config["similarity"],
        quality=config["quality"],
    )

    logger.info("Export complete")
    logger.info(f"Input records: {manifest['stats']['input_records']}")
    logger.info(
        f"Dropped: {manifest['stats']['dropped_validation']} validation, "
        f"{manifest['stats']['dropped_duplicates']} duplicates"
    )
    logger.info(f"Exported: {manifest['stats']['train']} train / {manifest['stats']['val']} val")
    logger.info(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
