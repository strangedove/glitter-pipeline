#!/usr/bin/env python3
"""
CLI script for running the full RP pipeline end-to-end.
Stages: generate -> analyze -> cleanup -> rewrite
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rp_pipeline.config.settings import get_settings
from rp_pipeline.utils.logging import StructuredLogger


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run full RP pipeline: generate -> analyze -> cleanup -> rewrite"
    )
    
    # Input
    parser.add_argument(
        "--cards", "-c",
        type=str,
        default=None,
        help="Path to cards JSONL file or directory"
    )
    
    # Stage toggles
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip generation stage"
    )
    parser.add_argument(
        "--skip-analyze",
        action="store_true",
        help="Skip analysis stage"
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip cleanup stage"
    )
    parser.add_argument(
        "--skip-rewrite",
        action="store_true",
        help="Skip rewrite stage"
    )
    
    # Generation options
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model for generation"
    )
    parser.add_argument(
        "--provider", "-p",
        type=str,
        default=None,
        help="Provider for generation"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=None,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--variants", "-v",
        type=int,
        default=None,
        help="Variants per card"
    )
    parser.add_argument(
        "--target-turns", "-t",
        type=int,
        default=None,
        help="Target turns per scene"
    )
    parser.add_argument(
        "--turn-length", "-l",
        type=str,
        choices=["short", "medium", "long"],
        default=None,
        help="Turn length strategy"
    )
    
    # Cleanup options
    parser.add_argument(
        "--no-rewrite",
        action="store_false",
        dest="use_rewrite",
        default=True,
        help="Disable LLM rewriting in cleanup"
    )
    parser.add_argument(
        "--rewrite-model",
        type=str,
        default=None,
        help="Model for rewriting"
    )
    
    # Checkpointing
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints"
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable all checkpointing"
    )
    
    # Output
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Base output directory (overrides config)"
    )
    
    # Config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to settings YAML file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--log-format",
        type=str,
        choices=["json", "text"],
        default=None,
        help="Log format"
    )
    
    return parser.parse_args()


def run_stage(
    stage: str,
    script_path: Path,
    args: List[str],
    logger: StructuredLogger,
) -> bool:
    """
    Run a pipeline stage.
    
    Args:
        stage: Stage name
        script_path: Path to script
        args: Arguments for script
        logger: Logger instance
    
    Returns:
        True if successful
    """
    cmd = [sys.executable, str(script_path)] + args
    
    logger.info(f"Starting {stage} stage: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        
        # Log output
        if result.stdout:
            for line in result.stdout.split('\n'):
                if line.strip():
                    logger.info(f"[{stage}] {line}")
        
        if result.stderr:
            for line in result.stderr.split('\n'):
                if line.strip():
                    logger.warning(f"[{stage} stderr] {line}")
        
        logger.info(f"{stage} stage completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"{stage} stage failed with exit code {e.returncode}")
        if e.stdout:
            logger.error(f"stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr}")
        return False


def build_generate_args(args: argparse.Namespace) -> List[str]:
    """Build arguments for generate.py."""
    generate_args = []
    
    if args.cards:
        generate_args.extend(["--cards", args.cards])
    if args.model:
        generate_args.extend(["--model", args.model])
    if args.provider:
        generate_args.extend(["--provider", args.provider])
    if args.batch_size:
        generate_args.extend(["--batch-size", str(args.batch_size)])
    if args.variants:
        generate_args.extend(["--variants", str(args.variants)])
    if args.target_turns:
        generate_args.extend(["--target-turns", str(args.target_turns)])
    if args.turn_length:
        generate_args.extend(["--turn-length", args.turn_length])
    if args.resume:
        generate_args.append("--resume")
    if args.no_checkpoint:
        generate_args.append("--no-checkpoint")
    if args.output_dir:
        generate_args.extend(["--output", str(Path(args.output_dir) / "raw")])
    if args.config:
        generate_args.extend(["--config", args.config])
    if args.log_level:
        generate_args.extend(["--log-level", args.log_level])
    if args.log_format:
        generate_args.extend(["--log-format", args.log_format])
    
    return generate_args


def build_analyze_args(args: argparse.Namespace, output_dir: str) -> List[str]:
    """Build arguments for analyze.py."""
    analyze_args = []
    
    # Input is output from generate
    generate_output = Path(output_dir) / "raw"
    analyze_args.extend(["--input", str(generate_output)])
    
    analyze_output = Path(output_dir) / "analyzed"
    analyze_args.extend(["--output", str(analyze_output)])
    
    if args.resume:
        analyze_args.append("--resume")
    if args.no_checkpoint:
        analyze_args.append("--no-checkpoint")
    if args.config:
        analyze_args.extend(["--config", args.config])
    if args.log_level:
        analyze_args.extend(["--log-level", args.log_level])
    if args.log_format:
        analyze_args.extend(["--log-format", args.log_format])
    
    return analyze_args


def build_cleanup_args(args: argparse.Namespace, output_dir: str) -> List[str]:
    """Build arguments for cleanup.py."""
    cleanup_args = []
    
    # Input is output from generate
    generate_output = Path(output_dir) / "raw"
    cleanup_args.extend(["--input", str(generate_output)])
    
    # Analysis input
    analyze_output = Path(output_dir) / "analyzed"
    cleanup_args.extend(["--analysis", str(analyze_output)])
    
    cleanup_output = Path(output_dir) / "cleaned"
    cleanup_args.extend(["--output", str(cleanup_output)])
    
    if not args.use_rewrite:
        cleanup_args.append("--no-rewrite")
    if args.rewrite_model:
        cleanup_args.extend(["--rewrite-model", args.rewrite_model])
    if args.resume:
        cleanup_args.append("--resume")
    if args.no_checkpoint:
        cleanup_args.append("--no-checkpoint")
    if args.config:
        cleanup_args.extend(["--config", args.config])
    if args.log_level:
        cleanup_args.extend(["--log-level", args.log_level])
    if args.log_format:
        cleanup_args.extend(["--log-format", args.log_format])
    
    return cleanup_args


def build_rewrite_args(args: argparse.Namespace, output_dir: str) -> List[str]:
    """Build arguments for rewrite.py."""
    rewrite_args = []
    
    # Input is output from cleanup
    cleanup_output = Path(output_dir) / "cleaned"
    rewrite_args.extend(["--input", str(cleanup_output)])
    
    rewrite_output = Path(output_dir) / "final"
    rewrite_args.extend(["--output", str(rewrite_output)])
    
    if args.resume:
        rewrite_args.append("--resume")
    if args.no_checkpoint:
        rewrite_args.append("--no-checkpoint")
    if args.config:
        rewrite_args.extend(["--config", args.config])
    if args.log_level:
        rewrite_args.extend(["--log-level", args.log_level])
    if args.log_format:
        rewrite_args.extend(["--log-format", args.log_format])
    
    return rewrite_args


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set up logging
    logger = StructuredLogger()
    logger.log("info", "Starting full pipeline")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        settings = get_settings()
        output_dir = Path(settings.paths.get("output", {}).get("base", "data/output"))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Get script paths
    scripts_dir = Path(__file__).parent
    generate_script = scripts_dir / "generate.py"
    analyze_script = scripts_dir / "analyze.py"
    cleanup_script = scripts_dir / "cleanup.py"
    # Note: rewrite.py not yet created, but we'll include it for completeness
    rewrite_script = scripts_dir / "rewrite.py"
    
    # Track overall success
    all_successful = True
    start_time = time.time()
    
    # Run stages in order
    if not args.skip_generate:
        generate_args = build_generate_args(args)
        success = run_stage("generate", generate_script, generate_args, logger)
        all_successful = all_successful and success
        if not success:
            logger.error("Generation failed, stopping pipeline")
            sys.exit(1)
    
    if not args.skip_analyze:
        analyze_args = build_analyze_args(args, output_dir)
        success = run_stage("analyze", analyze_script, analyze_args, logger)
        all_successful = all_successful and success
    
    if not args.skip_cleanup:
        cleanup_args = build_cleanup_args(args, output_dir)
        success = run_stage("cleanup", cleanup_script, cleanup_args, logger)
        all_successful = all_successful and success
    
    if not args.skip_rewrite:
        if rewrite_script.exists():
            rewrite_args = build_rewrite_args(args, output_dir)
            success = run_stage("rewrite", rewrite_script, rewrite_args, logger)
            all_successful = all_successful and success
        else:
            logger.warning("rewrite.py not found, skipping rewrite stage")
    
    duration = time.time() - start_time
    
    if all_successful:
        logger.info(f"Full pipeline completed successfully in {duration:.1f}s")
    else:
        logger.warning(f"Full pipeline completed with some failures in {duration:.1f}s")
        sys.exit(1)


if __name__ == "__main__":
    main()
