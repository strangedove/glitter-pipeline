#!/usr/bin/env python3
"""
CLI script for cleaning RP scenes (removing tics).
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rp_pipeline.config.settings import get_settings, reset_settings
from rp_pipeline.core.analysis import SceneAnalyzer, TicDetector
from rp_pipeline.core.cleanup import SceneCleaner
from rp_pipeline.data.schemas import Scene, Turn
from rp_pipeline.utils.caching import PipelineCheckpoint
from rp_pipeline.utils.logging import StructuredLogger


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Clean RP scenes by removing tics"
    )
    
    # Input/Output
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Input directory or file with scenes (JSONL format)"
    )
    parser.add_argument(
        "--analysis", "-a",
        type=str,
        default=None,
        help="Directory with analysis results (to use existing tic detection)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for cleaned scenes (overrides config)"
    )
    
    # Cleanup options
    parser.add_argument(
        "--use-rewrite",
        action="store_true",
        default=True,
        help="Use LLM rewriting for complex issues (default: True)"
    )
    parser.add_argument(
        "--no-rewrite",
        action="store_false",
        dest="use_rewrite",
        help="Disable LLM rewriting, use pattern matching only"
    )
    parser.add_argument(
        "--rewrite-model",
        type=str,
        default=None,
        help="Model to use for rewriting (overrides config)"
    )
    parser.add_argument(
        "--rewrite-provider",
        type=str,
        default=None,
        help="Provider for rewriting (openrouter, featherless, nvidia)"
    )
    
    # Checkpointing
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpointing"
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
        help="Log format (default: from config)"
    )
    
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Load and override configuration."""
    settings = get_settings()
    
    config = {
        "input_path": (
            args.input
            or settings.pipeline.get("cleanup", {}).get("output_dir", "data/output/analyzed")
        ),
        "analysis_path": (
            args.analysis
            or settings.pipeline.get("analyze", {}).get("output_dir", "data/output/analyzed")
        ),
        "output_dir": (
            args.output
            or settings.pipeline.get("cleanup", {}).get("output_dir", "data/output/cleaned")
        ),
        "use_rewrite": args.use_rewrite,
        "rewrite_model": args.rewrite_model,
        "rewrite_provider": args.rewrite_provider,
        "checkpoint_enabled": not args.no_checkpoint,
        "resume": args.resume,
        "log_level": args.log_level,
        "log_format": args.log_format or settings.logging.get("format", "json"),
    }
    
    return config


def load_scene_from_jsonl(file_path: Path) -> Optional[Scene]:
    """Load a scene from a JSONL file."""
    try:
        with open(file_path, 'r') as f:
            data = json.loads(f.read())
        
        # Parse OAI format back to Scene
        turns = []
        for idx, msg in enumerate(data.get("messages", [])):
            turns.append(Turn(
                role=msg["role"].upper(),
                turn_number=idx + 1,
                content=msg["content"],
                word_count=len(msg["content"].split()),
                token_count=int(len(msg["content"].split()) * 1.3),
            ))
        
        return Scene(
            card_id=data.get("card_id"),
            conversation="\n".join(f"[{t.role} - Turn {t.turn_number}] {t.content}" for t in turns),
            turns=turns,
            assistant_turns=[t.content for t in turns if t.role == "ASSISTANT"],
            metadata=data.get("metadata", {}),
            total_word_count=sum(t.word_count for t in turns),
            total_token_count=sum(t.token_count for t in turns),
            turn_count=len(turns),
        )
    except Exception as e:
        return None


def load_analysis(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load analysis result for a scene."""
    try:
        with open(file_path, 'r') as f:
            return json.loads(f.read())
    except Exception:
        return None


def format_cleanup_result(
    scene_id: str,
    cleanup_result: Any,
) -> Dict[str, Any]:
    """Format cleanup result for output."""
    cleaned_scene = cleanup_result.cleaned_scene
    
    # Format cleaned scene in OAI format
    messages = []
    for turn in cleaned_scene.turns:
        messages.append({
            "role": turn.role.lower(),
            "content": turn.content,
        })
    
    return {
        "scene_id": scene_id,
        "card_id": cleaned_scene.card_id,
        "messages": messages,
        "metadata": {
            **cleaned_scene.metadata,
            "turn_count": cleaned_scene.turn_count,
            "total_word_count": cleaned_scene.total_word_count,
            "total_token_count": cleaned_scene.total_token_count,
        },
        "cleanup_info": {
            "changes_made": cleanup_result.changes_made,
            "tics_removed": cleanup_result.tics_removed,
            "validation_passed": cleanup_result.validation_passed,
        },
    }


def main():
    """Main entry point."""
    args = parse_args()
    config = load_config(args)
    
    # Set up logging
    logger = StructuredLogger()
    logger.log("info", "Starting cleanup", **config)
    
    # Set up input/output
    input_path = Path(config["input_path"])
    analysis_path = Path(config["analysis_path"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect input files
    input_files = []
    if input_path.is_file():
        input_files = [input_path]
    elif input_path.is_dir():
        input_files = list(input_path.glob("*.jsonl"))
    
    if not input_files:
        logger.error(f"No input files found at {input_path}")
        sys.exit(1)
    
    logger.info(f"Found {len(input_files)} scene files to clean")
    
    # Set up analyzer and cleaner
    analyzer = SceneAnalyzer()
    
    # Configure cleaner with rewrite if enabled
    if config["use_rewrite"]:
        from rp_pipeline.models.base import ModelFactory
        from rp_pipeline.config.settings import Settings
        
        settings = get_settings()
        rewrite_config = settings.get_model_config("rewriting")
        
        if config["rewrite_provider"]:
            rewrite_config["provider"] = config["rewrite_provider"]
        if config["rewrite_model"]:
            rewrite_config["model"] = config["rewrite_model"]
        
        from rp_pipeline.models.base import BaseModelProvider
        provider = ModelFactory.create(
            rewrite_config.get("provider", "openrouter"),
            rewrite_config
        )
        cleaner = SceneCleaner(provider=provider)
    else:
        cleaner = SceneCleaner(provider=None)
    
    # Set up checkpoint
    checkpoint: Optional[PipelineCheckpoint] = None
    if config["checkpoint_enabled"]:
        checkpoint = PipelineCheckpoint(
            checkpoint_file=output_dir / ".." / ".." / "cache" / "cleanup_checkpoint.json"
        )
        if config["resume"] and checkpoint.is_resumable("cleanup"):
            start_idx, last_item = checkpoint.get_resume_position()
            logger.info(f"Resuming from {last_item}, processed {start_idx} items")
        else:
            checkpoint.start_stage("cleanup")
    
    # Process files
    processed = 0
    successful = 0
    failed = 0
    start_time = time.time()
    
    for file_path in input_files:
        scene_id = file_path.stem
        
        try:
            # Load scene
            scene = load_scene_from_jsonl(file_path)
            if scene is None:
                logger.warning(f"Failed to load scene from {file_path}")
                failed += 1
                if checkpoint:
                    checkpoint.update(scene_id, False)
                continue
            
            # Try to load existing analysis
            analysis_file = analysis_path / f"{scene_id}.analysis.jsonl"
            analysis_data = load_analysis(analysis_file)
            
            if analysis_data:
                # Use existing analysis
                from rp_pipeline.core.analysis import TicDetectionResult
                tic_result = TicDetectionResult(**analysis_data["tic_analysis"])
                quality_issues = analysis_data.get("quality_analysis", {})
            else:
                # Run analysis
                tic_result, quality_issues = analyzer.analyze(scene)
            
            # Clean
            cleanup_result = cleaner.clean_scene(
                scene,
                tic_result=tic_result,
                quality_issues=quality_issues,
                use_rewrite=config["use_rewrite"],
            )
            
            # Format and save
            result = format_cleanup_result(scene_id, cleanup_result)
            output_file = output_dir / f"{scene_id}.cleaned.jsonl"
            
            with open(output_file, 'w') as f:
                f.write(json.dumps(result) + '\n')
            
            logger.log_cleanup(
                scene_id=scene_id,
                changes_made=cleanup_result.changes_made,
                tics_removed=cleanup_result.tics_removed,
                validation_passed=cleanup_result.validation_passed,
            )
            
            successful += 1
            if checkpoint:
                checkpoint.update(scene_id, True)
            
        except Exception as e:
            logger.error(f"Error cleaning {file_path}: {e}")
            failed += 1
            if checkpoint:
                checkpoint.update(scene_id, False)
        
        processed += 1
    
    # Mark complete
    if checkpoint:
        checkpoint.mark_complete()
    
    duration = time.time() - start_time
    logger.log_pipeline(
        stage="cleanup",
        items_processed=processed,
        items_successful=successful,
        items_failed=failed,
        duration=duration,
    )
    
    logger.info(f"Cleanup complete: {successful} successful, {failed} failed in {duration:.1f}s")


if __name__ == "__main__":
    main()
