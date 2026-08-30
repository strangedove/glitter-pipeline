#!/usr/bin/env python3
"""
CLI script for analyzing RP scenes for tics and quality issues.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


from rp_pipeline.config.settings import get_settings, reset_settings
from rp_pipeline.core.analysis import SceneAnalyzer, TicDetector
from rp_pipeline.data.schemas import Scene, Turn
from rp_pipeline.utils.caching import PipelineCheckpoint
from rp_pipeline.utils.logging import StructuredLogger


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze RP scenes for tics and quality issues"
    )
    
    # Input/Output
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Input directory or file with scenes (JSONL format)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for analysis results (overrides config)"
    )
    
    # Analysis options
    parser.add_argument(
        "--tic-rate-threshold",
        type=float,
        default=None,
        help="Tic rate threshold for flagging (tics per 1000 words)"
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
        "input_path": args.input or settings.pipeline.get("analyze", {}).get("output_dir", "data/output/raw"),
        "output_dir": args.output or settings.pipeline.get("analyze", {}).get("output_dir", "data/output/analyzed"),
        "tic_rate_threshold": args.tic_rate_threshold or settings.quality.get("tic_rate_threshold", 5.0),
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


def format_analysis_result(
    scene_id: str,
    scene: Scene,
    tic_result: Any,
    quality_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Format analysis result for output."""
    return {
        "scene_id": scene_id,
        "card_id": scene.card_id,
        "metadata": scene.metadata,
        "scene_stats": {
            "turn_count": scene.turn_count,
            "total_word_count": scene.total_word_count,
            "total_token_count": scene.total_token_count,
        },
        "tic_analysis": {
            "tics": tic_result.tics,
            "emotion_tells": tic_result.emotion_tells,
            "total_tic_count": tic_result.total_tic_count,
            "tic_rate": tic_result.tic_rate,
            "needs_cleanup": tic_result.needs_cleanup,
        },
        "quality_analysis": quality_result,
    }


def main():
    """Main entry point."""
    args = parse_args()
    config = load_config(args)
    
    # Set up logging
    logger = StructuredLogger()
    logger.log("info", "Starting analysis", **config)
    
    # Set up input/output
    input_path = Path(config["input_path"])
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
    
    logger.info(f"Found {len(input_files)} scene files to analyze")
    
    # Set up analyzer
    analyzer = SceneAnalyzer()
    
    # Set up checkpoint
    checkpoint: Optional[PipelineCheckpoint] = None
    if config["checkpoint_enabled"]:
        checkpoint = PipelineCheckpoint(
            checkpoint_file=output_dir / ".." / ".." / "cache" / "analyze_checkpoint.json"
        )
        if config["resume"] and checkpoint.is_resumable("analyze"):
            start_idx, last_item = checkpoint.get_resume_position()
            logger.info(f"Resuming from {last_item}, processed {start_idx} items")
        else:
            checkpoint.start_stage("analyze")
    
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
            
            # Analyze
            tic_result, quality_result = analyzer.analyze(scene)
            
            # Format result
            result = format_analysis_result(scene_id, scene, tic_result, quality_result)
            
            # Save
            output_file = output_dir / f"{scene_id}.analysis.jsonl"
            with open(output_file, 'w') as f:
                f.write(json.dumps(result) + '\n')
            
            logger.log_analysis(
                scene_id=scene_id,
                tics_found=tic_result.total_tic_count,
                tic_rate=tic_result.tic_rate,
                needs_cleanup=tic_result.needs_cleanup,
            )
            
            successful += 1
            if checkpoint:
                checkpoint.update(scene_id, True)
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            failed += 1
            if checkpoint:
                checkpoint.update(scene_id, False)
        
        processed += 1
    
    # Mark complete
    if checkpoint:
        checkpoint.mark_complete()
    
    duration = time.time() - start_time
    logger.log_pipeline(
        stage="analyze",
        items_processed=processed,
        items_successful=successful,
        items_failed=failed,
        duration=duration,
    )
    
    logger.info(f"Analysis complete: {successful} successful, {failed} failed in {duration:.1f}s")


if __name__ == "__main__":
    main()
