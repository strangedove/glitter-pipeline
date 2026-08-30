#!/usr/bin/env python3
"""
CLI script for judging RP scenes using LLM-based evaluation.
Evaluates scenes on various quality dimensions.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rp_pipeline.config.settings import get_settings
from rp_pipeline.data.schemas import Scene, Turn
from rp_pipeline.models.base import ModelFactory
from rp_pipeline.utils.caching import PipelineCheckpoint
from rp_pipeline.utils.logging import StructuredLogger


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Judge RP scenes for quality"
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
        help="Output directory for judgments (overrides config)"
    )
    
    # Judge options
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model to use for judging (overrides config)"
    )
    parser.add_argument(
        "--provider", "-p",
        type=str,
        default=None,
        help="Provider to use (openrouter, featherless, nvidia)"
    )
    parser.add_argument(
        "--judge-type", "-j",
        type=str,
        choices=["behavioral", "style", "both"],
        default="behavioral",
        help="Type of judgment: behavioral (initiative, responsiveness) or style (prose quality)"
    )
    
    # Model parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens for judgment (default: from config)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for judgment (default: from config)"
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
        "input_path": args.input or settings.pipeline.get("judge", {}).get("output_dir", "data/output/final"),
        "output_dir": args.output or settings.pipeline.get("judge", {}).get("output_dir", "data/output/judged"),
        "model": args.model,
        "provider": args.provider,
        "judge_type": args.judge_type,
        "max_tokens": args.max_tokens or settings.defaults.get("judging", {}).get("max_tokens", 1200),
        "temperature": args.temperature or settings.defaults.get("judging", {}).get("temperature", 0.3),
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


def extract_assistant_turns(scene: Scene) -> str:
    """Extract just the assistant turns for judgment."""
    return "\n\n".join(
        f"[ASSISTANT - Turn {t.turn_number}]\n{t.content}"
        for t in scene.turns
        if t.role == "ASSISTANT"
    )


def main():
    """Main entry point."""
    args = parse_args()
    config = load_config(args)
    
    # Set up logging
    logger = StructuredLogger()
    logger.log("info", "Starting judgment", **config)
    
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
    
    logger.info(f"Found {len(input_files)} scene files to judge")
    
    # Set up model provider
    settings = get_settings()
    judge_config = settings.get_model_config("judging")
    
    if config["provider"]:
        judge_config["provider"] = config["provider"]
    if config["model"]:
        judge_config["model"] = config["model"]
    judge_config["max_tokens"] = config["max_tokens"]
    judge_config["temperature"] = config["temperature"]
    
    provider = ModelFactory.create(
        judge_config.get("provider", "featherless"),
        judge_config
    )
    
    # Get judge prompts
    from rp_pipeline.config.settings import load_prompts
    prompts = load_prompts()
    
    if config["judge_type"] in ["behavioral", "both"]:
        behavioral_system = prompts.get("judge_system", "")
    if config["judge_type"] in ["style", "both"]:
        style_system = prompts.get("style_judge_system", "")
    
    # Set up checkpoint
    checkpoint: Optional[PipelineCheckpoint] = None
    if config["checkpoint_enabled"]:
        checkpoint = PipelineCheckpoint(
            checkpoint_file=output_dir / ".." / ".." / "cache" / "judge_checkpoint.json"
        )
        if config["resume"] and checkpoint.is_resumable("judge"):
            start_idx, last_item = checkpoint.get_resume_position()
            logger.info(f"Resuming from {last_item}, processed {start_idx} items")
        else:
            checkpoint.start_stage("judge")
    
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
            
            # Extract assistant turns for judgment
            assistant_content = extract_assistant_turns(scene)
            
            # Determine which judgment to run
            judgments = {}
            
            if config["judge_type"] in ["behavioral", "both"]:
                response = provider.generate(
                    prompt=assistant_content,
                    system=behavioral_system,
                    max_tokens=config["max_tokens"],
                    temperature=config["temperature"],
                )
                if response.success:
                    judgments["behavioral"] = response.content
            
            if config["judge_type"] in ["style", "both"]:
                response = provider.generate(
                    prompt=assistant_content,
                    system=style_system,
                    max_tokens=config["max_tokens"],
                    temperature=config["temperature"],
                )
                if response.success:
                    judgments["style"] = response.content
            
            # Format result
            result = {
                "scene_id": scene_id,
                "card_id": scene.card_id,
                "metadata": scene.metadata,
                "scene_stats": {
                    "turn_count": scene.turn_count,
                    "total_word_count": scene.total_word_count,
                    "total_token_count": scene.total_token_count,
                },
                "judgments": judgments,
            }
            
            # Save
            output_file = output_dir / f"{scene_id}.judgment.jsonl"
            with open(output_file, 'w') as f:
                f.write(json.dumps(result) + '\n')
            
            logger.info(f"Judged {scene_id} with {len(judgments)} judgment(s)")
            
            successful += 1
            if checkpoint:
                checkpoint.update(scene_id, True)
            
        except Exception as e:
            logger.error(f"Error judging {file_path}: {e}")
            failed += 1
            if checkpoint:
                checkpoint.update(scene_id, False)
        
        processed += 1
    
    # Mark complete
    if checkpoint:
        checkpoint.mark_complete()
    
    duration = time.time() - start_time
    logger.log_pipeline(
        stage="judge",
        items_processed=processed,
        items_successful=successful,
        items_failed=failed,
        duration=duration,
    )
    
    logger.info(f"Judgment complete: {successful} successful, {failed} failed in {duration:.1f}s")


if __name__ == "__main__":
    main()
