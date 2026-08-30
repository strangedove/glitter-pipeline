#!/usr/bin/env python3
"""
CLI script for generating RP scenes from character cards.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


from rp_pipeline.config.settings import get_settings, reset_settings
from rp_pipeline.core.generation import SceneGenerator
from rp_pipeline.data.cards import CardDatabase
from rp_pipeline.data.schemas import CharacterCard, Scene
from rp_pipeline.utils.caching import PipelineCheckpoint, get_disk_cache
from rp_pipeline.utils.logging import StructuredLogger


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate RP scenes from character cards"
    )
    
    # Input/Output
    parser.add_argument(
        "--cards", "-c",
        type=str,
        default=None,
        help="Path to cards JSONL file or directory (overrides config)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for generated scenes (overrides config)"
    )
    
    # Generation options
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model to use for generation (overrides config)"
    )
    parser.add_argument(
        "--provider", "-p",
        type=str,
        default=None,
        help="Provider to use (openrouter, featherless, nvidia)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=None,
        help="Number of cards to process in batch (default: from config)"
    )
    parser.add_argument(
        "--variants", "-v",
        type=int,
        default=None,
        help="Variants to generate per card (default: from config)"
    )
    parser.add_argument(
        "--target-turns", "-t",
        type=int,
        default=None,
        help="Target number of turns per scene (default: from config)"
    )
    parser.add_argument(
        "--turn-length", "-l",
        type=str,
        choices=["short", "medium", "long"],
        default=None,
        help="Turn length strategy (default: from config)"
    )
    parser.add_argument(
        "--either-opener",
        action="store_true",
        default=None,
        help="Allow either character to open the scene"
    )
    parser.add_argument(
        "--no-either-opener",
        action="store_false",
        dest="either_opener",
        help="Force USER to open the scene"
    )
    
    # Model parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens for generation (default: from config)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for generation (default: from config)"
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
    if args.config:
        # Reset settings to use custom config
        reset_settings()
        # This will be picked up by get_settings()
        pass
    
    settings = get_settings()
    
    # Build effective config
    config = {
        "cards_path": args.cards or settings.paths.get("input", {}).get("default_cards"),
        "output_dir": args.output or settings.pipeline.get("generate", {}).get("output_dir", "data/output/raw"),
        "model": args.model,
        "provider": args.provider,
        "batch_size": args.batch_size or settings.generation.get("cards_per_batch", 5),
        "variants_per_card": args.variants or settings.generation.get("variants_per_card", 3),
        "target_turns": args.target_turns or settings.generation.get("target_turns", 8),
        "turn_length": args.turn_length or settings.generation.get("turn_length", "long"),
        "either_opener": (
            args.either_opener
            if args.either_opener is not None
            else settings.generation.get("either_opener", True)
        ),
        "max_tokens": args.max_tokens or settings.defaults.get("generation", {}).get("max_tokens", 4096),
        "temperature": args.temperature or settings.defaults.get("generation", {}).get("temperature", 0.85),
        "checkpoint_enabled": not args.no_checkpoint,
        "resume": args.resume,
        "log_level": args.log_level,
        "log_format": args.log_format or settings.logging.get("format", "json"),
    }
    
    return config


def format_scene_oai(scene: Scene) -> Dict[str, Any]:
    """Format a scene in OpenAI messages format."""
    messages = []
    for turn in scene.turns:
        role = turn.role.lower()
        messages.append({
            "role": role,
            "content": turn.content,
        })
    
    return {
        "id": scene.card_id or f"scene_{hash(str(scene.conversation)) % 10000:04d}",
        "card_id": scene.card_id,
        "messages": messages,
        "metadata": {
            **scene.metadata,
            "turn_count": scene.turn_count,
            "total_word_count": scene.total_word_count,
            "total_token_count": scene.total_token_count,
        },
    }


def main():
    """Main entry point."""
    args = parse_args()
    config = load_config(args)
    
    # Set up logging
    logger = StructuredLogger()
    logger.log("info", "Starting generation", **config)
    
    # Load cards
    cards_path = config["cards_path"]
    if not cards_path:
        logger.error("No cards path specified in config or --cards argument")
        sys.exit(1)
    
    card_db = CardDatabase(cards_path)
    if len(card_db) == 0:
        logger.error(f"No cards found at {cards_path}")
        sys.exit(1)
    
    logger.info(f"Loaded {len(card_db)} cards from {cards_path}")
    
    # Set up output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up generator
    gen_kwargs: Dict[str, Any] = {
        "turn_length": config["turn_length"],
        "target_turns": config["target_turns"],
        "either_opener": config["either_opener"],
    }
    
    # Override model config if specified
    if config["model"] or config["provider"]:
        from rp_pipeline.models.base import ModelFactory
        from rp_pipeline.config.settings import Settings
        
        settings = get_settings()
        model_config = settings.get_model_config("generation")
        
        if config["provider"]:
            model_config["provider"] = config["provider"]
        if config["model"]:
            model_config["model"] = config["model"]
        model_config["max_tokens"] = config["max_tokens"]
        model_config["temperature"] = config["temperature"]
        
        provider = ModelFactory.create(
            model_config.get("provider", "openrouter"),
            model_config
        )
        generator = SceneGenerator(provider=provider)
    else:
        generator = SceneGenerator()
    
    # Set up checkpoint
    checkpoint: Optional[PipelineCheckpoint] = None
    if config["checkpoint_enabled"]:
        checkpoint = PipelineCheckpoint(
            checkpoint_file=output_dir / ".." / ".." / "cache" / "generate_checkpoint.json"
        )
        if config["resume"] and checkpoint.is_resumable("generate"):
            start_idx, last_card = checkpoint.get_resume_position()
            logger.info(f"Resuming from card {last_card}, processed {start_idx} items")
        else:
            checkpoint.start_stage("generate")
    
    # Process cards
    processed = 0
    successful = 0
    failed = 0
    start_time = time.time()
    
    for card in card_db:
        card_id = f"{card.assistant_name}|{card.user_name}"
        
        for variant in range(config["variants_per_card"]):
            try:
                scene, response = generator.generate_scene(
                    card,
                    max_tokens=config["max_tokens"],
                    temperature=config["temperature"],
                    **gen_kwargs
                )
                
                if scene is None:
                    logger.warning(f"Failed to generate scene for {card_id} variant {variant}")
                    failed += 1
                    if checkpoint:
                        checkpoint.update(f"{card_id}_v{variant}", False)
                    continue
                
                # Format and save
                oai_scene = format_scene_oai(scene)
                output_file = output_dir / f"{card.assistant_name}_{card.user_name}_v{variant}.jsonl"
                
                with open(output_file, 'w') as f:
                    f.write(json.dumps(oai_scene) + '\n')
                
                logger.log_generation(
                    scene_id=oai_scene["id"],
                    card_id=card_id,
                    model=config["model"] or "default",
                    tokens_in=0,  # Would need to track from model
                    tokens_out=scene.total_token_count,
                    duration=0,
                    success=True,
                )
                
                successful += 1
                if checkpoint:
                    checkpoint.update(f"{card_id}_v{variant}", True)
                
            except Exception as e:
                logger.error(f"Error generating for {card_id} variant {variant}: {e}")
                failed += 1
                if checkpoint:
                    checkpoint.update(f"{card_id}_v{variant}", False)
            
            processed += 1
    
    # Mark complete
    if checkpoint:
        checkpoint.mark_complete()
    
    duration = time.time() - start_time
    logger.log_pipeline(
        stage="generate",
        items_processed=processed,
        items_successful=successful,
        items_failed=failed,
        duration=duration,
    )
    
    logger.info(f"Generation complete: {successful} successful, {failed} failed in {duration:.1f}s")


if __name__ == "__main__":
    main()
