#!/usr/bin/env python3
"""
CLI script for rewriting RP scenes using LLM for quality improvement.
This is a more advanced rewrite than the cleanup stage - focused on
improving prose quality, character voice, and scene structure.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


from rp_pipeline.config.settings import get_settings, load_prompts
from rp_pipeline.data.schemas import Scene, Turn
from rp_pipeline.core.generation import SceneGenerator
from rp_pipeline.core.verification import check_turn_labels
from rp_pipeline.models.base import ModelFactory
from rp_pipeline.utils.caching import PipelineCheckpoint
from rp_pipeline.utils.logging import StructuredLogger


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Rewrite RP scenes for quality improvement"
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
        help="Output directory for rewritten scenes (overrides config)"
    )
    
    # Rewrite options
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model to use for rewriting (overrides config)"
    )
    parser.add_argument(
        "--provider", "-p",
        type=str,
        default=None,
        help="Provider to use (openrouter, featherless, nvidia)"
    )
    parser.add_argument(
        "--style-rewrite",
        action="store_true",
        help="Style pass: fix only TIC-detected/style-judged issues, preserve events and structure"
    )
    parser.add_argument(
        "--judge-dir",
        type=str,
        default=None,
        help="Directory of *.judgment.jsonl behavioral verdicts. Gates rewriting: "
             "only scenes with VERDICT: NEEDS_REWRITE are rewritten; the verdict "
             "text is injected as feedback for the rewriter"
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Rewrite all scenes even when judge verdicts say PASS"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens for rewriting (default: from config)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for rewriting (default: from config)"
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
        "input_path": args.input or settings.pipeline.get("rewrite", {}).get("output_dir", "data/output/cleaned"),
        "output_dir": args.output or settings.pipeline.get("rewrite", {}).get("output_dir", "data/output/final"),
        "model": args.model,
        "provider": args.provider,
        "style_rewrite": args.style_rewrite,
        "judge_dir": args.judge_dir,
        "no_gate": args.no_gate,
        "max_tokens": args.max_tokens or settings.defaults.get("rewriting", {}).get("max_tokens", 5000),
        "temperature": args.temperature or settings.defaults.get("rewriting", {}).get("temperature", 0.7),
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


def format_scene_oai(scene: Scene) -> Dict[str, Any]:
    """Format a scene in OpenAI messages format."""
    messages = []
    for turn in scene.turns:
        role = turn.role.lower()
        messages.append({
            "role": role,
            "content": turn.content,
        })
    
    card_id = scene.card_id or scene.metadata.get("card_id")
    return {
        "id": card_id or f"scene_{hash(str(scene.conversation)) % 10000:04d}",
        "card_id": card_id,
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
    logger.log("info", "Starting rewrite", **config)
    
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
    
    logger.info(f"Found {len(input_files)} scene files to rewrite")
    
    # Set up model provider
    settings = get_settings()
    rewrite_config = settings.get_model_config("rewriting")
    
    if config["provider"]:
        rewrite_config["provider"] = config["provider"]
    if config["model"]:
        rewrite_config["model"] = config["model"]
    rewrite_config["max_tokens"] = config["max_tokens"]
    rewrite_config["temperature"] = config["temperature"]
    
    provider = ModelFactory.create(
        rewrite_config.get("provider", "openrouter"),
        rewrite_config
    )
    
    # Set up rewriter. Default: PrefRewriter (structure-preserving preference
    # rewrite: same turn count, same speaker order, edited content).
    # --style-rewrite keeps the legacy free-form rewrite_system flow.
    from rp_pipeline.core.pref_rewrite import PrefRewriter

    generator = SceneGenerator(provider=provider)
    pref_rewriter = None
    rewrite_system = ""
    if config["style_rewrite"]:
        prompts = load_prompts()
        rewrite_system = prompts.get("style_rewrite_system", "")
    else:
        pref_rewriter = PrefRewriter(provider)

    # Load behavioral verdicts for gating and feedback injection
    verdicts = {}
    if config["judge_dir"]:
        for jf in Path(config["judge_dir"]).glob("*.judgment.jsonl"):
            try:
                jrec = json.loads(jf.read_text().strip())
                beh = (jrec.get("judgments") or {}).get("behavioral") or ""
                verdicts[jf.stem.replace(".judgment", "")] = beh
            except (json.JSONDecodeError, OSError):
                continue
        logger.info(
            f"Loaded {len(verdicts)} behavioral verdicts from {config['judge_dir']}"
        )

    def gate_scene(scene_id: str):
        """Return (should_rewrite, feedback_text) for the scene."""
        verdict = verdicts.get(scene_id)
        if verdict is None:
            if config["judge_dir"] and not config["no_gate"]:
                # No verdict for this scene: can't confirm it's a problem.
                return False, None
            return True, None
        if "VERDICT: NEEDS_REWRITE" in verdict or config["no_gate"]:
            return True, verdict
        return False, None

    def style_critique(scene) -> str:
        """Build the flagged-issues critique for the style pass."""
        from rp_pipeline.core.analysis import TicDetector

        detector = TicDetector()
        text = "\n\n".join(t.content for t in scene.turns if t.role == "ASSISTANT")
        found = detector.detect_tics(text)
        lines = []
        for cat, matches in found.items():
            if matches:
                lines.append(f"- {cat}: {', '.join(repr(m) for m in matches[:6])}")
        return "\n".join(lines) if lines else ""

    # Set up checkpoint
    checkpoint: Optional[PipelineCheckpoint] = None
    if config["checkpoint_enabled"]:
        checkpoint = PipelineCheckpoint(
            checkpoint_file=output_dir / ".." / ".." / "cache" / "rewrite_checkpoint.json"
        )
        if config["resume"] and checkpoint.is_resumable("rewrite"):
            start_idx, last_item = checkpoint.get_resume_position()
            logger.info(f"Resuming from {last_item}, processed {start_idx} items")
        else:
            checkpoint.start_stage("rewrite")
    
    # Process files
    processed = 0
    successful = 0
    failed = 0
    start_time = time.time()
    
    for file_path in input_files:
        scene_id = file_path.stem
        processed += 1
        
        try:
            # Load scene
            scene = load_scene_from_jsonl(file_path)
            if scene is None:
                logger.warning(f"Failed to load scene from {file_path}")
                failed += 1
                if checkpoint:
                    checkpoint.update(scene_id, False)
                continue
            
            # Gate: only spend rewrite compute on scenes that need it.
            # Behavioral pass: judge verdict gates + feedback.
            # Style pass: TIC findings gate + critique injection.
            feedback = None
            if config["style_rewrite"]:
                critique = style_critique(scene)
                if not critique and not config["no_gate"]:
                    logger.info(f"Skipping {scene_id}: no style issues detected")
                    successful += 1
                    if checkpoint:
                        checkpoint.update(scene_id, True)
                    continue
                if critique:
                    feedback = (
                        "STYLE CRITIQUE - fix ONLY these flagged issues, change "
                        "nothing else:\n" + critique
                    )
            elif not config["no_gate"]:
                should, feedback = gate_scene(scene_id)
                if not should:
                    logger.info(f"Skipping {scene_id}: behavioral verdict PASS or missing")
                    successful += 1
                    if checkpoint:
                        checkpoint.update(scene_id, True)
                    continue

            if pref_rewriter is not None:
                rewritten_scene, response = pref_rewriter.rewrite_scene(
                    scene,
                    judge_feedback=feedback,
                    max_tokens=config["max_tokens"],
                    temperature=config["temperature"],
                )
            else:
                system = rewrite_system + (f"\n\n{feedback}" if feedback else "")
                response = provider.generate(
                    prompt=scene.conversation,
                    system=system,
                    max_tokens=config["max_tokens"],
                    temperature=config["temperature"],
                )
                if not response.success:
                    rewritten_scene = None
                else:
                    rewritten_scene = generator._parse_conversation(
                        response.content,
                        scene.metadata.get("card"),
                        scene.metadata.get("assistant_name", "Assistant"),
                        scene.metadata.get("user_name", "User"),
                    )
                    rewritten_scene.metadata = scene.metadata.copy()

            if not response.success or rewritten_scene is None:
                logger.warning(f"Failed to rewrite {scene_id}: {response.error}")
                failed += 1
                if checkpoint:
                    checkpoint.update(scene_id, False)
                continue

            # Structural verifier: a rewrite that drifts on turn count or
            # speaker order breaks the preference-pair contract — reject it.
            orig_roles = [t.role for t in scene.turns]
            if len(new_roles) != len(orig_roles) or new_roles != orig_roles:
                logger.warning(
                    f"Rejected rewrite {scene_id}: turn structure drifted "
                    f"({len(orig_roles)}->{len(new_roles)} turns)"
                )
                failed += 1
                if checkpoint:
                    checkpoint.update(scene_id, False)
                continue

            # Turn-label verifier: reject rewrites that poach the other
            # speaker's turns or leak role/turn labels into turn content.
            label_issues = check_turn_labels(rewritten_scene)
            if label_issues["poaching"] or label_issues["label_leak"]:
                kind = "poaching" if label_issues["poaching"] else "label leak"
                logger.warning(
                    f"Rejected rewrite {scene_id}: turn-label {kind} "
                    f"({(label_issues['poaching'] + label_issues['label_leak'])[:1]})"
                )
                failed += 1
                if checkpoint:
                    checkpoint.update(scene_id, False)
                continue
                logger.warning(
                    f"Rejected rewrite {scene_id}: turn structure drifted "
                    f"({len(orig_roles)}->{len(new_roles)} turns)"
                )
                failed += 1
                if checkpoint:
                    checkpoint.update(scene_id, False)
                continue
            
            result = format_scene_oai(rewritten_scene)
            output_file = output_dir / f"{scene_id}.rewritten.jsonl"
            
            with open(output_file, 'w') as f:
                f.write(json.dumps(result) + '\n')
            
            logger.info(f"Rewrote {scene_id}: {scene.total_word_count} -> {rewritten_scene.total_word_count} words")
            
            successful += 1
            if checkpoint:
                checkpoint.update(scene_id, True)
            
        except Exception as e:
            logger.error(f"Error rewriting {file_path}: {e}")
            failed += 1
            if checkpoint:
                checkpoint.update(scene_id, False)
    
    # Mark complete
    if checkpoint:
        checkpoint.mark_complete()
    
    duration = time.time() - start_time
    logger.log_pipeline(
        stage="rewrite",
        items_processed=processed,
        items_successful=successful,
        items_failed=failed,
        duration=duration,
    )
    
    logger.info(f"Rewrite complete: {successful} successful, {failed} failed in {duration:.1f}s")


if __name__ == "__main__":
    main()
