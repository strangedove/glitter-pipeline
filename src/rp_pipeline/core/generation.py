"""
Scene generation for RP Pipeline.
Handles the core generation logic using configured model providers.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from rp_pipeline.config.settings import get_settings, load_prompts
from rp_pipeline.data.cards import CardFormatter
from rp_pipeline.data.schemas import CharacterCard, Scene, Turn
from rp_pipeline.models.base import BaseModelProvider, ModelFactory, ModelResponse


class SceneGenerator:
    """
    Generates RP scenes from character cards.
    """

    def __init__(self, provider: Optional[BaseModelProvider] = None):
        """
        Initialize scene generator.

        Args:
            provider: Model provider to use. If None, uses default from config.
        """
        self.settings = get_settings()
        self.prompts = load_prompts()

        if provider is None:
            gen_config = self.settings.get_model_config("generation")
            provider = ModelFactory.create(
                gen_config.get("provider", "openrouter"),
                gen_config
            )
        self.provider = provider

        # Load turn length clauses
        self.turn_length_clauses = self.prompts.get("turn_length_clauses", {})

    def plan_directions(
        self,
        card: CharacterCard,
        n: int,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, str]], ModelResponse]:
        """
        Plan n fundamentally different scene trajectories for a card in a
        single LLM call (batch planning keeps the directions distinct).

        Returns (directions, response) where directions is a list of dicts
        with keys: direction, key_choice, emotional_arc, ending_state.
        Empty list on parse failure.
        """
        system = self.prompts.get("direction_system", "")
        if not system:
            return [], ModelResponse(
                content="", success=False, error="direction_system prompt missing"
            )
        system = system.replace("{n}", str(n))

        card_text, _, _ = CardFormatter.format_card(card)
        max_tokens = kwargs.pop("max_tokens", 3000)
        temperature = kwargs.pop("temperature", 0.9)
        response = self.provider.generate(
            prompt=card_text,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        if not response.success or not response.content:
            return [], response

        content = re.sub(r"```(?:json)?", "", response.content)
        m = re.search(r"\[.*\]", content, re.S)
        if not m:
            return [], response
        try:
            directions = json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            return [], response
        if not isinstance(directions, list):
            return [], response
        return [d for d in directions if isinstance(d, dict) and d.get("direction")], response

    def format_direction(self, d: Dict[str, str]) -> str:
        """Render a planned direction dict as generation-prompt text."""
        parts = ["TRAJECTORY OUTLINE (follow this direction for the scene):"]
        parts.append(f"- Direction: {d.get('direction', '')}")
        if d.get("key_choice"):
            parts.append(f"- Key divergence: a choice by the ASSISTANT character — {d['key_choice']}")
        if d.get("emotional_arc"):
            parts.append(f"- Emotional arc: {d['emotional_arc']}")
        if d.get("ending_state"):
            parts.append(f"- Ending state: {d['ending_state']}")
        return "\n".join(parts)

    def generate_scene(
        self,
        card: CharacterCard,
        direction: Optional[str] = None,
        turn_length: str = "long",
        target_turns: Optional[int] = None,
        either_opener: bool = True,
        **kwargs: Any,
    ) -> Tuple[Scene, ModelResponse]:
        """
        Generate a single scene from a character card.

        Args:
            card: Character card to use
            direction: Optional direction for the scene
            turn_length: Turn length strategy ("short", "medium", "long")
            target_turns: Target number of turns (overrides config)
            either_opener: Whether either character can open the scene
            **kwargs: Additional arguments for model call

        Returns:
            Tuple of (Scene, ModelResponse)
        """
        # Format the card
        card_text, assistant_name, user_name = CardFormatter.format_card(card)

        # Add direction if provided
        if direction:
            card_text = f"{card_text}\n\nDirection: {direction}"

        # Build the generation prompt
        gen_config = self.settings.generation
        target = target_turns or gen_config.get("target_turns", 8)

        # Get turn length clause
        length_clause = self.turn_length_clauses.get(
            turn_length,
            self.turn_length_clauses.get("long", "")
        )

        # Build the format line
        if either_opener:
            opener = (
                "The scene may open with EITHER character -- whichever the moment calls for; "
                "many scenes naturally open with the ASSISTANT character greeting or setting the scene."
            )
        else:
            opener = "Start with USER."

        count = (
            f"Write approximately {target} alternating turns in total "
            "(counting both speakers)"
        )
        fmt_line = (
            f"FORMAT:\n{count}. Label as [USER - Turn N] and "
            f"[ASSISTANT - Turn N], numbered sequentially across both "
            f"speakers (Turn 1, 2, 3, ...). {opener} {length_clause}"
        )

        # Build full system prompt
        gen_prompt_key = os.environ.get("RP_GEN_PROMPT_KEY", "gen_system_base")
        system_prompt = self.prompts.get(gen_prompt_key, "") + fmt_line

        # Add anti-tic recipe if configured
        if self.settings.get("COMBINED_RECIPE", "0") == "1":
            system_prompt += self.prompts.get("combined_recipe_block", "")

        # Generate
        max_tokens = kwargs.pop("max_tokens", gen_config.get("max_tokens", 4096))
        temperature = kwargs.pop("temperature", gen_config.get("temperature", 0.85))

        response = self.provider.generate(
            prompt=card_text,
            system=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

        if not response.success:
            return None, response

        # Parse the conversation
        scene = self._parse_conversation(
            response.content,
            card,
            assistant_name,
            user_name,
        )

        # Ensure scene ends with ASSISTANT turn for training
        scene = self._ensure_ends_with_assistant(
            scene, card, assistant_name, user_name, response
        )

        return scene, response

    def generate_batch(
        self,
        cards: List[CharacterCard],
        directions_per_card: int = 1,
        **kwargs: Any,
    ) -> List[Tuple[Scene, ModelResponse]]:
        """
        Generate multiple scenes from a batch of cards.

        Args:
            cards: List of character cards
            directions_per_card: Number of directions to generate per card
            **kwargs: Additional arguments for generate_scene

        Returns:
            List of (Scene, ModelResponse) tuples
        """
        results = []

        for card in cards:
            if directions_per_card <= 1:
                scene, response = self.generate_scene(card, **kwargs)
                if scene:
                    results.append((scene, response))
            else:
                # Generate multiple directions for this card
                # (Would need direction generation logic here)
                for i in range(directions_per_card):
                    scene, response = self.generate_scene(card, **kwargs)
                    if scene:
                        results.append((scene, response))

        return results

    def _parse_conversation(
        self,
        conversation: str,
        card: CharacterCard,
        assistant_name: str,
        user_name: str,
    ) -> Scene:
        """
        Parse a generated conversation into a Scene object.

        Args:
            conversation: Raw conversation text
            card: Source character card
            assistant_name: Name of assistant character
            user_name: Name of user character

        Returns:
            Parsed Scene object
        """
        # Parse turns; fallback when the model skipped the marker format
        # (common with reasoning models). Recover from raw prose first.
        # role_map lets the parser recognize character-name turn labels
        # ([Jubal - Turn 3], **Turn 4 — VERITY**, ...) as role markers.
        role_map: Dict[str, str] = {}
        for fragment, role in ((user_name, "USER"), (assistant_name, "ASSISTANT")):
            for part in re.split(r"\s+", fragment.strip()) if fragment else ():
                if len(part) >= 3:
                    role_map[part.lower()] = role
        turns, turn_parse = self._parse_turns(conversation, role_map), "native"
        if len(turns) < 4 and len(conversation.strip()) > 400:
            turns, turn_parse = self._rescue_turns(conversation, role_map)
            if turn_parse == "llm_structured" and len(turns) < 4:
                turns, turn_parse = self._synthesize_turns(conversation), "synthesized"
            if turn_parse == "llm_structured" and self._rescued_conversation:
                conversation = self._rescued_conversation

        # Calculate metrics
        total_words = sum(t.word_count for t in turns)
        total_tokens = sum(t.token_count for t in turns)

        # Extract assistant turns
        assistant_turns = [
            t.content for t in turns if t.role == "ASSISTANT"
        ]

        # Create scene
        scene = Scene(
            card_id=card.assistant_name if card else None,
            conversation=conversation,
            turns=turns,
            assistant_turns=assistant_turns,
            metadata={
                "card_id": card.assistant_name if card else None,
                "assistant_name": assistant_name,
                "user_name": user_name,
                "genre": card.genre if card else None,
                "tone": card.tone if card else None,
                "turn_parse": turn_parse,
            },
            total_word_count=total_words,
            total_token_count=total_tokens,
            turn_count=len(turns),
        )

        return scene

    def _rescue_turns(
        self, conversation: str, role_map: Optional[Dict[str, str]] = None
    ) -> Tuple[List[Turn], str]:
        """
        Recover turns from prose that ignored the marker format.

        1. Deterministic: split into paragraphs, batch them into turn-sized
           blocks, assign alternating roles.
        2. If that still yields <4 turns, ask the judge provider to insert
           markers, then re-parse.

        Returns (turns, provenance) where provenance is
        "synthesized" | "llm_structured" | "unrecoverable".
        """
        turns = self._synthesize_turns(conversation)
        if len(turns) >= 4:
            return turns, "synthesized"

        # LLM safety net: cheap structuring pass on the judge provider.
        try:
            from rp_pipeline.models.base import ModelFactory

            judge_cfg = dict(get_settings().get_model_config("judging"))
            if not judge_cfg.get("provider"):
                return turns, "unrecoverable"
            provider = ModelFactory.create(
                judge_cfg.get("provider", "featherless"), judge_cfg
            )
            response = provider.generate(
                prompt=(
                    "The following roleplay scene lost its turn markers. "
                    "Re-emit it EXACTLY as given — same words, same order, "
                    "no commentary — but insert turn markers so the speakers "
                    "strictly alternate. Each turn begins with a marker line "
                    "[USER - Turn N] or [ASSISTANT - Turn N] (sequential "
                    "numbering across both speakers), and each turn is 1-6 "
                    "paragraphs. Dialogue lines belong to whoever spoke them.\n\n"
                    "SCENE:\n\n" + conversation
                ),
                system=(
                    "You are a text-formatting tool. Re-emit the scene with "
                    "alternating turn markers. Do not rewrite, summarize, or "
                    "add anything."
                ),
                max_tokens=max(4096, int(len(conversation.split()) * 2.5)),
                temperature=0.1,
            )
            if response.success and response.content:
                llm_turns = self._parse_turns(response.content, role_map)
                if len(llm_turns) >= 4:
                    # Preserve the original text as the canonical conversation
                    rebuilt = "\n\n".join(
                        f"[{t.role} - Turn {i + 1}]\n{t.content}"
                        for i, t in enumerate(llm_turns)
                    )
                    for i, t in enumerate(llm_turns):
                        t.turn_number = i + 1
                    conversation = rebuilt
                    self._rescued_conversation = rebuilt
                    return llm_turns, "llm_structured"
        except Exception:
            pass
        return turns, "unrecoverable"

    def _synthesize_turns(self, conversation: str) -> List[Turn]:
        """
        Deterministic fallback: split prose into paragraphs, batch them into
        turn-sized blocks (>=80 words or a dialogue-heavy block, <=400 words),
        and assign alternating roles starting with USER.
        """
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", conversation) if p.strip()]
        if len(paragraphs) < 2:
            # split on single newlines as a weaker signal
            paragraphs = [p.strip() for p in conversation.split("\n") if p.strip()]
        if not paragraphs:
            return []

        blocks: List[List[str]] = []
        current: List[str] = []
        for p in paragraphs:
            has_dialogue = any(ch in p for ch in "\"“”")
            # A quoted-dialogue paragraph is a natural turn boundary;
            # narration accumulates into the current turn until then.
            if current and (has_dialogue or len(" ".join(current).split()) >= 400):
                blocks.append(current)
                current = []
            current.append(p)
        if current:
            blocks.append(current)

        # Merge down to a sane turn count (target_turns ± 2)
        target = int(self.settings.generation.get("target_turns", 8))
        while len(blocks) > target + 2 and len(blocks) > 2:
            sizes = [len(" ".join(b).split()) for b in blocks]
            i = min(range(len(blocks) - 1), key=lambda k: sizes[k] + sizes[k + 1])
            blocks[i:i + 2] = [blocks[i] + blocks[i + 1]]

        turns: List[Turn] = []
        for i, block in enumerate(blocks):
            role = "USER" if i % 2 == 0 else "ASSISTANT"
            content = "\n\n".join(block)
            turns.append(
                Turn(
                    role=role,
                    turn_number=i + 1,
                    content=content,
                    word_count=len(content.split()),
                    token_count=max(1, len(content) // 4),
                )
            )
        return turns

    def _parse_turns(
        self, conversation: str, role_map: Optional[Dict[str, str]] = None
    ) -> List[Turn]:
        """
        Parse conversation text into Turn objects.

        Accepts every label shape observed in the wild:
          [USER - Turn 3] / [assistant - Turn 3]      (case-insensitive)
          [Turn 3 - USER] / [Turn 3 — Assistant]      (reversed order)
          [Jubal - Turn 3] / [Turn 3 — VERITY]        (character-name labels,
          **Turn 3 — NAME**                            mapped via role_map)
          Turn 3 - NAME                               (bare header line)

        role_map maps lowercased name fragments to "USER"/"ASSISTANT".

        Args:
            conversation: Raw conversation text
            role_map: Optional name-to-role mapping from the card

        Returns:
            List of Turn objects
        """
        role_map = role_map or {}
        em = r"[-–—]"

        role_label = re.compile(
            rf"\[\s*(?:(USER|ASSISTANT)\s*{em}\s*Turn\s*(\d+)"
            rf"|Turn\s*(\d+)\s*{em}\s*(USER|ASSISTANT))\s*\]",
            re.I,
        )
        name_alt = "|".join(re.escape(n) for n in role_map) if role_map else r"\x00NOPE\x00"
        name_label = re.compile(
            rf"(?:\[\s*(?:(?P<n1>{name_alt})\s*{em}\s*Turn\s*(?P<num1>\d+)"
            rf"|Turn\s*(?P<num2>\d+)\s*{em}\s*(?P<n2>{name_alt}))\s*\]"
            rf"|\*\*\s*Turn\s*(?P<num3>\d+)\s*{em}\s*(?P<n3>{name_alt})\s*\*\*"
            rf"|^\s*Turn\s*(?P<num4>\d+)\s*{em}\s*(?P<n4>{name_alt})\s*$)",
            re.I | re.M,
        )
        # All label occurrences in one position-sorted list:
        # (position, end, role)
        marks: List[tuple] = []
        for m in role_label.finditer(conversation):
            role = (m.group(1) or m.group(4)).upper()
            marks.append((m.start(), m.end(), role))
        for m in name_label.finditer(conversation):
            name = m.group("n1") or m.group("n2") or m.group("n3") or m.group("n4")
            role = role_map.get(name.lower().strip())
            if role:
                marks.append((m.start(), m.end(), role))
        marks.sort()

        # Drop overlapping matches (a name match nested in a role match etc.)
        deduped: List[tuple] = []
        for s, e, role in marks:
            if deduped and s < deduped[-1][1]:
                continue
            deduped.append((s, e, role))
        marks = deduped

        turns: List[Turn] = []
        for i, (s, e, role) in enumerate(marks):
            end = marks[i + 1][0] if i + 1 < len(marks) else len(conversation)
            content = conversation[e:end]
            # Strip any residual label fragments (bold headers etc.)
            content = role_label.sub("", content)
            content = name_label.sub("", content)
            content = content.replace("\n", " ").strip()
            if not content:
                continue
            word_count = len(content.split())
            turns.append(
                Turn(
                    role=role,
                    turn_number=len(turns) + 1,
                    content=content,
                    word_count=word_count,
                    token_count=int(word_count * 1.3),
                )
            )
        return turns

    def _ensure_ends_with_assistant(
        self,
        scene: Scene,
        card: CharacterCard,
        assistant_name: str,
        user_name: str,
        response: ModelResponse,
    ) -> Scene:
        """
        Ensure scene ends with ASSISTANT turn for training.
        If it ends with USER, generate one more ASSISTANT turn to complete it.

        Args:
            scene: The parsed scene
            card: Source character card
            assistant_name: Name of assistant character
            user_name: Name of user character
            response: Original model response

        Returns:
            Scene guaranteed to end with ASSISTANT
        """
        if not scene.turns:
            return scene

        # Check if last turn is USER
        last_turn = scene.turns[-1]
        if last_turn.role == "ASSISTANT":
            return scene  # Already good

        # Need to add an ASSISTANT turn
        next_turn_num = scene.turn_count + 1
        last_user_content = last_turn.content

        continuation_prompt = (
            f"Continue as {assistant_name}. USER just said: {last_user_content}\n"
            f"Write one turn labeled [ASSISTANT - Turn {next_turn_num}]"
        )

        next_response = self.provider.generate(
            prompt=continuation_prompt,
            system="Respond as the ASSISTANT character. One turn only.",
            max_tokens=500,
            temperature=0.85,
        )

        if next_response.success and next_response.content:
            new_turns = self._parse_turns(next_response.content)
            if new_turns:
                # Add to scene
                scene.turns.extend(new_turns)
                scene.assistant_turns.append(new_turns[0].content)
                scene.turn_count = len(scene.turns)
                scene.total_word_count = sum(
                    t.word_count for t in scene.turns
                )
                scene.total_token_count = sum(
                    t.token_count for t in scene.turns
                )
                scene.conversation += (
                    f"\n[{new_turns[0].role} - Turn {next_turn_num}] "
                    f"{new_turns[0].content}"
                )

        return scene

    def validate_scene(self, scene: Scene) -> Tuple[bool, List[str]]:
        """
        Validate a generated scene.

        Args:
            scene: Scene to validate

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        quality_config = self.settings.quality

        # Check minimum word count
        min_words = quality_config.get("min_token_count", 50) * 0.77  # Rough words per token
        if scene.total_word_count < min_words:
            issues.append(
                f"Scene too short: {scene.total_word_count} words (min {min_words})"
            )

        # Check maximum word count
        max_words = quality_config.get("max_token_count", 6144) * 1.3
        if scene.total_word_count > max_words:
            issues.append(
                f"Scene too long: {scene.total_word_count} words (max {max_words})"
            )

        # Check minimum turn count
        min_turns = quality_config.get("min_turn_count", 4)
        if scene.turn_count < min_turns:
            issues.append(f"Too few turns: {scene.turn_count} (min {min_turns})")

        # Check maximum turn count
        max_turns = quality_config.get("max_turn_count", 12)
        if scene.turn_count > max_turns:
            issues.append(f"Too many turns: {scene.turn_count} (max {max_turns})")

        # Check for consistent turn numbering
        expected_turns = set(range(1, scene.turn_count + 1))
        actual_turns = {t.turn_number for t in scene.turns}
        if expected_turns != actual_turns:
            issues.append(
                f"Inconsistent turn numbering: expected {expected_turns}, got {actual_turns}"
            )

        # Check for alternating roles
        if len(scene.turns) > 1:
            for i in range(1, len(scene.turns)):
                if scene.turns[i].role == scene.turns[i - 1].role:
                    issues.append(
                        f"Consecutive turns with same role: Turn {scene.turns[i].turn_number}"
                    )
                    break

        return len(issues) == 0, issues
