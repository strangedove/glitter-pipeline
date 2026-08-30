"""
Scene generation for RP Pipeline.
Handles the core generation logic using configured model providers.
"""

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
            opener = ("The scene may open with EITHER character — whichever the moment calls for; "
                     "many scenes naturally open with the ASSISTANT character greeting or setting the scene.")
        else:
            opener = "Start with USER."
        
        count = f"Write approximately {target} alternating turns in total (counting both speakers)"
        fmt_line = f"FORMAT:\n{count}. Label as [USER - Turn N] and [ASSISTANT - Turn N], numbered sequentially across both speakers (Turn 1, 2, 3, ...). {opener} {length_clause}"
        
        # Build full system prompt
        system_prompt = self.prompts.get("gen_system_base", "") + fmt_line
        
        # Add anti-tic recipe if configured
        if self.settings.get("COMBINED_RECIPE", "0") == "1":
            system_prompt += self.prompts.get("combined_recipe_block", "")
        
        # Generate
        max_tokens = kwargs.get("max_tokens", gen_config.get("max_tokens", 4096))
        temperature = kwargs.get("temperature", gen_config.get("temperature", 0.85))
        
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
        # Parse turns
        turns = self._parse_turns(conversation)
        
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
            },
            total_word_count=total_words,
            total_token_count=total_tokens,
            turn_count=len(turns),
        )
        
        return scene
    
    def _parse_turns(self, conversation: str) -> List[Turn]:
        """
        Parse conversation text into Turn objects.
        
        Args:
            conversation: Raw conversation text
        
        Returns:
            List of Turn objects
        """
        turns = []
        
        # Pattern to match turn labels: [USER - Turn N] or [ASSISTANT - Turn N]
        turn_pattern = re.compile(
            r'\[(USER|ASSISTANT)\s*-\s*Turn\s*(\d+)\s*\]'
        )
        
        # Split by turn labels
        parts = turn_pattern.split(conversation)
        
        # Process parts (skip first empty part, then alternate between role/number and content)
        for i in range(2, len(parts), 3):
            role = parts[i-2].strip()
            turn_num = int(parts[i-1].strip())
            content = parts[i].strip()
            
            # Clean up content (remove leading/trailing whitespace, newlines)
            content = content.replace("\n", " ").strip()
            
            if content:
                word_count = len(content.split())
                # Estimate token count (roughly 1.3 words per token on average)
                token_count = int(word_count * 1.3)
                
                turns.append(Turn(
                    role=role,
                    turn_number=turn_num,
                    content=content,
                    word_count=word_count,
                    token_count=token_count,
                ))
        
        return turns
    
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
            issues.append(f"Scene too short: {scene.total_word_count} words (min {min_words})")
        
        # Check maximum word count
        max_words = quality_config.get("max_token_count", 6144) * 1.3
        if scene.total_word_count > max_words:
            issues.append(f"Scene too long: {scene.total_word_count} words (max {max_words})")
        
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
            issues.append(f"Inconsistent turn numbering: expected {expected_turns}, got {actual_turns}")
        
        # Check for alternating roles
        if len(scene.turns) > 1:
            for i in range(1, len(scene.turns)):
                if scene.turns[i].role == scene.turns[i-1].role:
                    issues.append(f"Consecutive turns with same role: Turn {scene.turns[i].turn_number}")
                    break
        
        return len(issues) == 0, issues
