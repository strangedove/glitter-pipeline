"""
Cleanup module for RP Pipeline.
Handles targeted rewriting to remove tics and improve quality.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from rp_pipeline.config.settings import get_settings, load_prompts
from rp_pipeline.data.schemas import (
    CharacterCard,
    CleanupResult,
    Scene,
    TicDetectionResult,
    Turn,
)
from rp_pipeline.models.base import BaseModelProvider, ModelFactory, ModelResponse
from rp_pipeline.core.pref_rewrite import PrefRewriter


class TicRemover:
    """
    Removes specific tic patterns from text.
    """
    
    # Patterns for direct text replacement
    REPLACEMENT_PATTERNS = [
        # Remove emotion telling
        (r'\b(felt|feels|feeling)\s+', ''),
        (r'\b(knew|knows|knowing)\s+', ''),
        (r'\b(realized|realizes|realizing)\s+', ''),
        (r'\b(understood|understands)\s+', ''),
        
        # Remove lazy comparisons
        (r'\blike\s+a\s+\w+', ''),
        (r'\bthe\s+way\s+a\s+\w+', ''),
        
        # Remove dialogue tags
        (r'\b(said,\s+(her|his|voice)\s+\w+)\b', 'said'),
        (r'\b(asked,\s+(her|his|voice)\s+\w+)\b', 'asked'),
    ]
    
    def __init__(self):
        """Initialize tic remover."""
        self.settings = get_settings()
        self.prompts = load_prompts()
    
    def remove_tics(
        self,
        text: str,
        tic_result: Optional[TicDetectionResult] = None,
    ) -> Tuple[str, Dict[str, int]]:
        """
        Remove detected tics from text.
        
        Args:
            text: Text to clean
            tic_result: Optional tic detection result to guide cleanup
        
        Returns:
            Tuple of (cleaned_text, tics_removed_counts)
        """
        cleaned = text
        counts: Dict[str, int] = {}
        
        # Apply replacement patterns
        for pattern, replacement in self.REPLACEMENT_PATTERNS:
            compiled = re.compile(pattern, re.IGNORECASE)
            original = cleaned
            cleaned = compiled.sub(replacement, cleaned)
            count = original.count(re.sub(r'\\b|\\s\+', '', pattern))
            if count > 0:
                counts[pattern] = count
        
        # If we have tic detection result, target specific categories
        if tic_result:
            # Remove emotion telling
            emotion_patterns = [
                r'\b(felt|feels|feeling)\s+\w+',
                r'\b(was|were|is|are)\s+\w+\s+({"|".join(TicDetector.EMOTION_WORDS)})\b',
            ]
            for pattern in emotion_patterns:
                compiled = re.compile(pattern, re.IGNORECASE)
                cleaned = compiled.sub('', cleaned)
                if pattern not in counts:
                    counts[pattern] = 0
                counts[pattern] += 1
        
        return cleaned, counts
    
    def fix_pronoun_repetition(
        self,
        text: str,
        max_consecutive: int = 3,
    ) -> str:
        """
        Fix repeated sentence openers.
        
        Args:
            text: Text to fix
            max_consecutive: Maximum allowed consecutive same openers
        
        Returns:
            Fixed text
        """
        sentences = re.split(r'([.!?])', text)
        
        # Reconstruct with sentence boundaries
        current_opener: Optional[str] = None
        streak = 0
        
        for i, part in enumerate(sentences):
            if not part or part in '.!?':
                continue
            
            first_word = part.split()[0] if part.split() else ""
            first_word_clean = re.sub(r'[^\w]', '', first_word).lower()
            
            if first_word_clean == current_opener:
                streak += 1
                if streak >= max_consecutive:
                    # Replace with alternative opener
                    sentences[i] = self._get_alternative_opener(part)
                    streak = 0
            else:
                current_opener = first_word_clean
                streak = 1
        
        # Reconstruct text
        result = []
        for i, part in enumerate(sentences):
            result.append(part)
            if i < len(sentences) - 1 and sentences[i+1] in '.!?':
                result.append(sentences[i+1])
        
        return ''.join(result)
    
    def _get_alternative_opener(self, sentence: str) -> str:
        """Get an alternative sentence opener."""
        # Simple approach: prepend with action or dialogue
        if sentence.startswith(('He ', 'She ', 'They ', 'The ', 'I ')):
            # Extract the rest
            rest = sentence[2:] if sentence.startswith('I ') else sentence[3:]
            # Prepend with action
            return f"Looking around, {rest}"
        return sentence


class SceneRewriter:
    """
    Uses LLM to rewrite scenes for quality improvement.
    Legacy rewriter - preserved for backwards compatibility.
    """
    
    def __init__(self, provider: Optional[BaseModelProvider] = None):
        """
        Initialize scene rewriter.
        
        Args:
            provider: Model provider to use. If None, uses default from config.
                     If no config available, provider remains None (lazy initialization).
        """
        self.settings = get_settings()
        self.prompts = load_prompts()
        self.provider = provider
        
        if provider is None:
            try:
                rewrite_config = self.settings.get_model_config("rewriting")
                provider = ModelFactory.create(
                    rewrite_config.get("provider", "openrouter"),
                    rewrite_config
                )
                self.provider = provider
            except (ValueError, KeyError):
                # No valid config, will need provider set later
                self.provider = None
    
    def rewrite_scene(
        self,
        scene: Scene,
        tic_result: Optional[TicDetectionResult] = None,
        quality_issues: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[Scene], ModelResponse]:
        """
        Rewrite a scene to fix identified issues.
        
        Args:
            scene: Scene to rewrite
            tic_result: Optional tic detection result
            quality_issues: Optional quality analysis issues
            **kwargs: Additional arguments for model call
        
        Returns:
            Tuple of (rewritten_scene, ModelResponse)
        """
        # Lazy initialize provider if needed
        if self.provider is None:
            try:
                rewrite_config = self.settings.get_model_config("rewriting")
                self.provider = ModelFactory.create(
                    rewrite_config.get("provider", "openrouter"),
                    rewrite_config
                )
            except (ValueError, KeyError):
                from rp_pipeline.models.base import ModelResponse
                return None, ModelResponse(
                    success=False,
                    content="",
                    error="No model provider configured for SceneRewriter"
                )

        # Build feedback for the model
        feedback_parts = []
        
        if tic_result:
            if tic_result.emotion_tells:
                feedback_parts.append(
                    "EMOTION TELLING: Remove narrator explanations of emotions. "
                    f"Found: {', '.join(tic_result.emotion_tells[:5])}"
                )
            if tic_result.tics:
                tic_summary = ", ".join(
                    f"{k} ({v}x)" for k, v in tic_result.tics.items()
                )
                feedback_parts.append(f"TICS: {tic_summary}")
        
        if quality_issues:
            if not quality_issues.get("turn_variety", {}).get("passes"):
                feedback_parts.append(
                    "TURN VARIETY: Vary turn lengths. "
                    f"Issues: {', '.join(quality_issues['turn_variety']['issues'])}"
                )
            if not quality_issues.get("responsiveness", {}).get("passes"):
                feedback_parts.append(
                    "RESPONSIVENESS: Each turn must respond to the previous. "
                    f"Issues: {', '.join(quality_issues['responsiveness']['issues'])}"
                )
        
        # Get rewrite prompt
        rewrite_system = self.prompts.get("rewrite_system", "")
        
        # Add specific feedback
        if feedback_parts:
            rewrite_system += "\n\nSPECIFIC FEEDBACK:\n"
            for fb in feedback_parts:
                rewrite_system += f"- {fb}\n"
        
        # Prepare conversation for rewrite
        conversation_text = scene.conversation
        
        rewrite_config = self.settings.get_model_config("rewriting")
        max_tokens = kwargs.get("max_tokens", rewrite_config.get("max_tokens", 5000))
        temperature = kwargs.get("temperature", rewrite_config.get("temperature", 0.7))
        
        response = self.provider.generate(
            prompt=conversation_text,
            system=rewrite_system,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        if not response.success:
            return None, response
        
        # Parse rewritten scene
        rewritten_scene = SceneGenerator(provider=self.provider)._parse_conversation(
            response.content,
            scene.metadata.get("card"),
            scene.metadata.get("assistant_name", "Assistant"),
            scene.metadata.get("user_name", "User"),
        )
        
        # Copy metadata
        rewritten_scene.metadata = scene.metadata.copy()
        
        return rewritten_scene, response


class SceneCleaner:
    """
    Combined scene cleaner that uses both direct replacement and LLM rewriting.
    Uses PrefRewriter by default for preference-aligned rewrites.
    """
    
    def __init__(self, provider: Optional[BaseModelProvider] = None, use_pref_rewriter: bool = True):
        """
        Initialize scene cleaner.
        
        Args:
            provider: Model provider for rewriting
            use_pref_rewriter: Whether to use PrefRewriter (default True)
                             Set to False to use legacy SceneRewriter
        """
        self.tic_remover = TicRemover()
        self.settings = get_settings()
        
        # Use PrefRewriter by default for preference-aligned rewrites
        if use_pref_rewriter:
            self.rewriter = PrefRewriter(provider)
        else:
            self.rewriter = SceneRewriter(provider)
    
    def clean_scene(
        self,
        scene: Scene,
        tic_result: Optional[TicDetectionResult] = None,
        quality_issues: Optional[Dict[str, Any]] = None,
        use_rewrite: bool = True,
        **kwargs: Any,
    ) -> CleanupResult:
        """
        Clean a scene by removing tics and optionally rewriting.
        
        Args:
            scene: Scene to clean
            tic_result: Optional tic detection result
            quality_issues: Optional quality analysis issues
            use_rewrite: Whether to use LLM rewriting for complex issues
            **kwargs: Additional arguments for rewriting
        
        Returns:
            CleanupResult with original and cleaned scene
        """
        changes_made: List[str] = []
        tics_removed: Dict[str, int] = {}
        
        # First, try direct tic removal on each turn
        cleaned_turns = []
        for turn in scene.turns:
            cleaned_content, removed = self.tic_remover.remove_tics(
                turn.content,
                tic_result
            )
            tics_removed.update(removed)
            
            if cleaned_content != turn.content:
                changes_made.append(
                    f"Turn {turn.turn_number}: removed {len(removed)} tics"
                )
            
            # Fix pronoun repetition
            fixed_content = self.tic_remover.fix_pronoun_repetition(cleaned_content)
            if fixed_content != cleaned_content:
                changes_made.append(
                    f"Turn {turn.turn_number}: fixed pronoun repetition"
                )
                cleaned_content = fixed_content
            
            # Create new turn with cleaned content
            cleaned_turn = Turn(
                role=turn.role,
                turn_number=turn.turn_number,
                content=cleaned_content,
                word_count=len(cleaned_content.split()),
                token_count=int(len(cleaned_content.split()) * 1.3),
            )
            cleaned_turns.append(cleaned_turn)
        
        # Build cleaned scene
        cleaned_scene = Scene(
            card_id=scene.card_id,
            conversation=scene.conversation,  # Will rebuild
            turns=cleaned_turns,
            assistant_turns=[
                t.content for t in cleaned_turns if t.role == "ASSISTANT"
            ],
            metadata=scene.metadata.copy(),
            total_word_count=sum(t.word_count for t in cleaned_turns),
            total_token_count=sum(t.token_count for t in cleaned_turns),
            turn_count=len(cleaned_turns),
        )
        
        # Rebuild conversation text
        cleaned_scene.conversation = self._rebuild_conversation(cleaned_turns)
        
        # If we have significant issues and rewrite is enabled, use LLM
        if use_rewrite and self._needs_rewrite(tic_result, quality_issues):
            # Extract judge feedback from quality_issues if present
            judge_feedback = quality_issues.get("judge_feedback") if quality_issues else None
            
            # PrefRewriter expects judge_feedback as a separate arg
            # SceneRewriter expects tic_result and quality_issues
            if hasattr(self.rewriter, 'rewrite_scene'):
                if isinstance(self.rewriter, PrefRewriter):
                    rewritten_scene, response = self.rewriter.rewrite_scene(
                        cleaned_scene,
                        judge_feedback=judge_feedback,
                        **kwargs
                    )
                else:
                    # Legacy SceneRewriter
                    rewritten_scene, response = self.rewriter.rewrite_scene(
                        cleaned_scene,
                        tic_result,
                        quality_issues,
                        **kwargs
                    )
            
            if rewritten_scene:
                changes_made.append("LLM rewrite applied (PrefRewriter)" if isinstance(self.rewriter, PrefRewriter) else "LLM rewrite applied")
                cleaned_scene = rewritten_scene
        
        # Validate the cleaned scene
        validation_passed = self._validate_cleaned(cleaned_scene, scene)
        
        return CleanupResult(
            original_scene=scene,
            cleaned_scene=cleaned_scene,
            changes_made=changes_made,
            tics_removed=tics_removed,
            validation_passed=validation_passed,
        )
    
    def _needs_rewrite(
        self,
        tic_result: Optional[TicDetectionResult],
        quality_issues: Optional[Dict[str, Any]],
    ) -> bool:
        """Determine if LLM rewrite is needed."""
        if tic_result and tic_result.needs_cleanup:
            return True
        if quality_issues and not quality_issues.get("overall_pass", True):
            return True
        return False
    
    def _validate_cleaned(self, cleaned: Scene, original: Scene) -> bool:
        """Validate that cleaned scene is acceptable."""
        # Check turn count matches
        if cleaned.turn_count != original.turn_count:
            return False
        
        # Check all turns present
        original_numbers = {t.turn_number for t in original.turns}
        cleaned_numbers = {t.turn_number for t in cleaned.turns}
        if original_numbers != cleaned_numbers:
            return False
        
        # Check word count is reasonable
        if cleaned.total_word_count < original.total_word_count * 0.5:
            return False
        
        return True
    
    def _rebuild_conversation(self, turns: List[Turn]) -> str:
        """Rebuild conversation text from cleaned turns."""
        parts = []
        for turn in turns:
            parts.append(f"[{turn.role} - Turn {turn.turn_number}]")
            parts.append(turn.content)
        return "\n".join(parts)


# Import SceneGenerator for parsing (circular import workaround)
from rp_pipeline.core.generation import SceneGenerator
