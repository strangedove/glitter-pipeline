"""
Preference rewrite module for RP Pipeline.
Creates preference pairs by rewriting scenes with same trajectory but improved quality.

The PrefRewriter preserves:
- Same number of turns
- Same starting speaker
- Same turn order/sequence
- Same characters and their personalities
- Same scene setup/direction from the card
- Same core trajectory (what the scene is about)

While improving:
- Character agency (ASSISTANT takes more initiative)
- Forward momentum (scene goes somewhere)
- Emotional shifts
- Actions and dialogue carry meaning without narrator explaining
- Turn length variation (don't normalize)
- Prose quality (remove tics, improve sentence structure)
"""

from typing import Any, Optional, Tuple

from rp_pipeline.config.settings import get_settings, load_prompts
from rp_pipeline.data.schemas import Scene
from rp_pipeline.models.base import BaseModelProvider, ModelFactory, ModelResponse


class PrefRewriter:
    """
    Rewrites scenes for preference pair generation.
    Preserves structure while improving character agency, momentum, and prose.
    """

    def __init__(self, provider: Optional[BaseModelProvider] = None):
        """
        Initialize preference rewriter.

        Args:
            provider: Model provider to use. If None, uses default from config.
                     If no config available, provider remains None (lazy initialization).
        """
        self.settings = get_settings()
        self.prompts = load_prompts()
        self.provider = provider
        self._owns_provider = False

        if provider is None:
            try:
                rewrite_config = self.settings.get_model_config("rewriting")
                provider = ModelFactory.create(
                    rewrite_config.get("provider", "openrouter"),
                    rewrite_config
                )
                self.provider = provider
                self._owns_provider = True
            except (ValueError, KeyError):
                # No valid config, will need provider set later
                self.provider = None
                self._owns_provider = False

    def rewrite_scene(
        self,
        scene: Scene,
        judge_feedback: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[Scene], ModelResponse]:
        """
        Rewrite a scene preserving structure but improving quality.

        Args:
            scene: Scene to rewrite
            judge_feedback: Optional judge feedback to incorporate
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
                self._owns_provider = True
            except (ValueError, KeyError):
                # Still no provider - return error
                from rp_pipeline.models.base import ModelResponse
                return None, ModelResponse(
                    success=False,
                    content="",
                    error="No model provider configured for PrefRewriter"
                )

        # Get the preference rewrite prompt
        pref_rewrite_system = self.prompts.get("pref_rewrite_system")
        if not pref_rewrite_system:
            # Fallback if prompt not found
            pref_rewrite_system = self._get_default_pref_prompt()

        # Add judge feedback if provided
        if judge_feedback:
            pref_rewrite_system += f"\n\nJUDGE FEEDBACK:\n{judge_feedback}"

        # Build constraints from original scene
        original_turn_count = scene.turn_count
        first_speaker = scene.turns[0].role if scene.turns else "USER"

        # Extract character names from metadata
        assistant_name = scene.metadata.get("assistant_name", "Assistant")
        user_name = scene.metadata.get("user_name", "User")
        card_id = scene.metadata.get("card_id", "unknown")

        # Build scene summary for context
        scene_summary = self._build_scene_summary(scene)

        # Add constraints to the prompt
        constraints = (
            f"\n\nCONSTRAINTS FOR THIS SCENE:\n"
            f"- Character card: {card_id}\n"
            f"- ASSISTANT: {assistant_name}\n"
            f"- USER: {user_name}\n"
            f"- Turn count: EXACTLY {original_turn_count} turns (no more, no less)\n"
            f"- First speaker: {first_speaker}\n"
            f"- Turn order: Must alternate properly\n"
            f"- Scene summary/trajectory: {scene_summary}\n\n"
            f"IMPORTANT: Rewrite the conversation below following ALL constraints above.\n"
            f"The scene does NOT need to finish - it can be an improved start.\n"
            f"Keep the same core trajectory but improve how it's presented."
        )

        full_system = pref_rewrite_system + constraints

        # Prepare the prompt with the original conversation
        conversation_text = scene.conversation

        rewrite_config = self.settings.get_model_config("rewriting")
        max_tokens = kwargs.get("max_tokens", rewrite_config.get("max_tokens", 5000))
        temperature = kwargs.get("temperature", rewrite_config.get("temperature", 0.7))

        response = self.provider.generate(
            prompt=conversation_text,
            system=full_system,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

        if not response.success:
            return None, response

        # Parse rewritten scene
        # Import here to avoid circular import
        from rp_pipeline.core.generation import SceneGenerator

        rewritten_scene = SceneGenerator._parse_conversation(
            response.content,
            None,
            assistant_name,
            user_name,
        )

        # Copy metadata from original
        rewritten_scene.metadata = scene.metadata.copy()

        # Add preference rewrite metadata
        rewritten_scene.metadata["pref_rewrite"] = True
        rewritten_scene.metadata["original_turn_count"] = original_turn_count
        rewritten_scene.metadata["original_scene_id"] = scene.card_id

        return rewritten_scene, response

    def _get_default_pref_prompt(self) -> str:
        """Get default preference rewrite prompt if not in config."""
        return """
You are revising a roleplay conversation to improve its quality while PRESERVING 
the core trajectory and setup.

CONSTRAINTS (MUST FOLLOW):
- Keep the EXACT same number of turns as the original
- Keep the SAME starting speaker (first turn role)
- Keep the SAME turn order/sequence (USER/ASSISTANT alternation)
- Keep the SAME characters and their established personalities
- Keep the SAME scene setup/direction from the card
- Keep the SAME core trajectory (what the scene is about)

IMPROVEMENTS TO MAKE:
- ASSISTANT character should take more initiative and drive the scene forward
- Each turn should change something - no spinning in place
- Emotions should shift in response to what happens
- Actions and dialogue should carry meaning without narrator explaining
- Scene should have forward momentum - something genuinely develops
- VARY TURN LENGTH with the beat (don't normalize toward uniform size)
- Some turns short and clipped, others longer as the moment dictates

PROSE FIXES:
- Remove narrator emotion-telling ("she felt", "he knew", etc.)
- Cut "though/despite/but" constructions that explain subtext
- Vary physical beats - don't rely on the same gestures repeatedly
- Don't wrap up neatly - leave tension unresolved
- Do NOT start turns with "I didn't look up" or similar - vary openings
- Reduce italicized emphasis - only when stress genuinely changes meaning
- Do NOT tag dialogue with voice descriptions ("said, his voice dropping")
- DIVERSIFY SENTENCE OPENERS - never start 3+ consecutive sentences with same pronoun

IMPORTANT: The scene does NOT need to finish within the turn limit. 
It can be an improved start that leaves room for continuation.

Write the full revised conversation with [USER - Turn N] and [ASSISTANT - Turn N] labels.
"""

    def _build_scene_summary(self, scene: Scene) -> str:
        """
        Build a brief summary of the scene's trajectory.

        Args:
            scene: Scene to summarize

        Returns:
            Brief summary string
        """
        if not scene.turns:
            return "Empty scene"

        # Get character names
        assistant_name = scene.metadata.get("assistant_name", "Assistant")
        user_name = scene.metadata.get("user_name", "User")

        # Extract setting from first turn content
        first_content = scene.turns[0].content
        setting_hint = self._extract_setting(first_content)

        # Extract trajectory from last turn content
        last_content = scene.turns[-1].content
        trajectory_hint = self._extract_trajectory(last_content, assistant_name)

        return f"{assistant_name} and {user_name} {setting_hint} - {trajectory_hint}"

    def _extract_setting(self, text: str) -> str:
        """Extract setting description from text."""
        text_lower = text.lower()
        if any(w in text_lower for w in ["bay", "dock", "hangar"]):
            return "in a bay/industrial setting"
        if any(w in text_lower for w in ["room", "office", "chamber"]):
            return "in a room"
        if any(w in text_lower for w in ["station", "ship", "space"]):
            return "on a station/ship"
        if any(w in text_lower for w in ["forest", "woods", "tree"]):
            return "in a forest/outdoors"
        if any(w in text_lower for w in ["street", "alley", "road"]):
            return "on a street"
        return "in a scene"

    def _extract_trajectory(self, text: str, assistant_name: str) -> str:
        """Extract trajectory/goal from text."""
        text_lower = text.lower()
        if any(w in text_lower for w in ["fix", "repair", "mend"]):
            return f"{assistant_name} needs to fix something"
        if any(w in text_lower for w in ["explain", "answer", "clarify"]):
            return f"{assistant_name} needs to explain something"
        if any(w in text_lower for w in ["help", "assist", "support"]):
            return f"{assistant_name} needs help with something"
        if any(w in text_lower for w in ["find", "locate", "discover"]):
            return f"{assistant_name} is searching for something"
        if any(w in text_lower for w in ["escape", "flee", "run"]):
            return f"{assistant_name} needs to escape"
        if any(w in text_lower for w in ["convince", "persuade", "argue"]):
            return f"{assistant_name} is trying to convince someone"
        return f"{assistant_name} is in a situation that needs resolution"

    def create_preference_pair(
        self,
        original_scene: Scene,
        judge_feedback: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[Scene], Optional[Scene], ModelResponse]:
        """
        Create a preference pair: original + rewritten version.

        Args:
            original_scene: The original first-draft scene
            judge_feedback: Optional judge feedback to guide rewriting
            **kwargs: Additional arguments for rewriting

        Returns:
            Tuple of (original_scene, rewritten_scene, ModelResponse)
        """
        rewritten_scene, response = self.rewrite_scene(
            original_scene,
            judge_feedback=judge_feedback,
            **kwargs
        )

        if rewritten_scene is None:
            return original_scene, None, response

        return original_scene, rewritten_scene, response
