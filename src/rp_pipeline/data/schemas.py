"""
Data schemas for RP Pipeline.
Uses Pydantic for validation and serialization.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class CharacterAppearance(BaseModel):
    """Physical appearance description for a character."""
    height: Optional[str] = None
    build: Optional[str] = None
    hair: Optional[str] = None  # color + length
    eyes: Optional[str] = None
    skin_tone: Optional[str] = None
    distinctive_features: Optional[List[str]] = None
    clothing: Optional[str] = None


class CharacterCard(BaseModel):
    """Character card for RP scene generation."""
    assistant_name: str = Field(..., min_length=1, description="Name of the AI character")
    user_name: str = Field(..., min_length=1, description="Name of the human character")
    assistant_character: str = Field(..., min_length=50, description="Full profile of AI character")
    user_character: str = Field(..., min_length=20, description="Profile of human character")
    scenario: str = Field(..., min_length=20, description="Scenario description")
    genre: str = Field(..., description="Literary genre")
    tone: str = Field(..., description="Desired tone")
    source_model: Optional[str] = Field(default=None, description="Model that generated this card")
    card_prompt: Optional[str] = Field(default=None, description="Generation mode")
    assistant_appearance: Optional[str] = Field(default=None, description="Physical appearance of AI character")
    user_appearance: Optional[str] = Field(default=None, description="Physical appearance of human character")
    
    @field_validator('assistant_character', 'user_character')
    @classmethod
    def validate_min_length(cls, v: str) -> str:
        if len(v.strip()) < 20:
            raise ValueError(f"Character description must be at least 20 characters, got {len(v)}")
        return v


class Direction(BaseModel):
    """A possible direction for a scene to take."""
    direction: str = Field(..., description="Description of this direction")
    key_choice: str = Field(..., description="Key character choice that defines this direction")
    emotional_arc: str = Field(..., description="Emotional journey")
    ending_state: str = Field(..., description="Final state of characters")


class Turn(BaseModel):
    """A single turn in a conversation."""
    role: str = Field(..., description="USER or ASSISTANT")
    turn_number: int = Field(..., ge=1, description="Turn number")
    content: str = Field(..., min_length=1, description="Turn text content")
    word_count: int = Field(..., ge=1, description="Word count")
    token_count: int = Field(..., ge=1, description="Token count")


class Scene(BaseModel):
    """A complete RP scene/conversation."""
    card_id: Optional[str] = Field(default=None, description="ID of source card")
    conversation: str = Field(..., description="Full conversation text")
    turns: List[Turn] = Field(default_factory=list, description="List of turns")
    assistant_turns: List[str] = Field(default_factory=list, description="Assistant turn texts")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Computed fields
    total_word_count: int = Field(default=0, description="Total words in scene")
    total_token_count: int = Field(default=0, description="Total tokens in scene")
    turn_count: int = Field(default=0, description="Number of turns")
    
    @property
    def assistant_word_counts(self) -> List[int]:
        """Get word counts for assistant turns only."""
        return [t.word_count for t in self.turns if t.role == "ASSISTANT"]


class SceneBatch(BaseModel):
    """A batch of generated scenes."""
    scenes: List[Scene] = Field(default_factory=list, description="List of scenes")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Batch metadata")
    
    @property
    def total_scenes(self) -> int:
        return len(self.scenes)
    
    @property
    def total_tokens(self) -> int:
        return sum(s.total_token_count for s in self.scenes)


class TicDetectionResult(BaseModel):
    """Result of tic detection analysis."""
    scene_id: str = Field(..., description="ID of analyzed scene")
    tics: Dict[str, int] = Field(default_factory=dict, description="Count of each tic pattern")
    emotion_tells: List[str] = Field(default_factory=list, description="Detected emotion-telling instances")
    total_tic_count: int = Field(default=0, description="Total tics found")
    tic_rate: float = Field(default=0.0, description="Tics per 1000 words")
    needs_cleanup: bool = Field(default=False, description="Whether cleanup is recommended")


class CleanupResult(BaseModel):
    """Result of scene cleanup."""
    original_scene: Scene = Field(..., description="Original scene before cleanup")
    cleaned_scene: Scene = Field(..., description="Scene after cleanup")
    changes_made: List[str] = Field(default_factory=list, description="List of changes made")
    tics_removed: Dict[str, int] = Field(default_factory=dict, description="Tics removed by type")
    validation_passed: bool = Field(default=True, description="Whether validation passed")


class GenerationMetrics(BaseModel):
    """Metrics for a generation run."""
    total_scenes: int = Field(default=0, description="Total scenes generated")
    total_tokens_in: int = Field(default=0, description="Total input tokens")
    total_tokens_out: int = Field(default=0, description="Total output tokens")
    total_cost: float = Field(default=0.0, description="Total API cost")
    total_time: float = Field(default=0.0, description="Total time in seconds")
    call_count: int = Field(default=0, description="Number of API calls")
    success_count: int = Field(default=0, description="Successful calls")
    failure_count: int = Field(default=0, description="Failed calls")
    scenes_by_genre: Dict[str, int] = Field(default_factory=dict, description="Scenes per genre")
    tics_detected: Dict[str, int] = Field(default_factory=dict, description="Tics found by type")


class PipelineConfig(BaseModel):
    """Configuration for pipeline execution."""
    generation: Dict[str, Any] = Field(default_factory=dict)
    judging: Dict[str, Any] = Field(default_factory=dict)
    rewriting: Dict[str, Any] = Field(default_factory=dict)
    paths: Dict[str, str] = Field(default_factory=dict)
    limits: Dict[str, Any] = Field(default_factory=dict)
    quality: Dict[str, Any] = Field(default_factory=dict)
