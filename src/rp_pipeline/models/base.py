"""
Base model interface for RP Pipeline.
All model providers implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from rp_pipeline.data.schemas import GenerationMetrics, Scene


class ModelResponse(BaseModel):
    """Response from a model call."""
    content: str = Field(..., description="Generated content")
    finish_reason: Optional[str] = Field(default=None, description="Why generation stopped")
    usage: Optional[Dict[str, Any]] = Field(default=None, description="Token usage")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    
    @property
    def success(self) -> bool:
        """Whether the call was successful."""
        return self.error is None and self.content is not None


class BaseModelProvider(ABC):
    """
    Abstract base class for model providers.
    All concrete providers (OpenRouter, Featherless, NVIDIA) inherit from this.
    """
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseModelProvider":
        """
        Create a provider instance from configuration.
        
        Args:
            config: Configuration dictionary with provider settings
        
        Returns:
            Configured provider instance
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.85,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Generate text from a prompt.
        
        Args:
            prompt: User prompt
            system: System prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional provider-specific arguments
        
        Returns:
            ModelResponse with generated content or error
        """
        pass
    
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.85,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Generate text from a chat conversation.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional provider-specific arguments
        
        Returns:
            ModelResponse with generated content or error
        """
        pass
    
    @abstractmethod
    def get_cost(self, response: ModelResponse) -> float:
        """
        Calculate cost for a model response.
        
        Args:
            response: The model response to price
        
        Returns:
            Cost in USD
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about this model.
        
        Returns:
            Dictionary with model name, provider, capabilities, etc.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass
    
    @property
    @abstractmethod
    def model_id(self) -> str:
        """Model ID being used."""
        pass


class ModelFactory:
    """Factory for creating model providers."""
    
    _providers: Dict[str, type[BaseModelProvider]] = {}
    
    @classmethod
    def register(cls, name: str) -> type[BaseModelProvider]:
        """
        Decorator to register a provider class.
        
        Args:
            name: Provider name (e.g., "openrouter", "featherless")
        
        Returns:
            Decorator function
        """
        def decorator(provider_class: type[BaseModelProvider]) -> type[BaseModelProvider]:
            cls._providers[name] = provider_class
            return provider_class
        return decorator
    
    @classmethod
    def create(cls, provider: str, config: Dict[str, Any]) -> BaseModelProvider:
        """
        Create a provider instance.
        
        Args:
            provider: Provider name
            config: Configuration dictionary
        
        Returns:
            Configured provider instance
        
        Raises:
            ValueError: If provider not found
        """
        if provider not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(f"Unknown provider '{provider}'. Available: {available}")
        
        return cls._providers[provider].from_config(config)
    
    @classmethod
    def get_provider_names(cls) -> List[str]:
        """Get list of registered provider names."""
        return list(cls._providers.keys())
