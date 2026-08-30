"""
OpenRouter model provider for RP Pipeline.
"""

import os
from typing import Any, Dict, List, Optional

import requests

from rp_pipeline.models.base import BaseModelProvider, ModelFactory, ModelResponse


# Pricing information (USD per 1M tokens) - update as needed
OPENROUTER_PRICING = {
    "input": {
        "xiaomi/mimo-v2.5-pro": 1.50,
        "openai/gpt-4o": 5.00,
        "anthropic/claude-3-5-sonnet": 3.00,
        "meta-llama/llama-3.1-70b-instruct": 0.59,
    },
    "output": {
        "xiaomi/mimo-v2.5-pro": 3.00,
        "openai/gpt-4o": 15.00,
        "anthropic/claude-3-5-sonnet": 15.00,
        "meta-llama/llama-3.1-70b-instruct": 0.79,
    }
}


@ModelFactory.register("openrouter")
class OpenRouterProvider(BaseModelProvider):
    """
    OpenRouter API provider.
    
    Supports any model available through OpenRouter.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_id: str = "xiaomi/mimo-v2.5-pro"):
        """
        Initialize OpenRouter provider.
        
        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            model_id: Model ID to use
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY env var.")
        self.model_id = model_id
        self.base_url = "https://openrouter.ai/api/v1"
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "OpenRouterProvider":
        """Create provider from configuration."""
        api_key = config.get("api_key") or os.environ.get("OPENROUTER_API_KEY")
        model_id = config.get("model_id", "xiaomi/mimo-v2.5-pro")
        return cls(api_key, model_id)
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.85,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate text from a prompt."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        return self.chat(messages, max_tokens, temperature, **kwargs)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.85,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate text from a chat conversation."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/strangedove/glitter-project",
            "X-Title": "RP Pipeline",
        }
        
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Add optional kwargs
        if "repetition_penalty" in kwargs:
            payload["repetition_penalty"] = kwargs["repetition_penalty"]
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            payload["top_k"] = kwargs["top_k"]
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=kwargs.get("timeout", 300),
            )
            
            if response.status_code == 429:
                return ModelResponse(
                    content="",
                    error="Rate limit exceeded",
                    finish_reason="error"
                )
            
            if response.status_code != 200:
                return ModelResponse(
                    content="",
                    error=f"API error: {response.status_code} - {response.text[:200]}",
                    finish_reason="error"
                )
            
            data = response.json()
            choice = data["choices"][0]
            
            return ModelResponse(
                content=choice["message"]["content"],
                finish_reason=choice.get("finish_reason", "stop"),
                usage=data.get("usage"),
                error=None,
            )
            
        except requests.exceptions.Timeout:
            return ModelResponse(
                content="",
                error="Request timeout",
                finish_reason="error"
            )
        except Exception as e:
            return ModelResponse(
                content="",
                error=str(e),
                finish_reason="error"
            )
    
    def get_cost(self, response: ModelResponse) -> float:
        """Calculate cost for a model response."""
        if response.usage is None:
            return 0.0
        
        model_id = self.model_id.lower()
        input_tokens = response.usage.get("prompt_tokens", 0)
        output_tokens = response.usage.get("completion_tokens", 0)
        
        # Get pricing or use default
        input_price = OPENROUTER_PRICING.get("input", {}).get(model_id, 1.50)
        output_price = OPENROUTER_PRICING.get("output", {}).get(model_id, 3.00)
        
        return (input_tokens / 1_000_000 * input_price + 
                output_tokens / 1_000_000 * output_price)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": "openrouter",
            "model_id": self.model_id,
            "base_url": self.base_url,
            "supports_streaming": True,
            "supports_tools": False,
        }
    
    @property
    def name(self) -> str:
        return "openrouter"
    
    @property
    def model_id(self) -> str:
        return self._model_id
