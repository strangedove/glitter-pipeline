"""
NVIDIA model provider for RP Pipeline.
"""

import os
from typing import Any, Dict, List, Optional

import requests

from rp_pipeline.models.base import BaseModelProvider, ModelFactory, ModelResponse


# Pricing information (USD per 1M tokens) - update as needed
NVIDIA_PRICING = {
    "input": {
        "deepseek-ai/deepseek-v4-pro": 0.70,
        "meta/llama3-2-90b": 0.59,
    },
    "output": {
        "deepseek-ai/deepseek-v4-pro": 2.80,
        "meta/llama3-2-90b": 0.79,
    }
}


@ModelFactory.register("nvidia")
class NVIDIAProvider(BaseModelProvider):
    """
    NVIDIA API provider.
    
    Fast and reliable for DeepSeek models.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_id: str = "deepseek-ai/deepseek-v4-pro"):
        """
        Initialize NVIDIA provider.
        
        Args:
            api_key: NVIDIA API key (defaults to NVIDIA_API_KEY env var)
            model_id: Model ID to use
        """
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA API key required. Set NVIDIA_API_KEY env var.")
        self._model_id = model_id
        self.base_url = "https://integrate.api.nvidia.com/v1"
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NVIDIAProvider":
        """Create provider from configuration."""
        api_key = config.get("api_key") or os.environ.get("NVIDIA_API_KEY")
        model_id = config.get("model_id", "deepseek-ai/deepseek-v4-pro")
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
        input_price = NVIDIA_PRICING.get("input", {}).get(model_id, 0.70)
        output_price = NVIDIA_PRICING.get("output", {}).get(model_id, 2.80)
        
        return (input_tokens / 1_000_000 * input_price + 
                output_tokens / 1_000_000 * output_price)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": "nvidia",
            "model_id": self.model_id,
            "base_url": self.base_url,
            "supports_streaming": True,
            "supports_tools": False,
        }
    
    @property
    def name(self) -> str:
        return "nvidia"
    
    @property
    def model_id(self) -> str:
        return self._model_id
