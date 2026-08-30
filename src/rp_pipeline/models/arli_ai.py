"""
Arli AI model provider for RP Pipeline.
OpenAI-compatible API at https://api.arliai.com/v1/chat/completions
"""

import os
import time
from typing import Any, Dict, List, Optional

import requests

from rp_pipeline.models.base import BaseModelProvider, ModelFactory, ModelResponse


# Pricing information (USD per 1M tokens) - placeholder, update with actual Arli pricing
ARLI_PRICING = {
    "input": {
        "default": 0.0,
    },
    "output": {
        "default": 0.0,
    }
}


@ModelFactory.register("arli_ai")
class ArliAIProvider(BaseModelProvider):
    """
    Arli AI API provider.
    OpenAI-compatible API endpoint for chat completions.
    Uses the /v1/chat/completions endpoint.
    """

    def __init__(self, api_key: Optional[str] = None, model_id: str = "DeepSeek-V4-Flash-0731"):
        """
        Initialize Arli AI provider.

        Args:
            api_key: Arli AI API key (defaults to ARLI_API_KEY env var)
            model_id: Model ID to use (Arli uses model IDs like "DeepSeek-V4-Flash-0731", etc.)
        """
        self.api_key = api_key or os.environ.get("ARLI_API_KEY") or os.environ.get("ARLI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Arli AI API key required. Set ARLI_API_KEY or ARLI_API_KEY env var."
            )
        self._model_id = model_id
        self.base_url = "https://api.arliai.com/v1"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ArliAIProvider":
        """
        Create provider from configuration dict.

        Args:
            config: Configuration dictionary with provider settings

        Returns:
            ArliAIProvider instance
        """
        api_key = config.get("api_key") or os.environ.get("ARLI_API_KEY")
        model_id = config.get("model_id", "DeepSeek-V4-Flash-0731")
        return cls(api_key, model_id)

    @property
    def model_id(self) -> str:
        """Get the current model ID."""
        return self._model_id

    @model_id.setter
    def model_id(self, value: str):
        """Set the model ID."""
        self._model_id = value

    @property
    def name(self) -> str:
        """Provider name."""
        return "arli_ai"

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _get_payload(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build the API request payload."""
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "repetition_penalty" in kwargs:
            payload["repetition_penalty"] = kwargs["repetition_penalty"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]
        for extra in ("thinking_token_budget", "reasoning_effort", "reasoning_config"):
            if extra in kwargs:
                payload[extra] = kwargs[extra]
        return payload

    def generate(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.85,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate a response from the model."""
        headers = self._get_headers()
        timeout = float(kwargs.pop("timeout", 300.0))
        retries = int(kwargs.pop("retries", 2))

        payload = self._get_payload(prompt, system, max_tokens, temperature, **kwargs)

        last_error = "unknown error"
        for attempt in range(retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.RequestException as e:
                body = ""
                if getattr(e, "response", None) is not None:
                    body = e.response.text[:200]
                last_error = f"API error: {e} {body}"
                if attempt < retries:
                    time.sleep(2 * (attempt + 1))
                continue

            # Some models (reasoning-style) return content: null when the
            # token budget goes to the reasoning channel — treat as failure.
            content = None
            if data.get("choices"):
                content = data["choices"][0].get("message", {}).get("content")
            content = str(content) if content else ""

            usage = {}
            if data.get("usage"):
                usage = {
                    "prompt_tokens": data["usage"].get("prompt_tokens", 0),
                    "completion_tokens": data["usage"].get("completion_tokens", 0),
                    "total_tokens": data["usage"].get("total_tokens", 0),
                }
            finish_reason = (data.get("choices") or [{}])[0].get("finish_reason")

            if content:
                return ModelResponse(
                    content=content,
                    success=True,
                    error=None,
                    usage=usage,
                    model_id=self.model_id,
                )

            last_error = (
                "empty content (finish_reason=length)"
                if finish_reason == "length"
                else "empty content from model"
            )
            if attempt < retries:
                time.sleep(2 * (attempt + 1))

        return ModelResponse(
            content="",
            success=False,
            error=last_error,
            usage={},
            model_id=self.model_id,
        )

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
            **kwargs: Additional model parameters

        Returns:
            ModelResponse with generated content
        """
        headers = self._get_headers()
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "repetition_penalty" in kwargs:
            payload["repetition_penalty"] = kwargs["repetition_penalty"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()

            content = ""
            if data.get("choices") and len(data["choices"]) > 0:
                content = data["choices"][0].get("message", {}).get("content", "")

            usage = {}
            if data.get("usage"):
                usage = {
                    "prompt_tokens": data["usage"].get("prompt_tokens", 0),
                    "completion_tokens": data["usage"].get("completion_tokens", 0),
                    "total_tokens": data["usage"].get("total_tokens", 0),
                }

            return ModelResponse(
                content=content,
                success=True,
                error=None,
                usage=usage,
                model_id=self.model_id,
            )

        except requests.exceptions.Timeout:
            return ModelResponse(
                content="",
                success=False,
                error="Request timeout",
                usage={},
                model_id=self.model_id,
            )
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get("message", error_msg)
                except (ValueError, KeyError):
                    error_msg = f"API error: {e.response.status_code} - {error_msg}"
            else:
                error_msg = f"API error: {error_msg}"
            return ModelResponse(
                content="",
                success=False,
                error=error_msg,
                usage={},
                model_id=self.model_id,
            )

    def get_cost(self, response: ModelResponse) -> float:
        """
        Calculate cost for a model response.

        Args:
            response: The model response to price

        Returns:
            Cost in USD
        """
        if not response.usage:
            return 0.0
        prompt_tokens = response.usage.get("prompt_tokens", 0)
        completion_tokens = response.usage.get("completion_tokens", 0)
        input_cost = ARLI_PRICING.get("input", {}).get(self.model_id, 0.0)
        output_cost = ARLI_PRICING.get("output", {}).get(self.model_id, 0.0)
        return (prompt_tokens * input_cost + completion_tokens * output_cost) / 1_000_000

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about this model.

        Returns:
            Dictionary with model name, provider, capabilities, etc.
        """
        return {
            "provider": "arli_ai",
            "model_id": self.model_id,
            "name": self.model_id,
            "capabilities": ["chat", "completions"],
            "base_url": self.base_url,
        }

    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "arli_ai"

    def get_model_list(self) -> List[str]:
        """
        Get list of available models.
        Note: This makes an API call to list models.
        """
        headers = self._get_headers()
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            models = [m.get("id", "") for m in data.get("data", []) if m.get("id")]
            return models
        except requests.exceptions.RequestException:
            return [
                "DeepSeek-V4-Flash-0731",
                "Gemma-4-31B-Agares-v1",
                "Gemma-4-31B-Animus-V14.1",
                "Gemma-4-31B-AssGuard",
                "Gemma-4-31B-Aura-4o-Rebirth-Merged",
                "Gemma-4-31B-Cognitive-Unshackled",
            ]
