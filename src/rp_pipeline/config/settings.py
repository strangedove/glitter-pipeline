"""
Settings management for RP Pipeline.
Loads from YAML config files with environment variable overrides.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Settings:
    """Pipeline settings loaded from config files and environment."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize settings.
        
        Args:
            config_path: Path to settings YAML file. If None, uses default.
        """
        self._config_path = config_path or self._find_config()
        self._config = self._load_config()
        self._override_from_env()
    
    @staticmethod
    def _find_config() -> str:
        """Find the settings config file."""
        # Check in restructured/pipeline/config/
        candidates = [
            "restructured/pipeline/config/settings.yaml",
            "config/settings.yaml",
            "settings.yaml",
        ]
        for candidate in candidates:
            if Path(candidate).exists():
                return candidate
        return "restructured/pipeline/config/settings.yaml"
    
    def _load_config(self) -> Dict[str, Any]:
        """Load config from YAML file."""
        try:
            with open(self._config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}
    
    def _override_from_env(self):
        """Override settings from environment variables."""
        # Map environment variables to config paths
        env_mapping = {
            "ARLI_API_KEY": "models.providers.arli_ai.api_key_env",
            "ARLI_API_KEY": "models.providers.arli_ai.api_key_env",
            "OPENROUTER_API_KEY": "models.providers.openrouter.api_key_env",
            "FEATHERLESS_API_KEY": "models.providers.featherless.api_key_env",
            "NVIDIA_API_KEY": "models.providers.nvidia.api_key_env",
            "GEN_MODEL": "defaults.generation.model",
            "JUDGE_MODEL": "defaults.judging.model",
            "MAX_TOKENS": "defaults.generation.max_tokens",
            "TEMPERATURE": "defaults.generation.temperature",
            "OUTPUT_DIR": "paths.output.base",
            "CARDS_FILE": "paths.input.default_cards",
        }
        
        for env_var, config_path in env_mapping.items():
            if env_var in os.environ:
                # Set the value
                self._set_nested(self._config, config_path, os.environ[env_var])
    
    @staticmethod
    def _set_nested(d: Dict, path: str, value: Any):
        """Set a nested dictionary value from dot notation path."""
        keys = path.split('.')
        current = d
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get a setting value by dot notation path.
        
        Args:
            path: Dot notation path (e.g., "defaults.generation.model")
            default: Default value if not found
        
        Returns:
            The setting value or default
        """
        keys = path.split('.')
        current = self._config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    
    def get_model_config(self, role: str) -> Dict[str, Any]:
        """
        Get model configuration for a specific role.
        
        Args:
            role: "generation", "judging", or "rewriting"
        
        Returns:
            Model configuration dict
        """
        return self.get(f"defaults.{role}", {})
    
    @property
    def generation(self) -> Dict[str, Any]:
        """Generation settings."""
        return self.get("generation", {})
    
    @property
    def pipeline(self) -> Dict[str, Any]:
        """Pipeline settings."""
        return self.get("pipeline", {})
    
    @property
    def paths(self) -> Dict[str, str]:
        """Path settings."""
        return self.get("paths", {})
    
    @property
    def limits(self) -> Dict[str, Any]:
        """Limit settings."""
        return self.get("limits", {})
    
    @property
    def quality(self) -> Dict[str, Any]:
        """Quality threshold settings."""
        return self.get("quality", {})
    
    @property
    def logging(self) -> Dict[str, Any]:
        """Logging settings."""
        return self.get("logging", {})
    
    @property
    def defaults(self) -> Dict[str, Any]:
        """Default model configurations."""
        return self.get("defaults", {})
    
    @property
    def cache(self) -> Dict[str, Any]:
        """Cache settings."""
        return self.get("cache", {})


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings():
    """Reset the global settings instance."""
    global _settings
    _settings = None


def load_prompts(config_path: Optional[str] = None) -> Dict[str, str]:
    """
    Load prompts from YAML config.
    
    Args:
        config_path: Path to prompts YAML file
    
    Returns:
        Dictionary of prompt name -> prompt text
    """
    if config_path is None:
        # Try to find prompts.yaml relative to current working directory
        candidates = [
            "restructured/pipeline/config/prompts.yaml",
            "config/prompts.yaml",
            "prompts.yaml",
        ]
        for candidate in candidates:
            if Path(candidate).exists():
                config_path = candidate
                break
        else:
            config_path = "restructured/pipeline/config/prompts.yaml"
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
