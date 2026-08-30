"""
Caching and checkpointing for RP Pipeline.
Handles persistent caching of intermediate results and pipeline state.
"""

import hashlib
import json
import os
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union

from rp_pipeline.config.settings import get_settings


T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """A cached value with metadata."""
    value: T
    created_at: float = field(default_factory=time.time)
    ttl: Optional[float] = None  # Time to live in seconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if this cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)


class DiskCache:
    """
    Disk-based cache with TTL support.
    Stores values as JSON or pickle files.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, default_ttl: int = 86400):
        """
        Initialize disk cache.
        
        Args:
            cache_dir: Directory for cache files. Uses config if not provided.
            default_ttl: Default time-to-live in seconds (24 hours default)
        """
        settings = get_settings()
        cache_config = settings.cache
        
        self.cache_dir = Path(
            cache_dir or cache_config.get("dir", "data/cache")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl if cache_config.get("enabled", True) else 0
        self._enabled = cache_config.get("enabled", True)
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        # Sanitize key for filesystem
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_key}.cache"
    
    def _make_key(self, *args: Any, **kwargs: Any) -> str:
        """Create a cache key from arguments."""
        # Convert args and kwargs to a stable string representation
        key_parts = []
        
        for arg in args:
            if isinstance(arg, (dict, list, tuple)):
                key_parts.append(json.dumps(arg, sort_keys=True, default=str))
            else:
                key_parts.append(str(arg))
        
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (dict, list, tuple)):
                key_parts.append(f"{k}={json.dumps(v, sort_keys=True, default=str)}")
            else:
                key_parts.append(f"{k}={str(v)}")
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found/expired
        """
        if not self._enabled:
            return None
        
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                entry = pickle.load(f)
            
            if isinstance(entry, CacheEntry) and entry.is_expired():
                # Remove expired entry
                cache_path.unlink(missing_ok=True)
                return None
            
            if isinstance(entry, CacheEntry):
                return entry.value
            return entry
        except Exception:
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Set a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
            metadata: Optional metadata to store with the value
        """
        if not self._enabled:
            return
        
        cache_path = self._get_cache_path(key)
        
        entry = CacheEntry(
            value=value,
            ttl=ttl if ttl is not None else self.default_ttl,
            metadata=metadata or {},
        )
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(entry, f)
        except Exception:
            pass  # Silently fail on cache write errors
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if deleted, False if not found
        """
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()
            return True
        return False
    
    def clear(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries cleared
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                cache_file.unlink()
                count += 1
            except Exception:
                pass
        return count
    
    def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        if not self._enabled:
            return False
        cache_path = self._get_cache_path(key)
        return cache_path.exists()
    
    def cached(
        self,
        ttl: Optional[int] = None,
        key_func: Optional[Callable[..., str]] = None,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """
        Decorator to cache function results.
        
        Args:
            ttl: Time-to-live for cached results
            key_func: Optional function to generate cache key from args
        
        Returns:
            Decorator function
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            def wrapper(*args: Any, **kwargs: Any) -> T:
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._make_key(func.__name__, args, kwargs)
                
                # Try to get from cache
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    return cached_value
                
                # Call function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl=ttl)
                return result
            
            return wrapper
        return decorator


@dataclass
class CheckpointState:
    """State for pipeline checkpointing."""
    stage: str
    items_processed: int
    items_successful: int
    items_failed: int
    last_item_id: Optional[str] = None
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_update: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class PipelineCheckpoint:
    """
    Manages pipeline checkpoints for resumable execution.
    """
    
    def __init__(self, checkpoint_file: Optional[str] = None):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_file: Path to checkpoint file. Uses config if not provided.
        """
        settings = get_settings()
        pipeline_config = settings.pipeline
        
        self.checkpoint_file = Path(
            checkpoint_file or 
            pipeline_config.get("generate", {}).get("checkpoint_file", 
                "data/cache/generate_checkpoint.json")
        )
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        self._state: Optional[CheckpointState] = None
        self._load()
    
    def _load(self) -> None:
        """Load checkpoint state from file."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    self._state = CheckpointState(**data)
            except Exception:
                self._state = None
        else:
            self._state = None
    
    def _save(self) -> None:
        """Save checkpoint state to file."""
        if self._state:
            try:
                with open(self.checkpoint_file, 'w') as f:
                    json.dump(self._state.__dict__, f, indent=2)
            except Exception:
                pass
    
    @property
    def state(self) -> Optional[CheckpointState]:
        """Get current checkpoint state."""
        return self._state
    
    def start_stage(self, stage: str) -> None:
        """
        Start a new pipeline stage.
        
        Args:
            stage: Stage name
        """
        self._state = CheckpointState(
            stage=stage,
            items_processed=0,
            items_successful=0,
            items_failed=0,
        )
        self._save()
    
    def update(
        self,
        item_id: str,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update checkpoint with new item processing result.
        
        Args:
            item_id: ID of the processed item
            success: Whether processing was successful
            metadata: Optional metadata to update
        """
        if self._state is None:
            self._state = CheckpointState(
                stage="unknown",
                items_processed=0,
                items_successful=0,
                items_failed=0,
            )
        
        self._state.items_processed += 1
        self._state.last_item_id = item_id
        self._state.last_update = datetime.utcnow().isoformat()
        
        if success:
            self._state.items_successful += 1
        else:
            self._state.items_failed += 1
        
        if metadata:
            self._state.metadata.update(metadata)
        
        self._save()
    
    def mark_complete(self) -> None:
        """Mark the current stage as complete."""
        if self._state:
            self._state.metadata["completed_at"] = datetime.utcnow().isoformat()
            self._save()
    
    def is_resumable(self, stage: str) -> bool:
        """
        Check if a stage can be resumed.
        
        Args:
            stage: Stage name to check
        
        Returns:
            True if checkpoint exists for this stage
        """
        return (
            self._state is not None and 
            self._state.stage == stage and
            self._state.items_processed > 0
        )
    
    def get_resume_position(self) -> Tuple[int, Optional[str]]:
        """
        Get the position to resume from.
        
        Returns:
            Tuple of (items_processed, last_item_id)
        """
        if self._state:
            return self._state.items_processed, self._state.last_item_id
        return 0, None
    
    def clear(self) -> None:
        """Clear the checkpoint."""
        self._state = None
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()


class MemoryCache:
    """
    In-memory cache for fast access during a session.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize memory cache.
        
        Args:
            max_size: Maximum number of entries
            default_ttl: Default time-to-live in seconds
        """
        self._cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        if entry.is_expired():
            del self._cache[key]
            return None
        return entry.value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Set a value in cache."""
        # Evict old entries if at capacity
        if len(self._cache) >= self.max_size:
            self._evict_oldest()
        
        self._cache[key] = CacheEntry(
            value=value,
            ttl=ttl if ttl is not None else self.default_ttl,
        )
    
    def _evict_oldest(self) -> None:
        """Evict the oldest entry."""
        if not self._cache:
            return
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
        del self._cache[oldest_key]
    
    def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> int:
        """Clear all cache entries."""
        count = len(self._cache)
        self._cache.clear()
        return count


# Global cache instances
_disk_cache: Optional[DiskCache] = None
_memory_cache: Optional[MemoryCache] = None


def get_disk_cache() -> DiskCache:
    """Get the global disk cache instance."""
    global _disk_cache
    if _disk_cache is None:
        _disk_cache = DiskCache()
    return _disk_cache


def get_memory_cache() -> MemoryCache:
    """Get the global memory cache instance."""
    global _memory_cache
    if _memory_cache is None:
        _memory_cache = MemoryCache()
    return _memory_cache


def get_checkpoint(stage: str) -> PipelineCheckpoint:
    """
    Get a checkpoint manager for a specific stage.
    
    Args:
        stage: Stage name
    
    Returns:
        PipelineCheckpoint instance
    """
    settings = get_settings()
    pipeline_config = settings.pipeline
    stage_config = pipeline_config.get("stages", {}).get(stage, {})
    checkpoint_file = stage_config.get("checkpoint_file")
    
    return PipelineCheckpoint(checkpoint_file)
