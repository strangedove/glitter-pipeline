"""
Structured logging for RP Pipeline.
Supports both JSON and text formatting for different use cases.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rp_pipeline.config.settings import get_settings


class PipelineFormatter(logging.Formatter):
    """
    Custom formatter that supports both JSON and text output.
    """
    
    def __init__(self, fmt: str = "text", include_timestamp: bool = True):
        """
        Initialize formatter.
        
        Args:
            fmt: Format type - "text" or "json"
            include_timestamp: Whether to include timestamp in output
        """
        super().__init__()
        self.fmt = fmt
        self.include_timestamp = include_timestamp
    
    def format(self, record: logging.LogRecord) -> str:
        """Format a log record."""
        if self.fmt == "json":
            return self._format_json(record)
        else:
            return self._format_text(record)
    
    def _format_json(self, record: logging.LogRecord) -> str:
        """Format as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in (
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'pathname', 'process', 'processName', 'relativeCreated',
                'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
                'message',
            ):
                try:
                    json.dumps(value)  # Check if serializable
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)
        
        return json.dumps(log_data, default=str)
    
    def _format_text(self, record: logging.LogRecord) -> str:
        """Format as text."""
        parts = []
        
        if self.include_timestamp:
            parts.append(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
        
        parts.append(f"[{record.levelname}]")
        parts.append(record.name)
        parts.append(record.getMessage())
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in (
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'pathname', 'process', 'processName', 'relativeCreated',
                'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
                'message',
            ):
                parts.append(f"{key}={value}")
        
        # Add exception info
        if record.exc_info:
            parts.append(f"\nException: {self.formatException(record.exc_info)}")
        
        return " ".join(str(p) for p in parts)


class PipelineHandler(logging.FileHandler):
    """
    Custom file handler that ensures directory exists.
    """
    
    def __init__(
        self,
        filename: str,
        mode: str = 'a',
        encoding: str = 'utf-8',
        delay: bool = False,
    ):
        """
        Initialize handler.
        
        Args:
            filename: Log file path
            mode: File mode
            encoding: File encoding
            delay: Delay file opening
        """
        # Ensure directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        super().__init__(filename, mode, encoding, delay)


class StructuredLogger:
    """
    High-level structured logger for the pipeline.
    """
    
    _instance: Optional["StructuredLogger"] = None
    
    def __new__(cls) -> "StructuredLogger":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize structured logger."""
        if hasattr(self, "_initialized") and self._initialized:
            return
        
        self.settings = get_settings()
        self.logger = self._setup_logger()
        self._initialized = True
    
    def _setup_logger(self) -> logging.Logger:
        """Set up the logger with configured settings."""
        logger = logging.getLogger("rp_pipeline")
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
        
        # Get logging config
        logging_config = self.settings.logging
        log_level = getattr(
            logging,
            logging_config.get("level", "INFO").upper(),
            logging.INFO
        )
        log_format = logging_config.get("format", "json")
        log_file = logging_config.get("file", "data/logs/pipeline.log")
        
        # Set level
        logger.setLevel(log_level)
        
        # Create formatter
        formatter = PipelineFormatter(fmt=log_format)
        
        # Create file handler
        try:
            file_handler = PipelineHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # Fallback to stderr if file logging fails
            stderr_handler = logging.StreamHandler(sys.stderr)
            stderr_handler.setLevel(log_level)
            stderr_handler.setFormatter(formatter)
            logger.addHandler(stderr_handler)
        
        # Also log to stderr for visibility
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(log_level)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
        
        return logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a named logger.
        
        Args:
            name: Logger name (e.g., "generation", "analysis", "cleanup")
        
        Returns:
            Configured logger
        """
        return logging.getLogger(f"rp_pipeline.{name}")
    
    def log(
        self,
        level: str,
        message: str,
        **kwargs: Any,
    ) -> None:
        """
        Log a message with optional extra fields.
        
        Args:
            level: Log level (debug, info, warning, error, critical)
            message: Log message
            **kwargs: Extra fields to include in log
        """
        logger = self.logger
        log_method = getattr(logger, level.lower(), logger.info)
        
        log_method(message, extra={**kwargs, "extra_fields": kwargs})
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self.log("debug", message, **kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self.log("info", message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self.log("warning", message, **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self.log("error", message, **kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self.log("critical", message, **kwargs)
    
    def log_generation(
        self,
        scene_id: str,
        card_id: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        duration: float,
        success: bool,
        **kwargs: Any,
    ) -> None:
        """Log a generation event."""
        self.info(
            "Generation completed",
            scene_id=scene_id,
            card_id=card_id,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration=duration,
            success=success,
            **kwargs,
        )
    
    def log_analysis(
        self,
        scene_id: str,
        tics_found: int,
        tic_rate: float,
        needs_cleanup: bool,
        **kwargs: Any,
    ) -> None:
        """Log an analysis event."""
        self.info(
            "Analysis completed",
            scene_id=scene_id,
            tics_found=tics_found,
            tic_rate=tic_rate,
            needs_cleanup=needs_cleanup,
            **kwargs,
        )
    
    def log_cleanup(
        self,
        scene_id: str,
        changes_made: List[str],
        tics_removed: Dict[str, int],
        validation_passed: bool,
        **kwargs: Any,
    ) -> None:
        """Log a cleanup event."""
        self.info(
            "Cleanup completed",
            scene_id=scene_id,
            changes_made=changes_made,
            tics_removed=tics_removed,
            validation_passed=validation_passed,
            **kwargs,
        )
    
    def log_pipeline(
        self,
        stage: str,
        items_processed: int,
        items_successful: int,
        items_failed: int,
        duration: float,
        **kwargs: Any,
    ) -> None:
        """Log a pipeline stage event."""
        self.info(
            f"Pipeline stage {stage} completed",
            stage=stage,
            items_processed=items_processed,
            items_successful=items_successful,
            items_failed=items_failed,
            duration=duration,
            **kwargs,
        )


# Convenience functions
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger for a specific component.
    
    Args:
        name: Optional logger name. If None, returns root pipeline logger.
    
    Returns:
        Configured logger
    """
    logger = StructuredLogger()
    if name:
        return logger.get_logger(name)
    return logger.logger


def setup_logging() -> StructuredLogger:
    """
    Set up pipeline logging.
    
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger()


# Initialize logging on import
_structured_logger: Optional[StructuredLogger] = None


def init_logging() -> StructuredLogger:
    """Initialize logging once."""
    global _structured_logger
    if _structured_logger is None:
        _structured_logger = setup_logging()
    return _structured_logger


# Module-level logger
logger = init_logging()
