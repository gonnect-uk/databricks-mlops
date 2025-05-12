"""
Strongly-typed logging utilities for the Databricks MLOps framework.
"""
import json
import logging
import sys
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field

from databricks_mlops.models.base import LogRecord


class LogLevel(str, Enum):
    """Standard log levels with string representation for type safety."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class StructuredLogFormatter(logging.Formatter):
    """
    Formatter that outputs logs in a structured JSON format.
    
    This ensures logs are machine-readable and can be easily parsed by
    log analysis tools.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            exc_type, exc_value, _ = record.exc_info
            log_entry["exception"] = {
                "type": exc_type.__name__,
                "message": str(exc_value),
            }
        
        # Add extra context fields from the record
        if hasattr(record, "context") and isinstance(record.context, dict):
            log_entry["context"] = record.context
        
        # Add trace information if available
        if hasattr(record, "trace_id"):
            log_entry["trace_id"] = record.trace_id
        if hasattr(record, "span_id"):
            log_entry["span_id"] = record.span_id
        
        return json.dumps(log_entry)


class StructuredLogFilter(logging.Filter):
    """
    Filter that adds structured context to log records.
    
    This ensures that all logs include standardized metadata.
    """
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        """
        Initialize with default context values.
        
        Args:
            context: Dictionary of default context values to add to all logs
        """
        super().__init__()
        self.context = context or {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add context to the log record.
        
        Args:
            record: The log record to modify
            
        Returns:
            bool: Always True (the record is always permitted)
        """
        # Only add context if it doesn't exist
        if not hasattr(record, "context"):
            record.context = self.context.copy()
        # If it exists but isn't a dict, convert it
        elif not isinstance(record.context, dict):
            record.context = {"value": str(record.context)}
        # If it exists and is a dict, merge with default context
        else:
            # Don't override existing values
            for key, value in self.context.items():
                if key not in record.context:
                    record.context[key] = value
        
        return True


def setup_logger(
    name: str,
    level: Union[LogLevel, str] = LogLevel.INFO,
    add_console_handler: bool = True,
    add_file_handler: bool = False,
    log_file: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> logging.Logger:
    """
    Set up a structured logger with the specified configuration.
    
    Args:
        name: The name of the logger
        level: The log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        add_console_handler: Whether to add a console handler
        add_file_handler: Whether to add a file handler
        log_file: Path to the log file (required if add_file_handler is True)
        context: Dictionary of context values to add to all logs
        
    Returns:
        A configured logger instance
    """
    # Convert string level to LogLevel if needed
    if isinstance(level, str):
        try:
            level = LogLevel(level.upper())
        except ValueError:
            level = LogLevel.INFO
    
    # Get or create the logger
    logger = logging.getLogger(name)
    logger.setLevel(level.value)
    
    # Add structured log filter
    logger.addFilter(StructuredLogFilter(context))
    
    # Create formatter
    formatter = StructuredLogFormatter()
    
    # Add console handler if requested
    if add_console_handler:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level.value)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if requested
    if add_file_handler:
        if not log_file:
            raise ValueError("log_file parameter is required when add_file_handler=True")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level.value)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class LoggingContext:
    """
    Context manager for temporarily adding context to logs.
    
    This allows adding structured context to logs within a specific scope.
    """
    
    def __init__(self, logger: logging.Logger, context: Dict[str, Any]):
        """
        Initialize with logger and context.
        
        Args:
            logger: The logger to add context to
            context: The context dictionary to add
        """
        self.logger = logger
        self.context = context
        self.filter = StructuredLogFilter(context)
    
    def __enter__(self) -> logging.Logger:
        """
        Add the context filter when entering the context.
        
        Returns:
            The logger with added context
        """
        self.logger.addFilter(self.filter)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Remove the context filter when exiting the context."""
        self.logger.removeFilter(self.filter)


def get_structured_log_record(
    logger_name: str,
    level: LogLevel,
    message: str,
    context: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
) -> LogRecord:
    """
    Create a strongly-typed LogRecord object.
    
    Args:
        logger_name: Name of the logger
        level: Log level
        message: Log message
        context: Additional context information
        trace_id: Optional trace ID for distributed tracing
        span_id: Optional span ID for distributed tracing
        
    Returns:
        A strongly-typed LogRecord model
    """
    return LogRecord(
        level=level.value,
        message=message,
        timestamp=datetime.now(),
        context=context or {},
        logger=logger_name,
        trace_id=trace_id,
        span_id=span_id,
    )
