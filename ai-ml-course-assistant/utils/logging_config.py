"""
Centralized logging configuration for AI/ML Course Assistant.

Provides consistent logging format across all modules.
Eliminates duplicate logging.basicConfig() calls.

Usage:
    from utils.logging_config import setup_logging
    setup_logging()  # Call once at module/script start
    
    # Or get a named logger:
    from utils.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Message")

Last Updated: 2026-01-22
"""

import logging
import sys
from pathlib import Path
from typing import Optional


# Default logging configuration
DEFAULT_LEVEL = logging.INFO
DEFAULT_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = '%H:%M:%S'

# Alternative formats for different use cases
DETAILED_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
SIMPLE_FORMAT = '%(levelname)s - %(message)s'
DEBUG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'


def setup_logging(
    level: int = DEFAULT_LEVEL,
    format_string: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    log_file: Optional[Path] = None,
    force: bool = False
) -> None:
    """
    Configure logging with standard format.
    
    This function centralizes logging configuration to avoid duplicate
    basicConfig() calls across modules. Call once at application/module start.
    
    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        format_string: Log message format
        date_format: Date/time format for timestamps
        log_file: Optional file path to write logs (in addition to console)
        force: Force reconfiguration even if already configured
        
    Example:
        # Basic usage (default INFO level):
        setup_logging()
        
        # Debug mode:
        setup_logging(level=logging.DEBUG, format_string=DEBUG_FORMAT)
        
        # With file output:
        setup_logging(log_file=Path("logs/app.log"))
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    
    # Add file handler if log_file specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt=date_format,
        handlers=handlers,
        force=force  # Python 3.8+ feature to reconfigure if already set
    )


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a named logger with optional level override.
    
    This is useful for getting module-specific loggers that can be
    configured independently while maintaining the centralized format.
    
    Args:
        name: Logger name (typically __name__ of the module)
        level: Optional logging level override for this logger
        
    Returns:
        Configured logger instance
        
    Example:
        logger = get_logger(__name__)
        logger.info("Module initialized")
        
        # With debug level override:
        debug_logger = get_logger(__name__, level=logging.DEBUG)
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


def set_level(level: int) -> None:
    """
    Change logging level for root logger.
    
    Useful for runtime level adjustment (e.g., enable debug mode).
    
    Args:
        level: New logging level
        
    Example:
        set_level(logging.DEBUG)  # Enable debug logging
        set_level(logging.WARNING)  # Only show warnings and errors
    """
    logging.getLogger().setLevel(level)


def disable_module_logging(module_name: str) -> None:
    """
    Disable logging for a specific module.
    
    Useful for silencing noisy third-party libraries.
    
    Args:
        module_name: Module name to disable (e.g., 'urllib3', 'openai')
        
    Example:
        disable_module_logging('urllib3')  # Silence HTTP request logs
        disable_module_logging('matplotlib')  # Silence matplotlib debug logs
    """
    logging.getLogger(module_name).setLevel(logging.WARNING)


def enable_debug_mode() -> None:
    """
    Enable debug mode with detailed logging format.
    
    Shows file names, line numbers, and full timestamps.
    Useful for development and troubleshooting.
    """
    setup_logging(
        level=logging.DEBUG,
        format_string=DEBUG_FORMAT,
        force=True
    )


def enable_production_mode() -> None:
    """
    Enable production mode with INFO level and standard format.
    
    Reduces log verbosity for production environments.
    """
    setup_logging(
        level=logging.INFO,
        format_string=DEFAULT_FORMAT,
        force=True
    )


# Convenience function for test environments
def enable_test_mode() -> None:
    """
    Enable test mode with simple format (no timestamps).
    
    Useful for pytest output - cleaner test logs.
    """
    setup_logging(
        level=logging.INFO,
        format_string=SIMPLE_FORMAT,
        force=True
    )


if __name__ == "__main__":
    """Demo of different logging configurations."""
    print("=" * 70)
    print("📋 Logging Configuration Demo")
    print("=" * 70)
    print()
    
    # Demo 1: Default configuration
    print("1️⃣  Default configuration (INFO level):")
    setup_logging()
    logger = get_logger(__name__)
    logger.debug("This debug message won't show")
    logger.info("This is an info message")
    logger.warning("This is a warning")
    logger.error("This is an error")
    print()
    
    # Demo 2: Debug mode
    print("2️⃣  Debug mode (detailed format):")
    enable_debug_mode()
    logger.debug("Now debug messages are visible!")
    logger.info("Info with file location")
    print()
    
    # Demo 3: Test mode
    print("3️⃣  Test mode (simple format):")
    enable_test_mode()
    logger.info("Clean test output")
    logger.warning("No timestamps in test mode")
    print()
    
    # Demo 4: Named loggers
    print("4️⃣  Named loggers:")
    setup_logging(force=True)
    app_logger = get_logger("app")
    db_logger = get_logger("database")
    app_logger.info("Application started")
    db_logger.info("Database connected")
    print()
    
    print("=" * 70)
    print("✅ All logging configurations working!")
    print("=" * 70)
