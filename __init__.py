"""
Bolsover District Council Analysis Package

This package provides tools for analyzing resident communications for Bolsover District Council.
"""

# Version of the package
__version__ = "1.0.0"

# Optional: List of what to import when someone does 'from bolsover_analysis import *'
__all__ = []  # Best practice is to keep this empty and be explicit in imports

# Optional: Package-level imports that should be available when importing the package
from .config.logging_config import setup_logger

# Optional: Initialize logging when package is imported
logger = setup_logger()
logger.info(f"Initialized bolsover_analysis package v{__version__}")