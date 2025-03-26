# config/__init__.py
"""
Configuration module for Bolsover District Council Analysis
"""

# Import order matters - basic config first

# config/__init__.py
from .logging_config import setup_logger
from .settings import (
    PAGE_CONFIG,
    UK_POSTCODE_REGEX,
    UK_ADDRESS_REGEX,
    OPENCAGE_API_KEY,
    NLTK_RESOURCES,
    CLASSIFICATION_LABELS
)
from .constants import CANDIDATE_LABELS_TOPIC

__all__ = [
    'setup_logger',
    'PAGE_CONFIG',
    'UK_POSTCODE_REGEX',
    'UK_ADDRESS_REGEX',
    'OPENCAGE_API_KEY',
    'NLTK_RESOURCES',
    'CLASSIFICATION_LABELS',
    'CANDIDATE_LABELS_TOPIC'
]