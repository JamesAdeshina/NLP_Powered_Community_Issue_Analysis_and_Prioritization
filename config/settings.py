# config/settings.py
import os
import ssl
import nltk

# SSL Context for NLTK Downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Page configuration
PAGE_CONFIG = {
    "layout": "wide",
    "page_title": "Bolsover District Council - Analysis",
    "page_icon": "🏛️"
}

# UK address patterns
UK_POSTCODE_REGEX = r'\b[A-Z]{1,2}\d{1,2}[A-Z]?\s\d[A-Z]{2}\b'
UK_ADDRESS_REGEX = r'\b\d+\s[\w\s]+\b,\s[\w\s]+,\s' + UK_POSTCODE_REGEX

# API Keys (use environment variables in production)
OPENCAGE_API_KEY = "e760785d8c7944888beefc24aa42eb66"

# NLTK resources
NLTK_RESOURCES = [
    'punkt',
    'stopwords',
    'wordnet',
    'vader_lexicon',
    'averaged_perceptron_tagger',
    'omw-1.4'
]

# Classification labels
CLASSIFICATION_LABELS = ["Local Problem", "New Initiatives"]