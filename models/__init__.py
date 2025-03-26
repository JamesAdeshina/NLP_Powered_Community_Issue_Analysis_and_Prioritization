"""
Models package for Bolsover District Council Analysis

Exposes:
- Classification models
- Sentiment analysis
- Summarization
- Embeddings
- Model loading utilities
"""

from .classification import classify_document
from .sentiment import sentiment_analysis
from .summarization import get_summaries
from .embeddings import get_embeddings, semantic_search
from .load_models import (
    get_zero_shot_classifier,
    get_abstractive_summarizer,
    get_sentiment_pipeline,
    get_qa_pipeline,
    get_embeddings_model,
    get_ner_pipeline,
    load_paraphrase_model
)

__all__ = [
    'classify_document',
    'sentiment_analysis',
    'get_summaries',
    'get_embeddings',
    'semantic_search',
    'get_zero_shot_classifier',
    'get_abstractive_summarizer',
    'get_sentiment_pipeline',
    'get_qa_pipeline',
    'get_embeddings_model',
    'get_ner_pipeline',
    'load_paraphrase_model'
]