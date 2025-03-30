from models.load_models import get_sentiment_pipeline
from rake_nltk import Rake
import logging

# 2. LOGGING SETUP
from config.logging_config import setup_logger


# Initialize logger at module level
logger = setup_logger()

# Create module-level logger that always exists
module_logger = logging.getLogger(__name__)

def generate_note(sentiment_label, keyphrases):
    """Generate human-readable summary note based on sentiment and keyphrases."""

    """Single source of truth for sentiment analysis"""
    logger.debug("Analyzing sentiment...")

    if not keyphrases:
        issues = "various issues"
    else:
        # Format as "X, Y, and Z"
        issues = ", ".join(keyphrases[:-1])
        if len(keyphrases) > 1:
            issues += f" and {keyphrases[-1]}"
        else:
            issues = keyphrases[0]

    sentiment_map = {
        "NEGATIVE": ("urgency and dissatisfaction", "health, safety, or environmental risks"),
        "POSITIVE": ("satisfaction and approval", "positive community impact"),
        "NEUTRAL": ("mixed implications", "potential concerns")
    }

    implication, risk = sentiment_map.get(sentiment_label, ("mixed implications", "potential concerns"))

    return f"The author is highlighting concerns about {issues}, which creates a sense of {implication} due to {risk}."


def sentiment_analysis(text):
    # Initialize logger first to prevent UnboundLocalError
    logger = module_logger

    try:
        # Attempt to get Streamlit logger if available
        import streamlit as st
        logger = st.session_state.get('logger', module_logger)
    except (ImportError, AttributeError, KeyError):
        pass  # Keep module logger if Streamlit isn't available

    try:
        # Input validation
        if not text or not isinstance(text, str) or not text.strip():
            error_msg = "Empty text provided for sentiment analysis"
            logger.warning(error_msg)
            return {
                "sentiment_label": "NEUTRAL",
                "confidence": 0.5,
                "explanation": error_msg,
                "note": "No meaningful text to analyze",
                "status": "warning"
            }

        # Sentiment analysis
        sentiment_pipeline = get_sentiment_pipeline()
        transformer_result = sentiment_pipeline(text)[0]

        if not transformer_result:
            error_msg = "Sentiment analysis returned no results"
            logger.error(error_msg)
            return {
                "sentiment_label": "NEUTRAL",
                "confidence": 0.5,
                "explanation": error_msg,
                "note": "Analysis failed to produce results",
                "status": "error"
            }

        # Process results
        best_result = max(transformer_result, key=lambda x: x['score'])
        sentiment_label = best_result['label'].upper()

        # Keyphrase extraction
        r = Rake()
        r.extract_keywords_from_text(text)
        keyphrases = r.get_ranked_phrases()[:3]

        # Generate output
        logger.info(f"Analyzed sentiment: {sentiment_label} (confidence: {best_result['score']:.2f})")

        return {
            "sentiment_label": sentiment_label,
            "confidence": best_result['score'],
            "explanation": f"Classified as {sentiment_label.lower()} with {best_result['score']:.0%} confidence",
            "note": generate_note(sentiment_label, keyphrases),
            "status": "success"
        }

    except Exception as e:
        error_msg = f"Sentiment analysis failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "sentiment_label": "NEUTRAL",
            "confidence": 0.5,
            "explanation": error_msg,
            "note": "System error during analysis",
            "status": "error"
        }



