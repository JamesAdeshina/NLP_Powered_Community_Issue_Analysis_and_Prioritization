from models.load_models import get_sentiment_pipeline
from config import setup_logger  # If using logging
import logging
# Initialize logger at module level
# logger = setup_logger()

# Create module-level logger that always exists
logger = logging.getLogger(__name__)

# Create module-level logger that always exists
def sentiment_analysis(text):

    """Single source of truth for sentiment analysis"""
    logger.debug("Analyzing sentiment...")

    try:
        # Remove all logger existence checks - just use it directly
        if not text or not isinstance(text, str) or not text.strip():
            error_msg = "Empty text provided for sentiment analysis"
            logger.warning(error_msg)
            return {
                "sentiment_label": "NEUTRAL",
                "confidence": 0.5,
                "explanation": error_msg,
                "status": "warning"
            }

        sentiment_pipeline = get_sentiment_pipeline()
        transformer_result = sentiment_pipeline(text)[0]

        if not transformer_result:
            error_msg = "Sentiment analysis returned no results"
            logger.error(error_msg)
            return {
                "sentiment_label": "NEUTRAL",
                "confidence": 0.5,
                "explanation": error_msg,
                "status": "error"
            }

        best_result = max(transformer_result, key=lambda x: x['score'])
        logger.info(f"Analyzed sentiment: {best_result['label']} (confidence: {best_result['score']:.2f})")

        return {
            "sentiment_label": best_result['label'].upper(),
            "confidence": best_result['score'],
            "explanation": f"Classified as {best_result['label']} with {best_result['score']:.0%} confidence",
            "status": "success"
        }

    except Exception as e:
        error_msg = f"Sentiment analysis failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "sentiment_label": "NEUTRAL",
            "confidence": 0.5,
            "explanation": error_msg,
            "status": "error"
        }
"""**
def sentiment_analysis(text):
    try:
        # Get logger from session state
        logger = st.session_state.get('logger')

        if not text or not isinstance(text, str) or not text.strip():
            error_msg = "Empty text provided for sentiment analysis"
            if logger:
                logger.warning(error_msg)
            return {
                "sentiment_label": "NEUTRAL",
                "confidence": 0.5,
                "explanation": error_msg,
                "status": "warning"
            }

        sentiment_pipeline = get_sentiment_pipeline()
        transformer_result = sentiment_pipeline(text)[0]

        if not transformer_result:
            error_msg = "Sentiment analysis returned no results"
            if logger:
                logger.error(error_msg)
            return {
                "sentiment_label": "NEUTRAL",
                "confidence": 0.5,
                "explanation": error_msg,
                "status": "error"
            }

        best_result = max(transformer_result, key=lambda x: x['score'])

        if logger:
            logger.info(
                f"Successfully analyzed sentiment: {best_result['label']} (confidence: {best_result['score']:.2f})")

        return {
            "sentiment_label": best_result['label'].upper(),
            "confidence": best_result['score'],
            "explanation": f"Classified as {best_result['label']} with {best_result['score']:.0%} confidence",
            "status": "success"
        }

    except Exception as e:
        error_msg = f"Sentiment analysis failed: {str(e)}"
        if logger:
            logger.error(error_msg, exc_info=True)
        return {
            "sentiment_label": "NEUTRAL",
            "confidence": 0.5,
            "explanation": error_msg,
            "status": "error"
        }
    
"""