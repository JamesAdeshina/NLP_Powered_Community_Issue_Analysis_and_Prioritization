import re
import logging
from typing import List, Tuple, Optional
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_sentiment_pipeline():
    # Load the sentiment analysis pipeline
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


def extract_positive_phrases(text: str) -> str:
    """Extract positive phrases from text using simple keyword matching."""
    positive_indicators = ["appreciate", "thank", "great", "improved", "happy", "pleased", "excellent"]
    phrases = []
    sentences = text.split('.')
    for sentence in sentences:
        if any(word.lower() in sentence.lower() for word in positive_indicators):
            phrases.append(sentence.strip())
    return ', '.join(phrases[:3]) if phrases else "positive aspects of the situation"


def extract_negative_phrases(text: str) -> str:
    """Extract negative phrases from text using simple keyword matching."""
    negative_indicators = ["concern", "problem", "issue", "worry", "disappoint", "poor", "lack"]
    phrases = []
    sentences = text.split('.')
    for sentence in sentences:
        if any(word.lower() in sentence.lower() for word in negative_indicators):
            phrases.append(sentence.strip())
    return ', '.join(phrases[:3]) if phrases else "specific problems or concerns"


def generate_sentiment_explanation(text: str, sentiment_label: str) -> str:
    """
    Generate a human-readable explanation for the sentiment.
    For positive sentiment, extracts positive phrases; for negative sentiment, negative phrases.
    """
    if sentiment_label == "POSITIVE":
        return f"The text expresses appreciation or satisfaction, mentioning positive aspects like: {extract_positive_phrases(text)}"
    elif sentiment_label == "NEGATIVE":
        return f"The text highlights concerns or problems, mentioning issues like: {extract_negative_phrases(text)}"
    else:
        return "The text appears neutral, presenting facts without strong positive or negative language."


def sentiment_analysis(text: str) -> dict:
    """Enhanced sentiment analysis with explanation."""
    logger.debug("Analyzing sentiment...")
    try:
        # Input validation
        if not text or not isinstance(text, str) or not text.strip():
            error_msg = "Empty text provided for sentiment analysis"
            logger.warning(error_msg)
            return {
                "sentiment_label": "NEUTRAL",
                "confidence": 0.5,
                "explanation": error_msg,
                "detailed_explanation": "No text was provided for analysis",
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
                "detailed_explanation": "The analysis couldn't determine sentiment",
                "status": "error"
            }

        sentiment_label = transformer_result["label"].upper()
        # Generate detailed explanation based on the text and sentiment label
        detailed_explanation = generate_sentiment_explanation(text, sentiment_label)

        logger.info(f"Analyzed sentiment: {sentiment_label} (confidence: {transformer_result['score']:.2f})")
        return {
            "sentiment_label": sentiment_label,
            "confidence": transformer_result["score"],
            "explanation": f"Classified as {sentiment_label} with {transformer_result['score']:.0%} confidence",
            "detailed_explanation": detailed_explanation,
            "status": "success"
        }
    except Exception as e:
        error_msg = f"Sentiment analysis failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "sentiment_label": "NEUTRAL",
            "confidence": 0.5,
            "explanation": error_msg,
            "detailed_explanation": "An error occurred during sentiment analysis",
            "status": "error"
        }


def main():
    text = input("Enter text for sentiment analysis:\n")
    result = sentiment_analysis(text)
    print("\nBrief Explanation:")
    print(result["explanation"])
    print("\nDetailed Explanation:")
    print(result["detailed_explanation"])
    print("\nFull Result:")
    print(result)


if __name__ == "__main__":
    main()
