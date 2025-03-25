from models.load_models import get_sentiment_pipeline


def sentiment_analysis(text):
    sentiment_pipeline = get_sentiment_pipeline()
    transformer_result = sentiment_pipeline(text)[0]

    sentiment_label = max(transformer_result, key=lambda x: x['score'])['label']
    confidence_score = max(transformer_result, key=lambda x: x['score'])['score']

    confidence = f"Classified as {sentiment_label} with {confidence_score:.0%} confidence"
    explanation = f"Classified as {sentiment_label} with {confidence_score:.0%} confidence"

    return {
        "Sentiment_label": sentiment_label,
        "Confidence score": confidence_score,
        "Confidence": confidence_score,
        "Explanation": explanation
    }