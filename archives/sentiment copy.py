import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.nn.functional import softmax


# Function to load the models and make predictions
def analyze_sentiment(model_name, text):
    if model_name == 'distilbert':
        # Load DistilBERT model and tokenizer
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    elif model_name == 'roberta':
        # Load RoBERTa model and tokenizer
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Get model predictions
    with torch.no_grad():
        logits = model(**inputs).logits

    # Apply softmax to get probabilities
    probs = softmax(logits, dim=-1)

    # Get predicted sentiment label
    sentiment_label = torch.argmax(probs, dim=-1).item()

    # Convert numerical sentiment to text
    sentiment = "positive" if sentiment_label == 1 else "negative"

    return sentiment, probs[0][sentiment_label].item()


# Example usage
text = "1 Church Street, Bolsover, S44 6JJ Dear Council Member, I am writing to express my growing frustration regarding the terrible condition of our local roads. Over the past several months, the once smooth surfaces of Main Road and its adjoining lanes have been severely damaged by numerous potholes and cracks. The recent heavy rains have worsened these conditions, causing water to pool in the damaged areas, making travel extremely dangerous for both drivers and pedestrians. The current state of our roads not only presents serious safety hazards but also ruins the overall image of our community and hinders local business activities. It is clear that the Bolsover District Council has neglected this issue for far too long. I strongly urge the Bolsover District Council to urgently conduct a thorough survey of the affected areas and allocate the necessary resources for prompt repairs and long-term maintenance. The longer repairs are delayed, the worse the situation will become, and this is no longer acceptable. A proper maintenance plan with clear schedules is essential to prevent further deterioration and to ensure that our infrastructure does not continue to decline. Additionally, better communication about repair schedules would help residents cope with these ongoing issues. I trust that you will finally treat this matter with the urgency it deserves and take immediate action to improve road safety in our area. The community members who rely on these roads daily are growing increasingly frustrated and need a prompt response. Yours faithfully, A Frustrated Resident"

# Using DistilBERT
distilbert_sentiment, distilbert_prob = analyze_sentiment('distilbert', text)
print(f"DistilBERT Sentiment: {distilbert_sentiment}, Probability: {distilbert_prob:.4f}")

# Using RoBERTa
roberta_sentiment, roberta_prob = analyze_sentiment('roberta', text)
print(f"RoBERTa Sentiment: {roberta_sentiment}, Probability: {roberta_prob:.4f}")

