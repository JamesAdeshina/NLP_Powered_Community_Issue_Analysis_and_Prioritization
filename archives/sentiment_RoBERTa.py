from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy

# Load the pre-trained RoBERTa model for sequence classification
model = RobertaForSequenceClassification.from_pretrained('roberta-base',
                                                         num_labels=3)  # 3 labels: Negative, Neutral, Positive
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Load spaCy model for named entity recognition (NER) and linguistic features (for key phrase extraction)
nlp = spacy.load("en_core_web_sm")

# Sample letter (less than 600 words)
letter = """
Dear Council,

I hope this letter finds you well. I am writing to express my concern regarding the current state of the road along Oxford Street. Over the past few months, several potholes have formed, making it very difficult and unsafe for drivers, especially at night when visibility is poor. I have personally witnessed multiple incidents of vehicles swerving to avoid these potholes, and this is becoming a serious safety issue for both drivers and pedestrians. 

It has also become increasingly difficult to park along the street due to the worsening condition of the road. The potholes are not only a driving hazard but also damage the vehicles themselves, which has been a growing complaint among residents in the area. 

I kindly request that the council take immediate action to repair the road and address the safety concerns. I would also appreciate any updates regarding the timeline for these repairs and how the residents of Oxford Street can contribute to this process, if possible.

Thank you for your attention to this matter. I look forward to your prompt response.

Sincerely,
[Your Name]
"""

# Tokenize the letter and prepare it for input into the model
inputs = tokenizer(letter, return_tensors='pt', truncation=True, padding=True, max_length=512)

# Predict sentiment using the pre-trained model
with torch.no_grad():
    logits = model(**inputs).logits

# Get the predicted sentiment (0: Negative, 1: Neutral, 2: Positive)
predicted_class_id = torch.argmax(logits, dim=-1).item()
sentiments = ["Negative", "Neutral", "Positive"]
predicted_sentiment = sentiments[predicted_class_id]


# Function to generate explanation dynamically
def generate_explanation(letter, sentiment):
    # Process the letter using spaCy
    doc = nlp(letter)

    # Extract entities, noun phrases, and important keywords
    important_keywords = []
    for ent in doc.ents:
        important_keywords.append(ent.text)  # Extract named entities like locations, people, etc.

    for np in doc.noun_chunks:
        important_keywords.append(np.text)  # Extract noun phrases (potential key topics in the letter)

    # Also, extract verbs and adjectives (which can provide tone and action)
    for token in doc:
        if token.pos_ in ["VERB", "ADJ"]:
            important_keywords.append(token.text)

    # Based on sentiment, summarize why the letter has that sentiment
    explanation = ""

    if sentiment == "Negative":
        explanation += "The letter expresses dissatisfaction with the current conditions, highlighting specific problems such as "
        explanation += ", ".join(
            important_keywords[:3]) + ". These issues seem to be causing safety concerns and inconvenience."
    elif sentiment == "Positive":
        explanation += "The letter is respectful and polite, expressing gratitude and a constructive tone, mentioning "
        explanation += ", ".join(
            important_keywords[:3]) + ". The author is requesting improvements or support in a positive manner."
    elif sentiment == "Neutral":
        explanation += "The letter provides a factual description of the current situation, focusing on "
        explanation += ", ".join(
            important_keywords[:3]) + ". It doesn't convey strong emotion or a call for urgent action."

    return explanation


# Generate an explanation for the sentiment prediction
explanation = generate_explanation(letter, predicted_sentiment)

# Output the predicted sentiment and explanation
print(f"Predicted Sentiment: {predicted_sentiment}")
print(f"Explanation: {explanation}")
