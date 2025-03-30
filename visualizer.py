import nltk
from transformers import pipeline
from textblob import TextBlob

# Download necessary nltk resources
nltk.download('punkt')

# Initialize sentiment analysis pipeline from Hugging Face's transformers library
sentiment_analyzer = pipeline('sentiment-analysis')


# Function to summarize and analyze sentiment of a text
def analyze_letter(letter_text):
    # Step 1: Sentiment Analysis
    sentiment_result = sentiment_analyzer(letter_text)
    sentiment = sentiment_result[0]['label']

    # Step 2: Extract main concerns (very basic, based on keyword matching or summarization)
    main_concerns = extract_main_concerns(letter_text)

    # Step 3: Generate a brief summary of the letter
    summary = summarize_text(letter_text)

    # Step 4: Use TextBlob for an additional sentiment score (polarity and subjectivity)
    blob = TextBlob(letter_text)
    polarity = blob.sentiment.polarity  # range from -1 (negative) to 1 (positive)
    subjectivity = blob.sentiment.subjectivity  # range from 0 (objective) to 1 (subjective)

    # Prepare a summary result
    analysis = {
        "Sentiment": sentiment,
        "Summary": summary,
        "Main Concerns": main_concerns,
        "Polarity": polarity,
        "Subjectivity": subjectivity
    }

    return analysis


# Function to extract main concerns using basic keyword search (can be enhanced with more advanced techniques)
def extract_main_concerns(text):
    concerns = []
    keywords = ['littering', 'trash', 'health risk', 'bins', 'overflowing', 'waste']

    for keyword in keywords:
        if keyword in text.lower():
            concerns.append(keyword)

    return concerns if concerns else ["No specific concerns found."]


# Function to summarize the text (simple summarization using nltk sentence tokenizer)
def summarize_text(text):
    sentences = nltk.sent_tokenize(text)
    summary = " ".join(sentences[:2])  # Return the first two sentences as a basic summary
    return summary


# Example Letter
letter = """
Dear Council,
I am writing to bring attention to the recent increase in littering in our neighborhood. In the
past few months, I've noticed an alarming amount of trash being left in public areas,
especially near the park and along Main Street. The current trash bins are overflowing and
appear to be insufficient for the amount of waste. This is not only unsightly but also poses a
serious health risk to local residents. I urge the council to increase the number of bins and
arrange for more frequent waste collection.
Thank you for your attention to this matter.
"""

# Analyze the letter
analysis = analyze_letter(letter)

# Output results
print(f"Sentiment: {analysis['Sentiment']}")
print(f"Summary: {analysis['Summary']}")
print(f"Main Concerns: {analysis['Main Concerns']}")
print(f"Polarity: {analysis['Polarity']}")
print(f"Subjectivity: {analysis['Subjectivity']}")
