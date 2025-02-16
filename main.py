import pandas as pd
from textblob import TextBlob
import spacy
from transformers import pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from concurrent.futures import ProcessPoolExecutor


# Function for Sentiment Analysis
def sentiment_analysis(text):
    blob = TextBlob(text)
    return blob.sentiment


# Function for Topic Extraction using spaCy NER
def extract_topics(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return [(entity.text, entity.label_) for entity in doc.ents]


# Function for Categorization
def categorize_text(text):
    problems_keywords = ['littering', 'overflowing', 'insufficient', 'trash bins', 'health risk']
    if any(keyword in text.lower() for keyword in problems_keywords):
        return 'Problem occurring locally'
    else:
        return 'New initiatives'


# Function for Abstractive Summarization
def abstractive_summarization(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']


# Function for Extractive Summarization using Sumy
def extractive_summarization(text):
    parser = PlaintextParser.from_string(text, PlaintextParser.from_string(text))
    summarizer = LsaSummarizer()
    summary = summarizer.parse(parser.document, sentences_count=3)
    return " ".join([str(sentence) for sentence in summary])


# Function for Query-based Summarization
def query_based_summarization(text, question):
    qa_pipeline = pipeline("question-answering")
    answer = qa_pipeline(question=question, context=text)
    return answer['answer']


# Main function to process a batch of letters
def process_letter(letter):
    # Perform all the analyses and summarizations
    sentiment = sentiment_analysis(letter['text'])
    topics = extract_topics(letter['text'])
    category = categorize_text(letter['text'])
    abstractive_summary = abstractive_summarization(letter['text'])
    extractive_summary = extractive_summarization(letter['text'])
    query_answer = query_based_summarization(letter['text'], "What is the writer addressing?")

    return {
        'sentiment': sentiment,
        'topics': topics,
        'category': category,
        'abstractive_summary': abstractive_summary,
        'extractive_summary': extractive_summary,
        'query_answer': query_answer
    }


# Example of processing multiple letters (e.g., from a CSV or database)
def process_batch_of_letters(letters):
    # Process letters in parallel using multiprocessing
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_letter, letters))

    # Create DataFrame for results
    df = pd.DataFrame(results)
    return df


# Sample input - a list of dictionaries, each representing a letter with a 'text' key
letters = [
    {
        "text": """Dear Council, I am writing to bring attention to the recent increase in littering in our neighborhood. In the past few months, I've noticed an alarming amount of trash being left in public areas, especially near the park and along Main Street. The current trash bins are overflowing and appear to be insufficient for the amount of waste. This is not only unsightly but also poses a serious health risk to local residents. I urge the council to increase the number of bins and arrange for more frequent waste collection."""},
    {
        "text": """Dear Council, I have observed several streetlights in our neighborhood are broken and need immediate repair. This has caused safety concerns for residents, particularly at night. I kindly request that the council prioritize this matter and ensure the lights are fixed as soon as possible."""}
]

# Process the batch of letters
df_results = process_batch_of_letters(letters)

# Print the DataFrame with results
print(df_results)
