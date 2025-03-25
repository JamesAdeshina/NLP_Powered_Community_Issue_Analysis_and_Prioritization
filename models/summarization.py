from models.load_models import get_abstractive_summarizer
from utils.nlp_utils import extractive_summarization, query_based_summarization

def abstractive_summarization(text):
    summarizer = get_abstractive_summarizer()
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
    return summary[0]['summary_text']

def get_summaries(text, query="What actions are being urged in the letter?"):
    abstractive = abstractive_summarization(text)
    extractive = extractive_summarization(text)
    query_based = query_based_summarization(text, query)
    return {
        "abstractive": abstractive,
        "extractive": extractive,
        "query_based": query_based
    }