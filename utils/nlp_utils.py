import nltk
from nltk.tokenize import sent_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def initialize_nltk():
    if 'nltk_initialized' not in st.session_state:
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            for resource in config.NLTK_RESOURCES:
                nltk.download(resource, quiet=True)
            st.session_state.nltk_initialized = True


def extractive_summarization(text, sentence_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join([str(sentence) for sentence in summary])


def query_based_summarization(text, query, threshold=0.1, top_n=2):
    sentences = sent_tokenize(text)
    if not sentences:
        return "No relevant information found for the query."

    action_indicators = ["urge", "request", "increase", "arrange", "immediate",
                         "control", "measure", "action", "implement", "improve", "take"]

    is_action_query = any(word in query.lower() for word in ["action", "request", "urge",
                                                             "increase", "immediate", "control", "measure"])
    if is_action_query:
        threshold = 0.05

    corpus = sentences + [query]
    vectorizer_q = TfidfVectorizer().fit(corpus)
    sentence_vectors = vectorizer_q.transform(sentences)
    query_vector = vectorizer_q.transform([query])
    scores = np.dot(sentence_vectors, query_vector.T).toarray().flatten()

    valid_indices = [i for i, score in enumerate(scores) if score >= threshold]
    if not valid_indices:
        return "No relevant information found for the query."

    if is_action_query:
        valid_indices = [i for i in valid_indices if any(kw in sentences[i].lower() for kw in action_indicators)]
        if not valid_indices:
            return "No relevant information found for the query."

    sorted_indices = sorted(valid_indices, key=lambda i: scores[i], reverse=True)[:top_n]
    selected_indices = sorted(sorted_indices)
    summary = " ".join(sentences[i] for i in selected_indices)
    return summary


def personalize_summary(summary, summary_type="general"):
    return f" {summary}"