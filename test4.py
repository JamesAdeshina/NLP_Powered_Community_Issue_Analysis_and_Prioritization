import ssl
import re
import emoji
import contractions
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline as hf_pipeline
import numpy as np
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# ------------------ SSL Context for NLTK Downloads ------------------
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# ------------------ NLTK Downloads ------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')


# ------------------ Preprocessing Functions ------------------
def remove_email_headers_and_footers(text):
    lines = text.split('\n')
    stripped_lines = [line.strip() for line in lines]
    if "" in stripped_lines:
        first_blank_index = stripped_lines.index("")
        content = "\n".join(lines[first_blank_index + 1:]).strip()
    else:
        content = text
    signature_markers = ['sincerely,', 'regards,', 'best regards,', 'thanks,', 'thank you,']
    final_lines = []
    for line in content.split('\n'):
        if any(line.lower().startswith(marker) for marker in signature_markers):
            break
        final_lines.append(line)
    return "\n".join(final_lines).strip()


def remove_emojis(text):
    return emoji.replace_emoji(text, replace="")


def expand_contractions(text):
    return contractions.fix(text)


def remove_urls(text):
    return re.sub(r'http\S+|www\S+', '', text)


def remove_mentions_hashtags(text):
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    return text


def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)


def remove_numbers(text):
    return re.sub(r'\d+', '', text)


def normalize_repeated_chars(text):
    return re.sub(r'(.)\1{2,}', r'\1', text)


def remove_extra_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()


def tokenize_and_lower(text):
    tokens = word_tokenize(text)
    return [token.lower() for token in tokens]


def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]


def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


def comprehensive_text_preprocessing(text, use_lemmatization=True):
    text = remove_email_headers_and_footers(text)
    text = remove_emojis(text)
    text = expand_contractions(text)
    text = remove_urls(text)
    text = remove_mentions_hashtags(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = normalize_repeated_chars(text)
    text = remove_extra_whitespace(text)
    tokens = tokenize_and_lower(text)
    tokens = remove_stopwords(tokens)
    if use_lemmatization:
        tokens = lemmatize_tokens(tokens)
    return ' '.join(tokens)


# ------------------ Unsupervised Classification Functions ------------------
def unsupervised_classification(texts, num_clusters=2):
    # Use n-grams for richer feature representation.
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans.labels_, vectorizer, kmeans


# ------------------ Dynamic Topic Labeling ------------------
candidate_labels = ["Local Problem", "New Initiatives"]
zero_shot_classifier = hf_pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def dynamic_label_clusters(vectorizer, kmeans):
    cluster_labels = {}
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    for i in range(kmeans.n_clusters):
        # Extract top 10 terms for each cluster.
        top_terms = [terms[ind] for ind in order_centroids[i, :10]]
        keyword_str = ", ".join(top_terms)
        result = zero_shot_classifier(keyword_str, candidate_labels)
        best_label = result["labels"][0]
        cluster_labels[i] = best_label
    return cluster_labels


# ------------------ Topic Modeling ------------------
def topic_modeling(texts, num_topics=1):
    # Using n-grams to capture multi-word phrases.
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42, learning_method='batch', max_iter=10)
    lda.fit(X)
    topics = []
    terms = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_terms = [terms[i] for i in topic.argsort()[:-6:-1]]
        topics.append(", ".join(top_terms))
    return topics


def dynamic_topic_label(keywords):
    result = zero_shot_classifier(keywords, candidate_labels)
    best_label = result["labels"][0]
    return best_label


# ------------------ Summarization Functions ------------------
# Cache the abstractive summarizer so it's not reloaded each time.
ABSTRACTIVE_SUMMARIZER = hf_pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", revision="a4f8f3e")


def abstractive_summarization(text):
    summary = ABSTRACTIVE_SUMMARIZER(text, max_length=50, min_length=25, do_sample=False)
    return summary[0]['summary_text']


def extractive_summarization(text, sentence_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join([str(sentence) for sentence in summary])


def query_based_summarization(text, query, threshold=0.1, top_n=2):
    sentences = sent_tokenize(text)
    if not sentences:
        return "No relevant information found for the query."
    corpus = sentences + [query]
    vectorizer = TfidfVectorizer().fit(corpus)
    sentence_vectors = vectorizer.transform(sentences)
    query_vector = vectorizer.transform([query])
    scores = np.dot(sentence_vectors, query_vector.T).toarray().flatten()
    valid_indices = [i for i, score in enumerate(scores) if score >= threshold]
    if not valid_indices:
        return "No relevant information found for the query."
    sorted_indices = sorted(valid_indices, key=lambda i: scores[i], reverse=True)[:top_n]
    selected_indices = sorted(sorted_indices)
    summary = " ".join(sentences[i] for i in selected_indices)
    return summary


# ------------------ Sentiment Analysis ------------------
def sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        sentiment_label = "Positive"
    elif compound <= -0.05:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"
    explanation = f"The sentiment of the text is {sentiment_label}."
    if sentiment_label == "Negative":
        details = []
        lower_text = text.lower()
        if "litter" in lower_text or "trash" in lower_text:
            details.append("an increase in littering")
        if "overflowing" in lower_text or "bins" in lower_text:
            details.append("overflowing trash bins")
        if "risk" in lower_text or "health" in lower_text:
            details.append("associated health risks")
        if details:
            explanation += " Note based on sentiment; the author is highlighting concerns about " + ", ".join(
                details) + ", which creates a sense of urgency and dissatisfaction."
    return {"scores": scores, "sentiment_label": sentiment_label, "explanation": explanation}


# ------------------ Data Loading ------------------
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print("Error loading data:", e)
        return None


# ------------------ Main Pipeline ------------------
def main():
    file_path = "Data/processed/community_issues_dataset_template.csv"
    df = load_data(file_path)
    if df is None:
        return

    # Preprocess texts
    df['clean_text'] = df['text'].apply(comprehensive_text_preprocessing)
    df = df[df['clean_text'].str.strip() != '']
    if df.empty:
        print("No valid texts for clustering after preprocessing.")
        return

    texts = df['clean_text'].tolist()
    if not texts:
        print("No valid texts for clustering.")
        return

    # Unsupervised clustering (using n-grams)
    labels, vectorizer, kmeans = unsupervised_classification(texts, num_clusters=2)
    # Dynamic labeling for clusters using zero-shot classification
    cluster_mapping = dynamic_label_clusters(vectorizer, kmeans)
    df['classification'] = [cluster_mapping[label] for label in labels]
    print("\nClassification Distribution:")
    print(df['classification'].value_counts())

    # Topic Modeling & Dynamic Topic Labeling for each category:
    for category in ["Local Problem", "New Initiatives"]:
        subset_texts = df[df['classification'] == category]['clean_text'].tolist()
        if subset_texts:
            topics = topic_modeling(subset_texts, num_topics=5)
            print(f"\nExtracted Topics for {category}:")
            for topic in topics:
                dynamic_label = dynamic_topic_label(topic)
                print(dynamic_label, f"(Keywords: {topic})")
        else:
            print(f"No letters classified as {category} for topic modeling.")

    # Process a sample letter for demonstration:
    sample_text = df['text'].iloc[0]
    print("\nSample Letter Original:")
    print(sample_text)
    print("\nSample Letter Summaries:")
    print("Abstractive Summary:", abstractive_summarization(sample_text))
    print("Extractive Summary:", extractive_summarization(sample_text))
    print("Query-based Summary 1 (for query 'air pollution'):",
          query_based_summarization(sample_text, query="air pollution"))
    print("Query-based Summary 2 (for query 'What are the main concerns raised in the letter?'):",
          query_based_summarization(sample_text, query="What are the main concerns raised in the letter?"))
    print("Query-based Summary 3 (for query 'What actions are being urged in the letter?'):",
          query_based_summarization(sample_text, query="What actions are being urged in the letter?"))

    sentiment_results = sentiment_analysis(sample_text)
    compound_score = sentiment_results["scores"]["compound"]
    if compound_score >= 0.1:
        overall_sentiment = "Positive"
    elif compound_score <= -0.1:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"
    print("\nSentiment Analysis of Sample Letter:")
    print(sentiment_results)
    print(f"Overall Sentiment: {overall_sentiment}")
    blob = TextBlob(sample_text)
    print(f"Sentiment polarity: {blob.sentiment.polarity}")


if __name__ == '__main__':
    main()
