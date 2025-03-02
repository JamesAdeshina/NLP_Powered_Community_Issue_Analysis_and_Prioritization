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
    """
    Remove common email headers and footers.
    If the text contains a blank line, assume everything before the first blank line is a header.
    Otherwise, return the original text.
    Also remove footers based on common signature markers.
    """
    lines = text.split('\n')
    # Check if there is a blank line (indicating end of header)
    stripped_lines = [line.strip() for line in lines]
    if "" in stripped_lines:
        # Find the first blank line and take the text after it
        first_blank_index = stripped_lines.index("")
        content = "\n".join(lines[first_blank_index + 1:]).strip()
    else:
        content = text  # No header detected, return original text

    # Remove footer: stop if a common signature marker is encountered.
    signature_markers = ['sincerely,', 'regards,', 'best regards,', 'thanks,', 'thank you,']
    final_lines = []
    for line in content.split('\n'):
        if any(line.lower().startswith(marker) for marker in signature_markers):
            break
        final_lines.append(line)
    return "\n".join(final_lines).strip()


def remove_emojis(text):
    """Remove emojis from the text."""
    return emoji.replace_emoji(text, replace="")


def expand_contractions(text):
    """Expand contractions (e.g., don't -> do not)."""
    return contractions.fix(text)


def remove_urls(text):
    """Remove URLs from the text."""
    return re.sub(r'http\S+|www\S+', '', text)


def remove_mentions_hashtags(text):
    """Remove social media mentions and hashtags."""
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    return text


def remove_punctuation(text):
    """Remove punctuation from the text."""
    return re.sub(r'[^\w\s]', '', text)


def remove_numbers(text):
    """Remove digits from the text."""
    return re.sub(r'\d+', '', text)


def normalize_repeated_chars(text):
    """Normalize words with excessive repeated characters (e.g., soooo -> so)."""
    return re.sub(r'(.)\1{2,}', r'\1', text)


def remove_extra_whitespace(text):
    """Remove extra whitespace from the text."""
    return re.sub(r'\s+', ' ', text).strip()


def tokenize_and_lower(text):
    """Tokenize the text and convert tokens to lowercase."""
    tokens = word_tokenize(text)
    return [token.lower() for token in tokens]


def remove_stopwords(tokens):
    """Remove stopwords from a list of tokens."""
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]


def lemmatize_tokens(tokens):
    """Lemmatize tokens to their base form."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


def comprehensive_text_preprocessing(text, use_lemmatization=True):
    """
    Comprehensive preprocessing of text:
      1. Remove email headers and footers.
      2. Remove emojis.
      3. Expand contractions.
      4. Remove URLs.
      5. Remove mentions and hashtags.
      6. Remove punctuation.
      7. Remove numbers.
      8. Normalize repeated characters.
      9. Remove extra whitespace.
      10. Tokenize & lowercase.
      11. Remove stopwords.
      12. Optionally, lemmatize tokens.
    """
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
    """
    Convert texts to TF-IDF vectors and cluster them using KMeans.
    Returns cluster labels, the fitted TF-IDF vectorizer, and KMeans model.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans.labels_, vectorizer, kmeans


def label_clusters(vectorizer, kmeans,
                   local_keywords=['problem', 'issue', 'concern', 'maintenance', 'litter', 'noise']):
    """
    Inspect cluster centroids and assign a label to each cluster.
    If top terms include any local-problem keywords, label as "Local Problem"; else, "New Initiatives".
    """
    cluster_labels = {}
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    for i in range(kmeans.n_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :10]]
        if any(keyword in top_terms for keyword in local_keywords):
            cluster_labels[i] = "Local Problem"
        else:
            cluster_labels[i] = "New Initiatives"
    return cluster_labels


# ------------------ Topic Modeling ------------------

def topic_modeling(texts, num_topics=5):
    """
    Apply LDA topic modeling on a list of texts to extract topics.
    Returns a list of topics with their top terms.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)
    topics = []
    terms = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_terms = [terms[i] for i in topic.argsort()[:-11:-1]]
        topics.append("Topic {}: {}".format(topic_idx, ", ".join(top_terms)))
    return topics


# ------------------ Summarization Functions ------------------

def abstractive_summarization(text):
    """Placeholder abstractive summarization: return the first sentence."""
    sentences = sent_tokenize(text)
    return sentences[0] if sentences else text


def extractive_summarization(text):
    """Placeholder extractive summarization: return the longest sentence."""
    sentences = sent_tokenize(text)
    return max(sentences, key=len) if sentences else text


def query_based_summarization(text, query):
    """Placeholder query-based summarization: return the first sentence that mentions the query."""
    sentences = sent_tokenize(text)
    for sentence in sentences:
        if query.lower() in sentence.lower():
            return sentence.strip()
    return "No relevant information found for the query."


# ------------------ Sentiment Analysis ------------------

def sentiment_analysis(text):
    """
    Perform sentiment analysis using VADER.
    Returns a dictionary with sentiment scores.
    """
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)


# ------------------ Data Loading ------------------

def load_data(file_path):
    """
    Load data from a CSV file into a DataFrame.
    """
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print("Error loading data:", e)
        return None


# ------------------ Main Pipeline ------------------

def main():
    # Load dataset (each row is a letter/correspondence)
    file_path = "Data/processed/community_issues_dataset_template.csv"
    df = load_data(file_path)
    if df is None:
        return

    # Preprocess each letter's text.
    df['clean_text'] = df['text'].apply(comprehensive_text_preprocessing)

    # Filter out any empty documents after preprocessing.
    df = df[df['clean_text'].str.strip() != '']
    if df.empty:
        print("No valid texts for clustering after preprocessing.")
        return

    # Unsupervised classification: cluster letters into 2 groups.
    texts = df['clean_text'].tolist()
    if not texts:
        print("No valid texts for clustering.")
        return

    labels, vectorizer, kmeans = unsupervised_classification(texts, num_clusters=2)
    cluster_mapping = label_clusters(vectorizer, kmeans)
    df['classification'] = [cluster_mapping[label] for label in labels]

    print("\nClassification Distribution:")
    print(df['classification'].value_counts())

    # Topic Modeling on each group:
    for category in ["Local Problem", "New Initiatives"]:
        subset_texts = df[df['classification'] == category]['clean_text'].tolist()
        if subset_texts:
            topics = topic_modeling(subset_texts, num_topics=5)
            print("\nExtracted Topics for {}:".format(category))
            for topic in topics:
                print(topic)
        else:
            print("No letters classified as {} for topic modeling.".format(category))

    # For demonstration, process a sample letter.
    sample_text = df['text'].iloc[0]
    print("\nSample Letter Original:")
    print(sample_text)
    print("\nSample Letter Summaries:")
    print("Abstractive Summary:", abstractive_summarization(sample_text))
    print("Extractive Summary:", extractive_summarization(sample_text))
    print("Query-based Summary (for query 'traffic'):", query_based_summarization(sample_text, query="traffic"))

    # Sentiment analysis on sample letter.
    sentiment_scores = sentiment_analysis(sample_text)

    # Determine sentiment category
    compound_score = sentiment_scores['compound']

    # Adjusted thresholds
    if compound_score >= 0.1:
        sentiment_label = "Positive"  # A stronger positive threshold
    elif compound_score <= -0.1:
        sentiment_label = "Negative"  # A stronger negative threshold
    else:
        sentiment_label = "Neutral"  # Neutral for scores between -0.1 and 0.1

    # Print sentiment results
    print("\nSentiment Analysis of Sample Letter:")
    print(sentiment_scores)
    print(f"Overall Sentiment: {sentiment_label}")

    # Create a TextBlob object
    blob = TextBlob(sample_text)

    # Sentiment polarity (-1 to 1, where 1 is positive and -1 is negative)
    print(f"Sentiment polarity: {blob.sentiment.polarity}")


if __name__ == '__main__':
    main()
