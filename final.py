import os
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
import plotly.graph_objects as go
import streamlit as st

# ------------------ Fix OpenMP Warning ------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

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
        top_terms = [terms[ind] for ind in order_centroids[i, :10]]
        keyword_str = ", ".join(top_terms)
        result = zero_shot_classifier(keyword_str, candidate_labels)
        best_label = result["labels"][0]
        cluster_labels[i] = best_label
    return cluster_labels


# ------------------ Topic Modeling ------------------
def topic_modeling(texts, num_topics=1):
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
    action_indicators = ["urge", "request", "increase", "arrange", "immediate", "control", "measure", "action",
                         "implement", "improve", "take"]
    is_action_query = any(
        word in query.lower() for word in ["action", "request", "urge", "increase", "immediate", "control", "measure"])
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


# ------------------ Sentiment Analysis ------------------
TRANSFORMER_SENTIMENT = hf_pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


def sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    vader_scores = sia.polarity_scores(text)
    transformer_result = TRANSFORMER_SENTIMENT(text)[0]
    if transformer_result["label"] == "NEGATIVE":
        sentiment_label = "Negative"
    elif transformer_result["label"] == "POSITIVE":
        sentiment_label = "Positive"
    else:
        sentiment_label = "Neutral"
    if vader_scores["compound"] <= -0.3:
        sentiment_label = "Negative"
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
    return {"vader_scores": vader_scores, "transformer_result": transformer_result, "sentiment_label": sentiment_label,
            "explanation": explanation}


# ------------------ Aggregated Analysis Plot Functions ------------------
import plotly.graph_objects as go


def plot_classification_distribution(class_counts):
    fig = go.Figure([go.Bar(x=class_counts.index, y=class_counts.values)])
    fig.update_layout(title="Classification Distribution", xaxis_title="Category", yaxis_title="Count")
    return fig


def plot_sentiment_distribution(avg_sentiment):
    fig = go.Figure([go.Bar(x=["Average Sentiment Polarity"], y=[avg_sentiment])])
    fig.update_layout(title="Average Sentiment Polarity", xaxis_title="Metric", yaxis_title="Polarity")
    return fig


def plot_sentiment_gauge(polarity):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=polarity,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={"text": "Sentiment Gauge"},
        gauge={
            "axis": {"range": [-1, 1]},
            "steps": [
                {"range": [-1, -0.3], "color": "red"},
                {"range": [-0.3, 0.3], "color": "yellow"},
                {"range": [0.3, 1], "color": "green"}
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": polarity
            }
        }
    ))
    return fig


# ------------------ Data Loading ------------------
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
        st.write("Data loaded successfully.")
        return df
    except Exception as e:
        st.write("Error loading data:", e)
        return None


# ------------------ Main UI Pipeline Using Tabs ------------------
def main():
    st.set_page_config(layout="wide")

    # Display the Bolsover District Council logo in the sidebar
    st.sidebar.image("src/img/Bolsover_District_Council_logo.svg", width=150)

    # Create tabs for navigation
    tab1, tab2, tab3 = st.tabs(["Data Entry", "Results", "Aggregated Analysis"])

    # ------------------ Data Entry Tab ------------------
    with tab1:
        st.title("Citizen Letter Data Entry")
        data_mode = st.radio("Choose Input Mode", ["Paste Text", "Upload File"])
        user_query = st.text_input("Enter Query for Summarization", "What actions are being urged in the letter?")

        if data_mode == "Paste Text":
            input_text = st.text_area("Paste your letter text here", height=200)
        else:
            uploaded_file = st.file_uploader("Upload a file (txt, csv, pdf, doc, docx)",
                                             type=["txt", "csv", "pdf", "doc", "docx"],
                                             accept_multiple_files=False)
            input_text = ""
            if uploaded_file is not None:
                if uploaded_file.type == "text/plain":
                    input_text = uploaded_file.read().decode("utf-8")
                elif uploaded_file.type == "text/csv":
                    df_file = pd.read_csv(uploaded_file)
                    input_text = " ".join(df_file['text'].astype(str).tolist())
                elif uploaded_file.type == "application/pdf":
                    try:
                        from PyPDF2 import PdfReader
                        reader = PdfReader(uploaded_file)
                        input_text = ""
                        for page in reader.pages:
                            input_text += page.extract_text()
                    except Exception as e:
                        st.write("Error processing PDF:", e)
                elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                            "application/msword"]:
                    try:
                        import docx2txt
                        input_text = docx2txt.process(uploaded_file)
                    except Exception as e:
                        st.write("Error processing DOC/DOCX:", e)
                else:
                    input_text = uploaded_file.read().decode("utf-8")

        if st.button("Submit"):
            st.session_state.input_text = input_text
            st.session_state.user_query = user_query
            st.session_state.data_mode = data_mode
            st.success("Data saved. Switch to the 'Results' tab to see the analysis.")

    # ------------------ Results Tab ------------------
    with tab2:
        st.title("Individual Letter Analysis")
        if "input_text" not in st.session_state or not st.session_state.input_text:
            st.warning("No input data found. Please go to the 'Data Entry' tab and provide a letter.")
        else:
            letter_text = st.session_state.input_text
            st.subheader("Original Text")
            st.write(letter_text)

            st.subheader("Abstractive Summary")
            st.write(abstractive_summarization(letter_text))

            st.subheader("Extractive Summary")
            st.write(extractive_summarization(letter_text))

            st.subheader("Query-based Summaries")
            user_query = st.session_state.user_query if "user_query" in st.session_state else "What actions are being urged in the letter?"
            st.write(f"For query '{user_query}':", query_based_summarization(letter_text, query=user_query))

            st.subheader("Sentiment Analysis")
            sentiment_results = sentiment_analysis(letter_text)
            st.write(sentiment_results)

            polarity = TextBlob(letter_text).sentiment.polarity
            st.write(f"Sentiment Polarity (TextBlob): {polarity}")

            st.subheader("Sentiment Gauge")
            gauge_fig = plot_sentiment_gauge(polarity)
            st.plotly_chart(gauge_fig)

    # ------------------ Aggregated Analysis Tab ------------------
    with tab3:
        st.title("Aggregated Analysis")
        uploaded_files = st.file_uploader("Upload multiple files for aggregated analysis",
                                          type=["txt", "csv", "pdf", "doc", "docx"],
                                          accept_multiple_files=True)
        if uploaded_files:
            all_texts = []
            for uploaded_file in uploaded_files:
                if uploaded_file.type == "text/plain":
                    text = uploaded_file.read().decode("utf-8")
                elif uploaded_file.type == "text/csv":
                    df_file = pd.read_csv(uploaded_file)
                    text = " ".join(df_file['text'].astype(str).tolist())
                elif uploaded_file.type == "application/pdf":
                    try:
                        from PyPDF2 import PdfReader
                        reader = PdfReader(uploaded_file)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text()
                    except Exception as e:
                        st.write("Error processing PDF:", e)
                        text = ""
                elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                            "application/msword"]:
                    try:
                        import docx2txt
                        text = docx2txt.process(uploaded_file)
                    except Exception as e:
                        st.write("Error processing DOC/DOCX:", e)
                        text = ""
                else:
                    text = uploaded_file.read().decode("utf-8")
                all_texts.append(text)

            # Create DataFrame for aggregated analysis.
            df_agg = pd.DataFrame({"text": all_texts})
            df_agg["clean_text"] = df_agg["text"].apply(comprehensive_text_preprocessing)
            texts_clean = df_agg["clean_text"].tolist()
            labels, vectorizer, kmeans = unsupervised_classification(texts_clean, num_clusters=2)
            cluster_mapping = dynamic_label_clusters(vectorizer, kmeans)
            df_agg["classification"] = [cluster_mapping[label] for label in labels]

            st.subheader("Classification Distribution")
            class_counts = df_agg["classification"].value_counts()
            st.write(class_counts)
            st.plotly_chart(plot_classification_distribution(class_counts))

            # Topic modeling per classification.
            for category in candidate_labels:
                subset_texts = df_agg[df_agg["classification"] == category]["clean_text"].tolist()
                if subset_texts:
                    topics = topic_modeling(subset_texts, num_topics=5)
                    st.subheader(f"Extracted Topics for {category}")
                    for topic in topics:
                        dynamic_label = dynamic_topic_label(topic)
                        st.write(f"{dynamic_label} (Keywords: {topic})")

            # Aggregated sentiment.
            df_agg["sentiment_polarity"] = df_agg["text"].apply(lambda x: TextBlob(x).sentiment.polarity)
            st.subheader("Average Sentiment Polarity")
            avg_sentiment = df_agg["sentiment_polarity"].mean()
            st.write(avg_sentiment)
            st.plotly_chart(plot_sentiment_distribution(avg_sentiment))
            st.subheader("Sentiment Gauge (Aggregated)")
            st.plotly_chart(plot_sentiment_gauge(avg_sentiment))

            # Simple CSV download example.
            report = df_agg.to_csv(index=False)
            st.download_button("Download Report (CSV)", report, file_name="aggregated_report.csv", mime="text/csv")
        else:
            st.warning("Please upload files for aggregated analysis.")


if __name__ == '__main__':
    main()
