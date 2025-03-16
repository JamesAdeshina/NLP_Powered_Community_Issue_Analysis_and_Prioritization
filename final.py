# ------------------ Standard Libraries ------------------
import os
import io
import ssl
import re

# ------------------ Data Handling and Numerical Processing ------------------
import pandas as pd
import numpy as np

# ------------------ NLP Libraries (General) ------------------
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import emoji
import contractions

# ------------------ Machine Learning and Text Processing ------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation

# ------------------ Summarization and Topic Extraction ------------------
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from rake_nltk import Rake

# ------------------ Transformers and Deep Learning Models ------------------
# We'll load heavy models using caching below.
from transformers import pipeline as hf_pipeline
import sentencepiece

# ------------------ Visualization ------------------
import plotly.graph_objects as go

# ------------------ Web App Framework ------------------
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



# ------------------ Caching Model Loading ------------------
@st.cache_resource
def get_zero_shot_classifier():
    return hf_pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@st.cache_resource
def get_abstractive_summarizer():
    return hf_pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", revision="a4f8f3e")

@st.cache_resource
def get_sentiment_pipeline():
    return hf_pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# ------------------ Helper: Sanitize Text for PDF ------------------
def sanitize_text(text: str) -> str:
    replacements = {
        "\u2014": "-",  # em-dash to hyphen
        "\u2013": "-",  # en-dash to hyphen
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote
        "\u201c": '"',  # left double quote
        "\u201d": '"',  # right double quote
        "\u2026": "..."  # ellipsis
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return ''.join(c if ord(c) < 256 else '?' for c in text)

# ------------------ Expanded Candidate Labels for Topic Detection ------------------
CANDIDATE_LABELS_TOPIC: tuple[str, ...] = (
    "Waste Management / Public Cleanliness",
    "Water Scarcity",
    "Food Insecurity",
    "Cybersecurity Threats",
    "Delays in NHS Treatment",
    "Underfunded Healthcare Services",
    "Decline in Local Shops / High Street Businesses",
    "High Cost of Living",
    "Overcrowded Public Transport",
    "Homelessness",
    "Lack of Affordable Housing",
    "Noise Pollution",
    "Potholes / Road Maintenance",
    "Traffic Congestion",
    "Air Pollution",
    "School Overcrowding",
    "Crime Rates in Urban Areas",
    "Limited Green Spaces",
    "Aging Infrastructure",
    "Digital Divide",
    "Rising Energy Costs",
    "Housing Quality Issues",
    "Lack of Social Mobility",
    "Climate Change Adaptation",
    "Elderly Care Shortages",
    "Rural Transport Accessibility",
    "Mental Health Service Shortages",
    "Drug and Alcohol Abuse",
    "Gender Pay Gap",
    "Age Discrimination in Employment",
    "Child Poverty",
    "Bureaucratic Delays in Government Services",
    "Lack of Public Restrooms in Urban Areas",
    "Unsafe Cycling Infrastructure",
    "Tackling Modern Slavery",
    "Gentrification and Displacement",
    "Rise in Anti-Social Behaviour",
    "Tackling Fake News and Misinformation",
    "Integration of Immigrant Communities",
    "Parking Problems",
    "Littering in Public Spaces",
    "Speeding Vehicles",
    "Crumbling Pavements",
    "Public Wi-Fi Gaps",
    "Youth Services Cuts",
    "Erosion of Coastal Areas",
    "Flooding in Residential Areas",
    "Loneliness and Social Isolation",
    "Domestic Violence and Abuse",
    "Racial Inequality and Discrimination",
    "LGBTQ+ Rights and Inclusion",
    "Disability Access",
    "Childcare Costs and Availability",
    "Veteran Support",
    "Community Cohesion",
    "Access to Arts and Culture",
    "Biodiversity Loss",
    "Urban Heat Islands",
    "Single-Use Plastics",
    "Education / Skills Development",
    "Community Workshops",
    "Renewable Energy Transition",
    "Food Waste",
    "Deforestation and Land Use",
    "Light Pollution",
    "Soil Degradation",
    "Marine Pollution",
    "Gig Economy Exploitation",
    "Regional Economic Disparities",
    "Skills Shortages",
    "Zero-Hours Contracts",
    "Pension Inequality",
    "Rising Inflation",
    "Small Business Struggles",
    "Post-Brexit Trade Challenges",
    "Automation and Job Loss",
    "Unpaid Internships",
    "Obesity Epidemic",
    "Dental Care Access",
    "Vaccine Hesitancy",
    "Pandemic Preparedness",
    "Nutritional Education",
    "Physical Inactivity",
    "Student Debt",
    "Teacher Shortages",
    "School Funding Cuts",
    "Bullying in Schools",
    "Access to Higher Education",
    "Vocational Training Gaps",
    "Digital Exclusion",
    "Extracurricular Activity Cuts",
    "Aging Public Buildings",
    "Smart City Development",
    "Electric Vehicle Infrastructure",
    "5G Rollout Delays",
    "Flood Defence Upgrades",
    "Rail Network Overcrowding",
    "AI Ethics and Regulation",
    "Space Debris Management",
    "Genetic Engineering Ethics",
    "Climate Migration",
    "Aging Population",
    "Urbanisation Pressures",
    "Data Privacy Concerns",
    "Sustainable Fashion"
)

# ------------------ Topic Functions ------------------
@st.cache_data
def dynamic_topic_label(keywords: str) -> str:
    classifier = get_zero_shot_classifier()
    result = classifier(keywords, CANDIDATE_LABELS_TOPIC)
    best_label = result["labels"][0]
    return best_label

@st.cache_data
def compute_topic(text: str, top_n: int = 5) -> tuple[str, str]:
    rake_extractor = Rake()  # RAKE uses default English stopwords.
    rake_extractor.extract_keywords_from_text(text)
    ranked_phrases = rake_extractor.get_ranked_phrases()
    top_terms = ranked_phrases[:top_n] if len(ranked_phrases) >= top_n else ranked_phrases
    keyword_str = ", ".join(top_terms)
    topic_label = dynamic_topic_label(keyword_str)
    return topic_label, keyword_str

# ------------------ Preprocessing Functions ------------------
@st.cache_data
def remove_email_headers_and_footers(text: str) -> str:
    lines = text.split('\n')
    stripped_lines = [line.strip() for line in lines]
    if "" in stripped_lines:
        first_blank_index = stripped_lines.index("")
        content = "\n".join(lines[first_blank_index + 1:]).strip()
    else:
        content = text
    signature_markers = ('sincerely,', 'regards,', 'best regards,', 'thanks,', 'thank you,')
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

# ------------------ Dynamic Topic Labeling for Clusters ------------------
candidate_labels = ["Local Problem", "New Initiatives"]

def dynamic_label_clusters(vectorizer, kmeans):
    cluster_labels = {}
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    for i in range(kmeans.n_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :10]]
        keyword_str = ", ".join(top_terms)
        classifier = get_zero_shot_classifier()
        result = classifier(keyword_str, candidate_labels)
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

# ------------------ Summarization Functions ------------------
abstractive_summarizer = get_abstractive_summarizer()

def abstractive_summarization(text):
    summary = abstractive_summarizer(text, max_length=50, min_length=25, do_sample=False)
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
    is_action_query = any(word in query.lower() for word in ["action", "request", "urge", "increase", "immediate", "control", "measure"])
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

# ------------------ Personalized Summary Wrapper ------------------
def personalize_summary(summary, summary_type="general"):
    return f" {summary}"

# ------------------ Sentiment Analysis ------------------
sentiment_pipeline = get_sentiment_pipeline()

def sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    vader_scores = sia.polarity_scores(text)
    transformer_result = sentiment_pipeline(text)[0]
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
            explanation += " Note based on sentiment; the author is highlighting concerns about " + ", ".join(details) + ", which creates a sense of urgency and dissatisfaction."
    return {"vader_scores": vader_scores, "transformer_result": transformer_result, "sentiment_label": sentiment_label, "explanation": explanation}

# ------------------ Aggregated Analysis Plot Functions ------------------
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

# ------------------ Report Generation Functions ------------------
from fpdf import FPDF
from docx import Document

def generate_pdf_report(original_text, abstractive_summary, extractive_summary, query_summary, sentiment_results):
    original_text = sanitize_text(original_text)
    abstractive_summary = sanitize_text(abstractive_summary)
    extractive_summary = sanitize_text(extractive_summary)
    query_summary = sanitize_text(query_summary)
    sentiment_results = sanitize_text(str(sentiment_results))
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt="Analysis Report", ln=True, align='C')
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=f"Original Text:\n{original_text}\n")
    pdf.ln(3)
    pdf.multi_cell(0, 10, txt=f"Abstractive Summary:\n{abstractive_summary}\n")
    pdf.ln(3)
    pdf.multi_cell(0, 10, txt=f"Extractive Summary:\n{extractive_summary}\n")
    pdf.ln(3)
    pdf.multi_cell(0, 10, txt=f"Query-based Summary:\n{query_summary}\n")
    pdf.ln(3)
    pdf.multi_cell(0, 10, txt=f"Sentiment Analysis:\n{sentiment_results}\n")
    return pdf.output(dest='S').encode('latin1', errors='replace')

def generate_docx_report(original_text, abstractive_summary, extractive_summary, query_summary, sentiment_results):
    doc = Document()
    doc.add_heading("Analysis Report", level=1)
    doc.add_heading("Original Text", level=2)
    doc.add_paragraph(original_text)
    doc.add_heading("Abstractive Summary", level=2)
    doc.add_paragraph(abstractive_summary)
    doc.add_heading("Extractive Summary", level=2)
    doc.add_paragraph(extractive_summary)
    doc.add_heading("Query-based Summary", level=2)
    doc.add_paragraph(query_summary)
    doc.add_heading("Sentiment Analysis", level=2)
    doc.add_paragraph(str(sentiment_results))
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()

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
    st.sidebar.image("src/img/Bolsover_District_Council_logo.svg", width=150)
    tabs = st.tabs(["Data Entry", "Results", "Aggregated Analysis"])

    # ------------------ Data Entry Tab ------------------
    with tabs[0]:
        st.title("Citizen Letter Data Entry")
        data_mode = st.radio("Choose Input Mode", ["Paste Text", "Upload File"])
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
            with st.spinner("Processing..."):
                st.session_state.input_text = input_text
                st.session_state.data_mode = data_mode
                st.session_state.data_submitted = True
            st.success("Data saved. Please switch to the 'Results' tab to see the analysis.")

    # ------------------ Paraphrasing Function ------------------
    def load_paraphrase_model():
        from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline as hf_pipeline
        tokenizer = T5Tokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws", use_fast=False)
        model = T5ForConditionalGeneration.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        paraphrase_pipeline = hf_pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=256
        )
        return paraphrase_pipeline

    paraphrase_model = load_paraphrase_model()

    def paraphrase_text(text):
        input_text = "paraphrase: " + text + " </s>"
        paraphrase = paraphrase_model(input_text, do_sample=False)
        return paraphrase[0]['generated_text']

    # ------------------ Results Tab ------------------
    with tabs[1]:
        st.title("Individual Letter Analysis")
        if "data_submitted" not in st.session_state or not st.session_state.data_submitted:
            st.warning("No data submitted yet. Please go to the 'Data Entry' tab, provide a letter, and click Submit.")
        else:
            letter_text = st.session_state.input_text
            st.subheader("Original Text")
            st.write(letter_text)

            letter_clean = comprehensive_text_preprocessing(letter_text)
            classifier = get_zero_shot_classifier()
            classification_result = classifier(letter_clean, candidate_labels)
            letter_class = classification_result["labels"][0]
            st.subheader("Classification")
            st.write(f"This letter is classified as: **{letter_class}**")

            topic_label, top_keywords = compute_topic(letter_clean)
            st.subheader("Topic")
            st.write(f"Topic: **{topic_label}**")

            user_query = st.text_input("Enter Query for Query-based Summarization", "What actions are being urged in the letter?")
            abstractive_res = abstractive_summarization(letter_text)
            extractive_res = extractive_summarization(letter_text)
            query_res = query_based_summarization(letter_text, query=user_query)
            refined_query_res = paraphrase_text(query_res)

            st.subheader("Abstractive Summary")
            st.write(personalize_summary(abstractive_res, "abstractive"))
            st.subheader("Extractive Summary")
            st.write(personalize_summary(extractive_res, "extractive"))
            st.subheader(f"Query-based Summary ('{user_query}')")
            st.write(personalize_summary(refined_query_res, "query"))
            st.subheader("Sentiment Analysis")
            sentiment_results = sentiment_analysis(letter_text)
            st.write(sentiment_results)
            polarity = TextBlob(letter_text).sentiment.polarity
            st.write(f"Sentiment Polarity (TextBlob): {polarity}")
            st.subheader("Sentiment Gauge")
            gauge_fig = plot_sentiment_gauge(polarity)
            st.plotly_chart(gauge_fig)

            export_format = st.selectbox("Select Export Format", ["PDF", "DOCX", "TXT", "CSV"])
            if export_format == "PDF":
                file_bytes = generate_pdf_report(letter_text, abstractive_res, extractive_res, query_res, sentiment_results)
                st.download_button("Download Report", file_bytes, file_name="analysis_report.pdf", mime="application/pdf")
            elif export_format == "DOCX":
                file_bytes = generate_docx_report(letter_text, abstractive_res, extractive_res, query_res, sentiment_results)
                st.download_button("Download Report", file_bytes, file_name="analysis_report.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            elif export_format == "TXT":
                txt_report = (
                    f"Analysis Report\n\nOriginal Text:\n{letter_text}\n\nAbstractive Summary:\n{abstractive_res}\n\n"
                    f"Extractive Summary:\n{extractive_res}\n\nQuery-based Summary:\n{query_res}\n\n"
                    f"Sentiment Analysis:\n{sentiment_results}"
                )
                st.download_button("Download Report", txt_report, file_name="analysis_report.txt", mime="text/plain")
            elif export_format == "CSV":
                df_report = pd.DataFrame({
                    "Original Text": [letter_text],
                    "Abstractive Summary": [abstractive_res],
                    "Extractive Summary": [extractive_res],
                    "Query-based Summary": [query_res],
                    "Sentiment Analysis": [str(sentiment_results)]
                })
                csv_report = df_report.to_csv(index=False)
                st.download_button("Download Report", csv_report, file_name="analysis_report.csv", mime="text/csv")

            if st.button("Back to Data Entry"):
                st.session_state.input_text = ""
                st.session_state.data_submitted = False
                st.experimental_rerun()

    # ------------------ Aggregated Analysis Tab ------------------
    with tabs[2]:
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

            for category in candidate_labels:
                subset_texts = df_agg[df_agg["classification"] == category]["clean_text"].tolist()
                if subset_texts:
                    topics = topic_modeling(subset_texts, num_topics=5)
                    st.subheader(f"Extracted Topics for {category}")
                    for topic in topics:
                        dynamic_label = dynamic_topic_label(topic)
                        st.write(f"{dynamic_label} (Keywords: {topic})")

            df_agg["sentiment_polarity"] = df_agg["text"].apply(lambda x: TextBlob(x).sentiment.polarity)
            st.subheader("Average Sentiment Polarity")
            avg_sentiment = df_agg["sentiment_polarity"].mean()
            st.write(avg_sentiment)
            st.plotly_chart(plot_sentiment_distribution(avg_sentiment))
            st.subheader("Sentiment Gauge (Aggregated)")
            st.plotly_chart(plot_sentiment_gauge(avg_sentiment))

            report_csv = df_agg.to_csv(index=False)
            st.download_button("Download Report (CSV)", report_csv, file_name="aggregated_report.csv", mime="text/csv")

            pdf_bytes = generate_pdf_report("Aggregated Report", "N/A", "N/A", "N/A", df_agg.to_dict())
            st.download_button("Download Report (PDF)", pdf_bytes, file_name="aggregated_report.pdf", mime="application/pdf")

            docx_bytes = generate_docx_report("Aggregated Report", "N/A", "N/A", "N/A", df_agg.to_dict())
            st.download_button("Download Report (DOCX)", docx_bytes, file_name="aggregated_report.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        else:
            st.warning("Please upload files for aggregated analysis.")

if __name__ == '__main__':
    main()
