import os
import io
import ssl
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import docx2txt

from preprocessing import comprehensive_text_preprocessing, remove_email_headers_and_footers, sanitize_text
from models import get_zero_shot_classifier, get_abstractive_summarizer, get_sentiment_pipeline, load_paraphrase_model
from summarization import abstractive_summarization, extractive_summarization, query_based_summarization, paraphrase_text
from analysis import compute_topic, unsupervised_classification, dynamic_label_clusters, sentiment_analysis, dynamic_topic_label, candidate_labels
# from reporting import generate_pdf_report, generate_docx_report
from reporting import generate_pdf_report as pdf_report_generator, generate_docx_report as docx_report_generator


# Fix OpenMP Warning
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# SSL Context for NLTK Downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# NLTK Downloads
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

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
            st.write(refined_query_res)
            st.subheader("Extractive Summary")
            st.write(extractive_res)
            st.subheader(f"Query-based Summary ('{user_query}')")
            st.write(refined_query_res)
            st.subheader("Sentiment Analysis")
            sentiment_results = sentiment_analysis(letter_text)
            st.write(sentiment_results)
            from textblob import TextBlob
            polarity = TextBlob(letter_text).sentiment.polarity
            st.write(f"Sentiment Polarity (TextBlob): {polarity}")
            st.subheader("Sentiment Gauge")
            gauge_fig = plot_sentiment_gauge(polarity)
            st.plotly_chart(gauge_fig)

            export_format = st.selectbox("Select Export Format", ["PDF", "DOCX", "TXT", "CSV"])
            if export_format == "PDF":
                file_bytes = pdf_report_generator(letter_text, abstractive_res, extractive_res, query_res,
                                                  sentiment_results)
                st.download_button("Download Report", file_bytes, file_name="analysis_report.pdf",
                                   mime="application/pdf")
            elif export_format == "DOCX":
                file_bytes = docx_report_generator(letter_text, abstractive_res, extractive_res, query_res,
                                                   sentiment_results)
                st.download_button("Download Report", file_bytes, file_name="analysis_report.docx",
                                   mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
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
            from analysis import unsupervised_classification, dynamic_label_clusters, topic_modeling
            labels, vectorizer, kmeans = unsupervised_classification(texts_clean, num_clusters=2)
            cluster_mapping = dynamic_label_clusters(vectorizer, kmeans)
            df_agg["classification"] = [cluster_mapping[label] for label in labels]

            st.subheader("Classification Distribution")
            class_counts = df_agg["classification"].value_counts()
            st.write(class_counts)
            st.plotly_chart(plot_classification_distribution(class_counts))

            # from analysis import candidate_labels
            for category in candidate_labels:
                subset_texts = df_agg[df_agg["classification"] == category]["clean_text"].tolist()
                if subset_texts:
                    topics = topic_modeling(subset_texts, num_topics=5)
                    st.subheader(f"Extracted Topics for {category}")
                    from analysis import dynamic_topic_label
                    for topic in topics:
                        dynamic_label = dynamic_topic_label(topic)
                        st.write(f"{dynamic_label} (Keywords: {topic})")

            from textblob import TextBlob
            df_agg["sentiment_polarity"] = df_agg["text"].apply(lambda x: TextBlob(x).sentiment.polarity)
            st.subheader("Average Sentiment Polarity")
            avg_sentiment = df_agg["sentiment_polarity"].mean()
            st.write(avg_sentiment)
            st.plotly_chart(plot_sentiment_distribution(avg_sentiment))
            st.subheader("Sentiment Gauge (Aggregated)")
            st.plotly_chart(plot_sentiment_gauge(avg_sentiment))

            report_csv = df_agg.to_csv(index=False)
            st.download_button("Download Report (CSV)", report_csv, file_name="aggregated_report.csv", mime="text/csv")

            from reporting import generate_pdf_report, generate_docx_report
            pdf_bytes = generate_pdf_report("Aggregated Report", "N/A", "N/A", "N/A", df_agg.to_dict())
            st.download_button("Download Report (PDF)", pdf_bytes, file_name="aggregated_report.pdf", mime="application/pdf")

            docx_bytes = generate_docx_report("Aggregated Report", "N/A", "N/A", "N/A", df_agg.to_dict())
            st.download_button("Download Report (DOCX)", docx_bytes, file_name="aggregated_report.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        else:
            st.warning("Please upload files for aggregated analysis.")

if __name__ == '__main__':
    main()
