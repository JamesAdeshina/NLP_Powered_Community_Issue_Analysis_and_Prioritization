import streamlit as st
from utils.file_utils import generate_pdf_report, generate_docx_report
import pandas as pd
import io


def get_export_options(text, summaries, sentiment_results):
    export_format = st.selectbox("Select Export Format", ["PDF", "DOCX", "TXT", "CSV"])

    if export_format == "PDF":
        file_bytes = generate_pdf_report(
            text,
            summaries["abstractive"],
            summaries["extractive"],
            summaries["query_based"],
            sentiment_results
        )
        st.download_button(
            "Download Report",
            file_bytes,
            file_name="analysis_report.pdf",
            mime="application/pdf"
        )
    elif export_format == "DOCX":
        file_bytes = generate_docx_report(
            text,
            summaries["abstractive"],
            summaries["extractive"],
            summaries["query_based"],
            sentiment_results
        )
        st.download_button(
            "Download Report",
            file_bytes,
            file_name="analysis_report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    elif export_format == "TXT":
        txt_report = (
            f"Analysis Report\n\nOriginal Text:\n{text}\n\nAbstractive Summary:\n{summaries['abstractive']}\n\n"
            f"Extractive Summary:\n{summaries['extractive']}\n\nQuery-based Summary:\n{summaries['query_based']}\n\n"
            f"Sentiment Analysis:\n{sentiment_results}"
        )
        st.download_button(
            "Download Report",
            txt_report,
            file_name="analysis_report.txt",
            mime="text/plain"
        )
    elif export_format == "CSV":
        df_report = pd.DataFrame({
            "Original Text": [text],
            "Abstractive Summary": [summaries["abstractive"]],
            "Extractive Summary": [summaries["extractive"]],
            "Query-based Summary": [summaries["query_based"]],
            "Sentiment Analysis": [str(sentiment_results)]
        })
        csv_report = df_report.to_csv(index=False)
        st.download_button(
            "Download Report",
            csv_report,
            file_name="analysis_report.csv",
            mime="text/csv"
        )