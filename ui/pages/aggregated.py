import re
import streamlit as st
import pandas as pd
import plotly.express as px
from processing.data_processing import process_uploaded_data
from models.summarization import get_summaries
from processing.topics import topic_modeling, dynamic_topic_label
from ui.components.sidebar import show_sidebar
from ui.components.cards import kpi_card, summary_card
from ui.components.maps import create_sentiment_map
from config import PAGE_CONFIG, CLASSIFICATION_LABELS, CANDIDATE_LABELS_TOPIC
from utils.preprocessing import comprehensive_text_preprocessing, extract_locations
from models.classification import classify_document
from models.sentiment import sentiment_analysis
from utils.geocoding import geocode_addresses
import ssl


def create_issues_dataframe(df):
    """Create dataframe of issues from topic modeling results"""
    topics = topic_modeling(df["clean_text"].tolist(), num_topics=3)
    issues_data = []

    for topic in topics:
        keywords = [re.escape(kw.strip()) for kw in topic.split(',')]
        pattern = r'\b(' + '|'.join(keywords) + r')\b'
        count = df['clean_text'].str.contains(pattern, regex=True, case=False, na=False).sum()

        issues_data.append({
            "Issue": dynamic_topic_label(topic),
            "Count": count,
            "Percentage": (count / len(df) * 100)
        })

    return pd.DataFrame(issues_data).sort_values('Count', ascending=False)


def aggregated_analysis_page():
    st.title("Comprehensive Letters Analysis")

    # 1. Validate session state and inputs
    if not st.session_state.get("data_submitted", False):
        st.warning("No data submitted yet. Please go to the 'Data Entry' page.")
        return
    if "uploaded_files_texts" not in st.session_state or len(st.session_state.uploaded_files_texts) < 2:
        st.warning("No multiple-file data found. Please upload multiple files.")
        return

    # Sidebar
    show_sidebar(st.session_state.get("uploaded_file_info", {}), st.session_state.get("input_text", ""))

    # Process data
    df_agg = process_uploaded_data(st.session_state.uploaded_files_texts)
    st.write("Processed Data:", df_agg)

    # Key Metrics
    st.markdown("### Key Metrics")
    total_letters = len(df_agg)
    class_counts = df_agg["classification"].value_counts(normalize=True) * 100
    local_problems_pct = class_counts.get("Local Problem", 0)
    new_initiatives_pct = class_counts.get("New Initiatives", 0)

    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    with kpi_col1:
        kpi_card("📩 Total Letters", total_letters, "")
    with kpi_col2:
        kpi_card("📍 Local Problems", f"{local_problems_pct:.1f}%", "")
    with kpi_col3:
        kpi_card("✨ New Initiatives", f"{new_initiatives_pct:.1f}%", "")

    # Most Common Issues
    st.subheader("Most Common Issues")
    issues_df = create_issues_dataframe(df_agg)
    fig = px.bar(
        issues_df,
        x="Issue",
        y="Count",
        text="Percentage",
        labels={'Count': 'Number of Complaints', 'Percentage': 'Percentage'},
        color="Issue"
    )
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>Count: %{y}<br>Percentage: %{text:.1f}%"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Classification & Sentiment Analysis
    st.subheader("📊 Classification Distribution & 😊 Sentiment Analysis")
    col4, col5 = st.columns(2)

    with col4:
        classification_counts = df_agg["classification"].value_counts().reset_index()
        classification_counts.columns = ['classification', 'count']
        fig_classification = px.pie(
            classification_counts,
            values='count',
            names='classification',
            title="Classification Distribution"
        )
        st.plotly_chart(fig_classification, use_container_width=True)

    with col5:
        sentiment_counts = df_agg["sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ['sentiment', 'count']
        fig_sentiment = px.bar(
            sentiment_counts,
            x='sentiment',
            y='count',
            title="Sentiment Analysis",
            color='sentiment',
            color_discrete_map={
                'Positive': 'green',
                'Neutral': 'gray',
                'Negative': 'red'
            },
            labels={'count': 'Number of Letters', 'sentiment': 'Sentiment'}
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)

    # Key Takeaways & Highlighted Sentences
    col6, col7 = st.columns(2)
    with col6:
        st.subheader("💡 Key Takeaways")
        key_takeaways = " ".join([
            get_summaries(text)["abstractive"]
            for text in st.session_state.uploaded_files_texts[:3]
        ])
        summary_card("Combined Abstracts", key_takeaways[:500])

    with col7:
        st.subheader("🔍 Highlighted Sentences")
        highlighted = " ".join([
            get_summaries(text)["extractive"]
            for text in st.session_state.uploaded_files_texts[:3]
        ])
        summary_card("Key Extracts", highlighted[:500])

    # AI Search Section
    st.subheader("🔍 AI Document Analyst")
    user_question = st.text_input(
        "Ask anything about the letters:",
        placeholder="e.g. What are the main complaints about waste management?"
    )
    if user_question:
        with st.spinner("Analyzing documents..."):
            response = ai_question_answer(
                user_question,
                st.session_state.uploaded_files_texts
            )
            st.markdown(f"""
            <div style='
                padding: 15px;
                border-radius: 10px;
                background-color: #f0f2f6;
                margin: 10px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            '>
                <p style='font-size: 16px; color: #333;'>{response}</p>
            </div>
            """, unsafe_allow_html=True)

    # Map Visualization
    st.subheader("📍 Geographic Issue Distribution")
    deck = create_sentiment_map(df_agg)
    if deck:
        st.pydeck_chart(deck)
    else:
        st.warning("No geographic data available for mapping.")

    # Export options
    st.subheader("Export Options")
    report_csv = df_agg.to_csv(index=False)
    st.download_button(
        "Download Report (CSV)",
        report_csv,
        file_name="aggregated_report.csv",
        mime="text/csv"
    )

    if st.button("Back to Data Entry"):
        st.session_state.input_text = ""
        st.session_state.data_submitted = False
        st.session_state.page = "data_entry"
        st.rerun()
