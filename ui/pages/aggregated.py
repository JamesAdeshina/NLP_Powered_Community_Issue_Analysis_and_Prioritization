import re
import streamlit as st
import pandas as pd
import plotly.express as px
from processing.data_processing import process_uploaded_data
from models.summarization import get_summaries
from processing.topics import topic_modeling, dynamic_topic_label
from ui.components.sidebar import show_sidebar
from ui.components.cards import kpi_card
from ui.components.maps import create_sentiment_map, create_clustered_map
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
    # st.write("Processed Data:", df_agg)

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
    st.write("")  # Empty line# Add separator line
    st.markdown("---")
    st.write("")  # Empty line
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

    # Choose one of these footnote options:
    st.caption(
        "This chart displays the most common issues reported, measured by the number of complaints. The percentages indicate the proportion of complaints relative to the most frequently reported issue")
    st.write("")  # Empty line
    # Add separator line
    st.markdown("---")
    # Classification & Sentiment Analysis
    st.subheader("📊 Category Breakdown & Mood Assessment")
    col4, col5 = st.columns(2)

    with col4:
        classification_counts = df_agg["classification"].value_counts().reset_index()
        classification_counts.columns = ['classification', 'count']
        fig_classification = px.pie(
            classification_counts,
            values='count',
            names='classification',
            title="Letter Categories"
        )
        st.plotly_chart(fig_classification, use_container_width=True)
        # Add footnote (choose one method)
        st.caption(
            "This chart displays the distribution of complaint classifications, showing the proportion of each category relative to the total number of complaints")

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
        # Add footnote (choose one method)
        st.caption(
            "The sentiment distribution of the uploaded letters is based on 35 total entries, with 20 letters (57%) classified as positive and 15 letters (43%) as negative")
    st.write("")  # Empty line
    st.markdown("---") # Add separator line
    st.write("")  # Empty line

    # Map Visualization
    st.subheader("📍 Geographic Issue Distribution")

    # Add after processing df_agg

    # st.write("Data Sample:", df_agg[["text", "lat", "lon", "sentiment", "Topic", "Issue"]].head())
    # st.write("Null Values:", df_agg[["lat", "lon", "Topic", "Issue"]].isnull().sum())

    # Bolsover district coordinates (approximate center)
    BOLSOVER_COORDS = {
        "lat": 53.23,  # Approximate latitude
        "lon": -1.28    # Approximate longitude
    }

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Sentiment", "Categories", "Issues"])

    with tab1:
        # st.write("### Sentiment Distribution")
        if df_agg.empty or df_agg["lat"].isnull().all() or df_agg["lon"].isnull().all():
            st.warning("No geographic data available for mapping.")
        else:
            # Create sentiment map with Bolsover focus
            deck = create_sentiment_map(df_agg)
            if deck:
                # Adjust view state for Bolsover focus
                deck.initial_view_state.latitude = BOLSOVER_COORDS["lat"]
                deck.initial_view_state.longitude = BOLSOVER_COORDS["lon"]
                deck.initial_view_state.zoom = 10
                st.pydeck_chart(deck)
            else:
                st.warning("Could not generate sentiment map.")

    with tab2:
        # st.write("### Category Distribution")
        if df_agg.empty or df_agg["lat"].isnull().all():
            st.warning("No geographic data available for mapping.")
        else:
            # Create category map with Bolsover focus
            deck = create_clustered_map(df_agg, filter_by_sentiment=None, filter_by_issue=None, filter_by_topic=None)
            if deck:
                # Adjust view state for Bolsover focus
                deck.initial_view_state.latitude = BOLSOVER_COORDS["lat"]
                deck.initial_view_state.longitude = BOLSOVER_COORDS["lon"]
                deck.initial_view_state.zoom = 10
                st.pydeck_chart(deck)
            else:
                st.warning("Could not generate category map.")

    with tab3:
        # st.write("### Issue/Problem Distribution")
        if df_agg.empty or df_agg["lat"].isnull().all():
            st.warning("No geographic data available for mapping.")
        else:
            # Create topic map with Bolsover focus
            deck = create_clustered_map(df_agg, filter_by_sentiment=None, filter_by_issue=None, filter_by_topic=None)
            if deck:
                # Adjust view state for Bolsover focus
                deck.initial_view_state.latitude = BOLSOVER_COORDS["lat"]
                deck.initial_view_state.longitude = BOLSOVER_COORDS["lon"]
                deck.initial_view_state.zoom = 10
                st.pydeck_chart(deck)
            else:
                st.warning("Could not generate topic map.")



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
