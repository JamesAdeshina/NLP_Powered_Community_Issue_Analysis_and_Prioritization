import streamlit as st
from models.classification import classify_document
from processing.topics import compute_topic
from models.summarization import get_summaries
from models.sentiment import sentiment_analysis
from ui.components.sidebar import show_sidebar
from ui.components.cards import keytakeaways_card, highlighted_card
from ui.components.reports import get_export_options
from utils.visualization import plot_sentiment_gauge
from utils.nlp_utils import query_based_summarization, personalize_summary  # Added missing imports
from config import CLASSIFICATION_LABELS  # For classify_document







def results_page():
    st.title("Individual Letter Analysis")

    if "data_submitted" not in st.session_state or not st.session_state.data_submitted:
        st.warning("No data submitted yet. Please go to the 'Data Entry' page.")
        return

    # Get text based on input method
    if st.session_state.data_mode == "Upload File":
        if "uploaded_files_texts" in st.session_state and st.session_state.uploaded_files_texts:
            letter_text = st.session_state.uploaded_files_texts[0]
        else:
            st.error("No text found in uploaded file")
            return
    else:
        letter_text = st.session_state.get("input_text", "")

    # Sidebar
    show_sidebar(st.session_state.get("uploaded_file_info", {}), letter_text)

    # Classification
    st.subheader("🏷️ Classification")
    letter_class = classify_document(letter_text)
    st.write(f"This letter is classified as: **{letter_class}**")

    # Topic
    st.subheader("Topic")
    topic_label, top_keywords = compute_topic(letter_text)
    st.write(f"The main topic extracted is: **{topic_label}**")

    # Summaries
    summaries = get_summaries(letter_text)

    col1, col2 = st.columns(2)
    with col1:
        keytakeaways_card("💡 Key Takeaways", summaries["abstractive"])
    with col2:
        highlighted_card("🔍 Highlighted Sentences", summaries["extractive"])

    # Query-based summary
    st.subheader("❓ Inquiry-Driven Insights")
    user_query = st.text_input("Ask anything about the letters:", "What actions are being urged in the letter?")
    query_summary = query_based_summarization(letter_text, query=user_query)
    st.write(personalize_summary(query_summary, "query"))

    # Sentiment analysis
    st.subheader("🗣️ Tone of Letter")
    sentiment_results = sentiment_analysis(letter_text)

    def format_sentiment_label(label: str) -> str:
        label = label.upper()
        if label == "NEGATIVE":
            return "🔴 Negative"
        elif label == "POSITIVE":
            return "🟢 Positive"
        elif label == "NEUTRAL":
            return "🟡 Neutral"
        else:
            return label

    # Safely get sentiment label with fallback
    sentiment_label = sentiment_results.get('sentiment_label', 'NEUTRAL')
    formatted_label = format_sentiment_label(sentiment_label)

    explanation = sentiment_results.get('explanation', 'Sentiment analysis not available')
    confidence = sentiment_results.get('confidence', 0.5)
    #🟢🔴🟡

    col_mood, col_gauge = st.columns(2)
    with col_mood:
        st.write(sentiment_results['explanation'])
        st.write(f"**The Mood of the text is:** {formatted_label}")
        st.write(f"{sentiment_results['note']}")
    with col_gauge:
        gauge_fig = plot_sentiment_gauge(sentiment_results['confidence'])
        st.plotly_chart(gauge_fig)

    # Export options
    get_export_options(letter_text, summaries, sentiment_results)

    if st.button("Back to Data Entry"):
        st.session_state.input_text = ""
        st.session_state.data_submitted = False
        st.session_state.page = "data_entry"
        st.rerun()