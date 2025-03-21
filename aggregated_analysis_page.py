
def aggregated_analysis_page():
    st.title("Comprehensive Letters Analysis")

    # 1) Ensure user has actually uploaded multiple files
    if not st.session_state.get("data_submitted", False):
        st.warning("No data submitted yet. Please go to the 'Data Entry' page and upload multiple files.")
        return

    if "uploaded_files_texts" not in st.session_state or len(st.session_state.uploaded_files_texts) < 2:
        st.warning("No multiple-file data found. Please go to the 'Data Entry' page and upload multiple files.")
        return

    # 2) Show an icon & original letter in the sidebar
    with st.sidebar:
        file_info = st.session_state.get("uploaded_file_info", {})
        num_files = file_info.get("num_files", 0)
        ext_set = file_info.get("file_extensions", set())

        icon_path = pick_sidebar_icon(num_files, ext_set)
        st.image(icon_path, width=150)

        with st.expander("Original Letter", expanded=False):
            st.write(st.session_state.get("input_text", "No text available."))

    # 3) Prepare data
    uploaded_texts = st.session_state.uploaded_files_texts
    df_agg = pd.DataFrame({"text": uploaded_texts})
    df_agg["clean_text"] = df_agg["text"].apply(comprehensive_text_preprocessing)
    texts_clean = df_agg["clean_text"].tolist()

    # 4) Classification
    labels, vectorizer, kmeans = unsupervised_classification(texts_clean, num_clusters=2)
    cluster_mapping = dynamic_label_clusters(vectorizer, kmeans)
    df_agg["classification"] = [cluster_mapping[label] for label in labels]

    st.subheader("Classification Distribution")
    class_counts = df_agg["classification"].value_counts()
    st.write(class_counts)
    st.plotly_chart(plot_classification_distribution(class_counts))


    # 5) Topic Modeling for each classification
    for category in candidate_labels:
        subset_texts = df_agg[df_agg["classification"] == category]["clean_text"].tolist()
        if subset_texts:
            topics = topic_modeling(subset_texts, num_topics=5)
            st.subheader(f"Extracted Topics for {category}")
            for topic in topics:
                dynamic_label = dynamic_topic_label(topic)
                st.write(f"{dynamic_label} (Keywords: {topic})")

    # 6) Aggregated sentiment
    def get_vader_compound(txt):
        return sentiment_analysis(txt)["vader_scores"]["compound"]

    df_agg["sentiment_polarity"] = df_agg["text"].apply(get_vader_compound)
    st.subheader("Average Sentiment Polarity")
    avg_sentiment = df_agg["sentiment_polarity"].mean()
    st.write(avg_sentiment)
    st.plotly_chart(plot_sentiment_distribution(avg_sentiment))

    st.subheader("Sentiment Gauge (Aggregated)")
    st.plotly_chart(plot_sentiment_gauge(avg_sentiment))

    # 7) Export options
    report_csv = df_agg.to_csv(index=False)
    st.download_button("Download Report (CSV)", report_csv, file_name="aggregated_report.csv", mime="text/csv")

    pdf_bytes = generate_pdf_report("Aggregated Report", "N/A", "N/A", "N/A", df_agg.to_dict())
    st.download_button("Download Report (PDF)", pdf_bytes, file_name="aggregated_report.pdf", mime="application/pdf")

    docx_bytes = generate_docx_report("Aggregated Report", "N/A", "N/A", "N/A", df_agg.to_dict())
    st.download_button("Download Report (DOCX)", docx_bytes, file_name="aggregated_report.docx",
                       mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

    # 8) Navigation
    if st.button("Back to Data Entry"):
        st.session_state.input_text = ""
        st.session_state.data_submitted = False
        st.session_state.page = "data_entry"
        st.rerun()








    # 6) Most Common Issues
    st.subheader("Most Common Issues")

    # Extract topics
    topics = topic_modeling(texts_clean, num_topics=3)

    # Count the occurrences of each topic
    issues_data = []
    for topic in topics:
        # Split the topic into individual keywords
        keywords = topic.split(", ")
        # Count how many texts contain any of the keywords
        count = sum(1 for text in df_agg["clean_text"] if any(keyword in text for keyword in keywords))
        issues_data.append({"Issue": dynamic_topic_label(topic), "Count": count})

    # Convert to DataFrame
    issues_data = pd.DataFrame(issues_data)
    issues_data["Percentage"] = (issues_data["Count"] / len(df_agg) * 100).round(1)

    # Plot the data
    fig_issues = px.bar(issues_data, x="Issue", y="Count", text="Percentage", title="Most Common Issues")
    st.plotly_chart(fig_issues, use_container_width=True)