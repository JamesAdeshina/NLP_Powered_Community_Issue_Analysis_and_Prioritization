
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




def aggregated_analysis_page():
    st.title("Comprehensive Letters Analysis")


    # 1) Check data availability
    if not st.session_state.get("data_submitted", False):
        st.warning("No data submitted yet. Please go to the 'Data Entry' page and upload multiple files.")
        return

    if "uploaded_files_texts" not in st.session_state or len(st.session_state.uploaded_files_texts) < 2:
        st.warning("No multiple-file data found. Please go to the 'Data Entry' page and upload multiple files.")
        return

    # 2) Sidebar content
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

    # 5) Calculate Key Metrics
    total_letters = len(df_agg)
    class_counts = df_agg["classification"].value_counts(normalize=True) * 100
    local_problems_pct = class_counts.get("Local Problem", 0)
    new_initiatives_pct = class_counts.get("New Initiatives", 0)

    # Key Metrics Cards
    st.markdown("### Key Metrics")
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

    theme_base = st.get_option("theme.base")
    bg_color = "#FFFFFF" if theme_base == "light" else "#222"
    text_color = "#000000" if theme_base == "light" else "#FFFFFF"

    with kpi_col1:
        st.markdown(f"""
        <div style='background-color:{bg_color}; padding:15px; border-radius:10px; text-align:center; box-shadow: 0px 4px 6px rgba(0,0,0,0.1);'>
            <h3 style='color:{text_color};'>📩 Total Letters</h3>
            <h2 style='color:{text_color};'>{total_letters}</h2>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col2:
        st.markdown(f"""
        <div style='background-color:{bg_color}; padding:15px; border-radius:10px; text-align:center; box-shadow: 0px 4px 6px rgba(0,0,0,0.1);'>
            <h3 style='color:{text_color};'>📍 Local Problems</h3>
            <h2 style='color:{text_color};'>{local_problems_pct:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        st.markdown(f"""
        <div style='background-color:{bg_color}; padding:15px; border-radius:10px; text-align:center; box-shadow: 0px 4px 6px rgba(0,0,0,0.1);'>
            <h3 style='color:{text_color};'>✨ New Initiatives</h3>
            <h2 style='color:{text_color};'>{new_initiatives_pct:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    # 6) Most Common Issues
    st.subheader("Most Common Issues")

    # Get topics and their keywords
    topics = topic_modeling(texts_clean, num_topics=3)

    # Create regex patterns for exact word matching
    issues_data = []
    for topic in topics:
        keywords = [re.escape(kw.strip()) for kw in topic.split(',')]
        pattern = r'\b(' + '|'.join(keywords) + r')\b'

        # Count matches with case insensitivity
        count = df_agg['clean_text'].str.contains(
            pattern,
            regex=True,
            case=False,
            na=False
        ).sum()

        issues_data.append({
            "Issue": dynamic_topic_label(topic),
            "Count": count,
            "Percentage": (count / len(df_agg) * 100)
        })

    # Create and sort DataFrame
    issues_df = pd.DataFrame(issues_data).sort_values('Count', ascending=False)

    # Format percentages
    issues_df['Percentage'] = issues_df['Percentage'].round(1)

    # Create visualization with combined labels
    fig = px.bar(
        issues_df,
        x="Issue",
        y="Count",
        text="Percentage",
        labels={'Count': 'Number of Complaints', 'Percentage': 'Percentage'},
        color="Issue"
    )

    # Improve label formatting
    fig.update_traces(
        texttemplate='%{text}%',
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>Count: %{y}<br>Percentage: %{text}%"
    )

    # Set axis limits based on data
    max_count = issues_df['Count'].max()
    fig.update_layout(
        yaxis_range=[0, max_count * 1.2],
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # 7) Classification & Sentiment Analysis
    st.subheader("📊 Classification Distribution & 😊 Sentiment Analysis")
    col4, col5 = st.columns(2)

    with col4:
        # Classification Distribution
        class_counts = df_agg["classification"].value_counts()
        fig_classification = px.pie(
            class_counts,
            values=class_counts.values,
            names=class_counts.index,
            title="Classification Distribution"
        )
        st.plotly_chart(fig_classification, use_container_width=True)

    with col5:
        # Sentiment Analysis
        df_agg["sentiment"] = df_agg["text"].apply(lambda x: sentiment_analysis(x)["sentiment_label"])
        sentiment_counts = df_agg["sentiment"].value_counts()
        fig_sentiment = px.bar(
            sentiment_counts,
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            title="Sentiment Analysis",
            color=sentiment_counts.index
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)

    # 8) Key Takeaways & Highlighted Sentences
    col6, col7 = st.columns(2)

    with col6:
        st.subheader("💡 Key Takeaways")
        try:
            key_takeaways = " ".join([abstractive_summarization(text)
                                      for text in st.session_state.uploaded_files_texts[:3]])
            st.markdown(f"""
                <div class="card">
                    <p>{key_takeaways[:500]}</p>  
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Key Takeaways Error: {str(e)}")
            st.session_state.page = "data_entry"
            st.rerun()

    with col7:
        st.subheader("🔍 Highlighted Sentences")
        try:
            highlighted = " ".join([extractive_summarization(text, 1)
                                    for text in st.session_state.uploaded_files_texts[:3]])
            st.markdown(f"""
                <div class="card">
                    <p>{highlighted[:500]}</p>  
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Highlight Error: {str(e)}")
            st.session_state.page = "data_entry"
            st.rerun()

    # AI Search Section
    st.subheader("🔍 AI Document Analyst")
    user_question = st.text_input("Ask anything about the letters:",
                                  placeholder="e.g. What are the main complaints about waste management?")

    if user_question:
        with st.spinner("Analyzing documents..."):
            try:
                # Get all uploaded texts
                documents = st.session_state.uploaded_files_texts

                # Get AI response
                response = ai_question_answer(user_question, documents)

                # Display response
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

            except Exception as e:
                st.error(f"Error processing question: {str(e)}")

    # Create the map
    st.subheader("📍 Geographic Issue Distribution")










    # 9) Export options
    st.subheader("Export Options")
    report_csv = df_agg.to_csv(index=False)
    st.download_button("Download Report (CSV)", report_csv, file_name="aggregated_report.csv", mime="text/csv")

    # # 10) Navigation
    if st.button("Back to Data Entry"):
        st.session_state.input_text = ""
        st.session_state.data_submitted = False
        st.session_state.page = "data_entry"
        st.rerun()
