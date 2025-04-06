import streamlit as st
import logging
from utils.file_utils import extract_text_from_file
from utils.preprocessing import comprehensive_text_preprocessing
from config import PAGE_CONFIG

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def data_entry_page():
    st.title("Letter Submission (Data Entry)")

    # Log page load
    logger.info("Loading data entry page")

    data_mode = st.radio("Choose Input Mode", ["Paste Text", "Upload File"], index=1)  # Default to Upload
    logger.info(f"Selected data mode: {data_mode}")

    input_text = ""
    uploaded_files = []

    if data_mode == "Paste Text":
        input_text = st.text_area("Paste your letter text here", height=200)
        logger.debug("User selected 'Paste Text' mode.")

        # Analysis options preview
        with st.expander("Available Analyses for Individual Letters"):
            st.markdown("""
            - **Letter Classification**: Identifies as "Local Problem" or "New Initiative"
            - **Topic Detection**: Key problem/issue highlighted
            - **Key Takeaways**: Abstractive summary of key points
            - **Highlighted Sentences**: Extractive summarization
            - **Query-Based Summarization**: Responses to specific queries
            - **Sentiment Analysis**: Positive/Negative with confidence levels
            """)

    else:  # Upload File mode
        uploaded_files = st.file_uploader(
            "Upload files (txt, pdf, doc, docx)",
            type=["txt", "pdf", "doc", "docx"],
            accept_multiple_files=True,
            key="file_uploader"  # Add key for better session state management
        )
        logger.debug(f"User selected 'Upload File' mode. Files uploaded: {len(uploaded_files)}")

        # Bulk analysis preview
        with st.expander("Available Analyses for Bulk Uploads"):
            st.markdown("""
            - **Key Metrics**: Total letters, breakdown by category
            - **Common Issues**: Bar chart of most frequent problems
            - **Classification Distribution**: Pie charts by category/sentiment
            - **Geographic Distribution**: Map visualization of issues
            - **AI Document Analyst**: Interactive Q&A about letters
            """)

    if st.button("Submit"):
        logger.info("Submit button clicked, processing data...")

        with st.spinner("Processing..."):
            # Reset relevant session state variables
            st.session_state.clear()
            st.session_state.data_mode = data_mode
            logger.debug(f"Session state cleared and data_mode set to {data_mode}")

            if data_mode == "Paste Text":
                if not input_text.strip():
                    st.warning("Please paste some text before submitting.")
                    logger.warning("User attempted to submit without pasting any text.")
                    return

                # Store single text as a list for consistency
                st.session_state.uploaded_files_texts = [input_text.strip()]
                st.session_state.input_text = input_text.strip()
                st.session_state.data_submitted = True
                st.session_state.page = "results"
                logger.info("Text submitted successfully. Redirecting to 'results' page.")

            else:  # Upload File mode
                if not uploaded_files:
                    st.warning("Please upload at least one file.")
                    logger.warning("User attempted to submit with no files uploaded.")
                    return

                extracted_texts = []
                valid_files = 0

                for file in uploaded_files:
                    try:
                        text = extract_text_from_file(file)
                        if text and text.strip():
                            extracted_texts.append(text.strip())
                            valid_files += 1
                            logger.debug(f"File {file.name} processed successfully.")
                        else:
                            logger.warning(f"File {file.name} is empty or failed to extract text.")
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")
                        logger.error(f"Error processing {file.name}: {str(e)}")
                        continue

                if not extracted_texts:
                    st.error("No valid text content found in any uploaded files")
                    logger.error("No valid text content found in uploaded files.")
                    return

                # Store all processing results
                st.session_state.uploaded_files_texts = extracted_texts
                st.session_state.input_text = "\n\n".join(extracted_texts)
                st.session_state.data_submitted = True
                st.session_state.uploaded_file_info = {
                    "num_files": valid_files,
                    "file_names": [f.name for f in uploaded_files],
                    "file_types": [f.type for f in uploaded_files]
                }
                logger.info(f"{valid_files} files processed successfully.")

                # Route to appropriate page
                if valid_files > 1:
                    st.session_state.page = "aggregated_analysis"
                    logger.info("Multiple files uploaded, routing to 'aggregated_analysis' page.")
                else:
                    st.session_state.page = "results"
                    logger.info("Single file uploaded, routing to 'results' page.")

            st.rerun()

