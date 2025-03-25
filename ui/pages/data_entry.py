import streamlit as st
from utils.file_utils import extract_text_from_file
from utils.preprocessing import comprehensive_text_preprocessing

def data_entry_page():
    st.title("Letter Submission (Data Entry)")
    data_mode = st.radio("Choose Input Mode", ["Paste Text", "Upload File"])

    input_text = ""
    uploaded_files = []

    if data_mode == "Paste Text":
        input_text = st.text_area("Paste your letter text here", height=200)
    else:
        uploaded_files = st.file_uploader(
            "Upload files (txt, pdf, doc, docx)",
            type=["txt", "pdf", "doc", "docx"],
            accept_multiple_files=True
        )

    if st.button("Submit"):
        with st.spinner("Processing..."):
            if data_mode == "Paste Text":
                if not input_text.strip():
                    st.warning("Please paste some text before submitting.")
                    return

                st.session_state.input_text = input_text
                st.session_state.data_submitted = True
                st.session_state.data_mode = data_mode
                st.session_state.uploaded_file_info = {
                    "num_files": 1,
                    "file_extensions": {"paste"}
                }
                st.session_state.page = "results"
                st.rerun()

            elif data_mode == "Upload File":
                if not uploaded_files:
                    st.warning("Please upload at least one file.")
                    return

                file_types = []
                extracted_texts = []
                combined_text = ""

                for file in uploaded_files:
                    text = extract_text_from_file(file)
                    if text:
                        extracted_texts.append(text.strip())
                        combined_text += f"\n\n{text.strip()}"

                if not extracted_texts:
                    st.error("Could not extract any text from uploaded files")
                    return

                # Update session state
                st.session_state.uploaded_files_texts = extracted_texts
                st.session_state.input_text = combined_text.strip()
                st.session_state.data_submitted = True
                st.session_state.data_mode = data_mode

                # Route to correct page
                if len(uploaded_files) > 1:
                    st.session_state.page = "aggregated_analysis"
                else:
                    st.session_state.page = "results"

                st.rerun()