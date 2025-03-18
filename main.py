import streamlit as st
import os
import docx2txt
import pandas as pd
# Import your custom modules for analysis, summarization, etc.
# from analysis import ...
# from summarization import ...
# from preprocessing import comprehensive_text_preprocessing
# from reporting import ...
# from models import get_zero_shot_classifier, ...

# 1) This must be the very first Streamlit call in your script.
st.set_page_config(
    layout="centered",
    page_title="Bolsover District Council - Letter Submission"
)

# 2) Optionally hide Streamlit's default menu/footer for a cleaner look
HIDE_STYLE = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        max-width: 800px;
        margin: auto;
    }
    </style>
"""
st.markdown(HIDE_STYLE, unsafe_allow_html=True)

def main():
    # Initialize session state variables
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False
    if "letter_text" not in st.session_state:
        st.session_state.letter_text = ""

    # Decide which page to show based on session state
    if not st.session_state.analysis_done:
        show_submission_page()
    else:
        show_preview_page()

def show_submission_page():
    """Page 1: 'Letter Submission (Data Entry)' UI."""
    # Bolsover logo at the top
    cols = st.columns([1, 2, 1])
    with cols[1]:
        st.image("src/img/Bolsover_District_Council_logo.svg", width=120)

    # Page title
    st.markdown("<h2 style='text-align: center;'>Letter Submission (Data Entry)</h2>", unsafe_allow_html=True)
    st.write("")

    # Radio for "Paste Letter" vs. "Upload File"
    col1, col2, col3 = st.columns([2, 6, 2])
    with col2:
        data_mode = st.radio(
            label="Upload or Paste a Letter",
            options=["Paste Letter", "Upload File"],
            index=0,
            horizontal=True
        )

    input_text = ""
    uploaded_file = None

    if data_mode == "Paste Letter":
        input_text = st.text_area("Paste letter text here...", height=200)
    else:
        uploaded_file = st.file_uploader(
            "Upload or Drag & Drop a file (txt, csv, pdf, doc, docx)",
            type=["txt", "csv", "pdf", "doc", "docx"],
            accept_multiple_files=False
        )

    # Large button to trigger analysis
    if st.button("Upload & Analyse Letter", use_container_width=True):
        if data_mode == "Paste Letter":
            if not input_text.strip():
                st.warning("Please paste some text before clicking 'Upload & Analyse Letter'.")
                return
            else:
                st.session_state.letter_text = input_text
                st.session_state.analysis_done = True
                st.experimental_rerun()
        else:
            if uploaded_file is None:
                st.warning("Please upload a file before clicking 'Upload & Analyse Letter'.")
                return
            else:
                # Read file contents
                text_from_file = read_uploaded_file(uploaded_file)
                if not text_from_file.strip():
                    st.warning("Uploaded file is empty or could not be read.")
                    return
                st.session_state.letter_text = text_from_file
                st.session_state.analysis_done = True
                st.experimental_rerun()

def read_uploaded_file(uploaded_file):
    """Helper to read contents from the uploaded file."""
    if uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        return " ".join(df['text'].astype(str).tolist())
    elif uploaded_file.type == "application/pdf":
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return ""
    elif uploaded_file.type in [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword"
    ]:
        try:
            return docx2txt.process(uploaded_file)
        except Exception as e:
            st.error(f"Error processing DOC/DOCX: {e}")
            return ""
    else:
        return uploaded_file.read().decode("utf-8")

def show_preview_page():
    """
    Page 2: 'Letter Preview' style UI
    after the user has provided text and clicked 'Upload & Analyse Letter'.
    """
    letter_text = st.session_state.letter_text

    # Example: run your analysis logic here
    # letter_clean = comprehensive_text_preprocessing(letter_text)
    # classification_result = ...
    # topic_result = ...
    # summarization, etc.

    # For demonstration, let's just show a placeholder UI.
    # 1) Bolsover logo
    st.image("src/img/Bolsover_District_Council_logo.svg", width=120)

    # 2) Title: e.g. "Appreciation for Road Maintenance Improvements..."
    st.markdown("<h3 style='text-align: center;'>Appreciation for Road Maintenance Improvements on High Street, Bristol</h3>", unsafe_allow_html=True)

    # 3) Category, Priority, Date
    colA, colB = st.columns([2,2])
    with colA:
        st.write("**Category**: Road Maintenance")
        st.write("**Priority**: Urgent")
    with colB:
        st.write("**Date**: 12 March 2025")

    # 4) Summarization Options
    st.markdown("### Summarization Options")
    with st.expander("Key Takeaways"):
        st.write("- Overflowing bins addressed\n- Litter causing health concerns")

    with st.expander("Highlighted Sentences"):
        st.write("- Example highlight 1\n- Example highlight 2")

    with st.expander("Inquisitive Summary"):
        st.write("Sample inquisitive summary text, e.g. 'What improvements have been made...'")

    # 5) Sidebar with letter preview or reassign actions
    st.sidebar.markdown("## Letter Preview")
    st.sidebar.write(letter_text)

    if st.sidebar.button("Mark as Reviewed"):
        st.info("Marked as reviewed (placeholder).")

    if st.sidebar.button("Reassign Category"):
        st.info("Category reassign flow... (placeholder).")

    if st.sidebar.button("Add Internal Notes"):
        st.info("Add internal notes flow... (placeholder).")

    # 6) A "Back" or "Return to Data Entry" button
    if st.button("Return to Data Entry"):
        st.session_state.analysis_done = False
        st.session_state.letter_text = ""
        st.experimental_rerun()

if __name__ == "__main__":
    main()
