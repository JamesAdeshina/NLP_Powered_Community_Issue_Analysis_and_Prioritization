import sys
import os
import streamlit as st
from PIL import Image

# Add root project folder to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ✅ Now this will work if your file is named `ocr_file_reader.py`
from processing.ocr_file_reader import (
    extract_text_from_image,
    extract_text_from_pdf,
)

st.set_page_config(page_title="OCR Text Extractor", layout="centered")
st.title("📄 OCR Text Extractor")

uploaded_file = st.file_uploader("Upload a PDF, DOCX, or image file", type=["pdf", "docx", "png", "jpg", "jpeg"])

if uploaded_file:
    file_type = uploaded_file.type
    st.info(f"Processing file: {uploaded_file.name}")

    with st.spinner("Extracting text..."):
        if "pdf" in file_type:
            text = extract_text_from_pdf(uploaded_file.read())
        elif "image" in file_type:
            image = Image.open(uploaded_file)
            text = extract_text_from_image(image)
        else:
            st.error("Unsupported file type.")
            st.stop()

    st.subheader("📜 Extracted Text:")
    st.text_area("Output", value=text, height=400)
