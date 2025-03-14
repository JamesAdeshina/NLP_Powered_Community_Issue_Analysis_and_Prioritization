import streamlit as st
import base64
from PIL import Image
import io

# Function to display a single document (image or PDF)
def preview_single_document(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            st.write("### PDF Preview")
            # Display PDF in the app
            pdf_data = uploaded_file.getvalue()
            st.markdown(f'<iframe src="data:application/pdf;base64,{base64.b64encode(pdf_data).decode()}" width="700" height="500"></iframe>', unsafe_allow_html=True)
        elif uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
            # Show image preview
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        else:
            st.error("Unsupported file type.")

# Function to display multiple documents and allow navigation with "Next"
def preview_multiple_documents():
    # File uploader to upload multiple documents
    uploaded_files = st.file_uploader("Choose files", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        # Initialize session state to store the index for navigation
        if 'index' not in st.session_state:
            st.session_state.index = 0

        # Display current document
        current_file = uploaded_files[st.session_state.index]
        st.write(f"### Document {st.session_state.index + 1} Preview")
        preview_single_document(current_file)

        # Navigation buttons (Next and Previous)
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.session_state.index > 0:
                if st.button("Previous", key="prev_button"):
                    st.session_state.index -= 1
        with col2:
            if st.session_state.index < len(uploaded_files) - 1:
                if st.button("Next", key="next_button"):
                    st.session_state.index += 1
            else:
                st.warning("You have reached the last document!")

# Main app
def main():
    st.title("Document Preview App")

    # Option to preview a single or multiple documents
    option = st.radio("Choose an option", ("Preview a Single Document", "Preview Multiple Documents"))

    if option == "Preview a Single Document":
        st.subheader("Upload and Preview a Document")
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "png", "jpg", "jpeg"])
        preview_single_document(uploaded_file)

    elif option == "Preview Multiple Documents":
        st.subheader("Upload and Preview Multiple Documents")
        preview_multiple_documents()

if __name__ == "__main__":
    main()
