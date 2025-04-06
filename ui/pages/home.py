import streamlit as st
import base64
import os
import logging

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def get_base64_of_bin_file(img_path):
    """
    Convert an image file to base64 encoding.
    """
    try:
        logger.info(f"Loading image file from: {img_path}")
        with open(img_path, 'rb') as f:
            data = f.read()
        encoded_image = base64.b64encode(data).decode()
        logger.info(f"Successfully loaded image: {img_path}")
        return encoded_image
    except Exception as e:
        logger.error(f"Error loading image file {img_path}: {e}")
        raise


def set_bg_from_local(img_path):
    """
    Set the background image from a local file path.
    """
    logger.info(f"Setting background image from: {img_path}")
    img_base64 = get_base64_of_bin_file(img_path)
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
    logger.info(f"Background image set successfully from: {img_path}")


def home_page():
    """
    The home page of the Bolsover District Council application.
    """
    logger.info("Loading home page...")

    # Set the full-page background
    set_bg_from_local("src/img/background_img.png")

    # Custom CSS for layout and button styling
    logger.info("Applying custom CSS for layout and button styling...")
    st.markdown(
        """
        <style>
        .block-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
        .stButton button {
            padding: 0.75rem 2rem;
            font-size: 1.2rem;
            background-color: #3778c2;
            color: white;
            border-radius: 8px;
        }
        /* Center all images */
        .stImage {
            display: flex;
            justify-content: center !important;
        }
        .rounded-image {
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin: 20px 0;
            transition: transform 0.3s;
        }
        .rounded-image:hover {
            transform: scale(1.02);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Get base64 of the logo image
    logo_path = os.path.join("src", "img", "Bolsover_District_Council_logo.png")
    logger.info(f"Loading logo image from: {logo_path}")
    logo_base64 = get_base64_of_bin_file(logo_path)

    # Get base64 of the Bolsover image
    bolsover_path = os.path.join("src", "img", "Bolsover.png")
    logger.info(f"Loading Bolsover image from: {bolsover_path}")
    bolsover_base64 = get_base64_of_bin_file(bolsover_path)

    # Display the logo image
    st.markdown(
        f"""
        <div style="text-align:center;">
            <img src="data:image/png;base64,{logo_base64}" width="150" />
        </div>
        """,
        unsafe_allow_html=True
    )

    st.title("Public Letters and Correspondence")

    # Display images in columns (update paths as needed)
    st.markdown(
        f"""
        <div style="text-align:center;">
            <img class="rounded-image" src="data:image/png;base64,{bolsover_base64}" width="500" />
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    st.write("")

    # Log button click event
    def go_to_data_entry():
        logger.info("Continue button clicked, navigating to data entry page.")
        st.session_state.page = "data_entry"

    if st.button("Continue", on_click=go_to_data_entry):
        pass

    logger.info("Home page loaded successfully.")


if __name__ == "__main__":
    home_page()
