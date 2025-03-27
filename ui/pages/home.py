import streamlit as st
import base64
import os

def get_base64_of_bin_file(img_path):
    with open(img_path, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg_from_local(img_path):
    """
    Set the background image from a local file path.
    """
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

def home_page():
    # Set the full-page background
    set_bg_from_local("src/img/background_img.png")

    # Custom CSS for layout and button styling
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
    logo_base64 = get_base64_of_bin_file(logo_path)

    # Get base64 of the Bolsover image
    bolsover_path = os.path.join("src", "img", "Bolsover.png")
    bolsover_base64 = get_base64_of_bin_file(bolsover_path)

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
    # st.write("Welcome to Bolsover District Council’s Public Letters Portal. Click below to continue.")

    def go_to_data_entry():
        st.session_state.page = "data_entry"

    if st.button("Continue", on_click=go_to_data_entry):
        pass


if __name__ == "__main__":
    home_page()
