import streamlit as st
from config import PAGE_CONFIG
from ui.pages.data_entry import data_entry_page
from ui.pages.results import results_page
from ui.pages.aggregated import aggregated_analysis_page
from config.logging_config import setup_logger

# Initialize logging at the start of your application
logger = setup_logger()


def main():
    logger.info("Starting Bolsover District Council application")

    st.set_page_config(**PAGE_CONFIG)
    st.sidebar.image("src/img/Bolsover_District_Council_logo.svg", width=150)

    if "page" not in st.session_state:
        st.session_state.page = "data_entry"

    if st.session_state.page == "data_entry":
        data_entry_page()
    elif st.session_state.page == "results":
        results_page()
    elif st.session_state.page == "aggregated_analysis":
        aggregated_analysis_page()

if __name__ == '__main__':
    main()