import sys
from pathlib import Path
import streamlit as st
import logging

# 1. PATH CONFIGURATION
sys.path.append(str(Path(__file__).parent))

# 2. LOGGING SETUP
from config.logging_config import setup_logger

logger = setup_logger()

# 3. STREAMLIT CONFIG
from config import PAGE_CONFIG

st.set_page_config(**PAGE_CONFIG)

# 4. UI IMPORTS
from ui.pages.home import home_page
from ui.pages.data_entry import data_entry_page
from ui.pages.results import results_page
from ui.pages.aggregated import aggregated_analysis_page


def main():
    logger.info("Starting Bolsover District Council application")

    # Initialize session state page if not exists; default to "home"
    if "page" not in st.session_state:
        st.session_state.page = "home"
        logger.debug("Initialized session state with default page 'home'")

    # Only display the sidebar image if we're not on the landing page
    if st.session_state.page != "home":
        st.sidebar.image("src/img/Bolsover_District_Council_logo.svg", width=150)

    # Page Routing
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "data_entry":
        data_entry_page()
    elif st.session_state.page == "results":
        results_page()
    elif st.session_state.page == "aggregated_analysis":
        aggregated_analysis_page()
    else:
        st.error(f"Unknown page: {st.session_state.page}")
        st.session_state.page = "home"
        home_page()


if __name__ == '__main__':
    main()
