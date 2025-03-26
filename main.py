import sys
from pathlib import Path
import logging
import streamlit as st

# 1. PATH CONFIGURATION - Should be first
# Add project root to Python path (before any other imports)
sys.path.append(str(Path(__file__).parent))

# 2. LOGGING SETUP - Should be early
from config.logging_config import setup_logger
logger = setup_logger()

# 3. STREAMLIT CONFIG - After logging
from config import PAGE_CONFIG
st.set_page_config(**PAGE_CONFIG)

# 4. UI IMPORTS - After all configurations
from ui.pages.data_entry import data_entry_page
from ui.pages.results import results_page
from ui.pages.aggregated import aggregated_analysis_page



def main():
    logger.info("Starting Bolsover District Council application")
    logger.info("Application started...")

    # Initialize page in session state if not exists
    if "page" not in st.session_state:
        st.session_state.page = "data_entry"
        logger.debug("Initialized session state with default page")


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