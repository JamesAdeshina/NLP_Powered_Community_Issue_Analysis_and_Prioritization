import streamlit as st
from datetime import datetime


def show_feedback(message, level="info", details=None, expandable=False):
    """
    Display consistent feedback messages to users
    Parameters:
    - message: Main message to display
    - level: 'info', 'warning', 'error', or 'success'
    - details: Additional details for debugging
    - expandable: Whether to show details in an expander
    """
    if level == "error":
        st.error(message)
    elif level == "warning":
        st.warning(message)
    elif level == "success":
        st.success(message)
    else:
        st.info(message)

    if details and expandable:
        with st.expander("Technical details"):
            st.text(details)

    # Log the feedback
    log_message = f"{level.upper()}: {message}"
    if details:
        log_message += f"\nDetails: {details}"

    # Get logger from streamlit's session state or create new
    if 'logger' not in st.session_state:
        from config.logging_config import setup_logger
        st.session_state.logger = setup_logger()

    if level == "error":
        st.session_state.logger.error(log_message)
    elif level == "warning":
        st.session_state.logger.warning(log_message)
    else:
        st.session_state.logger.info(log_message)