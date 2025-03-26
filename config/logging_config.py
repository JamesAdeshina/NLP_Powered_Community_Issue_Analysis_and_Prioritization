# config/logging_config.py
import logging
import os
from datetime import datetime


def setup_logger(name=__name__):
    """Configure and return a logger instance"""
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    # Configure logging
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(logs_dir, f"bolsover_{current_date}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(name)


# Test the function works when run directly
if __name__ == "__main__":
    test_logger = setup_logger()
    test_logger.info("Logger test successful")