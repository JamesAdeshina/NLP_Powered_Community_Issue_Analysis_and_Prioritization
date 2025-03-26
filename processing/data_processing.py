import re
import pandas as pd
from utils.preprocessing import comprehensive_text_preprocessing
from models.classification import classify_document
from models.sentiment import sentiment_analysis
# from utils.geocoding import extract_locations, geocode_location, geocode_addresses
from processing.geocoding import extract_locations, geocode_location, geocode_addresses
from config import (
    UK_POSTCODE_REGEX,
    UK_ADDRESS_REGEX
)


def process_uploaded_data(uploaded_files_texts):
    df = pd.DataFrame({"text": uploaded_files_texts})

    # Extract all unique locations first
    all_locations = set()
    for text in uploaded_files_texts:
        addresses = set(re.findall(UK_ADDRESS_REGEX, text, flags=re.IGNORECASE))
        postcodes = set(re.findall(UK_POSTCODE_REGEX, text, flags=re.IGNORECASE))
        all_locations.update(addresses)
        all_locations.update(postcodes)

    # Geocode all unique locations
    location_data = []
    for loc in all_locations:
        try:
            lat, lon = geocode_location(loc)
            if lat and lon:
                location_data.append({
                    "location": loc,
                    "lat": lat,
                    "lon": lon
                })
        except Exception as e:
            print(f"Geocoding failed for '{loc}': {str(e)}")

    locations_df = pd.DataFrame(location_data)

    # Match texts to locations
    result = []
    for idx, text in enumerate(uploaded_files_texts):
        clean_text = comprehensive_text_preprocessing(text)
        classification = classify_document(clean_text)
        sentiment = sentiment_analysis(text)["sentiment_label"]

        text_addresses = set(re.findall(UK_ADDRESS_REGEX, text, flags=re.IGNORECASE))
        text_postcodes = set(re.findall(UK_POSTCODE_REGEX, text, flags=re.IGNORECASE))
        text_locations = text_addresses.union(text_postcodes)

        for loc in text_locations:
            loc_info = locations_df[locations_df["location"] == loc]
            if not loc_info.empty:
                result.append({
                    "text": text,
                    "clean_text": clean_text,
                    "classification": classification,
                    "sentiment": sentiment,
                    "location": loc,
                    "lat": loc_info["lat"].values[0],
                    "lon": loc_info["lon"].values[0]
                })

    return pd.DataFrame(result)



