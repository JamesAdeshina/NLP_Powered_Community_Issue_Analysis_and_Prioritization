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
    # Initialize DataFrame with text
    df = pd.DataFrame({"text": uploaded_files_texts})

    # Debug: Show sample of input texts
    print(f"Processing {len(uploaded_files_texts)} documents")
    print("Sample text:", uploaded_files_texts[0][:100] + "...")

    # Extract all unique locations first
    all_locations = set()
    for text in uploaded_files_texts:
        try:
            addresses = set(re.findall(UK_ADDRESS_REGEX, text, flags=re.IGNORECASE))
            postcodes = set(re.findall(UK_POSTCODE_REGEX, text, flags=re.IGNORECASE))
            all_locations.update(addresses)
            all_locations.update(postcodes)
        except Exception as e:
            print(f"Error extracting locations from text: {str(e)}")

    print(f"Found {len(all_locations)} unique locations in texts")

    # Geocode all unique locations with enhanced error handling
    location_data = []
    for loc in all_locations:
        try:
            lat, lon = geocode_location(loc)
            if lat is not None and lon is not None:  # Explicit None check
                location_data.append({
                    "location": loc,
                    "lat": float(lat),  # Ensure numeric
                    "lon": float(lon)  # Ensure numeric
                })
                print(f"Successfully geocoded: {loc} → {lat},{lon}")
            else:
                print(f"Geocoding returned None for: {loc}")
        except Exception as e:
            print(f"Geocoding failed for '{loc}': {str(e)}")

    # Create locations DataFrame
    locations_df = pd.DataFrame(location_data)

    # If no locations geocoded, initialize with empty columns
    if locations_df.empty:
        print("Warning: No locations were successfully geocoded")
        locations_df = pd.DataFrame(columns=["location", "lat", "lon"])

    # Process texts with enhanced location handling
    result = []
    for text in uploaded_files_texts:
        try:
            clean_text = comprehensive_text_preprocessing(text)
            classification = classify_document(clean_text)
            sentiment = sentiment_analysis(text)["sentiment_label"]

            # Extract locations from this text
            text_locations = set()
            try:
                addresses = set(re.findall(UK_ADDRESS_REGEX, text, flags=re.IGNORECASE))
                postcodes = set(re.findall(UK_POSTCODE_REGEX, text, flags=re.IGNORECASE))
                text_locations = addresses.union(postcodes)
            except Exception as e:
                print(f"Location extraction failed for text: {str(e)}")

            # If no locations found in text
            if not text_locations:
                result.append({
                    "text": text,
                    "clean_text": clean_text,
                    "classification": classification,
                    "sentiment": sentiment,
                    "location": None,
                    "lat": None,
                    "lon": None
                })
                continue

            # Process each found location
            for loc in text_locations:
                if not locations_df.empty and "location" in locations_df.columns:
                    loc_info = locations_df[locations_df["location"] == loc]
                    if not loc_info.empty:
                        result.append({
                            "text": text,
                            "clean_text": clean_text,
                            "classification": classification,
                            "sentiment": sentiment,
                            "location": loc,
                            "lat": float(loc_info["lat"].values[0]),
                            "lon": float(loc_info["lon"].values[0])
                        })
                else:
                    result.append({
                        "text": text,
                        "clean_text": clean_text,
                        "classification": classification,
                        "sentiment": sentiment,
                        "location": loc,
                        "lat": None,
                        "lon": None
                    })

        except Exception as e:
            print(f"Error processing document: {str(e)}")
            continue

    final_df = pd.DataFrame(result)

    # Debug: Show final data stats
    print(f"Final DataFrame shape: {final_df.shape}")
    print(f"Records with coordinates: {final_df[['lat', 'lon']].notna().all(axis=1).sum()}")

    return final_df





