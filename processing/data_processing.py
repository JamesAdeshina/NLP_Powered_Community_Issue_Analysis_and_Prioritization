import pandas as pd
from utils.preprocessing import comprehensive_text_preprocessing
from models.classification import classify_document
from models.sentiment import sentiment_analysis
from utils.geocoding import extract_locations, geocode_location


def process_uploaded_data(uploaded_files_texts):
    df = pd.DataFrame({"text": uploaded_files_texts})

    # Text processing
    df["clean_text"] = df["text"].apply(comprehensive_text_preprocessing)
    df["classification"] = df["clean_text"].apply(classify_document)
    df["sentiment"] = df["text"].apply(lambda x: sentiment_analysis(x)["sentiment_label"])

    # Location processing
    df["locations"] = df["text"].apply(
        lambda t: re.findall(UK_ADDRESS_REGEX, t, flags=re.IGNORECASE)
    )
    df["postcodes"] = df["text"].apply(
        lambda t: re.findall(UK_POSTCODE_REGEX, t, flags=re.IGNORECASE)
    )
    df["all_locations"] = df.apply(
        lambda row: list(set(row["locations"] + row["postcodes"])),
        axis=1
    )
    df["geocoded"] = df["all_locations"].apply(
        lambda locs: [geocode_location(loc) for loc in locs if loc]
    )

    # Explode locations into individual rows
    df = df.explode("geocoded").reset_index(drop=True)
    df[["lat", "lon"]] = pd.DataFrame(
        df["geocoded"].tolist(),
        index=df.index
    )

    return df.dropna(subset=["lat", "lon"])