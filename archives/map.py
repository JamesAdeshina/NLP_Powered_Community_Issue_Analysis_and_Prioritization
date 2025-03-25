import pandas as pd
import streamlit as st
from opencage.geocoder import OpenCageGeocode
import pydeck as pdk

# Function to geocode addresses
def geocode_addresses(addresses):
    api_key = "e760785d8c7944888beefc24aa42eb66"
    geocoder = OpenCageGeocode(api_key)

    locations = []
    for address in addresses:
        try:
            results = geocoder.geocode(address + ", Bolsover, UK")
            if results and len(results) > 0:
                lat = results[0]['geometry']['lat']
                lon = results[0]['geometry']['lng']
                locations.append((lat, lon))
                st.write(f"Geocoded {address}: ({lat}, {lon})")  # Debug output
            else:
                st.warning(f"Geocoding failed for address: {address}")
                locations.append((None, None))
        except Exception as e:
            st.error(f"Geocoding error for {address}: {str(e)}")
            locations.append((None, None))
    return locations

# Function to create the clustered map using PyDeck
def create_clustered_map(df):
    # Geocode addresses
    df[['lat', 'lon']] = pd.DataFrame(geocode_addresses(df['Address']), columns=['lat', 'lon'])

    # Debug: Show the DataFrame after geocoding
    st.write("DataFrame after geocoding:")
    st.write(df)

    # Drop rows where geocoding failed
    df = df.dropna(subset=['lat', 'lon'])

    # Debug: Show the DataFrame after dropping invalid rows
    st.write("DataFrame after dropping invalid rows:")
    st.write(df)

    if df.empty:
        st.error("No valid geocoded addresses found. Please check the address format.")
        return None

    # Create a basic ScatterplotLayer
    layer = pdk.Layer(
        "ScatterplotLayer",  # Use ScatterplotLayer for simplicity
        df,
        get_position=["lon", "lat"],
        get_color=[255, 0, 0, 160],  # Red color with transparency
        get_radius=100,  # Radius of the points
        pickable=True,
    )

    # Set the viewport location
    view_state = pdk.ViewState(
        longitude=df["lon"].mean(),
        latitude=df["lat"].mean(),
        zoom=12,
        pitch=0,  # No tilt for now
        bearing=0,
    )

    # Create the PyDeck deck
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v10",
        tooltip={
            "html": "<b>Issue:</b> {Issue}<br><b>Address:</b> {Address}<br><b>Text:</b> {text}",
            "style": {
                "backgroundColor": "steelblue",
                "color": "white",
            },
        },
    )

    return deck

# Main function to run the Streamlit app
def main():
    st.title("Bolsover District Council - Issue Map (Clustered with PyDeck)")

    # Example data
    df_letters = pd.DataFrame({
        "Address": [
            "1 Church Street, Bolsover, S44 6JJ, UK",
            "5 Station Road, Bolsover, S44 8GH, UK",
            "12 High Street, Bolsover, S44 0AA, UK",
        ],
        "text": [
            "Road conditions complaint...",
            "Waste management issue...",
            "Noise pollution complaint...",
        ],
        "Issue": [
            "Road Conditions",
            "Waste Management",
            "Noise Pollution",
        ]
    })

    # Create and display the clustered map
    st.subheader("📍 Geographic Issue Distribution (Clustered with PyDeck)")
    deck = create_clustered_map(df_letters)
    if deck:
        st.pydeck_chart(deck)

# Run the app
if __name__ == "__main__":
    main()