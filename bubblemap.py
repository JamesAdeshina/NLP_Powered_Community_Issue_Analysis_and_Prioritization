import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np

# Initialize session state
if 'selected_category' not in st.session_state:
    st.session_state.selected_category = None

# Bolsover District coordinates
BOLSOVER_CENTER = (53.23, -1.29)

# Generate sample data with categories
def generate_data():
    return pd.DataFrame({
        'latitude': np.random.normal(BOLSOVER_CENTER[0], 0.01, 10),
        'longitude': np.random.normal(BOLSOVER_CENTER[1], 0.01, 10),
        'size': np.random.randint(10, 50, 10),
        'category': np.random.choice(['local_problem', 'new_initiative'], 10),
        'description': [f"Issue {i+1}" for i in range(10)]
    })

data = generate_data()

# Create separate layers for each category
layers = [
    pdk.Layer(
        "ScatterplotLayer",
        data[data['category'] == 'local_problem'],
        id='local_problem',
        get_position=["longitude", "latitude"],
        get_radius="size",
        get_fill_color="[255, 0, 0, 200]",  # Red
        radius_scale=5,
        pickable=True,
    ),
    pdk.Layer(
        "ScatterplotLayer",
        data[data['category'] == 'new_initiative'],
        id='new_initiative',
        get_position=["longitude", "latitude"],
        get_radius="size",
        get_fill_color="[0, 0, 255, 200]",  # Blue
        radius_scale=7,
        pickable=True,
    )
]

# Create map view
view_state = pdk.ViewState(
    latitude=BOLSOVER_CENTER[0],
    longitude=BOLSOVER_CENTER[1],
    zoom=11,
    pitch=0
)

# Create deck with click handler
deck = pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    tooltip={"text": "{description}"},
    map_style="road",
)

# Modified click handling with error prevention
result = st.pydeck_chart(deck)

# Safely handle click events
clicked_layer = None
if result and hasattr(result, 'last_clicked'):
    click_data = result.last_clicked
    if isinstance(click_data, dict):  # Ensure it's a dictionary
        clicked_layer = click_data.get('layer', {}).get('id')

# Update session state
st.session_state.selected_category = clicked_layer if clicked_layer in ['local_problem', 'new_initiative'] else None

# Display filtered data based on selection
if st.session_state.selected_category:
    filtered_data = data[data['category'] == st.session_state.selected_category]
    st.subheader(f"Showing {st.session_state.selected_category.replace('_', ' ').title()}")
else:
    filtered_data = data
    st.subheader("All Data Points")

# Display data table
st.dataframe(filtered_data[['latitude', 'longitude', 'size', 'description', 'category']],
            column_config={
                "latitude": "Latitude",
                "longitude": "Longitude",
                "size": "Severity/Size",
                "description": "Description",
                "category": "Category"
            })

# Add reset instruction
st.caption("Click on map points to filter by category. Click outside points to reset view.")