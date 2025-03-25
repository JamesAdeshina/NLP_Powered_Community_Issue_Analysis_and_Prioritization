import pydeck as pdk
import streamlit as st
import pandas as pd
import plotly.express as px


def create_clustered_map(df, filter_by_sentiment=None, filter_by_issue=None, filter_by_topic=None):
    if filter_by_sentiment:
        df = df[df["sentiment"] == filter_by_sentiment]
    if filter_by_issue:
        df = df[df["Issue"] == filter_by_issue]
    if filter_by_topic:
        df = df[df["Topic"] == filter_by_topic]

    df = df.dropna(subset=['lat', 'lon'])
    if df.empty:
        st.error("No data found for the selected filters.")
        return None

    color_mapping = {
        "POSITIVE": [0, 0, 255, 160],
        "NEGATIVE": [255, 0, 0, 160],
    }

    layer = pdk.Layer(
        "ScatterplotLayer",
        df,
        get_position=["lon", "lat"],
        get_color="[color_mapping[sentiment][0], color_mapping[sentiment][1], color_mapping[sentiment][2], color_mapping[sentiment][3]]",
        get_radius=100,
        pickable=True,
    )

    view_state = pdk.ViewState(
        longitude=df["lon"].mean(),
        latitude=df["lat"].mean(),
        zoom=12,
        pitch=0,
        bearing=0,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v10",
        tooltip={
            "html": "<b>Issue:</b> {Issue}<br><b>Sentiment:</b> {sentiment}<br><b>Topic:</b> {Topic}<br><b>Address:</b> {Address}<br><b>Text:</b> {text}",
            "style": {
                "backgroundColor": "steelblue",
                "color": "white",
            },
        },
    )

    return deck


def create_sentiment_map(df):
    df = df.dropna(subset=["lat", "lon", "sentiment"])
    if df.empty:
        return None

    df["color"] = df["sentiment"].apply(
        lambda x: [0, 255, 0, 160] if x == "POSITIVE" else [255, 0, 0, 160])

    min_lat, max_lat = df["lat"].min(), df["lat"].max()
    min_lon, max_lon = df["lon"].min(), df["lon"].max()

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["lon", "lat"],
        get_color="color",
        get_radius=200,
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=(min_lat + max_lat) / 2,
        longitude=(min_lon + max_lon) / 2,
        zoom=10,
        min_zoom=5,
        max_zoom=15,
        pitch=0,
        bearing=0
    )

    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v10" if st.get_option(
            "theme.base") == "light" else "mapbox://styles/mapbox/dark-v10",
        tooltip={"text": "Sentiment: {sentiment}\nAddress: {Address}"}
    )