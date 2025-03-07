import streamlit as st
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()


def get_sentiment_score(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']  # Compound score ranges from -1 (negative) to 1 (positive)


def create_gauge_with_needle(sentiment_score):
    """Creates a gauge chart with a moving needle for sentiment analysis"""

    # Convert sentiment score to angle (for needle placement)
    min_angle, max_angle = -90, 90  # Semi-circle range
    needle_angle = min_angle + (sentiment_score + 1) * (max_angle - min_angle) / 2

    # Define the gauge
    fig = go.Figure()

    # Gauge background
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=sentiment_score,
        title={'text': "Sentiment Score"},
        gauge={
            'axis': {'range': [-1, 1], 'tickvals': [-1, -0.5, 0, 0.5, 1]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [-1, -0.5], 'color': "red"},
                {'range': [-0.5, 0], 'color': "orange"},
                {'range': [0, 0.5], 'color': "lightgreen"},
                {'range': [0.5, 1], 'color': "green"}
            ],
        }
    ))

    # Needle Calculation
    r = 0.75  # Needle length
    x = r * np.cos(np.radians(needle_angle))
    y = r * np.sin(np.radians(needle_angle))

    # Add needle (triangle marker)
    fig.add_trace(go.Scatter(
        x=[0, x],
        y=[0, y],
        mode="lines",
        line=dict(color="black", width=4),
        name="Needle"
    ))

    # Update layout to make it look better
    fig.update_layout(
        width=500, height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=False,
        polar=dict(
            radialaxis=dict(visible=False),
        )
    )

    return fig


# Streamlit App
st.title("Sentiment Analysis Gauge with Needle")

# User Input
user_text = st.text_area("Enter your text:", "I love this amazing app!")

if user_text:
    sentiment_score = get_sentiment_score(user_text)
    st.write(f"Sentiment Score: {sentiment_score}")

    # Display gauge with needle
    st.plotly_chart(create_gauge_with_needle(sentiment_score))
