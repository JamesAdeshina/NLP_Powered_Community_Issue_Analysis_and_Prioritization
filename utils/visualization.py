import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import pydeck as pdk


def plot_classification_distribution(class_counts):
    fig = go.Figure([go.Bar(x=class_counts.index, y=class_counts.values)])
    fig.update_layout(title="Classification Distribution", xaxis_title="Category", yaxis_title="Count")
    return fig

def plot_sentiment_distribution(avg_sentiment):
    fig = go.Figure([go.Bar(x=["Average Sentiment Polarity"], y=[avg_sentiment])])
    fig.update_layout(title="Average Sentiment Polarity", xaxis_title="Metric", yaxis_title="Polarity")
    return fig

def plot_sentiment_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={"text": "Sentiment Confidence"},
        gauge={
            "axis": {"range": [0, 1]},
            "steps": [
                {"range": [0, 0.3], "color": "red"},
                {"range": [0.3, 0.7], "color": "yellow"},
                {"range": [0.7, 1], "color": "green"}
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": confidence
            }
        }
    ))
    fig.update_layout(
        annotations=[
            dict(x=0.15, y=0.1, text="<b>Low</b>", showarrow=False,
                 font=dict(color="red", size=12)),
            dict(x=0.50, y=0.1, text="<b>Medium</b>", showarrow=False,
                 font=dict(color="yellow", size=12)),
            dict(x=0.85, y=0.1, text="<b>High</b>", showarrow=False,
                 font=dict(color="green", size=12))
        ]
    )
    return fig

def create_bolsover_map(df):
    fig = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        color="Issue",
        hover_name="Address",
        hover_data=["text"],
        zoom=12,
        height=600,
        title="Geographical Distribution of Issues in Bolsover"
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig