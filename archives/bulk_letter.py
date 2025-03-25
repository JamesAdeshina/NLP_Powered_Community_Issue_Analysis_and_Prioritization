import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import random

# Set page config
st.set_page_config(page_title="Bulk Letter Insights Dashboard", layout="wide")

# Sidebar - Letter Preview
st.sidebar.image("https://via.placeholder.com/150", caption="Review the 238 letters")
st.sidebar.subheader("Original Letter")
st.sidebar.write("""
Dear Sir/Madam, I am writing to express my sincere appreciation for the recent improvements in road maintenance on High Street, Bristol...
""")

# Date Selection
st.markdown("### 📅 Select Date Range")
date_range = st.date_input(" ", [datetime.date(2025, 3, 12), datetime.date(2025, 3, 13)], label_visibility='collapsed')

# KPI Metrics in Cards with Theme-based Colors
theme_base = st.get_option("theme.base")
bg_color = "#FFFFFF" if theme_base == "light" else "#222"
text_color = "#000000" if theme_base == "light" else "#FFFFFF"

st.markdown("### Key Metrics")
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
with kpi_col1:
    st.markdown(f"""
    <div style='background-color:{bg_color}; padding:15px; border-radius:10px; text-align:center; box-shadow: 0px 4px 6px rgba(0,0,0,0.1);'>
        <h3 style='color:{text_color};'>📩 Total Letters</h3>
        <h2 style='color:{text_color};'>238</h2>
    </div>
    """, unsafe_allow_html=True)

with kpi_col2:
    st.markdown(f"""
    <div style='background-color:{bg_color}; padding:15px; border-radius:10px; text-align:center; box-shadow: 0px 4px 6px rgba(0,0,0,0.1);'>
        <h3 style='color:{text_color};'>📍 Local Problems</h3>
        <h2 style='color:{text_color};'>85%</h2>
    </div>
    """, unsafe_allow_html=True)

with kpi_col3:
    st.markdown(f"""
    <div style='background-color:{bg_color}; padding:15px; border-radius:10px; text-align:center; box-shadow: 0px 4px 6px rgba(0,0,0,0.1);'>
        <h3 style='color:{text_color};'>✨ New Initiatives</h3>
        <h2 style='color:{text_color};'>22%</h2>
    </div>
    """, unsafe_allow_html=True)

# Most Common Issues
st.subheader("Most Common Issues")
data_issues = pd.DataFrame({
    "Issue": ["Road Potholes", "Housing Complaints", "Waste Management"],
    "Percentage": [30, 20, 15],
    "Urgent Reports": [5, 3, 2]
})
fig_issues = px.bar(data_issues, x="Issue", y="Percentage", color="Urgent Reports", text="Percentage")
st.plotly_chart(fig_issues, use_container_width=True)

# Classification & Sentiment Side by Side
st.subheader("📊 Classification Distribution & 😊 Sentiment Analysis")
col4, col5 = st.columns(2)
with col4:
    classification_data = pd.DataFrame({
        "Category": ["Local Problems", "New Initiatives", "Others"],
        "Count": [150, 52, 36]
    })
    fig_classification = px.pie(classification_data, values='Count', names='Category', title="Classification Distribution")
    st.plotly_chart(fig_classification, use_container_width=True)

with col5:
    sentiment_data = pd.DataFrame({
        "Sentiment": ["Positive", "Neutral", "Negative"],
        "Count": [123, 120, 110]
    })
    fig_sentiment = px.bar(sentiment_data, x='Sentiment', y='Count', title="Sentiment Analysis", color='Sentiment')
    st.plotly_chart(fig_sentiment, use_container_width=True)

# Key Takeaways & Highlighted Sentences Side by Side
st.subheader("💡 Key Takeaways & 🔍 Highlighted Sentences")
col6, col7 = st.columns(2)
with col6:
    st.text_area("Paste key takeaways here...", key="takeaways")
with col7:
    st.text_area("Paste highlighted sentences here...", key="highlighted")

# AI Search
txt_search = st.text_input("❓Ask AI a Question About the Letters")
st.write(f"Search results for: {txt_search}")

# Scatter Plot with Tabs
st.subheader("📍 Letter Sentiment Analysis")
tabs = st.tabs(["Sentiment", "Categories", "Topics"])
with tabs[0]:
    latitudes = [51.45 + random.uniform(-0.01, 0.01) for _ in range(100)]
    longitudes = [-2.58 + random.uniform(-0.01, 0.01) for _ in range(100)]
    sentiments = [random.choice(["Positive", "Negative"]) for _ in range(100)]
    df_map = pd.DataFrame({"lat": latitudes, "lon": longitudes, "sentiment": sentiments})
    fig_map = px.scatter_mapbox(df_map, lat="lat", lon="lon", color="sentiment", zoom=12,
                                mapbox_style="carto-positron", title="Sentiment Map")
    st.plotly_chart(fig_map, use_container_width=True)
