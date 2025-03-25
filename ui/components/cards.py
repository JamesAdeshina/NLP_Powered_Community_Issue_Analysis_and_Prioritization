import streamlit as st


def card_style():
    st.markdown(
        """
        <style>
        .card {
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            transition: 0.3s;
            border-radius: 5px;
            padding: 16px;
            margin: 10px 0;
            background-color: var(--background-color);
            color: var(--text-color);
        }
        .card:hover {
            box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
        }
        [data-theme="light"] .card {
            --background-color: #F0F2F6;
            --text-color: #000000;
        }
        [data-theme="dark"] .card {
            --background-color: #262730;
            --text-color: #FFFFFF;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def summary_card(title, content):
    card_style()
    st.markdown(f"""
        <div class="card">
            <h3>{title}</h3>
            <p>{content}</p>
        </div>
    """, unsafe_allow_html=True)


def kpi_card(title, value, icon):
    theme_base = st.get_option("theme.base")
    bg_color = "#FFFFFF" if theme_base == "light" else "#222"
    text_color = "#000000" if theme_base == "light" else "#FFFFFF"

    st.markdown(f"""
    <div style='background-color:{bg_color}; padding:15px; border-radius:10px; text-align:center; box-shadow: 0px 4px 6px rgba(0,0,0,0.1);'>
        <h3 style='color:{text_color};'>{icon} {title}</h3>
        <h2 style='color:{text_color};'>{value}</h2>
    </div>
    """, unsafe_allow_html=True)