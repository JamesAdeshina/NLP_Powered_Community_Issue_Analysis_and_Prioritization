import streamlit as st


def card_style():
    st.markdown(
        """
        <style>
        .keytakeaways-card {
            display: flex;
            flex-direction: row;
            align-items: flex-start;
            padding: 23px 20px;
            gap: 10px;
            isolation: isolate;
            width: 100%;
            min-height: 277px;
            background: linear-gradient(0deg, #87DB5A, #87DB5A), #FAFAFA;
            border-radius: 20px;
            margin-bottom: 20px;
        }

        .highlighted-card {
            display: flex;
            flex-direction: row;
            align-items: flex-start;
            padding: 23px 20px;
            gap: 10px;
            isolation: isolate;
            width: 100%;
            min-height: 277px;
            background: #FFD76A;
            border-radius: 20px;
            margin-bottom: 20px;
        }

        .card-title {
            font-weight: 600;
            font-size: 18px;
            margin-bottom: 15px;
        }

        .card-content {
            font-size: 16px;
            line-height: 1.5;
        }

        .highlighted-content {
            font-size: 16px;
            line-height: 1.5;
            padding-left: 0;
        }


        </style>
        """,
        unsafe_allow_html=True,
    )


def keytakeaways_card(title, content):
    card_style()
    st.markdown(f"""
        <div class="keytakeaways-card">
            <div>
                <div class="card-title">{title}</div>
                <div class="card-content">{content}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def highlighted_card(title, content):
    card_style()
    st.markdown(f"""
        <div class="highlighted-card">
            <div>
                <div class="card-title">{title}</div>
                <ul class="highlighted-content">{content}</ul>
            </div>
        </div>
    """, unsafe_allow_html=True)

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