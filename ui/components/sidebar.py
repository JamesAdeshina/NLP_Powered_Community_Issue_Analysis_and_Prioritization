import streamlit as st


def pick_sidebar_icon(num_files, file_types):
    if num_files == 0:
        return "src/img/Multiple_Default.svg"
    if num_files == 1:
        ft = next(iter(file_types))
        if ft == "pdf":
            return "src/img/Single_Pdf.svg"
        elif ft == "doc":
            return "src/img/Single_Doc.svg"
        else:
            return "src/img/Single_Default.svg"
    if file_types == {"pdf"}:
        return "src/img/Multiple_Pdf.svg"
    elif file_types == {"doc"}:
        return "src/img/Multiple_Doc.svg"
    elif len(file_types) > 1:
        return "src/img/Multiple_Both.svg"
    else:
        return "src/img/Multiple_Default.svg"


def show_sidebar(file_info, text):
    with st.sidebar:
        num_files = file_info.get("num_files", 0)
        ext_set = file_info.get("file_extensions", set())

        icon_path = pick_sidebar_icon(num_files, ext_set)
        st.image(icon_path, width=150)

        with st.expander("Original Letter", expanded=False):
            st.write(text if text.strip() else "No text available")