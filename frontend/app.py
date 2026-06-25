import streamlit as st

from api import ask_question
from components import (
    render_answer,
    render_chunks,
    render_header,
    render_intent,
    render_iot,
    render_pipeline,
    render_prompt,
    render_search_box,
)

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="Fleet Management RAG",
    page_icon="🚚",
    layout="wide",
)

# ----------------------------------------------------
# Load CSS
# ----------------------------------------------------

try:
    with open("styles.css") as css:
        st.markdown(
            f"<style>{css.read()}</style>",
            unsafe_allow_html=True,
        )
except FileNotFoundError:
    pass

# ----------------------------------------------------
# Header
# ----------------------------------------------------

render_header()

query = render_search_box()

# ----------------------------------------------------
# Execute Query
# ----------------------------------------------------

if query:

    with st.spinner("Analyzing fleet data..."):
        result = ask_question(query)

    if result.get("error") and not result.get("response"):
        st.error(result["error"])
        st.stop()

    # ----------------------------------------------------
    # Always show final answer first
    # ----------------------------------------------------

    render_answer(result)

    st.divider()

    intent = result.get("intent", "").lower()

    # ----------------------------------------------------
    # Dynamic tabs based on intent
    # ----------------------------------------------------

    if intent == "iot":

        tab_iot, tab_trace, tab_prompt = st.tabs(
            [
                "📡 Live IoT Data",
                "⚙️ Pipeline Trace",
                "💻 Raw Prompt",
            ]
        )

        with tab_iot:
            render_iot(result)

        with tab_trace:
            col1, col2 = st.columns([1, 2])

            with col1:
                render_intent(result)

            with col2:
                render_pipeline(result)

        with tab_prompt:
            render_prompt(result)

    elif intent == "manual":

        tab_trace, tab_chunks, tab_prompt = st.tabs(
            [
                "⚙️ Pipeline Trace",
                "📄 Retrieved Chunks",
                "💻 Raw Prompt",
            ]
        )

        with tab_trace:
            col1, col2 = st.columns([1, 2])

            with col1:
                render_intent(result)

            with col2:
                render_pipeline(result)

        with tab_chunks:
            render_chunks(result)

        with tab_prompt:
            render_prompt(result)

    elif intent == "hybrid":

        tab_iot, tab_trace, tab_chunks, tab_prompt = st.tabs(
            [
                "📡 Live IoT Data",
                "⚙️ Pipeline Trace",
                "📄 Retrieved Chunks",
                "💻 Raw Prompt",
            ]
        )

        with tab_iot:
            render_iot(result)

        with tab_trace:
            col1, col2 = st.columns([1, 2])

            with col1:
                render_intent(result)

            with col2:
                render_pipeline(result)

        with tab_chunks:
            render_chunks(result)

        with tab_prompt:
            render_prompt(result)

    else:

        tab_trace = st.tabs(["⚙️ Pipeline Trace"])[0]

        with tab_trace:
            col1, col2 = st.columns([1, 2])

            with col1:
                render_intent(result)

            with col2:
                render_pipeline(result)