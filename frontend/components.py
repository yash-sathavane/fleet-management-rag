import pandas as pd
import streamlit as st

def render_header():
    st.title("🚚 Fleet Management Dashboard")
    st.caption("LangGraph • Gemini • ChromaDB • IoT Fleet Monitoring")
    st.divider()

def render_search_box():
    query = st.text_input(
        "Fleet Management Query",
        placeholder="Example: How do I troubleshoot TruckA overheating?",
        label_visibility="collapsed",
    )
    # Make the button visually distinct
    if st.button("🚀 Analyze Query", type="primary", use_container_width=True):
        return query
    return None

def render_answer(result):
    st.subheader("Response")
    response = result.get("response", "No response generated.")
    # Use st.info or st.markdown with a custom box to make it pop
    st.info(response, icon="🧠")

def render_iot(result):
    truck_data = result.get("truck_data", {})
    if not truck_data:
        st.warning("No live IoT data required for this query.")
        return

    st.subheader("Active Fleet Telemetry")
    
    # Render each truck as a row of metric cards instead of a dataframe
    for truck, values in truck_data.items():
        st.markdown(f"**{truck}**")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Fuel Level", f"{values.get('fuel', '--')}%")
        
        # Add visual warnings for high temps
        temp = values.get('temperature', 0)
        temp_color = "normal" if temp < 90 else "inverse" 
        col2.metric("Engine Temp", f"{temp}°C", delta="High" if temp >= 90 else "Normal", delta_color=temp_color)
        
        col3.metric("Current Speed", f"{values.get('speed', '--')} km/h")
        col4.metric("Location", str(values.get('location', '--')))
        st.write("---")

def render_pipeline(result):
    st.subheader("Execution Path")
    intent = result.get('intent', 'unknown').upper()
    
    # Condense the success boxes into a clean markdown flow
    steps = [
        "**User Query Received**",
        "**Supervisor Agent** routed request",
        f"**Intent Classified:** `{intent}`",
        "**Information Agent** triggered"
    ]
    
    intent_raw = result.get("intent")
    if intent_raw == "manual":
        steps.append("**Manual Retrieval** executed")
    elif intent_raw == "iot":
        steps.append("**IoT Retrieval** executed")
    elif intent_raw == "hybrid":
        steps.append("**Hybrid (Manual + IoT) Retrieval** executed")
        
    steps.append("**Gemini** synthesized final response")

    # Render as a sleek numbered list
    for i, step in enumerate(steps, 1):
        st.markdown(f"{i}. {step}")

def render_intent(result):
    st.subheader("Routing Metrics")
    intent_raw = result.get("intent", "unknown")
    
    sources = {
        "manual": "Manuals & Docs",
        "iot": "Live Sensors",
        "hybrid": "Docs + Sensors"
    }
    
    st.metric("Detected Intent", intent_raw.upper())
    st.metric("Knowledge Source", sources.get(intent_raw, "-"))
def render_chunks(result):
    chunks = result.get("retrieved_chunks", [])
    if not chunks:
        return

    st.subheader("Raw Text Chunks")
    for i, chunk in enumerate(chunks):
        with st.expander(f"📖 Context Chunk {i+1}"):
            st.write(chunk)

def render_prompt(result):
    prompt = result.get("prompt", "")
    if not prompt:
        st.write("No prompt available.")
        return

    st.subheader("Compiled LLM Prompt")
    st.code(prompt, language="markdown")