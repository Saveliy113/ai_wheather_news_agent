import streamlit as st
from agents.orchestrator import Orchestrator
# ENV-variables
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="Weather & News AI",
    page_icon="🌦️",
    layout="centered"
)

st.title("🌦️ Weather & News Assistant")

# ---------------------------
# Init session state
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize orchestrator (store in session state to avoid re-initialization)
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = Orchestrator()

# ---------------------------
# User input (PROCESS FIRST)
# ---------------------------
user_input = st.chat_input("Ask about weather or news...")

if user_input:
    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Process query through orchestrator
    with st.spinner("Analyzing your query..."):
        result = st.session_state.orchestrator.run(user_input)
    
    # Format response based on orchestrator result
    if result.get("success"):
        intent = result.get("intent", "unknown")
        location = result.get("location")
        topic = result.get("topic")
        time_ref = result.get("time")
        
        # Build response message
        response_parts = [
            f"🤖 **Intent Detected**: {intent.upper()}\n\n"
        ]
        
        if location:
            response_parts.append(f"📍 **Location**: {location}\n")
        if topic:
            response_parts.append(f"📰 **Topic**: {topic}\n")
        if time_ref:
            response_parts.append(f"⏰ **Time**: {time_ref}\n")
        
        response_parts.append("\n---\n")
        response_parts.append("_MCP servers will be connected next to fetch actual data._")
        
        response = "".join(response_parts)
    else:
        # Error case
        error_msg = result.get("error", "Unknown error occurred")
        response = f"❌ **Error**: {error_msg}\n\n"
        if result.get("raw_response"):
            response += f"_Raw response: {result['raw_response']}_"

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

# ---------------------------
# Render chat history (AFTER)
# ---------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
