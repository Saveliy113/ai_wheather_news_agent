import streamlit as st
from agents.orchestrator import Orchestrator
# ENV-variables
from dotenv import load_dotenv
load_dotenv()


def _is_connection_error(error_msg: str) -> bool:
    """
    Check if an error message indicates a connection-related error.
    
    Args:
        error_msg: Error message to check
        
    Returns:
        True if it's a connection error, False otherwise
    """
    if not error_msg:
        return False
    
    error_lower = error_msg.lower()
    connection_keywords = [
        "connection",
        "timeout",
        "network",
        "unable to connect",
        "failed to connect",
        "connectionerror",
        "connection timed out",
        "no internet",
        "internet connection"
    ]
    return any(keyword in error_lower for keyword in connection_keywords)


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
        # Use OpenAI-generated response if available
        if result.get("response"):
            response = result["response"]
        else:
            # Fallback: show basic info if no response generated
            intent = result.get("intent", "unknown")
            location = result.get("location")
            topic = result.get("topic")
            time_ref = result.get("time")
            weather_data = result.get("weather_data")
            news_data = result.get("news_data")
            
            response_parts = []
            
            # Handle errors
            if weather_data and not weather_data.get("success"):
                error = weather_data.get("error", "Unknown error")
                error_type = weather_data.get("error_type", "")
                # Check if it's a connection-related error
                if error_type == "connection_error" or _is_connection_error(error):
                    response_parts.append("Sorry, but I am not able to provide you information due to connection problems. Please check your internet connection and try again.\n\n")
                else:
                    response_parts.append(f"❌ **Weather Error**: {error}\n\n")
            
            if news_data and not news_data.get("success"):
                error = news_data.get("error", "Unknown error")
                error_type = news_data.get("error_type", "")
                # Check if it's a connection-related error
                if error_type == "connection_error" or _is_connection_error(error):
                    response_parts.append("Sorry, but I am not able to provide you information due to connection problems. Please check your internet connection and try again.\n\n")
                else:
                    response_parts.append(f"❌ **News Error**: {error}\n\n")
            
            if not response_parts:
                response_parts.append(f"🤖 **Intent Detected**: {intent.upper()}\n\n")
                if location:
                    response_parts.append(f"📍 **Location**: {location}\n")
                if topic:
                    response_parts.append(f"📰 **Topic**: {topic}\n")
                if time_ref:
                    response_parts.append(f"⏰ **Time**: {time_ref}\n")
            
            response = "".join(response_parts) if response_parts else "Processing your request..."
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
