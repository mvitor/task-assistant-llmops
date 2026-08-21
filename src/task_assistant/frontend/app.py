# src/task_assistant/frontend/app.py
import os
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

st.set_page_config(
    page_title="Task Assistant",
    page_icon="✅",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_id" not in st.session_state:
    st.session_state.user_id = "default-user"

if "session_id" not in st.session_state:
    st.session_state.session_id = "default-session"

if "pending_message" not in st.session_state:
    st.session_state.pending_message = None


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

with st.sidebar:
    st.title("Task Assistant")
    st.markdown("MLOps/LLMOps work task assistant.")

    st.session_state.user_id = st.text_input(
        "User ID",
        value=st.session_state.user_id,
    )
    st.session_state.session_id = st.text_input(
        "Session ID",
        value=st.session_state.session_id,
    )

    st.markdown("---")
    st.markdown("**Quick actions**")
    if st.button("List my tasks"):
        st.session_state.pending_message = "List my current tasks."
        st.rerun()

    if st.button("Summarize my day"):
        st.session_state.pending_message = "Summarize my day."
        st.rerun()

    st.markdown("---")
    if st.button("🗑️ New conversation"):
        session_id = st.session_state.session_id
        try:
            requests.delete(f"{API_BASE_URL}/session/{session_id}", timeout=2)
        except Exception:
            pass
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    # Health check
    try:
        resp = requests.get(HEALTH_ENDPOINT, timeout=2)
        health_status = resp.json().get("status", "unknown")
    except Exception:
        health_status = "unreachable"

    st.caption(f"API health: {health_status}")


# -----------------------------------------------------------------------------
# Main chat UI
# -----------------------------------------------------------------------------

st.title("Task Assistant")

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# Chat input — also picks up quick-action button presses via pending_message
prompt = st.chat_input("Ask me to create, list, or update tasks...")
if not prompt and st.session_state.pending_message:
    prompt = st.session_state.pending_message
    st.session_state.pending_message = None

if prompt:
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {
                    "message": prompt,
                    "user_id": st.session_state.user_id,
                    "session_id": st.session_state.session_id,
                }
                resp = requests.post(CHAT_ENDPOINT, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                assistant_message = data.get("response", "No response from API.")
            except Exception as e:
                assistant_message = f"Error calling API: {e}"

        st.write(assistant_message)

    # Add assistant message to state
    st.session_state.messages.append({"role": "assistant", "content": assistant_message})
