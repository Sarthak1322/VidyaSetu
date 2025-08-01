# app.py

import streamlit as st
from VidyaSetu import VidyaSetuTutor  # Imports your main class

# --- Page Configuration ---
st.set_page_config(
    page_title="VidyaSetu AI Bridge of knowledge",
    page_icon="🤖"
)

# --- Page Title and Description ---
st.title("🤖 VidyaSetu AI Bridge of knowledge")
st.markdown("""
Welcome to your personal AI tutor! Ask questions about your documents and get answers directly from the source material.
""")

# --- Application Logic ---

# Initialize the tutor in Streamlit's session state.
# This crucial step ensures the VidyaSetuTutor object is created only once,
# preventing it from reloading every time you interact with the UI.
if 'tutor' not in st.session_state:
    with st.spinner("📚 Preparing the tutor... This may take a moment."):
        st.session_state.tutor = VidyaSetuTutor()

# Initialize the chat message history.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat messages from history.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input and Response Handling ---

# The st.chat_input function creates a text input field at the bottom of the page.
if prompt := st.chat_input("Ask a question about your documents..."):

    # Display the user's message in the chat.
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add the user's message to our history.
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get the AI's response using the .ask() method from your tutor.
    with st.spinner("🤖 Thinking..."):
        response = st.session_state.tutor.ask(prompt)

    # Display the AI's response.
    with st.chat_message("assistant"):
        st.markdown(response)

    # Add the AI's response to our history.
    st.session_state.messages.append(
        {"role": "assistant", "content": response})
