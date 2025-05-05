import os

import streamlit as st
from utils.config import load_config
from utils.generator import OpenAIGenerator
from utils.retriever import ChromaDBRetriever


def main():
    st.set_page_config(
        page_title="Chat",
        page_icon="💬",
        layout="centered",
    )

    st.title("Chat with Your Documents")

    # Check if retriever and generator are initialized
    if "retriever" not in st.session_state or "generator" not in st.session_state:
        st.error("Please start the application from the main page.")
        return

    # Initialize chat history if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load configuration
    config = load_config()

    # Display chat interface
    with st.sidebar:
        st.subheader("Chat Settings")

        # Allow user to adjust top_k
        top_k = st.slider(
            "Number of documents to retrieve",
            min_value=1,
            max_value=10,
            value=config.get("top_k", 3),
        )

        # Allow user to adjust temperature
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=config.get("temperature", 0.7),
            step=0.1,
        )

        # Apply settings button
        if st.button("Apply Settings"):
            # Update session state
            st.session_state.generator.temperature = temperature

            # Update config
            config["top_k"] = top_k
            config["temperature"] = temperature

            st.session_state.config = config
            st.success("Settings applied!")

        # Clear chat history
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input for new question
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Retrieve relevant documents
                relevant_docs = st.session_state.retriever.retrieve(prompt, top_k=top_k)

                # Show retrieved documents in expander
                with st.expander("Retrieved Documents"):
                    for i, doc in enumerate(relevant_docs):
                        st.markdown(f"**Document {i+1}:**")
                        st.text(doc)
                        st.divider()

                # Generate response
                response = st.session_state.generator.generate_response(
                    prompt, relevant_docs
                )

                # Display response
                st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
