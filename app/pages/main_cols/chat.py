import time

import streamlit as st


def render_chat_column():
    """
    Renders the right column with embedding chat functionality.
    """
    st.title("Embedding Chat")

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
                # Retrieve relevant documents with metadata
                relevant_docs = st.session_state.retriever.retrieve(prompt, top_k=3)

                # Generate response with document sources and metadata
                response = st.session_state.generator.generate_response(
                    prompt, relevant_docs
                )

                # Display response
                st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # # Display chat messages
    # for message in st.session_state.messages:
    #     with st.chat_message(message["role"]):
    #         st.write(message["content"])

    # # Chat input
    # if prompt := st.chat_input("Type a message..."):
    #     # Add user message to chat history
    #     st.session_state.messages.append({"role": "user", "content": prompt})

    #     # Display user message
    #     with st.chat_message("user"):
    #         st.write(prompt)

    #     # Simulate document retrieval
    #     with st.spinner("Retrieving documents..."):
    #         retrieved_docs = simulate_retrieval(prompt)
    #         time.sleep(1)  # Simulate processing time

    #     # Simulate RAG response
    #     with st.spinner("Generating response..."):
    #         response = simulate_rag_response(prompt, retrieved_docs)
    #         time.sleep(1)  # Simulate processing time

    #     # Add assistant response to chat history
    #     st.session_state.messages.append({"role": "assistant", "content": response})

    #     # Display assistant response
    #     with st.chat_message("assistant"):
    #         st.write(response)

    #     # Force a rerun to update the visualization with the new query point
    #     st.rerun()
