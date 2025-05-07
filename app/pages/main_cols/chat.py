import time

import streamlit as st


def render_chat_column():
    """
    Renders the right column with embedding chat functionality.
    """
    st.title("Pergunte ao Passado")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input for new question

    if prompt := st.chat_input("Introduza a sua mensagem:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Retrieve relevant documents with metadata
                relevant_docs = st.session_state.retriever.retrieve(prompt, top_k=5)

                # Generate response with document sources and metadata
                response = st.session_state.generator.generate_response(
                    prompt, relevant_docs
                )

                df = st.session_state.df
                df["_highlighted"] = False
                highlight_ids = [int(doc["id"].split("_")[-1]) for doc in relevant_docs]
                df.loc[highlight_ids, "_highlighted"] = True
                st.session_state.df = df
                st.session_state.highlight_active = False
                st.session_state.highlighted_indices = []

                # print(highlight_ids)
                # print("-------------")
                # print(relevant_docs)

                if len(highlight_ids) > 0:
                    st.session_state.highlight_active = True
                    st.session_state.highlighted_indices = highlight_ids

                # Display response
                st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
