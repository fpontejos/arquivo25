import time

import streamlit as st

# import streamlit as st


# def render_chat_column():
#     """
#     Renders the right column with embedding chat functionality.
#     """
#     st.title("Pergunte ao Passado")

#     # Display chat messages
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # Input for new question

#     if prompt := st.chat_input("Introduza a sua mensagem:"):
#         # Add user message to chat history
#         st.session_state.messages.append({"role": "user", "content": prompt})

#         # Display user message
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         # Get response
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 # Retrieve relevant documents with metadata
#                 relevant_docs = st.session_state.retriever.retrieve(prompt, top_k=5)

#                 # Generate response with document sources and metadata
#                 response, relevant = st.session_state.generator.generate_response(
#                     prompt, relevant_docs
#                 )

#                 st.markdown(response)
#                 df = st.session_state.df
#                 df["_highlighted"] = False

#                 if relevant:

#                     highlight_ids = [
#                         int(doc["id"].split("_")[-1]) for doc in relevant_docs
#                     ]
#                     df.loc[highlight_ids, "_highlighted"] = True
#                     st.session_state.df = df
#                     st.session_state.highlight_active = False  #
#                     st.session_state.highlighted_indices = []  #

#                     relevant_text = (
#                         "<div style='border: 1px solid #5f5f5f; padding:1em;'><p>"
#                     )
#                     relevant_text += "<strong>Fontes:</strong>"

#                     doc_num = 1

#                     for reli in relevant_docs:
#                         doc_link = reli["metadata"]["link"]

#                         relevant_text += f""" <a href="{doc_link}">[{doc_num}] </a>"""
#                         doc_num += 1

#                     relevant_text += "</p></div>"

#                     if len(highlight_ids) > 0:
#                         st.session_state.highlight_active = True
#                         st.session_state.highlighted_indices = highlight_ids

#                     # Display response
#                     st.markdown(
#                         relevant_text,
#                         unsafe_allow_html=True,
#                     )

#         # Add assistant response to chat history
#         st.session_state.messages.append({"role": "assistant", "content": response})


def render_chat_column():
    """
    Renders the right column with embedding chat functionality.
    Chat input at top, conversation history below.
    """
    st.title("Pergunte ao Passado")

    # Chat input at the top
    prompt = st.chat_input("Introduza a sua mensagem:")

    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get response
        with st.spinner("Thinking..."):
            # Retrieve relevant documents with metadata
            relevant_docs = st.session_state.retriever.retrieve(prompt, top_k=5)

            # Generate response with document sources and metadata
            response, relevant = st.session_state.generator.generate_response(
                prompt, relevant_docs
            )

            # Update dataframe highlighting
            df = st.session_state.df
            df["_highlighted"] = False

            if relevant:
                highlight_ids = [int(doc["id"].split("_")[-1]) for doc in relevant_docs]
                df.loc[highlight_ids, "_highlighted"] = True
                st.session_state.df = df
                st.session_state.highlight_active = True
                st.session_state.highlighted_indices = highlight_ids

                # Prepare sources text
                relevant_text = (
                    "<div style='border: 1px solid #5f5f5f; padding:1em;'><p>"
                )
                relevant_text += "<strong>Fontes:</strong>"

                for i, doc in enumerate(relevant_docs, 1):
                    doc_link = doc["metadata"]["link"]
                    relevant_text += f""" <a href="{doc_link}">[{i}] </a>"""

                relevant_text += "</p></div>"

                # Combine response with sources
                full_response = response + "\n\n" + relevant_text
            else:
                full_response = response
                st.session_state.highlight_active = False
                st.session_state.highlighted_indices = []

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

    # Display conversation history below input
    st.markdown("---")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "<div style=" in message["content"]:
                # Split response and sources for proper rendering
                parts = message["content"].split("<div style=")
                st.markdown(parts[0])  # Main response
                if len(parts) > 1:
                    st.markdown(
                        "<div style=" + parts[1], unsafe_allow_html=True
                    )  # Sources
            else:
                st.markdown(message["content"])


# def render_chat_column():
#     """
#     Renders the right column with embedding chat functionality.
#     Chat input at top, conversation history below.
#     """
#     st.title("Pergunte ao Passado")

#     # Chat input at the top
#     prompt = st.chat_input("Introduza a sua mensagem:")

#     if prompt:
#         # Add user message to chat history
#         st.session_state.messages.append({"role": "user", "content": prompt})

#         # Add user message to chat history
#         #         st.session_state.messages.append({"role": "user", "content": prompt})

#         # Display user message
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         # Get response
#         with st.spinner("Thinking..."):
#             # Retrieve relevant documents with metadata
#             relevant_docs = st.session_state.retriever.retrieve(prompt, top_k=5)

#             # Generate response with document sources and metadata
#             response, relevant = st.session_state.generator.generate_response(
#                 prompt, relevant_docs
#             )

#             # Update dataframe highlighting
#             df = st.session_state.df
#             df["_highlighted"] = False

#             if relevant:

#                 highlight_ids = [int(doc["id"].split("_")[-1]) for doc in relevant_docs]
#                 df.loc[highlight_ids, "_highlighted"] = True
#                 st.session_state.df = df
#                 st.session_state.highlight_active = False  #
#                 st.session_state.highlighted_indices = []  #

#                 relevant_text = (
#                     "<div style='border: 1px solid #5f5f5f; padding:1em;'><p>"
#                 )
#                 relevant_text += "<strong>Fontes:</strong>"

#                 doc_num = 1

#                 for reli in relevant_docs:
#                     doc_link = reli["metadata"]["link"]

#                     relevant_text += f""" <a href="{doc_link}">[{doc_num}] </a>"""
#                     doc_num += 1

#                 relevant_text += "</p></div>"

#                 if len(highlight_ids) > 0:
#                     st.session_state.highlight_active = True
#                     st.session_state.highlighted_indices = highlight_ids

#                 # Display response
#                 st.markdown(
#                     relevant_text,
#                     unsafe_allow_html=True,
#                 )

#             # if relevant:
#             #     highlight_ids = [int(doc["id"].split("_")[-1]) for doc in relevant_docs]
#             #     df.loc[highlight_ids, "_highlighted"] = True
#             #     st.session_state.df = df
#             #     st.session_state.highlight_active = True
#             #     st.session_state.highlighted_indices = highlight_ids

#             #     # Prepare sources text
#             #     relevant_text = (
#             #         "<div style='border: 1px solid #5f5f5f; padding:1em;'><p>"
#             #     )
#             #     relevant_text += "<strong>Fontes:</strong>"

#             #     for i, doc in enumerate(relevant_docs, 1):
#             #         doc_link = doc["metadata"]["link"]
#             #         relevant_text += f""" <a href="{doc_link}">[{i}] </a>"""

#             #     relevant_text += "</p></div>"

#             #     # Combine response with sources
#             #     full_response = response + "\n\n" + relevant_text
#             # else:
#             #     full_response = response
#             #     st.session_state.highlight_active = False
#             #     st.session_state.highlighted_indices = []

#         # Add assistant response to chat history
#         st.session_state.messages.append({"role": "assistant", "content": response})

#     # Display conversation history below input
#     st.markdown("---")

#     # Chat history container with scrolling
#     chat_container = st.container()

#     with chat_container:
#         # Create scrollable container for messages
#         st.markdown(
#             """
#         <style>
#         .chat-container {
#             max-height: 60vh;
#             overflow-y: auto;
#             padding: 1rem;
#             border: 1px solid #e0e0e0;
#             border-radius: 8px;
#         }
#         </style>
#         """,
#             unsafe_allow_html=True,
#         )

#         messages_html = '<div class="chat-container" id="chat-messages">'

#         for message in st.session_state.messages:
#             with st.chat_message(message["role"]):
#                 if (
#                     message["role"] == "assistant"
#                     and "<div style=" in message["content"]
#                 ):
#                     # Split response and sources for proper rendering
#                     parts = message["content"].split("<div style=")
#                     messages_html += parts[0]  # Main response
#                     if len(parts) > 1:
#                         messages_html += "<div style=" + parts[1]  # Sources
#                 else:
#                     messages_html += message["content"]

#         messages_html += "</div>"

#         # Auto-scroll JavaScript
#         messages_html += """
#         <script>
#         setTimeout(function() {
#             var chatContainer = document.getElementById('chat-messages');
#             if (chatContainer) {
#                 chatContainer.scrollTop = chatContainer.scrollHeight;
#             }
#         }, 100);
#         </script>
#         """

#         st.markdown(messages_html, unsafe_allow_html=True)
