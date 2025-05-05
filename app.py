import argparse
import os

import streamlit as st
from dotenv import load_dotenv
from utils.config import load_config, save_config
from utils.embeddings import OpenAIEmbedding
from utils.generator import OpenAIGenerator
from utils.retriever import ChromaDBRetriever

# Load environment variables
load_dotenv()


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="RAG Application with ChromaDB")
    parser.add_argument(
        "--db_path", type=str, default="./chromadb", help="Path to ChromaDB directory"
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="documents",
        help="Name of the ChromaDB collection",
    )
    return parser.parse_args()


# Initialize session state for chat history
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retriever" not in st.session_state:
        args = parse_args()

        # Load configuration
        config = load_config()

        # Initialize components
        embedding = OpenAIEmbedding(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=config.get("embedding_model", "text-embedding-3-large"),
        )

        retriever = ChromaDBRetriever(
            db_path=args.db_path,
            collection_name=args.collection_name,
            embedding=embedding,
        )

        generator = OpenAIGenerator(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=config.get("model", "gpt-4o"),
            temperature=config.get("temperature", 0.7),
        )

        st.session_state.retriever = retriever
        st.session_state.generator = generator
        st.session_state.config = config


def main():
    st.set_page_config(
        page_title="RAG Chat Assistant",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="auto",
    )

    # Initialize session state
    init_session_state()

    # App header
    st.title("RAG Chat Assistant")

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


if __name__ == "__main__":
    main()
