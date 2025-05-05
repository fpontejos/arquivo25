import os

import streamlit as st
from utils.config import load_config
from utils.retriever import ChromaDBRetriever


def main():
    st.set_page_config(
        page_title="About RAG Chat",
        page_icon="🤖",
        layout="centered",
    )

    st.title("About RAG Chat Assistant")

    # Load configuration
    config = load_config()

    # Get database info if retriever is initialized
    if "retriever" in st.session_state:
        retriever = st.session_state.retriever
        collection_info = retriever.get_collection_info()

        st.subheader("Database Information")

        # Display collection information
        st.markdown(f"**Collection Name:** {retriever.collection_name}")
        st.markdown(f"**Database Path:** {retriever.db_path}")
        st.markdown(f"**Document Count:** {collection_info.get('count', 'Unknown')}")

        # Display collection metadata if available
        if collection_info.get("metadata"):
            with st.expander("Collection Metadata"):
                st.json(collection_info.get("metadata", {}))

        # Display a sample of document IDs if available
        if collection_info.get("ids") and len(collection_info.get("ids", [])) > 0:
            with st.expander("Sample Document IDs"):
                for doc_id in collection_info.get("ids", [])[:5]:  # Show only first 5
                    st.text(doc_id)

        # Display sample documents with metadata
        if (
            collection_info.get("sample_documents")
            and len(collection_info.get("sample_documents", [])) > 0
        ):
            with st.expander("Sample Documents"):
                for i, doc in enumerate(
                    collection_info.get("sample_documents", [])[:3]
                ):
                    metadata = {}
                    if collection_info.get("sample_metadatas") and i < len(
                        collection_info.get("sample_metadatas", [])
                    ):
                        metadata = collection_info.get("sample_metadatas", [])[i]

                    st.markdown(f"**Document {i+1}:**")
                    st.text(doc)

                    if metadata:
                        st.markdown("**Metadata:**")
                        st.json(metadata)

                    st.divider()

    # Configuration information
    st.subheader("Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**OpenAI Settings**")
        st.markdown(
            f"**Embedding Model:** {config.get('embedding_model', 'text-embedding-3-large')}"
        )
        st.markdown(f"**Completion Model:** {config.get('model', 'gpt-4o')}")
        st.markdown(f"**Temperature:** {config.get('temperature', 0.7)}")

    with col2:
        st.markdown("**Retrieval Settings**")
        st.markdown(f"**Top K Documents:** {config.get('top_k', 3)}")

    # About the application
    st.subheader("About This Application")
    st.markdown(
        """
    This is a simple RAG (Retrieval-Augmented Generation) application that connects to an existing ChromaDB vector database.
    
    It allows you to ask questions about your documents and get responses based on the content of those documents.
    
    The application uses:
    - ChromaDB for vector storage and retrieval
    - OpenAI for embeddings and text generation
    - Streamlit for the user interface
    """
    )


if __name__ == "__main__":
    main()
