import argparse
import os

import numpy as np
import pandas as pd
import streamlit as st
import umap
from dotenv import load_dotenv
from pages.main_cols.chat import render_chat_column
from pages.main_cols.scatter import render_visualization_column
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
        st.session_state.df = None

        initialize_data()


def main():
    st.set_page_config(
        page_title="RAG Chat Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="auto",
    )

    # Initialize session state
    init_session_state()

    col1, col2 = st.columns([1, 1])

    with st.container():
        with col1:
            render_chat_column()

        with col2:
            st.title("Scatter Col")
            render_visualization_column()


def initialize_data():
    """Initialize data if not already in session state."""
    if st.session_state.df is None:
        with st.spinner("Generating documents and embeddings..."):

            retriever = st.session_state.retriever
            client = retriever.client
            # db_path = "./../data/chroma_db"
            # client = chromadb.PersistentClient(path=db_path)

            # Get your collection
            collection = retriever.collection

            # Retrieve all documents and their embeddings
            results = collection.get(include=["embeddings", "documents", "metadatas"])

            # Access all embeddings
            embeddings = results["embeddings"]

            # Print information about the embeddings
            print(f"Retrieved {len(embeddings)} embeddings")
            print(f"Dimension of embeddings: {embeddings.shape}")

            # If you need to access the documents and metadata as well
            documents = results["documents"]
            metadatas = results["metadatas"]

            # documents = generate_sample_documents(100)
            # embeddings = create_embeddings(documents)
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                metric="cosine",
                random_state=42,
            )
            projections = reducer.fit_transform(embeddings)
            df = pd.DataFrame(documents, columns=["document"])
            df = pd.concat([df, pd.DataFrame(metadatas)], axis=1)
            df["x"] = projections[:, 0]
            df["y"] = projections[:, 1]

            categs = [
                "Technology",
                "Science",
                "Arts",
                "History",
                "Current Query",
            ]
            rng = np.random.default_rng(seed=42)  # Set random seed for reproducibility
            df["category"] = df["x"].apply(lambda x: categs[rng.integers(0, 5)])

            st.session_state.embeddings = embeddings
            st.session_state.df = df
            st.session_state.umap_projection = projections


if __name__ == "__main__":
    main()
