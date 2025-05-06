import argparse
import os
import sys

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
        "--db_path",
        type=str,
        default="./data/chroma_cravo",
        help="Path to ChromaDB directory",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="cravo",
        help="Name of the ChromaDB collection",
    )
    return parser.parse_args()


# Initialize session state for chat history
def init_session_state():
    args = st.session_state.args
    config = st.session_state.config

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "retriever" not in st.session_state:
        retriever, generator = get_retriever(args, config)

        st.session_state.retriever = retriever
        st.session_state.generator = generator

    if "df" not in st.session_state:
        try:
            client = retriever.client

            collection = client.get_collection(args.collection_name)

            # Retrieve all documents and their embeddings
            results = collection.get(include=["embeddings", "documents", "metadatas"])

            # Access all embeddings
            embeddings = results["embeddings"]

            # If you need to access the documents and metadata as well
            documents = results["documents"]
            metadatas = results["metadatas"]

            embeddings_path = "./data/embeddings/"

            embeddings, umap_df, projections = initialize_data(
                embeddings_path, collection
            )
            st.session_state.embeddings = embeddings
            st.session_state.df = umap_df
            st.session_state.umap_projection = projections

        except Exception as e:
            print(3)
            df = None
            st.session_state.df = None

        embeddings, umap_df, projections = initialize_data(embeddings_path, collection)
        st.session_state.embeddings = embeddings
        st.session_state.df = umap_df
        st.session_state.umap_projection = projections


def get_retriever(args, config):

    # Initialize components
    embedding = OpenAIEmbedding(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=config.get("embedding_model", os.getenv("DEFAULT_EMBEDDING_MODEL")),
    )

    retriever = ChromaDBRetriever(
        db_path=args.db_path,
        collection_name=args.collection_name,
        embedding=embedding,
    )

    generator = OpenAIGenerator(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=config.get("model", os.getenv("DEFAULT_COMPLETION_MODEL")),
        temperature=config.get("temperature", 0.7),
    )

    return retriever, generator


def initialize_data(embeddings_path, collection):
    """Initialize data if not already in session state."""
    if st.session_state.df is None:
        with st.spinner("Generating documents and embeddings..."):

            umap_path = os.path.join(embeddings_path, "umap_metadata.csv")
            umap_df = pd.read_csv(umap_path)

            projections = umap_df[["x", "y"]].values

            categs = umap_df["source_name"].unique().tolist() + ["Current Query"]
            umap_df["_highlighted"] = False

            print(umap_df.columns, categs)

            print("-------------------------------")

            # Retrieve all documents and their embeddings
            results = collection.get(include=["embeddings", "documents", "metadatas"])
            # Access all embeddings
            embeddings = results["embeddings"]

            # embeddings, df, projections = initialize_data()

            # Print information about the embeddings
            print(f"Retrieved {len(embeddings)} embeddings")
            print(f"Dimension of embeddings: {embeddings.shape}")
            # st.session_state.embeddings = embeddings

            return embeddings, umap_df, projections


def main():
    # Get the parent directory
    parent_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    # print(parent_directory)
    # Add the parent directory to the Python path if it is not already included
    if parent_directory not in sys.path:
        sys.path.append(parent_directory)

    st.set_page_config(
        page_title="RAG Chat Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="auto",
    )
    st.session_state.dark = True
    with open("./app/assets/style.css") as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    # print(12, os.getcwd())
    # Load configuration
    args = parse_args()
    config = load_config()
    st.session_state.config = config
    st.session_state.args = args

    # Initialize session state
    init_session_state()
    # print(11)

    col1, col2 = st.columns([1, 1])

    with st.container():
        with col1:
            render_chat_column()

        with col2:
            # Initialize session state variables if they don't exist
            if "highlighted_indices" not in st.session_state:
                st.session_state.highlighted_indices = []
            if "highlight_active" not in st.session_state:
                st.session_state.highlight_active = False

            color_palette = {
                "carnation_red": "#D72638",
                "deep_gold": "#F4A300",
                "Expresso": "#FF4C4C",
                "Web": "#E07B39",
                "Wikipedia PT": "#556B2F",
                "Publico": "#2c6b7e",
                "x": "#2E9CCA",
                "Current Query": "#6F1D1B",
            }
            st.session_state.color_palette = color_palette

            render_visualization_column()


if __name__ == "__main__":
    main()
