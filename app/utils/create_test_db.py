import argparse
import os
import sys

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="Create a test ChromaDB collection")
    parser.add_argument(
        "--db_path", type=str, default="./chromadb", help="Path to ChromaDB directory"
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="documents",
        help="Name of the ChromaDB collection",
    )
    parser.add_argument(
        "--sample_docs",
        type=str,
        nargs="+",
        default=[
            "This is a sample document about artificial intelligence.",
            "ChromaDB is a vector database for storing and querying embeddings.",
            "Retrieval-Augmented Generation combines search and text generation.",
            "Streamlit is a framework for building data applications quickly.",
            "OpenAI provides APIs for embedding and text generation.",
        ],
        help="Sample documents to add to the collection",
    )
    return parser.parse_args()


def get_embeddings(api_key, texts, model="text-embedding-3-large"):
    """
    Get embeddings for the provided texts using OpenAI API.
    """
    client = OpenAI(api_key=api_key)

    response = client.embeddings.create(input=texts, model=model)

    # Extract the embeddings from the response
    embeddings = [item.embedding for item in response.data]

    return embeddings


def main():
    # Load environment variables
    load_dotenv()

    # Parse arguments
    args = parse_args()

    # Check if OpenAI API key is available
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it in the .env file or export it in your environment.")
        sys.exit(1)

    # Create database directory if it doesn't exist
    os.makedirs(args.db_path, exist_ok=True)

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=args.db_path)

    # Create or get collection
    try:
        collection = client.get_collection(name=args.collection_name)
        print(
            f"Collection '{args.collection_name}' already exists. Adding documents..."
        )
    except ValueError:
        collection = client.create_collection(name=args.collection_name)
        print(f"Created new collection '{args.collection_name}'.")

    # Get embeddings for the sample documents
    docs = args.sample_docs
    try:
        embeddings = get_embeddings(openai_api_key, docs)
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        sys.exit(1)

    # Add documents to the collection
    try:
        collection.add(
            documents=docs,
            embeddings=embeddings,
            ids=[f"doc_{i}" for i in range(len(docs))],
        )
        print(f"Added {len(docs)} documents to the collection.")
    except Exception as e:
        print(f"Error adding documents to collection: {e}")
        sys.exit(1)

    # Print collection information
    count = collection.count()
    print(f"Collection now has {count} documents.")

    print("\nTest database created successfully!")
    print(
        f"To use it with the RAG app, run: streamlit run app.py -- --db_path {args.db_path} --collection_name {args.collection_name}"
    )


if __name__ == "__main__":
    main()
