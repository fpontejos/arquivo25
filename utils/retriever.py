from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings
from utils.embeddings import OpenAIEmbedding


class ChromaDBRetriever:
    """
    Class for retrieving documents from ChromaDB.
    """

    def __init__(self, db_path: str, collection_name: str, embedding: OpenAIEmbedding):
        """
        Initialize the ChromaDB retriever.

        Args:
            db_path (str): Path to the ChromaDB directory
            collection_name (str): Name of the ChromaDB collection
            embedding (OpenAIEmbedding): OpenAI embedding client
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding = embedding

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=db_path, settings=Settings(anonymized_telemetry=False)
        )

        # Get existing collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except ValueError as e:
            raise ValueError(
                f"Collection '{collection_name}' not found. Make sure it exists: {str(e)}"
            )

        # stats = {"document_count": 0, "collection_name": "", "error": None}

        # try:
        #     client = self.client

        #     # Get all collections
        #     collections = client.list_collections()

        #     if not collections:
        #         stats["error"] = "No collections found in the database"

        #     # Use the first collection
        #     collection = client.get_collection(collections[0].name)
        #     stats["collection_name"] = collections[0].name

        #     # Get collection count
        #     collection_count = collection.count()
        #     stats["document_count"] = collection_count

        #     print(stats)

        # except Exception as e:
        #     raise ValueError(
        #         f"{db_path}\nCollection '{collection_name}' not found. Make sure it exists: {str(e)}\n{stats}"
        #     )

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve relevant documents based on the query.

        Args:
            query (str): Query to search for
            top_k (int, optional): Number of documents to retrieve. Defaults to 3.

        Returns:
            List[str]: List of retrieved documents
        """
        # Get query embedding
        query_embedding = self.embedding.get_embedding(query)

        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k
        )

        # Extract documents
        if results and "documents" in results and len(results["documents"]) > 0:
            documents = results["documents"][0]  # First query result
            return documents

        return []

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.

        Returns:
            Dict[str, Any]: Collection information
        """
        # Query with empty embedding to get collection info
        results = self.collection.peek(limit=5)

        info = {
            "count": self.collection.count(),
            "ids": results.get("ids", []) if results else [],
        }

        return info
