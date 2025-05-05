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

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents based on the query.

        Args:
            query (str): Query to search for
            top_k (int, optional): Number of documents to retrieve. Defaults to 3.

        Returns:
            List[Dict[str, Any]]: List of documents with their metadata and sources
        """
        # Get query embedding
        query_embedding = self.embedding.get_embedding(query)

        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Combine documents with their metadata
        documents_with_metadata = []

        if (
            results
            and "documents" in results
            and "metadatas" in results
            and "distances" in results
            and "ids" in results
            and len(results["documents"]) > 0
        ):

            documents = results["documents"][0]  # First query result
            metadatas = results["metadatas"][0]  # First query result
            distances = results["distances"][0]  # First query result
            ids = results["ids"][0]  # First query result

            for i, doc in enumerate(documents):
                document_info = {
                    "content": doc,
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "distance": distances[i] if i < len(distances) else None,
                    "id": ids[i] if i < len(ids) else None,
                }
                documents_with_metadata.append(document_info)

        return documents_with_metadata

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.

        Returns:
            Dict[str, Any]: Collection information
        """
        # Query with empty embedding to get collection info
        results = self.collection.peek(limit=5)

        # Get collection metadata
        collection_metadata = self.collection.metadata or {}

        info = {
            "count": self.collection.count(),
            "ids": results.get("ids", []) if results else [],
            "collection_name": self.collection_name,
            "metadata": collection_metadata,
            "sample_documents": results.get("documents", []) if results else [],
            "sample_metadatas": results.get("metadatas", []) if results else [],
        }

        return info

    def get_all_documents(self, limit: int = 100) -> Dict[str, Any]:
        """
        Get all documents from the collection.

        Args:
            limit (int, optional): Maximum number of documents to return. Defaults to 100.

        Returns:
            Dict[str, Any]: All documents with their metadata
        """
        # Get all documents
        results = self.collection.get(
            limit=limit, include=["documents", "metadatas", "embeddings"]
        )

        return results
