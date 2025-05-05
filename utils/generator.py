import json
from typing import Any, Dict, List

import openai
from tenacity import retry, stop_after_attempt, wait_exponential


class OpenAIGenerator:
    """
    Class for generating responses using OpenAI API.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o", temperature: float = 0.7):
        """
        Initialize the OpenAI generator.

        Args:
            api_key (str): OpenAI API key
            model (str, optional): OpenAI model name. Defaults to "gpt-4o".
            temperature (float, optional): Temperature for generation. Defaults to 0.7.
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        openai.api_key = api_key

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def generate_response(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """
        Generate a response based on the query and retrieved documents.

        Args:
            query (str): User query
            documents (List[Dict[str, Any]]): Retrieved documents with metadata

        Returns:
            str: Generated response
        """
        client = openai.OpenAI(api_key=self.api_key)

        # Create the context from documents with metadata
        context_items = []
        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            doc_id = doc.get("id", f"doc_{i}")
            similarity = 1.0 - (
                doc.get("distance", 0) or 0
            )  # Convert distance to similarity score

            # Format metadata as string
            metadata_str = ""
            if metadata:
                try:
                    metadata_str = "\nMetadata: " + json.dumps(metadata, indent=2)
                except:
                    metadata_str = "\nMetadata: " + str(metadata)

            # Format the document entry with its metadata and similarity score
            context_item = f"Document ID: {doc_id} (Relevance: {similarity:.2f})\nContent: {content}{metadata_str}\n"
            context_items.append(context_item)

        context = "\n\n".join(context_items)

        # Create the prompt
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided documents.
        
        CONTEXT:
        {context}
        
        USER QUESTION:
        {query}
        
        Please answer the question based only on the provided documents. Include sources in your response by referencing the Document IDs. The document sources and metadata should be included in your response.
        
        If the documents don't contain the information needed to answer the question, say so clearly.
        Do not make up information or use knowledge beyond what's provided in the documents.
        
        ANSWER:
        """

        # Generate response
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided context. Always cite your sources by referencing the Document IDs.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=1000,
        )

        return response.choices[0].message.content
