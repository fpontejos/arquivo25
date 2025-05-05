from typing import List

import openai
from tenacity import retry, stop_after_attempt, wait_exponential


class OpenAIEmbedding:
    """
    Class for creating embeddings using OpenAI API.
    """

    def __init__(self, api_key: str, model: str = "text-embedding-3-large"):
        """
        Initialize the OpenAI embedding client.

        Args:
            api_key (str): OpenAI API key
            model (str, optional): OpenAI embedding model name.
                Defaults to "text-embedding-3-large".
        """
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text.

        Args:
            text (str): Text to embed

        Returns:
            List[float]: Embedding vector
        """
        client = openai.OpenAI(api_key=self.api_key)

        response = client.embeddings.create(input=text, model=self.model)

        # Extract the embedding from the response
        embedding = response.data[0].embedding

        return embedding

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts.

        Args:
            texts (List[str]): List of texts to embed

        Returns:
            List[List[float]]: List of embedding vectors
        """
        client = openai.OpenAI(api_key=self.api_key)

        response = client.embeddings.create(input=texts, model=self.model)

        # Extract the embeddings from the response
        embeddings = [item.embedding for item in response.data]

        return embeddings
