from typing import List

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
    def generate_response(self, query: str, documents: List[str]) -> str:
        """
        Generate a response based on the query and retrieved documents.

        Args:
            query (str): User query
            documents (List[str]): Retrieved documents

        Returns:
            str: Generated response
        """
        client = openai.OpenAI(api_key=self.api_key)

        # Create the context from documents
        context = "\n\n".join([f"Document: {doc}" for doc in documents])

        # Create the prompt
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided documents.
        
        CONTEXT:
        {context}
        
        USER QUESTION:
        {query}
        
        Please answer the question based only on the provided documents. If the documents don't contain the information needed to answer the question, say so clearly. Do not make up information or use knowledge beyond what's provided in the documents.
        
        ANSWER:
        """

        # Generate response
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided context.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=1000,
        )

        return response.choices[0].message.content
