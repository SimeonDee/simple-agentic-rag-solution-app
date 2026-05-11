"""This implements the EmbeddingModel class.

This is responsible for generating embeddings for documents and queries
using a specified Hugging Face model.

NOTE: This class is designed to be used in conjunction with the VectorStore class.
Not used in current implementation, just provided in-case of future use.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from typing import List


class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the EmbeddingModel with the specified Hugging Face model.

        Args:
            model_name (str): The name of the Hugging Face model to use for generating embeddings. Default is "sentence-transformers/all-MiniLM-L6-v2".
        """
        self.model_name = model_name
        self.model = HuggingFaceEmbeddings(model_name=self.model_name)

    def generate_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of input texts.

        Args:
            texts (List[str]): A list of input texts for which to generate embeddings.

        Returns:
            List[List[float]]: A list of embeddings corresponding to the input texts.
        """
        return self.model.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        """Generate an embedding for a single query string.

        Args:
            query (str): The input query string for which to generate an embedding.

        Returns:
            List[float]: The embedding corresponding to the input query.
        """
        return self.model.embed_query(query)
