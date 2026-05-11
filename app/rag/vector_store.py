from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from uuid import uuid4
from typing import List

uuid = lambda: str(uuid4())


class VectorStore:
    def __init__(self, embeddings: HuggingFaceEmbeddings):
        """Initialize the VectorStore with the specified embedding model.

        Args:
            embeddings (HuggingFaceEmbeddings): An instance of the HuggingFaceEmbeddings class to use for generating embeddings.
        """
        self.embedding = embeddings
        self.store = None

    def create_vector_store(self, documents: List[Document]):
        """Create a FAISS vector store from the input documents.

        Args:
            documents (List[Document]): A list of documents to be added to the vector store.
        """
        texts = [doc.page_content for doc in documents]
        self.store: FAISS = FAISS.from_texts(
            texts=texts,
            metadatas=[doc.metadata for doc in documents],
            embedding=self.embedding,
            ids=[uuid() for _ in documents],  # generate unique IDs for each document
        )

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """Search the vector store for relevant documents based on the input query.

        Args:
            query (str): The input query string for which to search the vector store.
            top_k (int): The number of top relevant documents to return. Default is 5.
        Returns:
            List[Document]: A list of the top_k most relevant documents from the vector store.
        """
        if self.store is None:
            raise ValueError(
                "Vector store has not been created. Please create the vector store before searching."
            )
        return self.store.similarity_search(query, k=top_k, return_metadata=True)
