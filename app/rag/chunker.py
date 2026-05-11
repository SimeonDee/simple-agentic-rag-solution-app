from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List


class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the TextChunker with the specified chunk size and overlap.

        Args:
            chunk_size (int): The maximum size of each text chunk. Default is 1000 characters.
            chunk_overlap (int): The number of overlapping characters between chunks. Default is 200 characters.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        """Chunk the input documents into smaller pieces based on the specified chunk size and overlap.

        Args:
            documents (List[Document]): A list of documents to be chunked.

        Returns:
            List[Document]: A list of chunked documents.
        """
        chunked_documents = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                chunked_doc = Document(
                    page_content=chunk, metadata={**doc.metadata, "chunk_index": i}
                )
                chunked_documents.append(chunked_doc)
        return chunked_documents
