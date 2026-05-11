from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document
from typing import List
from pathlib import Path


class PDFLoader:
    def __init__(self, file_path: Path):
        """Initialize the PDFLoader with the path to the PDF file or directory.

        Args:
            file_path (Path): The path to the PDF file or directory containing PDF files.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File or directory {file_path} does not exist.")
        self.file_path = file_path

    def load(self) -> List[Document]:
        """Load the PDF file(s) and return a list of documents.

        Returns:
            List[Document]: A list of documents loaded from the PDF file(s).
        """
        if self.file_path.is_dir():
            loader = DirectoryLoader(
                self.file_path, glob="*.pdf", loader_cls=PyPDFLoader
            )
        else:
            loader = PyPDFLoader(self.file_path)
        return loader.load()
