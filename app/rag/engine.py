from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from app.core.config import Config
from app.rag.loader import PDFLoader
from app.rag.chunker import TextChunker
from app.rag.vector_store import VectorStore
from app.rag.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


class RAGEngine:
    def __init__(self, config: Config | None = None):
        """Initialize the RAGEngine with the specified configuration.

        Args:
            config (Config): An instance of the Config class containing
                configuration settings for the RAG engine.
        """
        self.config = config or Config()
        self.store = None
        self._initialize()

    def _initialize(self):
        """Initialize the components of the RAG engine.

        This includes the loader, chunker, embedding model, and vector store.
        """
        load_dotenv()
        self.loader = PDFLoader(file_path=self.config.PDF_DIR)
        self.chunker = TextChunker(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
        )
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.config.EMBEDDING_MODEL_NAME
        )
        self.vector_store = VectorStore(embeddings=self.embedding_model)
        self.llm = ChatGroq(
            model=self.config.LLM_MODEL_NAME,
            temperature=self.config.LLM_TEMPERATURE,
        )

    def process_documents(self):
        """Load, chunk, and create a vector store from the input documents."""
        documents = self.loader.load()
        chunked_documents = self.chunker.chunk(documents)
        self.vector_store.create_vector_store(chunked_documents)

    def generate_answer(self, question: str) -> str:
        """Generate a response to the input query based on the relevant documents from the vector store.

        Args:
            question (str): The input question string for which to generate a response.
        Returns:
            str: The generated response based on the input question and relevant documents.
        """
        # Search the vector store for relevant documents based on the input question
        relevant_docs = self.vector_store.search(
            query=question, top_k=self.config.TOP_K
        )
        context = "\n\n---\n\n".join(
            [f"{doc.page_content} \nMetadata: {doc.metadata}" for doc in relevant_docs]
        )

        # Format the prompt with the retrieved context and the input question
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=USER_PROMPT_TEMPLATE.strip(),
        )
        formatted_prompt = prompt.format(context=context, question=question)

        # Create an agent with the specified system prompt and invoke it with the formatted prompt
        agent = create_agent(
            model=self.llm,
            system_prompt=SYSTEM_PROMPT.strip(),
        )
        result = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": formatted_prompt},
                ]
            }
        )
        return result["messages"][-1].content
