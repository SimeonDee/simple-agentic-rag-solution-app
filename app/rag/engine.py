import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from app.core.config import Config
from app.rag.loader import PDFLoader
from app.rag.chunker import TextChunker
from app.rag.vector_store import VectorStore
from app.rag.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


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
        self._process_documents()

    def _initialize(self):
        """Initialize the components of the RAG engine.

        This includes the loader, chunker, embedding model, and vector store.
        """
        load_dotenv()
        logger.info("Initializing RAG engine components")
        self.loader = PDFLoader(file_path=self.config.PDF_DIR)
        self.chunker = TextChunker(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
        )
        logger.info("Loading embedding model: %s", self.config.EMBEDDING_MODEL_NAME)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.config.EMBEDDING_MODEL_NAME
        )
        self.vector_store = VectorStore(embeddings=self.embedding_model)
        logger.info("Configuring LLM: %s (temperature=%.1f)", self.config.LLM_MODEL_NAME, self.config.LLM_TEMPERATURE)
        self.llm = ChatGroq(
            model=self.config.LLM_MODEL_NAME,
            temperature=self.config.LLM_TEMPERATURE,
            max_tokens=self.config.LLM_MAX_TOKENS,
        )

    def _process_documents(self):
        """Load, chunk, and create a vector store from the input documents."""
        logger.info("Loading documents from %s", self.config.PDF_DIR)
        documents = self.loader.load()
        logger.info("Loaded %d document(s)", len(documents))
        chunked_documents = self.chunker.chunk(documents)
        logger.info("Created %d chunk(s) from %d document(s)", len(chunked_documents), len(documents))
        self.vector_store.create_vector_store(chunked_documents)
        logger.info("Vector store built successfully")

    def generate_answer(self, question: str) -> str:
        """Generate a response to the input query based on the relevant documents from the vector store.

        Args:
            question (str): The input question string for which to generate a response.
        Returns:
            str: The generated response based on the input question and relevant documents.
        """
        logger.info("Generating answer for question: %.100s...", question)
        relevant_docs = self.vector_store.search(
            query=question, top_k=self.config.TOP_K
        )
        logger.info("Retrieved %d relevant document(s)", len(relevant_docs))
        context = "\n\n---\n\n".join(
            [f"{doc.page_content} \nMetadata: {doc.metadata}" for doc in relevant_docs]
        )

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=USER_PROMPT_TEMPLATE.strip(),
        )
        formatted_prompt = prompt.format(context=context, question=question)

        logger.info("Invoking LLM agent")
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
        logger.info("Answer generated successfully")
        return result["messages"][-1].content
