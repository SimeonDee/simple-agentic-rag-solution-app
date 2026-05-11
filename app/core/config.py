from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Config:
    PDF_DIR = BASE_DIR / "data"
    LLM_MODEL_NAME = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    VECTOR_STORE_PATH = BASE_DIR / "vector_store"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    TOP_K = 5
    TEMPERATURE = 0.3
    MAX_TOKENS = 2048
