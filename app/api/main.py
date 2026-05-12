import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag.engine import RAGEngine

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

app = FastAPI()

logger.info("Starting RAG Engine API")
rag_engine = RAGEngine()
logger.info("RAG Engine initialized and ready")


class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, description="The user question to ask the RAG engine."
    )


@app.get("/")
async def health_check():
    return {"health": "Okay"}


@app.get("/info")
async def get_info():
    return {"app": "RAG Engine API", "version": "1.0.0"}


@app.post("/query")
async def query(request: QueryRequest):
    logger.info("Received query: %.100s...", request.question)
    try:
        answer = rag_engine.generate_answer(request.question)
        logger.info("Query answered successfully")
        return {
            "question": request.question,
            "answer": answer,
        }
    except ValueError as e:
        logger.error(f"Validation error processing query: {e}")
        raise HTTPException(
            status_code=400, detail="Invalid query or system not ready."
        )
    except Exception as e:
        logger.exception(f"Unexpected error generating answer: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")
