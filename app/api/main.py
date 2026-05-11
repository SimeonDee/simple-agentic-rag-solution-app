from fastapi import FastAPI, Query
from app.rag.engine import RAGEngine
from typing import List, Optional

app = FastAPI()

rag_engine = RAGEngine()


@app.get("/")
async def health_check():
    return {"health": "Okay"}


@app.get("/info")
async def get_info():
    return {"app": "RAG Engine API", "version": "1.0.0"}


@app.get("/query")
async def query(
    question: str = Query(..., description="The user question to ask the RAG engine.")
):
    try:
        answer = rag_engine.generate_answer(question)
        return {
            "question": question,
            "answer": answer,
        }
    except Exception as e:
        return {
            "question": question,
            "answer": None,
            "error": str(e),
        }
