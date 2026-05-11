"""This module defines the prompts used for the RAG system."""

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the provided context."
)

USER_PROMPT_TEMPLATE = """
Use only the information provided in the context below to answer the question.
If the answer is not in the context say "I don't know the answer".
Do not use any information that is not provided in the context.

Context: {context}

Question: {question}

Answer:
"""
