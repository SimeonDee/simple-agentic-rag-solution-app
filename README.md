# Simple Agentic RAG Solution

A simple agentic Retrieval-Augmented Generation (RAG) application that answers questions from PDF documents using a LangChain agent, Groq-hosted Llama 3.3 70B model, FAISS vector store, and HuggingFace embeddings — served via a FastAPI backend.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [License](#license)

## Features

- **PDF Ingestion** — Load single PDFs or entire directories of PDFs
- **Recursive Text Chunking** — Split documents into overlapping chunks for better retrieval
- **Semantic Embeddings** — Generate embeddings using the `all-MiniLM-L6-v2` sentence-transformer model
- **FAISS Vector Store** — Fast similarity search over document embeddings
- **Agentic RAG** — LangChain agent powered by Groq Llama 3.3 70B for context-aware, grounded answers
- **FastAPI Backend** — RESTful API for querying the RAG pipeline

## Architecture

```
PDF Documents
     │
     ▼
┌──────────┐    ┌──────────────┐    ┌─────────────────┐
│ PDFLoader │───▶│ TextChunker  │───▶│  FAISS Vector   │
│           │    │ (Recursive)  │    │     Store        │
└──────────┘    └──────────────┘    └────────┬────────┘
                                             │
                                      similarity search
                                             │
                                             ▼
                                    ┌────────────────┐
                  User Query ──────▶│  LangChain     │──────▶ Answer
                                    │  Agent (Groq)  │
                                    └────────────────┘
```

## Tech Stack

| Component        | Technology                              |
| ---------------- | --------------------------------------- |
| LLM              | Llama 3.3 70B via Groq                  |
| Agent Framework  | LangChain                               |
| Embeddings       | HuggingFace `all-MiniLM-L6-v2`          |
| Vector Store     | FAISS (faiss-cpu)                        |
| Document Loader  | PyPDF via LangChain Community            |
| Text Splitting   | RecursiveCharacterTextSplitter           |
| API Framework    | FastAPI + Uvicorn                        |
| Package Manager  | uv                                       |
| Language         | Python 3.12+                             |

## Project Structure

```
simple-agentic-rag-solution-app/
├── main.py                  # Application entry point
├── requirements.txt         # pip-compatible dependency list
├── pyproject.toml           # Project metadata and uv dependencies
├── data/                    # PDF documents for ingestion
├── vector_store/            # Persisted FAISS index (generated)
└── app/
    ├── api/                 # API route handlers
    ├── core/
    │   └── config.py        # Centralised configuration
    ├── models/              # Pydantic request/response models
    └── rag/
        ├── loader.py        # PDF document loading
        ├── chunker.py       # Text chunking logic
        ├── embeddings.py    # Embedding model wrapper
        ├── vector_store.py  # FAISS vector store operations
        ├── prompts.py       # System and user prompt templates
        └── engine.py        # RAG engine orchestrating the pipeline
```

## Prerequisites

- **Python 3.12** (macOS x86_64 users: 3.13+ is not supported due to PyTorch wheel availability)
- **[uv](https://docs.astral.sh/uv/)** package manager
- **[Groq API key](https://console.groq.com/)** for LLM inference
- **[HuggingFace access token](https://huggingface.co/settings/tokens)** for downloading embedding models

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/simple-agentic-rag-solution-app.git
   cd simple-agentic-rag-solution-app
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   uv venv --python 3.12
   source .venv/bin/activate
   uv add -r requirements.txt
   ```

## Configuration

Create a `.env` file in the project root with your API keys:

```env
GROQ_API_KEY="your-groq-api-key"
HUGGINGFACEHUB_ACCESS_TOKEN="your-huggingface-token"
```

Default model and chunking settings can be adjusted in [app/core/config.py](app/core/config.py):

| Parameter              | Default                | Description                          |
| ---------------------- | ---------------------- | ------------------------------------ |
| `LLM_MODEL_NAME`       | `llama-3.3-70b-versatile` | Groq-hosted LLM model            |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2`    | HuggingFace sentence-transformer    |
| `CHUNK_SIZE`           | `500`                  | Max characters per chunk             |
| `CHUNK_OVERLAP`        | `50`                   | Overlapping characters between chunks|
| `TOP_K`                | `5`                    | Number of documents retrieved        |
| `TEMPERATURE`          | `0.3`                  | LLM sampling temperature             |
| `MAX_TOKENS`           | `2048`                 | Max tokens in LLM response           |

## Usage

1. **Add PDF documents** to the `data/` directory.

2. **Run the application**

   ```bash
   uvicorn main:app --reload
   ```

   The API will be available at `http://localhost:8000`.

3. **Interactive docs** — Visit `http://localhost:8000/docs` for the Swagger UI.

## License

This project is licensed under the [MIT License](LICENSE).
