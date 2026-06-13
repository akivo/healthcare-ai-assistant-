"""
FastAPI application entry point.
Exposes endpoints for document ingestion, healthcare Q&A, and health checks.
"""

import logging
import sys
import os
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.ingest import ingest_documents, get_vector_store
from app.agent import process_question
from app.llm import check_ollama_health

settings = get_settings()

# Ensure required directories exist before logging setup
os.makedirs("logs", exist_ok=True)
os.makedirs("vector_store", exist_ok=True)

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/app.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle handler."""
    logger.info("Healthcare AI Assistant starting up...")

    # Pre-load embedding model to avoid cold start on first request
    try:
        from app.embeddings import get_embedding_model
        get_embedding_model()
        logger.info("Embedding model pre-loaded.")
    except Exception as e:
        logger.error(f"Failed to pre-load embedding model: {e}")

    # Check LLM connectivity
    ollama_status = check_ollama_health()
    if ollama_status["status"] == "connected":
        logger.info(f"Ollama connected. Model: {settings.llm_model}")
    else:
        logger.warning(f"Ollama not connected: {ollama_status.get('error')}")

    logger.info(f"API running at http://localhost:{settings.api_port}")
    logger.info(f"Swagger docs at http://localhost:{settings.api_port}/docs")

    yield

    logger.info("Healthcare AI Assistant shutting down.")


app = FastAPI(
    title="Healthcare AI Assistant",
    description="RAG-powered AI assistant for healthcare document Q&A. Built with LangChain, ChromaDB, Ollama, and FastAPI.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ──────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Healthcare question for the AI assistant",
        example="Can a patient request a medication refill through telehealth?",
    )


class SourceCitation(BaseModel):
    document: str
    chunk: str
    relevance_score: float = 0.0


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    confidence: str
    route: str


class IngestRequest(BaseModel):
    data_dir: str = Field(default="./data", description="Path to healthcare documents folder")


class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    chunks_created: int
    chunks_stored: int
    source_files: list[str]
    vector_store: str
    message: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health Check", tags=["System"])
async def health_check():
    """Returns the operational status of the API, LLM, and vector store."""
    logger.info("Health check requested.")

    ollama_status = check_ollama_health()

    try:
        vector_store = get_vector_store()
        existing = vector_store.get()
        doc_count = len(existing.get("ids", []))
        chroma_status = "initialized"
    except Exception as e:
        doc_count = 0
        chroma_status = f"error: {str(e)}"

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "api_version": "1.0.0",
        "llm": ollama_status,
        "vector_store": {
            "status": chroma_status,
            "chunks_stored": doc_count,
            "backend": "ChromaDB",
        },
        "config": {
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
        },
    }


@app.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest Documents",
    description="Load .txt documents from the data folder, generate embeddings, and store in ChromaDB.",
    tags=["Document Management"],
)
async def ingest_documents_endpoint(request: IngestRequest = None):
    """Triggers the document ingestion pipeline. Run this before using /ask."""
    data_dir = request.data_dir if request else "./data"
    logger.info(f"Ingestion requested: {data_dir}")

    try:
        result = ingest_documents(data_dir=data_dir)
        logger.info(f"Ingestion complete: {result['documents_processed']} docs, {result['chunks_created']} chunks")
        return result
    except FileNotFoundError as e:
        logger.error(f"Directory not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"No documents found: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion pipeline error: {str(e)}")


@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a Healthcare Question",
    description="Submit a question. Appointment queries are routed to the mock scheduler; all others use the RAG pipeline.",
    tags=["Q&A"],
)
async def ask_question(request: AskRequest):
    """Processes a healthcare question and returns a grounded answer with source citations."""
    logger.info(f"Question: {request.question[:80]}...")

    try:
        result = process_question(question=request.question)
        result.pop("tool_data", None)
        logger.info(f"Response | route={result.get('route')} | confidence={result.get('confidence')}")
        return result
    except Exception as e:
        logger.error(f"Question processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")


# ── Frontend ───────────────────────────────────────────────────────────────────

if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse("frontend/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
