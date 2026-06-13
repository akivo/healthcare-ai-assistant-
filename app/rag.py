"""
RAG (Retrieval-Augmented Generation) pipeline.
Handles semantic retrieval from ChromaDB and answer generation via Ollama.
"""

import logging
import os
from typing import List, Dict, Any

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser

from app.config import get_settings
from app.embeddings import get_embedding_model
from app.llm import get_llm
from app.prompts import get_rag_prompt

logger = logging.getLogger(__name__)

NO_ANSWER_MESSAGE = (
    "I could not find this information in the provided documents. "
    "Please contact our care team at (800) 555-CARE for assistance."
)


def get_retriever():
    """Create and return a ChromaDB similarity search retriever."""
    settings = get_settings()

    vector_store = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embedding_model(),
        persist_directory=settings.chroma_persist_dir,
    )

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.top_k_results},
    )


def calculate_confidence(docs_with_scores: List[tuple]) -> str:
    """
    Determine confidence level based on retrieval similarity scores.

    Returns:
        "high", "medium", or "low"
    """
    settings = get_settings()

    if not docs_with_scores:
        return "low"

    best_score = max(score for _, score in docs_with_scores)

    if best_score >= settings.confidence_high_threshold:
        return "high"
    elif best_score >= settings.confidence_low_threshold:
        return "medium"
    return "low"


def format_docs(docs) -> str:
    """Format retrieved document chunks into a single context string."""
    parts = []
    for i, doc in enumerate(docs, 1):
        filename = doc.metadata.get("filename", "unknown document")
        parts.append(f"[Source {i}: {filename}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def run_rag_query(question: str) -> Dict[str, Any]:
    """
    Execute the full RAG pipeline for a given question.

    Steps:
        1. Search ChromaDB for semantically similar chunks
        2. Format retrieved chunks as context
        3. Generate a grounded answer using the LLM
        4. Return answer with source citations and confidence score

    Args:
        question: User's input question

    Returns:
        dict with answer, sources, confidence, and route
    """
    logger.info(f"Processing RAG query: {question[:80]}...")
    settings = get_settings()

    vector_store = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embedding_model(),
        persist_directory=settings.chroma_persist_dir,
    )

    # Check vector store is populated
    try:
        existing = vector_store.get()
        if not existing or not existing.get("ids"):
            return {
                "answer": "The knowledge base is empty. Please call POST /ingest first.",
                "sources": [],
                "confidence": "low",
                "route": "rag",
            }
    except Exception as e:
        logger.error(f"Vector store check failed: {e}")

    # Retrieve relevant chunks with similarity scores
    docs_with_scores = vector_store.similarity_search_with_relevance_scores(
        question, k=settings.top_k_results
    )

    logger.info(f"Retrieved {len(docs_with_scores)} chunks.")
    confidence = calculate_confidence(docs_with_scores)
    docs = [doc for doc, _ in docs_with_scores]
    context = format_docs(docs)

    if not context.strip():
        return {
            "answer": NO_ANSWER_MESSAGE,
            "sources": [],
            "confidence": "low",
            "route": "rag",
        }

    # Generate answer
    chain = get_rag_prompt() | get_llm() | StrOutputParser()
    logger.info("Generating answer with LLM...")
    answer = chain.invoke({"context": context, "question": question})

    if "could not find" in answer.lower():
        confidence = "low"

    # Build source citations
    sources = []
    seen = set()
    for doc, score in docs_with_scores:
        filename = doc.metadata.get("filename", "unknown")
        if filename not in seen:
            seen.add(filename)
            sources.append({
                "document": filename,
                "chunk": doc.page_content[:200].strip() + "...",
                "relevance_score": round(score, 3),
            })

    return {
        "answer": answer.strip(),
        "sources": sources,
        "confidence": confidence,
        "route": "rag",
    }
