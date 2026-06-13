"""
Document ingestion pipeline.
Loads .txt documents from /data, splits into chunks,
generates embeddings, and stores them in ChromaDB.
"""

import os
import logging
from typing import List, Dict, Any

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import get_settings
from app.embeddings import get_embedding_model

logger = logging.getLogger(__name__)


def load_documents(data_dir: str) -> List[Document]:
    """Load all .txt files from the specified directory."""
    logger.info(f"Loading documents from: {data_dir}")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    loader = DirectoryLoader(
        path=data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
    )

    documents = loader.load()

    if not documents:
        raise ValueError(f"No .txt documents found in {data_dir}")

    logger.info(f"Loaded {len(documents)} documents.")
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for retrieval.
    Uses RecursiveCharacterTextSplitter to preserve sentence boundaries.
    """
    settings = get_settings()
    logger.info(f"Splitting documents: chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Enrich metadata for source citation
    for i, chunk in enumerate(chunks):
        source_path = chunk.metadata.get("source", "unknown")
        chunk.metadata["filename"] = os.path.basename(source_path)
        chunk.metadata["chunk_index"] = i

    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents.")
    return chunks


def get_vector_store() -> Chroma:
    """Initialize and return the ChromaDB vector store instance."""
    settings = get_settings()

    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embedding_model(),
        persist_directory=settings.chroma_persist_dir,
    )


def ingest_documents(data_dir: str = None) -> Dict[str, Any]:
    """
    Run the full ingestion pipeline: load → split → embed → store.

    Args:
        data_dir: Path to documents folder. Defaults to config value.

    Returns:
        Dictionary with ingestion results and statistics.
    """
    settings = get_settings()
    data_dir = data_dir or settings.data_dir

    logger.info("Starting document ingestion pipeline...")

    documents = load_documents(data_dir)
    chunks = split_documents(documents)

    logger.info("Storing chunks in ChromaDB...")

    vector_store = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embedding_model(),
        persist_directory=settings.chroma_persist_dir,
    )

    # Clear existing data to prevent duplicates on re-ingestion
    try:
        existing = vector_store.get()
        if existing and existing.get("ids"):
            logger.info(f"Clearing {len(existing['ids'])} existing chunks...")
            vector_store.delete(ids=existing["ids"])
    except Exception as e:
        logger.warning(f"Could not clear existing data: {e}")

    vector_store.add_documents(chunks)

    doc_count = len(vector_store.get()["ids"])
    source_files = list(set(
        os.path.basename(doc.metadata.get("source", "unknown"))
        for doc in documents
    ))

    logger.info(f"Ingestion complete. {doc_count} chunks stored.")

    return {
        "status": "success",
        "documents_processed": len(documents),
        "chunks_created": len(chunks),
        "chunks_stored": doc_count,
        "source_files": source_files,
        "vector_store": "chromadb",
        "message": f"Successfully ingested {len(documents)} documents into vector store.",
    }
