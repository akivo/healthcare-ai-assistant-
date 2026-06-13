"""
Embedding model loader for semantic search.
Uses sentence-transformers/all-MiniLM-L6-v2 for generating document embeddings.
"""

import logging
from functools import lru_cache
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load and cache the HuggingFace embedding model.
    Uses lru_cache to avoid reloading the model on every request.
    """
    settings = get_settings()
    logger.info(f"Loading embedding model: {settings.embedding_model}")

    model = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("Embedding model loaded successfully.")
    return model
