"""
config.py — Centralized Configuration
======================================
Reads all settings from the .env file.
Using pydantic-settings means every setting is type-checked and validated
automatically. No hardcoded values anywhere in the codebase.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """
    All application settings loaded from environment variables / .env file.
    The 'model_config' tells pydantic to read from a .env file automatically.
    """

    # LLM Settings
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "phi3:mini"
    llm_temperature: float = 0.1

    # Embedding Model
    embedding_model: str = "all-MiniLM-L6-v2"

    # ChromaDB (Vector Database)
    chroma_persist_dir: str = "./vector_store"
    chroma_collection_name: str = "healthcare_docs"

    # Document Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # API Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Logging
    log_level: str = "INFO"

    # RAG Retrieval
    top_k_results: int = 4
    confidence_high_threshold: float = 0.75
    confidence_low_threshold: float = 0.50

    # Data directory
    data_dir: str = "./data"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached singleton instance of Settings.
    lru_cache ensures we only read the .env file ONCE, not on every request.
    This is important for performance in a web API.
    """
    return Settings()
