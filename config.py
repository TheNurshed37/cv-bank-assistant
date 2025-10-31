# config.py - ENHANCED FOR PHASE 2
import os
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # API Configuration
    GOOGLE_API_KEY: str
    UPLOAD_DIR: str = "data"
    FAISS_DIR: str = "faiss_index"
    HASH_INDEX_FILE: str = "hash_index.json"
    
    # Security
    # ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "https://your-hr-domain.com"]
    MAX_FILE_SIZE: int = 50 * 1024 * 1024
    
    # Model Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL: str = "gemini-2.5-flash"
    LLM_VALIDATION_MODEL: str = "gemini-2.5-flash-lite"  # Faster for validation
    LLM_TEMPERATURE: float = 0.1  # Lower for more consistent validation
    
    # RAG Configuration 
    CHUNK_SIZE: int = 500  # Larger for comprehensive chunks
    CHUNK_OVERLAP: int = 100
    SEARCH_K: int = 4

    class Config:
        env_file = ".env"

settings = Settings()