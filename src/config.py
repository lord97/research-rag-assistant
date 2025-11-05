"""
Configuration module for the RAG system.
This file centralizes all configuration settings and loads environment variables.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Configuration class containing all settings for the RAG system.
    Uses environment variables with fallback to default values.
    """
    
    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Paths - where we store uploaded files and vector database
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
    UPLOADS_PATH = os.getenv("UPLOADS_PATH", "./data/uploads")
    
    # Model configurations
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    
    # Text chunking settings
    # CHUNK_SIZE: how many characters in each text chunk
    # CHUNK_OVERLAP: how many characters overlap between chunks (for context)
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    
    # Retrieval settings
    # How many relevant chunks to retrieve when answering a question
    TOP_K_RESULTS = 5
    
    @classmethod
    def validate(cls):
        """
        Validates that all required configurations are set.
        Raises an error if API key is missing.
        """
        if not cls.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY not found. Please set it in .env file"
            )
        
        # Create directories if they don't exist
        os.makedirs(cls.VECTOR_DB_PATH, exist_ok=True)
        os.makedirs(cls.UPLOADS_PATH, exist_ok=True)