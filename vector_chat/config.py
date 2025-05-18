"""
Configuration settings for the vector_chat package.
Handles environment variables, API keys, and default settings.
"""

import logging
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

# Qdrant settings
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "openai_embeddings")

# OpenAI models
DEFAULT_CHAT_MODEL: str = os.getenv("DEFAULT_CHAT_MODEL", "gpt-4o")
DEFAULT_EMBEDDING_MODEL: str = os.getenv(
    "DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small"
)

# Available embedding models
AVAILABLE_EMBEDDING_MODELS: List[str] = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]

# Embedding dimensions by model
EMBEDDING_DIMENSIONS: Dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# Emoji indicators for different information sources
EMOJI_SEARCH: str = "🔍"  # Searching
EMOJI_CONTEXT: str = "📚"  # Using context from Qdrant
EMOJI_AI: str = "🤖"  # AI general knowledge
EMOJI_ERROR: str = "⚠️"  # Error indicator

# Default chunking settings
DEFAULT_MAX_SENTENCES_PER_CHUNK: int = 3

# Text file extensions for auto-detection
TEXT_FILE_EXTENSIONS: List[str] = [
    ".txt",
    ".md",
    ".py",
    ".js",
    ".html",
    ".css",
    ".json",
    ".csv",
    ".xml",
    ".yaml",
    ".yml",
]


# Validate required environment variables
def validate_environment() -> bool:
    """
    Validate that required environment variables are set.

    Returns:
        bool: True if all required variables are set, False otherwise
    """
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY environment variable is not set")
        return False

    logger.info("Environment validation successful")
    return True
