"""
vector_chat - A package for embedding text and creating a chat interface with OpenAI
"""

__version__ = "0.1.0"

from vector_chat.clients import OpenAIClient
from vector_chat.services.chunker import chunk_by_sentences, chunk_text
from vector_chat.services.qdrant_service import QdrantService
