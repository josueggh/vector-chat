"""
Text chunking utilities for embedding.
"""

import logging
import os
from typing import List, Optional

import nltk
from nltk.tokenize import sent_tokenize

from vector_chat.config import DEFAULT_MAX_SENTENCES_PER_CHUNK, TEXT_FILE_EXTENSIONS

# Ensure NLTK punkt tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

logger = logging.getLogger(__name__)


def chunk_by_sentences(
    text: str, max_sents: int = DEFAULT_MAX_SENTENCES_PER_CHUNK
) -> List[str]:
    """
    Split text into chunks of sentences.

    Args:
        text: Text to split
        max_sents: Maximum number of sentences per chunk

    Returns:
        List of text chunks
    """
    sents = sent_tokenize(text)
    chunks: List[str] = []
    current: List[str] = []

    for sent in sents:
        current.append(sent)
        if len(current) >= max_sents:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_text(
    text: str,
    max_sents: int = DEFAULT_MAX_SENTENCES_PER_CHUNK,
    source_name: str = "unknown",
) -> List[dict]:
    """
    Process text into chunks with metadata.

    Args:
        text: Text to process
        max_sents: Maximum number of sentences per chunk
        source_name: Name of the source (file or description)

    Returns:
        List of dictionaries with chunk text and metadata
    """
    chunks = chunk_by_sentences(text, max_sents)

    return [
        {
            "chunk_text": chunk,
            "source": source_name,
            "chunk_index": i,
            "total_chunks": len(chunks),
        }
        for i, chunk in enumerate(chunks)
    ]


def list_text_files(directory: str = ".") -> List[str]:
    """
    List all text files in the directory.

    Args:
        directory: Directory to search

    Returns:
        List of file paths
    """
    files = []

    try:
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(file)
                if ext.lower() in TEXT_FILE_EXTENSIONS:
                    files.append(file_path)

        logger.info(f"Found {len(files)} text files in {directory}")
        return files
    except Exception as e:
        logger.error(f"Error listing text files: {str(e)}")
        return []


def read_file_content(file_path: str) -> Optional[str]:
    """
    Read the content of a file.

    Args:
        file_path: Path to the file

    Returns:
        File content as string or None if error
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(f"Read {len(content)} characters from {file_path}")
        return content
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return None


def process_file(
    file_path: str, max_sents: int = DEFAULT_MAX_SENTENCES_PER_CHUNK
) -> List[dict]:
    """
    Process a file into chunks with metadata.

    Args:
        file_path: Path to the file
        max_sents: Maximum number of sentences per chunk

    Returns:
        List of dictionaries with chunk text and metadata
    """
    content = read_file_content(file_path)
    if not content:
        logger.warning(f"No content read from {file_path}")
        return []

    source_name = os.path.basename(file_path)
    return chunk_text(content, max_sents, source_name)
