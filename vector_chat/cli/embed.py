"""
Command-line interface for embedding text into vector database.
"""

import argparse
import copy
import logging
import sys
from typing import List, Optional

from vector_chat.clients import OpenAIClient
from vector_chat.config import (
    AVAILABLE_EMBEDDING_MODELS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_SENTENCES_PER_CHUNK,
    QDRANT_COLLECTION,
    validate_environment,
)
from vector_chat.services.chunker import (
    chunk_text,
    list_text_files,
    process_file,
    read_file_content,
)
from vector_chat.services.qdrant_service import QdrantService

logger = logging.getLogger(__name__)


def setup_argparse() -> argparse.ArgumentParser:
    """
    Set up command-line argument parser.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Embed text chunks into vector database"
    )

    parser.add_argument("-f", "--file", help="Path to text file to embed", type=str)

    parser.add_argument("-t", "--text", help="Text to embed directly", type=str)

    parser.add_argument(
        "-m",
        "--model",
        help=f"Embedding model to use (default: {DEFAULT_EMBEDDING_MODEL})",
        choices=AVAILABLE_EMBEDDING_MODELS,
        default=DEFAULT_EMBEDDING_MODEL,
    )

    parser.add_argument(
        "-c",
        "--collection",
        help=f"Qdrant collection name (default: {QDRANT_COLLECTION})",
        default=QDRANT_COLLECTION,
    )

    parser.add_argument(
        "-s",
        "--sentences",
        help=f"Maximum sentences per chunk (default: {DEFAULT_MAX_SENTENCES_PER_CHUNK})",
        type=int,
        default=DEFAULT_MAX_SENTENCES_PER_CHUNK,
    )

    parser.add_argument(
        "-l",
        "--list-files",
        help="List available text files in current directory",
        action="store_true",
    )

    parser.add_argument("--verbose", help="Enable verbose logging", action="store_true")

    return parser


def get_input_text(args: argparse.Namespace) -> Optional[tuple]:
    """
    Get input text from file or direct input.

    Args:
        args: Command-line arguments

    Returns:
        Tuple of (text, source_name) or None if no input
    """
    # Check for file input
    if args.file:
        content = read_file_content(args.file)
        if content:
            return content, args.file
        else:
            logger.error(f"Could not read file: {args.file}")
            return None

    # Check for direct text input
    if args.text:
        return args.text, "command_line_input"

    # If no input provided, prompt user
    if not args.list_files:
        # List available files
        files = list_text_files()

        if files:
            logger.info("Available text files:")
            for i, file in enumerate(files):
                logger.info(f"{i+1}. {file}")

            try:
                choice = input(
                    "\nSelect a file number (or press Enter for manual input): "
                )
                if choice.strip():
                    idx = int(choice) - 1
                    if 0 <= idx < len(files):
                        content = read_file_content(files[idx])
                        if content:
                            return content, files[idx]
            except (ValueError, IndexError):
                logger.warning("Invalid selection")

        # Manual input
        logger.info(
            "Enter text to embed (press Ctrl+D or Ctrl+Z on Windows to finish):"
        )
        try:
            lines = sys.stdin.readlines()
            text = "".join(lines)
            if text.strip():
                return text, "manual_input"
        except KeyboardInterrupt:
            logger.warning("Input cancelled")

    return None


def embed_text(
    text: str,
    source_name: str,
    model_name: str,
    collection_name: str,
    max_sentences: int,
) -> bool:
    """
    Embed text chunks and store in vector database.

    Args:
        text: Text to embed
        source_name: Name of the source
        model_name: Name of the embedding model
        collection_name: Name of the Qdrant collection
        max_sentences: Maximum sentences per chunk

    Returns:
        True if successful, False otherwise
    """
    try:
        # Initialize OpenAI client
        openai_client = OpenAIClient(embedding_model=model_name)

        # Process text into chunks
        chunks_data = chunk_text(text, max_sentences, source_name)
        chunks = [item["chunk_text"] for item in chunks_data]

        if not chunks:
            logger.error("No chunks generated from text")
            return False

        logger.info(f"Text chunked into {len(chunks)} segments")
        for i, chunk in enumerate(chunks):
            logger.debug(f"Chunk {i+1}: {chunk[:50]}...")

        # Generate embeddings
        logger.info(f"Generating embeddings using {model_name}...")
        vectors = openai_client.embed(chunks)

        # Prepare payloads with metadata
        ids = list(range(1, len(chunks) + 1))
        payloads = [
            {
                "chunk_text": chunk["chunk_text"],
                "source": chunk["source"],
                "model_name": model_name,
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"],
            }
            for chunk in chunks_data
        ]

        # Initialize Qdrant and store vectors
        qdrant = QdrantService(
            collection_name=collection_name,
            vector_size=openai_client.embedding_dimension,
        )

        qdrant.upsert(ids, vectors, copy.deepcopy(payloads))

        logger.info(
            f"Successfully embedded {len(chunks)} chunks into collection '{collection_name}'"
        )
        return True

    except Exception as e:
        logger.error(f"Error embedding text: {str(e)}", exc_info=True)
        return False


def main() -> int:
    """
    Main entry point for the embed command.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Parse arguments
    parser = setup_argparse()
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed")
        return 1

    # List files if requested
    if args.list_files:
        files = list_text_files()
        if files:
            logger.info("Available text files:")
            for file in files:
                logger.info(f"- {file}")
        else:
            logger.info("No text files found in current directory")
        return 0

    # Get input text
    input_data = get_input_text(args)
    if not input_data:
        logger.error("No input text provided")
        return 1

    text, source = input_data

    # Embed text
    success = embed_text(
        text=text,
        source_name=source,
        model_name=args.model,
        collection_name=args.collection,
        max_sentences=args.sentences,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
