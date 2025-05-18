"""
Command-line interface for chatting with OpenAI using vector context.
"""

import argparse
import logging
import sys
from typing import List, Optional, Tuple

from vector_chat.clients import OpenAIClient
from vector_chat.config import (
    AVAILABLE_EMBEDDING_MODELS,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    EMOJI_AI,
    EMOJI_CONTEXT,
    EMOJI_ERROR,
    EMOJI_SEARCH,
    QDRANT_COLLECTION,
    validate_environment,
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
        description="Chat with OpenAI using vector context"
    )

    parser.add_argument(
        "-c",
        "--chat-model",
        help=f"Chat model to use (default: {DEFAULT_CHAT_MODEL})",
        default=DEFAULT_CHAT_MODEL,
    )

    parser.add_argument(
        "-e",
        "--embedding-model",
        help=f"Embedding model to use (default: {DEFAULT_EMBEDDING_MODEL})",
        choices=AVAILABLE_EMBEDDING_MODELS,
        default=DEFAULT_EMBEDDING_MODEL,
    )

    parser.add_argument(
        "--collection",
        help=f"Qdrant collection name (default: {QDRANT_COLLECTION})",
        default=QDRANT_COLLECTION,
    )

    parser.add_argument(
        "-k",
        "--top-k",
        help="Number of context chunks to retrieve (default: 3)",
        type=int,
        default=3,
    )

    parser.add_argument(
        "-t",
        "--threshold",
        help="Similarity threshold for context retrieval (default: 0.3)",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--no-context",
        help="Disable context retrieval, use only general knowledge",
        action="store_true",
    )

    parser.add_argument("--verbose", help="Enable verbose logging", action="store_true")

    return parser


def initialize_clients(
    args: argparse.Namespace,
) -> Tuple[OpenAIClient, Optional[QdrantService]]:
    """
    Initialize OpenAI and Qdrant clients.

    Args:
        args: Command-line arguments

    Returns:
        Tuple of (OpenAIClient, QdrantService or None)
    """
    # Initialize OpenAI client
    openai_client = OpenAIClient(
        chat_model=args.chat_model, embedding_model=args.embedding_model
    )

    # Add system message
    openai_client.add_system_message(
        "You are a helpful assistant that can answer questions based on provided context or general knowledge. "
        "If context is provided, prioritize that information in your answers. "
        "If no context is provided or the question is outside the scope of the context, "
        "use your general knowledge to provide a helpful response. "
        "Always be honest about what you know and don't know."
    )

    # Initialize Qdrant client if context is enabled
    qdrant_client = None
    if not args.no_context:
        try:
            qdrant_client = QdrantService(collection_name=args.collection)
            logger.info(f"Connected to Qdrant collection: {args.collection}")
        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {str(e)}")
            logger.info("Continuing without context retrieval")

    return openai_client, qdrant_client


def get_context(
    query: str,
    openai_client: OpenAIClient,
    qdrant_client: QdrantService,
    top_k: int = 3,
    score_threshold: float = 0.3,
) -> Tuple[bool, Optional[str]]:
    """
    Get relevant context for a query.

    Args:
        query: User query
        openai_client: OpenAI client
        qdrant_client: Qdrant client
        top_k: Number of results to retrieve
        score_threshold: Similarity threshold

    Returns:
        Tuple of (context_found, context_text)
    """
    try:
        # Generate query embedding
        logger.info(f"{EMOJI_SEARCH} Searching for relevant information...")
        q_vec = openai_client.embed([query])[0]

        # Search for relevant chunks
        results = qdrant_client.search(
            q_vec, top_k=top_k, score_threshold=score_threshold
        )

        if not results:
            logger.info(f"{EMOJI_SEARCH} No relevant context found")
            return False, None

        # Prepare context from search results
        context_parts = []
        for i, result in enumerate(results):
            score = result[1]
            text = result[2]["chunk_text"]
            source_info = (
                f" (from {result[2].get('source', 'unknown source')})"
                if "source" in result[2]
                else ""
            )
            model_info = (
                f" [model: {result[2].get('model_name', 'unknown model')}]"
                if "model_name" in result[2]
                else ""
            )
            context_parts.append(
                f"Context {i+1} (Relevance: {score:.2f}){source_info}{model_info}: {text}"
            )

        context = "\n\n".join(context_parts)
        logger.info(f"{EMOJI_CONTEXT} Found {len(results)} relevant context chunks")

        return True, context

    except Exception as e:
        logger.error(f"{EMOJI_ERROR} Error retrieving context: {str(e)}")
        return False, None


def chat_loop(
    openai_client: OpenAIClient,
    qdrant_client: Optional[QdrantService] = None,
    top_k: int = 3,
    score_threshold: float = 0.3,
) -> None:
    """
    Run the interactive chat loop.

    Args:
        openai_client: OpenAI client
        qdrant_client: Qdrant client (or None to disable context)
        top_k: Number of context chunks to retrieve
        score_threshold: Similarity threshold for context retrieval
    """
    print(
        "\nChat with OpenAI (type 'exit' to quit, 'reset' to clear conversation history):"
    )

    if qdrant_client:
        print(
            f"\n{EMOJI_CONTEXT} = Using saved context | {EMOJI_AI} = AI knowledge | {EMOJI_SEARCH} = Searching"
        )
    else:
        print(f"\n{EMOJI_AI} = AI knowledge (no context retrieval enabled)")

    while True:
        # Get user query
        try:
            query = input("\nYou: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat...")
            break

        # Check for special commands
        if query.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        elif query.lower() == "reset":
            openai_client.reset_conversation()
            print(f"\n{EMOJI_AI} Conversation history has been reset.")
            continue

        # Add user query to conversation
        openai_client.add_user_message(query)

        # Try to find relevant context if available
        context_found = False
        if qdrant_client:
            context_found, context = get_context(
                query,
                openai_client,
                qdrant_client,
                top_k=top_k,
                score_threshold=score_threshold,
            )

            if context_found:
                # Add context to chat as system message
                openai_client.add_system_message(
                    f"Here is some relevant context to help answer the question. "
                    f"Use this information if it's helpful for answering the question:\n{context}"
                )

        # Get response from the model
        try:
            response = openai_client.get_response(temperature=0.7)
            if context_found:
                print(f"\n{EMOJI_CONTEXT} AI: {response}")
            else:
                print(f"\n{EMOJI_AI} AI: {response}")

        except Exception as e:
            print(f"{EMOJI_ERROR} Error: {str(e)}")
            continue


def main() -> int:
    """
    Main entry point for the chat command.

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

    try:
        # Initialize clients
        openai_client, qdrant_client = initialize_clients(args)

        # Run chat loop
        chat_loop(
            openai_client=openai_client,
            qdrant_client=qdrant_client,
            top_k=args.top_k,
            score_threshold=args.threshold,
        )

        return 0

    except Exception as e:
        logger.error(f"Error in chat application: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
