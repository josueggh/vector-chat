"""
Main entry point for the vector_chat package.
"""

import argparse
import sys
from typing import List, Optional

from vector_chat.cli.chat import main as chat_main
from vector_chat.cli.embed import main as embed_main


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the vector_chat package.

    Args:
        args: Command-line arguments

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Vector Chat - Text embedding and chat with context"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Embed command
    embed_parser = subparsers.add_parser(
        "embed", help="Embed text into vector database"
    )
    embed_parser.add_argument("-f", "--file", help="Path to text file to embed")
    embed_parser.add_argument("-t", "--text", help="Text to embed directly")
    embed_parser.add_argument(
        "-l", "--list-files", help="List available text files", action="store_true"
    )

    # Chat command
    chat_parser = subparsers.add_parser(
        "chat", help="Chat with OpenAI using vector context"
    )
    chat_parser.add_argument(
        "--no-context", help="Disable context retrieval", action="store_true"
    )

    # Parse arguments
    parsed_args = parser.parse_args(args)

    # Run command
    if parsed_args.command == "embed":
        return embed_main()
    elif parsed_args.command == "chat":
        return chat_main()
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
