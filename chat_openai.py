#!/usr/bin/env python3
"""
Chat with OpenAI using vector context from Qdrant.
This script provides a command-line interface for the vector_chat package.
"""

import sys
from vector_chat.cli.chat import main

if __name__ == "__main__":
    sys.exit(main()) 