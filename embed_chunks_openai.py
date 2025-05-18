#!/usr/bin/env python3
"""
Embed text chunks into vector database.
This script provides a command-line interface for the vector_chat package.
"""

import sys
from vector_chat.cli.embed import main

if __name__ == "__main__":
    sys.exit(main()) 