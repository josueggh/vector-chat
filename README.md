# Vector Chat

![PyPI](https://img.shields.io/pypi/v/vector-chat)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python toolkit for text embedding with OpenAI and Qdrant, semantic search, and AI-powered chat.

![Demo](data/demo.gif)

The demo reads the text file stored in `data`, whose story is related to a robot named Naro and his AI companion Matita, 
exploring the mysteries of the Moon.

## Features

- Create embeddings from text files or direct input
- Store embeddings in a Qdrant vector database
- Chat with OpenAI models using relevant context from stored embeddings
- Command-line interfaces for embedding and chatting

## Requirements

- Qdrant 

**Note:** You can run Qdrant using docker, or visit the [official website](https://qdrant.tech/) for more information.
```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Installation

### Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```


### Using Poetry (recommended)

```bash
# Install with Poetry
poetry install
```

## Configuration

Create the `.env` file in your project directory. Then, set the following variables:

```
# Required
OPENAI_API_KEY=your_openai_api_key
# Optional
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key_if_needed
QDRANT_COLLECTION=openai_embeddings
DEFAULT_CHAT_MODEL=gpt-4o
DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
```

## Usage

### Command Line Interface

#### Embedding Text

```bash
# Embed a text file (an example is provided in data/demo.txt)
poetry run embed --file path/to/file.txt

# Embed text directly
poetry run embed --text "Your text to embed"

# Specify embedding model
poetry run embed --file path/to/file.txt --model text-embedding-3-large

# List available text files
poetry run embed --list-files

# Show help
poetry run embed --help
```

You can also use the standalone script:

```bash
python embed_chunks_openai.py --file path/to/file.txt
```

#### Chatting with Context

```bash
# Start chat with default settings
poetry run chat

# Use a specific chat model
poetry run chat --chat-model gpt-4o

# Use a specific embedding model for context search
poetry run chat --embedding-model text-embedding-3-large

# Disable context retrieval
poetry run chat --no-context

# Show help
poetry run chat --help
```

You can also use the standalone script:

```bash
python chat_openai.py
```

### Python API

```python
from vector_chat import OpenAIClient, QdrantService, chunk_text

# Initialize clients
openai_client = OpenAIClient(embedding_model="text-embedding-3-small")
qdrant_client = QdrantService(collection_name="openai_embeddings")

# Embed text
text = "Your text to embed"
chunks = chunk_text(text, max_sents=3, source_name="example")
chunk_texts = [chunk["chunk_text"] for chunk in chunks]
vectors = openai_client.embed(chunk_texts)

# Store embeddings
ids = list(range(1, len(chunks) + 1))
payloads = [
    {
        "chunk_text": chunk["chunk_text"],
        "source": chunk["source"],
        "chunk_index": chunk["chunk_index"],
        "total_chunks": chunk["total_chunks"]
    }
    for chunk in chunks
]
qdrant_client.upsert(ids, vectors, payloads)

# Chat with context
query = "What is the main topic of the text?"
query_vector = openai_client.embed([query])[0]
results = qdrant_client.search(query_vector, top_k=3, score_threshold=0.3)

# Use context in chat
if results:
    context = "\n\n".join([result[2]["chunk_text"] for result in results])
    openai_client.add_system_message(f"Context: {context}")
    
openai_client.add_user_message(query)
response = openai_client.get_response()
print(response)
```

## Development

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_chunker.py
```

### Code Formatting

```bash
# Format code with Black
poetry run black vector_chat tests

# Sort imports with isort
poetry run isort vector_chat tests
```

## License

MIT 