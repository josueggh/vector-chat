"""
OpenAI client for both chat completions and embeddings.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI

from vector_chat.config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    EMOJI_ERROR,
    OPENAI_API_KEY,
)

logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    Client for interacting with OpenAI APIs for both chat completions and embeddings.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        chat_model: str = DEFAULT_CHAT_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        """
        Initialize OpenAI client for both chat completions and embeddings.

        Args:
            api_key: OpenAI API key, defaults to environment variable
            chat_model: Model name for chat completions
            embedding_model: Model name for embeddings
        """
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass as parameter."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.conversation_history = []

        # Get embedding dimension based on model
        self.embedding_dimension = EMBEDDING_DIMENSIONS.get(embedding_model, 1536)

    def add_system_message(self, content: str) -> None:
        """
        Add a system message to the conversation history.

        Args:
            content: The message content
        """
        self.conversation_history.append({"role": "system", "content": content})

    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the conversation history.

        Args:
            content: The message content
        """
        self.conversation_history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """
        Add an assistant message to the conversation history.

        Args:
            content: The message content
        """
        self.conversation_history.append({"role": "assistant", "content": content})

    def get_response(self, temperature: float = 0.7) -> str:
        """
        Get a response from the chat model based on conversation history.

        Args:
            temperature: Sampling temperature (0-1)

        Returns:
            The model's response text

        Raises:
            Exception: If there's an error getting a response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=self.conversation_history,
                temperature=temperature,
            )
            message = response.choices[0].message.content
            self.add_assistant_message(message)
            return message
        except Exception as e:
            logger.error(f"Error getting chat response: {str(e)}")
            raise

    def get_structured_response(
        self, prompt: str, json_structure: Dict[str, Any], temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Get a structured JSON response from the model.

        Args:
            prompt: The prompt to send to the model
            json_structure: Default JSON structure to return on error
            temperature: Sampling temperature (0-1)

        Returns:
            Structured data as a dictionary
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides structured data in JSON format.",
                },
                {"role": "user", "content": prompt},
            ]

            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
            )

            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            logger.error(f"Error getting structured response: {str(e)}")
            return json_structure  # Return the default structure on error

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings using OpenAI's embedding model.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors

        Raises:
            Exception: If there's an error creating embeddings
        """
        try:
            vectors = []
            batch_size = 64
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch,
                )
                vectors.extend([item.embedding for item in response.data])
            return vectors
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise

    def reset_conversation(self, keep_system_messages: bool = True) -> None:
        """
        Reset the conversation history, optionally keeping system messages.

        Args:
            keep_system_messages: Whether to keep system messages
        """
        if keep_system_messages:
            system_messages = [
                msg for msg in self.conversation_history if msg["role"] == "system"
            ]
            self.conversation_history = system_messages
        else:
            self.conversation_history = []

    def ask(self, query: str, temperature: float = 0.7) -> str:
        """
        Simple helper to ask a single question and get a response.

        Args:
            query: The question to ask
            temperature: Sampling temperature (0-1)

        Returns:
            The model's response
        """
        self.add_user_message(query)
        return self.get_response(temperature=temperature)
