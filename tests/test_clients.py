"""
Tests for the OpenAIClient class.
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest

from vector_chat.clients import OpenAIClient
from vector_chat.config import DEFAULT_CHAT_MODEL, DEFAULT_EMBEDDING_MODEL


class TestOpenAIClient(unittest.TestCase):
    """Tests for the OpenAIClient class."""

    @patch("vector_chat.clients.OpenAI")
    def test_init_with_api_key(self, mock_openai):
        """Test client initialization with API key."""
        # Test with default parameters
        with patch("vector_chat.clients.OPENAI_API_KEY", "test_key"):
            client = OpenAIClient()

            # Assert
            self.assertEqual(client.api_key, "test_key")
            self.assertEqual(client.chat_model, DEFAULT_CHAT_MODEL)
            self.assertEqual(client.embedding_model, DEFAULT_EMBEDDING_MODEL)
            self.assertTrue(mock_openai.called)

    @patch("vector_chat.clients.OpenAI")
    def test_init_with_custom_models(self, mock_openai):
        """Test client initialization with custom models."""
        with patch("vector_chat.clients.OPENAI_API_KEY", "test_key"):
            client = OpenAIClient(
                chat_model="custom-chat-model", embedding_model="custom-embedding-model"
            )

            # Assert
            self.assertEqual(client.chat_model, "custom-chat-model")
            self.assertEqual(client.embedding_model, "custom-embedding-model")
            self.assertEqual(client.embedding_dimension, 1536)
            self.assertTrue(mock_openai.called)

    @patch("vector_chat.clients.OpenAI")
    def test_init_without_api_key(self, mock_openai):
        """Test error when API key is missing."""
        with patch("vector_chat.clients.OPENAI_API_KEY", None):
            with self.assertRaises(ValueError):
                OpenAIClient()

    @patch("vector_chat.clients.OpenAI")
    def test_add_messages(self, mock_openai):
        """Test conversation history management methods."""
        with patch("vector_chat.clients.OPENAI_API_KEY", "test_key"):
            client = OpenAIClient()

            # Test adding messages
            client.add_system_message("System message")
            client.add_user_message("User message")
            client.add_assistant_message("Assistant message")

            # Assert
            self.assertEqual(len(client.conversation_history), 3)
            self.assertEqual(
                client.conversation_history[0],
                {"role": "system", "content": "System message"},
            )
            self.assertEqual(
                client.conversation_history[1],
                {"role": "user", "content": "User message"},
            )
            self.assertEqual(
                client.conversation_history[2],
                {"role": "assistant", "content": "Assistant message"},
            )

            # Test reset with keeping system messages
            client.reset_conversation(keep_system_messages=True)
            self.assertEqual(len(client.conversation_history), 1)
            self.assertEqual(
                client.conversation_history[0],
                {"role": "system", "content": "System message"},
            )

            # Test reset without keeping system messages
            client.reset_conversation(keep_system_messages=False)
            self.assertEqual(len(client.conversation_history), 0)

    @patch("vector_chat.clients.OpenAI")
    def test_get_response(self, mock_openai):
        """Test getting a response from the chat model."""
        # Setup mock response
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()

        mock_message.content = "Test response"
        mock_choice.message = mock_message
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        # Test get_response
        with patch("vector_chat.clients.OPENAI_API_KEY", "test_key"):
            client = OpenAIClient()
            client.client = mock_client

            # Add a message to history
            client.add_user_message("Test question")

            # Get response
            response = client.get_response()

            # Assert
            self.assertEqual(response, "Test response")
            self.assertTrue(mock_client.chat.completions.create.called)
            self.assertEqual(len(client.conversation_history), 2)
            self.assertEqual(client.conversation_history[1]["role"], "assistant")
            self.assertEqual(client.conversation_history[1]["content"], "Test response")

    @patch("vector_chat.clients.OpenAI")
    def test_get_structured_response(self, mock_openai):
        """Test getting a structured response from the chat model."""
        # Setup mock response
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()

        mock_message.content = '{"key": "value"}'
        mock_choice.message = mock_message
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        # Test get_structured_response
        with patch("vector_chat.clients.OPENAI_API_KEY", "test_key"):
            client = OpenAIClient()
            client.client = mock_client

            result = client.get_structured_response(
                "Test prompt", {"default": "default_value"}
            )

            # Assert
            self.assertEqual(result, {"key": "value"})
            self.assertTrue(mock_client.chat.completions.create.called)
            call_args = mock_client.chat.completions.create.call_args[1]
            self.assertEqual(call_args["model"], DEFAULT_CHAT_MODEL)
            self.assertEqual(call_args["response_format"], {"type": "json_object"})
            self.assertEqual(len(call_args["messages"]), 2)
            self.assertEqual(call_args["messages"][0]["role"], "system")
            self.assertEqual(call_args["messages"][1]["role"], "user")
            self.assertEqual(call_args["messages"][1]["content"], "Test prompt")

    @patch("vector_chat.clients.OpenAI")
    def test_embed(self, mock_openai):
        """Test creating embeddings."""
        # Setup mock response
        mock_client = MagicMock()
        mock_embedding = MagicMock()
        mock_data1 = MagicMock()
        mock_data1.embedding = [0.1, 0.2]
        mock_data2 = MagicMock()
        mock_data2.embedding = [0.3, 0.4]

        mock_embedding.data = [mock_data1, mock_data2]
        mock_client.embeddings.create.return_value = mock_embedding
        mock_openai.return_value = mock_client

        # Test embed
        with patch("vector_chat.clients.OPENAI_API_KEY", "test_key"):
            client = OpenAIClient()
            client.client = mock_client

            vectors = client.embed(["text1", "text2"])

            # Assert
            self.assertEqual(vectors, [[0.1, 0.2], [0.3, 0.4]])
            mock_client.embeddings.create.assert_called_once()
            call_args = mock_client.embeddings.create.call_args[1]
            self.assertEqual(call_args["model"], DEFAULT_EMBEDDING_MODEL)
            self.assertEqual(call_args["input"], ["text1", "text2"])

    @patch("vector_chat.clients.OpenAI")
    def test_reset_conversation(self, mock_openai):
        """Test resetting the conversation history."""
        with patch("vector_chat.clients.OPENAI_API_KEY", "test_key"):
            client = OpenAIClient()

            # Add messages
            client.add_system_message("System message")
            client.add_user_message("User message")
            client.add_assistant_message("Assistant message")

            # Reset keeping system messages
            client.reset_conversation(keep_system_messages=True)

            # Assert
            self.assertEqual(len(client.conversation_history), 1)
            self.assertEqual(
                client.conversation_history[0],
                {"role": "system", "content": "System message"},
            )

            # Reset without keeping system messages
            client.add_user_message("New user message")
            client.reset_conversation(keep_system_messages=False)

            # Assert
            self.assertEqual(len(client.conversation_history), 0)

    @patch("vector_chat.clients.OpenAI")
    def test_ask(self, mock_openai):
        """Test the ask helper method."""
        # Setup mock response
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()

        mock_message.content = "Test response"
        mock_choice.message = mock_message
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        # Test ask
        with patch("vector_chat.clients.OPENAI_API_KEY", "test_key"):
            client = OpenAIClient()
            client.client = mock_client

            response = client.ask("Test question")

            # Assert
            self.assertEqual(response, "Test response")
            self.assertEqual(client.conversation_history[-2]["role"], "user")
            self.assertEqual(
                client.conversation_history[-2]["content"], "Test question"
            )
            self.assertEqual(client.conversation_history[-1]["role"], "assistant")
            self.assertEqual(
                client.conversation_history[-1]["content"], "Test response"
            )
