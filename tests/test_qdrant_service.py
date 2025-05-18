"""
Tests for the QdrantService class.
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.http import models

from vector_chat.services.qdrant_service import QdrantService


class TestQdrantService(unittest.TestCase):
    """Tests for the QdrantService class."""

    @patch("vector_chat.services.qdrant_service.QdrantClient")
    def test_init_existing_collection(self, mock_client):
        """Test initialization with existing collection."""
        # Set up mock client
        mock_client_instance = mock_client.return_value
        mock_client_instance.collection_exists.return_value = True

        # Initialize service
        service = QdrantService(collection_name="test_collection")

        # Verify
        mock_client_instance.collection_exists.assert_called_once_with(
            "test_collection"
        )
        mock_client_instance.create_collection.assert_not_called()
        self.assertEqual(service.collection_name, "test_collection")

    @patch("vector_chat.services.qdrant_service.QdrantClient")
    def test_init_new_collection(self, mock_client):
        """Test initialization with new collection."""
        # Set up mock client
        mock_client_instance = mock_client.return_value
        mock_client_instance.collection_exists.return_value = False

        # Initialize service
        service = QdrantService(collection_name="test_collection", vector_size=1536)

        # Verify
        mock_client_instance.collection_exists.assert_called_once_with(
            "test_collection"
        )
        mock_client_instance.create_collection.assert_called_once()

        # Check create_collection arguments
        call_args = mock_client_instance.create_collection.call_args[1]
        self.assertEqual(call_args["collection_name"], "test_collection")
        self.assertEqual(call_args["vectors_config"].size, 1536)
        self.assertEqual(call_args["vectors_config"].distance, models.Distance.COSINE)

    @patch("vector_chat.services.qdrant_service.QdrantClient")
    def test_init_missing_vector_size(self, mock_client):
        """Test error when creating new collection without vector_size."""
        # Set up mock client
        mock_client_instance = mock_client.return_value
        mock_client_instance.collection_exists.return_value = False

        # Initialize service should raise error
        with self.assertRaises(ValueError):
            QdrantService(collection_name="test_collection")

    @patch("vector_chat.services.qdrant_service.QdrantClient")
    def test_upsert(self, mock_client):
        """Test upserting vectors."""
        # Set up mock client
        mock_client_instance = mock_client.return_value
        mock_client_instance.collection_exists.return_value = True

        # Initialize service
        service = QdrantService(collection_name="test_collection")

        # Test data
        ids = [1, 2]
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        payloads = [{"text": "test1"}, {"text": "test2"}]

        # Upsert vectors
        service.upsert(ids, vectors, payloads)

        # Verify
        mock_client_instance.upsert.assert_called_once()

        # Check upsert arguments
        call_args = mock_client_instance.upsert.call_args[1]
        self.assertEqual(call_args["collection_name"], "test_collection")
        self.assertEqual(len(call_args["points"]), 2)

        # Check points
        points = call_args["points"]
        self.assertEqual(points[0].id, 1)
        self.assertEqual(points[0].vector, [0.1, 0.2])
        self.assertEqual(points[0].payload, {"text": "test1"})

        self.assertEqual(points[1].id, 2)
        self.assertEqual(points[1].vector, [0.3, 0.4])
        self.assertEqual(points[1].payload, {"text": "test2"})

    @patch("vector_chat.services.qdrant_service.QdrantClient")
    def test_upsert_without_payloads(self, mock_client):
        """Test upserting vectors without payloads."""
        # Set up mock client
        mock_client_instance = mock_client.return_value
        mock_client_instance.collection_exists.return_value = True

        # Initialize service
        service = QdrantService(collection_name="test_collection")

        # Test data
        ids = [1, 2]
        vectors = [[0.1, 0.2], [0.3, 0.4]]

        # Upsert vectors
        service.upsert(ids, vectors)

        # Verify
        mock_client_instance.upsert.assert_called_once()

        # Check points
        points = mock_client_instance.upsert.call_args[1]["points"]
        self.assertEqual(points[0].payload, {})
        self.assertEqual(points[1].payload, {})

    @patch("vector_chat.services.qdrant_service.QdrantClient")
    def test_search(self, mock_client):
        """Test searching for vectors."""
        # Create mock search results
        mock_hit1 = MagicMock()
        mock_hit1.id = 1
        mock_hit1.score = 0.9
        mock_hit1.payload = {"text": "test1"}

        mock_hit2 = MagicMock()
        mock_hit2.id = 2
        mock_hit2.score = 0.8
        mock_hit2.payload = {"text": "test2"}

        # Set up mock client
        mock_client_instance = mock_client.return_value
        mock_client_instance.collection_exists.return_value = True
        mock_client_instance.search.return_value = [mock_hit1, mock_hit2]

        # Initialize service
        service = QdrantService(collection_name="test_collection")

        # Search for vectors
        query_vector = [0.1, 0.2]
        results = service.search(query_vector, top_k=5, score_threshold=0.7)

        # Verify
        mock_client_instance.search.assert_called_once()

        # Check search arguments
        call_args = mock_client_instance.search.call_args[1]
        self.assertEqual(call_args["collection_name"], "test_collection")
        self.assertEqual(call_args["query_vector"], query_vector)
        self.assertEqual(call_args["limit"], 5)
        self.assertEqual(call_args["with_payload"], True)
        self.assertEqual(call_args["score_threshold"], 0.7)

        # Check results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], (1, 0.9, {"text": "test1"}))
        self.assertEqual(results[1], (2, 0.8, {"text": "test2"}))

    @patch("vector_chat.services.qdrant_service.QdrantClient")
    def test_check_collection_exists(self, mock_client):
        """Test checking if collection exists."""
        # Set up mock client
        mock_client_instance = mock_client.return_value
        mock_client_instance.collection_exists.return_value = True

        # Initialize service
        service = QdrantService(collection_name="test_collection")

        # Check collection exists
        exists = service.check_collection_exists()

        # Verify
        self.assertTrue(exists)
        mock_client_instance.collection_exists.assert_called_with("test_collection")

        # Test error handling
        mock_client_instance.collection_exists.side_effect = Exception("Test error")
        exists = service.check_collection_exists()
        self.assertFalse(exists)
