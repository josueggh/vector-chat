"""
Tests for the chunker module.
"""

import os
import tempfile
import unittest
from typing import List
from unittest.mock import mock_open, patch

import pytest

from vector_chat.config import DEFAULT_MAX_SENTENCES_PER_CHUNK
from vector_chat.services.chunker import (
    chunk_by_sentences,
    chunk_text,
    list_text_files,
    process_file,
    read_file_content,
)


class TestChunker(unittest.TestCase):
    """Tests for the chunker module."""

    def test_chunk_by_sentences_default(self):
        """Test chunking text by sentences with default settings."""
        text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four. This is sentence five."
        chunks = chunk_by_sentences(text)

        # Should create chunks with DEFAULT_MAX_SENTENCES_PER_CHUNK sentences each
        self.assertEqual(len(chunks), 2)
        self.assertEqual(
            chunks[0],
            "This is sentence one. This is sentence two. This is sentence three.",
        )
        self.assertEqual(chunks[1], "This is sentence four. This is sentence five.")

    def test_chunk_by_sentences_custom(self):
        """Test chunking text by sentences with custom max_sents."""
        text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four. This is sentence five."
        chunks = chunk_by_sentences(text, max_sents=2)

        # Should create chunks with 2 sentences each
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "This is sentence one. This is sentence two.")
        self.assertEqual(chunks[1], "This is sentence three. This is sentence four.")
        self.assertEqual(chunks[2], "This is sentence five.")

    def test_chunk_by_sentences_empty(self):
        """Test chunking empty text."""
        text = ""
        chunks = chunk_by_sentences(text)

        # Should return empty list
        self.assertEqual(chunks, [])

    def test_chunk_text(self):
        """Test chunk_text function."""
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = chunk_text(text, max_sents=3, source_name="test_source")

        # Should return list of dictionaries with chunk text and metadata
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_text"], text)
        self.assertEqual(chunks[0]["source"], "test_source")
        self.assertEqual(chunks[0]["chunk_index"], 0)
        self.assertEqual(chunks[0]["total_chunks"], 1)

    def test_chunk_text_multiple(self):
        """Test chunk_text function with multiple chunks."""
        text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        chunks = chunk_text(text, max_sents=2, source_name="test_source")

        # Should return list of dictionaries with chunk text and metadata
        self.assertEqual(len(chunks), 2)
        self.assertEqual(
            chunks[0]["chunk_text"], "This is sentence one. This is sentence two."
        )
        self.assertEqual(chunks[0]["source"], "test_source")
        self.assertEqual(chunks[0]["chunk_index"], 0)
        self.assertEqual(chunks[0]["total_chunks"], 2)

        self.assertEqual(
            chunks[1]["chunk_text"], "This is sentence three. This is sentence four."
        )
        self.assertEqual(chunks[1]["source"], "test_source")
        self.assertEqual(chunks[1]["chunk_index"], 1)
        self.assertEqual(chunks[1]["total_chunks"], 2)

    @patch("os.listdir")
    @patch("os.path.isfile")
    def test_list_text_files(self, mock_isfile, mock_listdir):
        """Test listing text files."""
        # Setup mocks
        mock_listdir.return_value = [
            "file1.txt",
            "file2.md",
            "file3.py",
            "file4.jpg",
            "file5",
        ]
        mock_isfile.side_effect = lambda x: x != "./file5"

        # Call function
        files = list_text_files()

        # Should return list of text files
        self.assertEqual(len(files), 3)
        self.assertIn("./file1.txt", files)
        self.assertIn("./file2.md", files)
        self.assertIn("./file3.py", files)
        self.assertNotIn("./file4.jpg", files)
        self.assertNotIn("./file5", files)

    @patch("os.listdir")
    def test_list_text_files_error(self, mock_listdir):
        """Test error handling in list_text_files."""
        # Setup mock to raise exception
        mock_listdir.side_effect = Exception("Test error")

        # Call function
        files = list_text_files()

        # Should return empty list on error
        self.assertEqual(files, [])

    @patch("builtins.open", new_callable=mock_open, read_data="Test file content")
    def test_read_file_content(self, mock_file):
        """Test reading file content."""
        # Call function
        content = read_file_content("test_file.txt")

        # Should return file content
        self.assertEqual(content, "Test file content")
        mock_file.assert_called_once_with("test_file.txt", "r", encoding="utf-8")

    @patch("builtins.open")
    def test_read_file_content_error(self, mock_file):
        """Test error handling in read_file_content."""
        # Setup mock to raise exception
        mock_file.side_effect = Exception("Test error")

        # Call function
        content = read_file_content("test_file.txt")

        # Should return None on error
        self.assertIsNone(content)

    @patch("vector_chat.services.chunker.read_file_content")
    @patch("vector_chat.services.chunker.chunk_text")
    def test_process_file(self, mock_chunk_text, mock_read_file_content):
        """Test processing a file."""
        # Setup mocks
        mock_read_file_content.return_value = "Test file content"
        mock_chunk_text.return_value = [
            {
                "chunk_text": "Test file content",
                "source": "test_file.txt",
                "chunk_index": 0,
                "total_chunks": 1,
            }
        ]

        # Call function
        chunks = process_file("test_file.txt")

        # Should return chunks from chunk_text
        self.assertEqual(chunks, mock_chunk_text.return_value)
        mock_read_file_content.assert_called_once_with("test_file.txt")
        mock_chunk_text.assert_called_once_with(
            "Test file content", DEFAULT_MAX_SENTENCES_PER_CHUNK, "test_file.txt"
        )

    @patch("vector_chat.services.chunker.read_file_content")
    def test_process_file_no_content(self, mock_read_file_content):
        """Test processing a file with no content."""
        # Setup mock to return None
        mock_read_file_content.return_value = None

        # Call function
        chunks = process_file("test_file.txt")

        # Should return empty list
        self.assertEqual(chunks, [])
