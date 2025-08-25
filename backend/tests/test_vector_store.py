import os
import sys
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models import Course, CourseChunk, Lesson
from vector_store import SearchResults, VectorStore


class TestVectorStore(unittest.TestCase):
    """Test suite for VectorStore, especially focusing on MAX_RESULTS=0 bug"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a mock ChromaDB client and collections
        self.mock_client = Mock()
        self.mock_catalog = Mock()
        self.mock_content = Mock()

        # Setup collections
        self.mock_client.get_or_create_collection.side_effect = [
            self.mock_catalog,
            self.mock_content,
        ]

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_search_with_max_results_zero(self, mock_embedding_func, mock_client_class):
        """Test the critical bug where MAX_RESULTS=0 prevents any results"""
        # Setup mocks
        mock_client_class.return_value = self.mock_client

        # Create VectorStore with MAX_RESULTS=0 (the bug)
        vector_store = VectorStore(
            chroma_path="test_path",
            embedding_model="test_model",
            max_results=0,  # This is the bug we're testing
        )
        vector_store.course_catalog = self.mock_catalog
        vector_store.course_content = self.mock_content

        # Mock search results
        self.mock_content.query.return_value = {
            "documents": [["Result 1", "Result 2"]],
            "metadatas": [
                [
                    {"course_title": "Test Course", "lesson_number": 1},
                    {"course_title": "Test Course", "lesson_number": 2},
                ]
            ],
            "distances": [[0.1, 0.2]],
        }

        # Execute search
        results = vector_store.search("test query")

        # Verify search was called with n_results=0 (the bug!)
        self.mock_content.query.assert_called_once()
        call_args = self.mock_content.query.call_args
        self.assertEqual(call_args[1]["n_results"], 0)  # This is the problem!

        # With MAX_RESULTS=0, ChromaDB returns no results even if data exists
        # This test demonstrates the bug

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_search_with_proper_max_results(
        self, mock_embedding_func, mock_client_class
    ):
        """Test search with proper MAX_RESULTS value"""
        # Setup mocks
        mock_client_class.return_value = self.mock_client

        # Create VectorStore with proper MAX_RESULTS value
        vector_store = VectorStore(
            chroma_path="test_path",
            embedding_model="test_model",
            max_results=5,  # Proper value
        )
        vector_store.course_catalog = self.mock_catalog
        vector_store.course_content = self.mock_content

        # Mock search results
        self.mock_content.query.return_value = {
            "documents": [["Result 1", "Result 2"]],
            "metadatas": [
                [
                    {"course_title": "Test Course", "lesson_number": 1},
                    {"course_title": "Test Course", "lesson_number": 2},
                ]
            ],
            "distances": [[0.1, 0.2]],
        }

        # Execute search
        results = vector_store.search("test query")

        # Verify search was called with correct n_results
        self.mock_content.query.assert_called_once()
        call_args = self.mock_content.query.call_args
        self.assertEqual(call_args[1]["n_results"], 5)

        # Verify results are returned properly
        self.assertEqual(len(results.documents), 2)
        self.assertEqual(results.documents[0], "Result 1")
        self.assertEqual(results.documents[1], "Result 2")

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_search_with_limit_override(self, mock_embedding_func, mock_client_class):
        """Test search with explicit limit parameter overriding max_results"""
        # Setup mocks
        mock_client_class.return_value = self.mock_client

        # Create VectorStore with MAX_RESULTS=0 (the bug)
        vector_store = VectorStore(
            chroma_path="test_path",
            embedding_model="test_model",
            max_results=0,  # Bug value
        )
        vector_store.course_catalog = self.mock_catalog
        vector_store.course_content = self.mock_content

        # Mock search results
        self.mock_content.query.return_value = {
            "documents": [["Result 1"]],
            "metadatas": [[{"course_title": "Test Course", "lesson_number": 1}]],
            "distances": [[0.1]],
        }

        # Execute search with explicit limit (should override MAX_RESULTS=0)
        results = vector_store.search("test query", limit=3)

        # Verify search was called with the explicit limit, not MAX_RESULTS
        self.mock_content.query.assert_called_once()
        call_args = self.mock_content.query.call_args
        self.assertEqual(call_args[1]["n_results"], 3)

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_course_name_resolution(self, mock_embedding_func, mock_client_class):
        """Test course name resolution functionality"""
        # Setup mocks
        mock_client_class.return_value = self.mock_client

        vector_store = VectorStore("test_path", "test_model", max_results=5)
        vector_store.course_catalog = self.mock_catalog
        vector_store.course_content = self.mock_content

        # Mock course catalog search for name resolution
        self.mock_catalog.query.return_value = {
            "documents": [["Full Course Title"]],
            "metadatas": [[{"title": "Full Course Title"}]],
        }

        # Mock content search
        self.mock_content.query.return_value = {
            "documents": [["Course content"]],
            "metadatas": [[{"course_title": "Full Course Title", "lesson_number": 1}]],
            "distances": [[0.1]],
        }

        # Execute search with course name
        results = vector_store.search("test query", course_name="Partial Name")

        # Verify course catalog was queried for name resolution
        self.mock_catalog.query.assert_called_once_with(
            query_texts=["Partial Name"], n_results=1
        )

        # Verify content search was called with resolved course name
        self.mock_content.query.assert_called_once()
        call_args = self.mock_content.query.call_args
        expected_filter = {"course_title": "Full Course Title"}
        self.assertEqual(call_args[1]["where"], expected_filter)

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_course_name_resolution_failure(
        self, mock_embedding_func, mock_client_class
    ):
        """Test handling when course name resolution fails"""
        # Setup mocks
        mock_client_class.return_value = self.mock_client

        vector_store = VectorStore("test_path", "test_model", max_results=5)
        vector_store.course_catalog = self.mock_catalog

        # Mock course catalog search returning no results
        self.mock_catalog.query.return_value = {"documents": [[]], "metadatas": [[]]}

        # Execute search with non-existent course name
        results = vector_store.search("test query", course_name="Nonexistent Course")

        # Verify error is returned
        self.assertEqual(results.error, "No course found matching 'Nonexistent Course'")
        self.assertTrue(results.is_empty())

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_search_with_filters(self, mock_embedding_func, mock_client_class):
        """Test search with various filter combinations"""
        # Setup mocks
        mock_client_class.return_value = self.mock_client

        vector_store = VectorStore("test_path", "test_model", max_results=5)
        vector_store.course_catalog = self.mock_catalog
        vector_store.course_content = self.mock_content

        # Mock successful course resolution
        self.mock_catalog.query.return_value = {
            "documents": [["Test Course"]],
            "metadatas": [[{"title": "Test Course"}]],
        }

        # Mock content search
        self.mock_content.query.return_value = {
            "documents": [["Filtered content"]],
            "metadatas": [[{"course_title": "Test Course", "lesson_number": 3}]],
            "distances": [[0.1]],
        }

        # Test course + lesson filter
        results = vector_store.search(
            "test query", course_name="Test Course", lesson_number=3
        )

        call_args = self.mock_content.query.call_args
        expected_filter = {
            "$and": [{"course_title": "Test Course"}, {"lesson_number": 3}]
        }
        self.assertEqual(call_args[1]["where"], expected_filter)

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_search_exception_handling(self, mock_embedding_func, mock_client_class):
        """Test handling of search exceptions"""
        # Setup mocks
        mock_client_class.return_value = self.mock_client

        vector_store = VectorStore("test_path", "test_model", max_results=5)
        vector_store.course_content = self.mock_content

        # Mock content search to raise exception
        self.mock_content.query.side_effect = Exception("Database error")

        # Execute search
        results = vector_store.search("test query")

        # Verify error is handled and returned
        self.assertEqual(results.error, "Search error: Database error")
        self.assertTrue(results.is_empty())

    @patch("vector_store.chromadb.PersistentClient")
    @patch(
        "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    )
    def test_add_course_content(self, mock_embedding_func, mock_client_class):
        """Test adding course content chunks"""
        # Setup mocks
        mock_client_class.return_value = self.mock_client

        vector_store = VectorStore("test_path", "test_model", max_results=5)
        vector_store.course_content = self.mock_content

        # Create test chunks
        chunks = [
            CourseChunk(
                content="Chunk 1 content",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0,
            ),
            CourseChunk(
                content="Chunk 2 content",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=1,
            ),
        ]

        # Add chunks
        vector_store.add_course_content(chunks)

        # Verify add was called with correct parameters
        self.mock_content.add.assert_called_once()
        call_args = self.mock_content.add.call_args

        # Check documents
        expected_docs = ["Chunk 1 content", "Chunk 2 content"]
        self.assertEqual(call_args[1]["documents"], expected_docs)

        # Check metadatas
        expected_metadata = [
            {"course_title": "Test Course", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "Test Course", "lesson_number": 1, "chunk_index": 1},
        ]
        self.assertEqual(call_args[1]["metadatas"], expected_metadata)

        # Check IDs
        expected_ids = ["Test_Course_0", "Test_Course_1"]
        self.assertEqual(call_args[1]["ids"], expected_ids)

    def test_search_results_class(self):
        """Test SearchResults utility class"""
        # Test from_chroma class method
        chroma_results = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [["meta1", "meta2"]],
            "distances": [[0.1, 0.2]],
        }

        results = SearchResults.from_chroma(chroma_results)
        self.assertEqual(results.documents, ["doc1", "doc2"])
        self.assertEqual(results.metadata, ["meta1", "meta2"])
        self.assertEqual(results.distances, [0.1, 0.2])
        self.assertIsNone(results.error)

        # Test empty results
        empty_results = SearchResults.empty("Test error")
        self.assertEqual(empty_results.error, "Test error")
        self.assertTrue(empty_results.is_empty())

        # Test is_empty method
        non_empty_results = SearchResults(["doc"], ["meta"], [0.1])
        self.assertFalse(non_empty_results.is_empty())


if __name__ == "__main__":
    unittest.main()
