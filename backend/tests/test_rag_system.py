import os
import sys
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ai_generator import AIGenerator
from document_processor import DocumentProcessor
from models import Course, CourseChunk, Lesson
from rag_system import RAGSystem
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from session_manager import SessionManager
from vector_store import SearchResults, VectorStore


class MockConfig:
    """Mock configuration for testing"""

    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    CHROMA_PATH = "./test_chroma_db"
    EMBEDDING_MODEL = "test_model"
    MAX_RESULTS = 0  # The bug we're testing
    ANTHROPIC_API_KEY = "test_key"
    ANTHROPIC_MODEL = "test_model"
    MAX_HISTORY = 2


class TestRAGSystem(unittest.TestCase):
    """Test suite for RAGSystem end-to-end functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = MockConfig()

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    def test_rag_system_initialization(
        self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc
    ):
        """Test RAG system initialization with all components"""
        # Initialize RAG system
        rag_system = RAGSystem(self.mock_config)

        # Verify all components were initialized
        mock_doc_proc.assert_called_once_with(800, 100)
        mock_vector_store.assert_called_once_with("./test_chroma_db", "test_model", 0)
        mock_ai_gen.assert_called_once_with("test_key", "test_model")
        mock_session_mgr.assert_called_once_with(2)

        # Verify tools are registered
        self.assertIsInstance(rag_system.tool_manager, ToolManager)
        self.assertIsInstance(rag_system.search_tool, CourseSearchTool)
        self.assertIsInstance(rag_system.outline_tool, CourseOutlineTool)

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    def test_query_processing_with_tool_usage(
        self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc
    ):
        """Test end-to-end query processing with tool usage"""
        # Setup mocks
        mock_ai_generator = Mock()
        mock_ai_gen.return_value = mock_ai_generator
        mock_ai_generator.generate_response.return_value = (
            "AI response with tool results"
        )

        mock_tool_manager = Mock()
        mock_tool_manager.get_last_sources.return_value = [
            {"text": "Test Course - Lesson 1", "link": None}
        ]

        # Initialize RAG system
        rag_system = RAGSystem(self.mock_config)
        rag_system.tool_manager = mock_tool_manager  # Replace with mock

        # Execute query
        response, sources = rag_system.query("What is MCP?")

        # Verify AI generator was called with correct parameters
        mock_ai_generator.generate_response.assert_called_once()
        call_args = mock_ai_generator.generate_response.call_args

        self.assertIn("What is MCP?", call_args[1]["query"])  # Query in keyword args
        self.assertEqual(call_args[1]["conversation_history"], None)
        self.assertIsNotNone(call_args[1]["tools"])  # Tools should be provided
        self.assertEqual(call_args[1]["tool_manager"], mock_tool_manager)

        # Verify response and sources
        self.assertEqual(response, "AI response with tool results")
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["text"], "Test Course - Lesson 1")

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    def test_query_processing_with_session_history(
        self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc
    ):
        """Test query processing with conversation history"""
        # Setup mocks
        mock_ai_generator = Mock()
        mock_ai_gen.return_value = mock_ai_generator
        mock_ai_generator.generate_response.return_value = (
            "Response with history context"
        )

        mock_session_manager = Mock()
        mock_session_mgr.return_value = mock_session_manager
        mock_session_manager.get_conversation_history.return_value = (
            "Previous conversation context"
        )

        mock_tool_manager = Mock()
        mock_tool_manager.get_last_sources.return_value = []

        # Initialize RAG system
        rag_system = RAGSystem(self.mock_config)
        rag_system.tool_manager = mock_tool_manager

        # Execute query with session ID
        response, sources = rag_system.query(
            "Follow-up question", session_id="test_session"
        )

        # Verify session history was retrieved
        mock_session_manager.get_conversation_history.assert_called_once_with(
            "test_session"
        )

        # Verify AI generator received history
        mock_ai_generator.generate_response.assert_called_once()
        call_args = mock_ai_generator.generate_response.call_args
        self.assertEqual(
            call_args[1]["conversation_history"], "Previous conversation context"
        )

        # Verify session was updated
        mock_session_manager.add_exchange.assert_called_once_with(
            "test_session", "Follow-up question", "Response with history context"
        )

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    def test_query_processing_max_results_zero_bug(
        self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc
    ):
        """Test that MAX_RESULTS=0 bug affects query processing"""
        # Setup mocks
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance

        # Simulate the MAX_RESULTS=0 bug - search returns empty results
        mock_search_results = SearchResults.empty("No results due to MAX_RESULTS=0")

        mock_ai_generator = Mock()
        mock_ai_gen.return_value = mock_ai_generator
        mock_ai_generator.generate_response.return_value = (
            "I couldn't find any relevant information"
        )

        # Initialize RAG system with bug
        rag_system = RAGSystem(self.mock_config)

        # Create a real CourseSearchTool that will be affected by the bug
        rag_system.search_tool = CourseSearchTool(mock_vector_store_instance)
        mock_vector_store_instance.search.return_value = mock_search_results

        # Register the tool
        rag_system.tool_manager = ToolManager()
        rag_system.tool_manager.register_tool(rag_system.search_tool)

        # Mock AI to simulate tool usage
        def mock_generate_response(
            query, conversation_history=None, tools=None, tool_manager=None
        ):
            if tools and tool_manager:
                # Simulate AI deciding to use the search tool
                tool_result = tool_manager.execute_tool(
                    "search_course_content", query="MCP basics"
                )
                return f"Based on search: {tool_result}"
            return "No tools used"

        mock_ai_generator.generate_response = mock_generate_response

        # Execute query that should trigger search
        response, sources = rag_system.query("What is MCP?")

        # Verify the bug affects the response
        self.assertIn("No results due to MAX_RESULTS=0", response)

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    def test_add_course_document(
        self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc
    ):
        """Test adding a single course document"""
        # Setup mocks
        mock_document_processor = Mock()
        mock_doc_proc.return_value = mock_document_processor

        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance

        # Mock course and chunks
        mock_course = Course(title="Test Course", lessons=[])
        mock_chunks = [
            CourseChunk(content="Chunk 1", course_title="Test Course", chunk_index=0),
            CourseChunk(content="Chunk 2", course_title="Test Course", chunk_index=1),
        ]

        mock_document_processor.process_course_document.return_value = (
            mock_course,
            mock_chunks,
        )

        # Initialize RAG system
        rag_system = RAGSystem(self.mock_config)

        # Add course document
        course, chunk_count = rag_system.add_course_document("test_file.pdf")

        # Verify document processing
        mock_document_processor.process_course_document.assert_called_once_with(
            "test_file.pdf"
        )

        # Verify vector store operations
        mock_vector_store_instance.add_course_metadata.assert_called_once_with(
            mock_course
        )
        mock_vector_store_instance.add_course_content.assert_called_once_with(
            mock_chunks
        )

        # Verify return values
        self.assertEqual(course, mock_course)
        self.assertEqual(chunk_count, 2)

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    @patch("rag_system.os.path.exists")
    @patch("rag_system.os.listdir")
    @patch("rag_system.os.path.isfile")
    def test_add_course_folder(
        self,
        mock_isfile,
        mock_listdir,
        mock_exists,
        mock_session_mgr,
        mock_ai_gen,
        mock_vector_store,
        mock_doc_proc,
    ):
        """Test adding multiple course documents from folder"""
        # Setup mocks
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.pdf", "course2.pdf"]
        mock_isfile.return_value = True  # All paths are files

        mock_document_processor = Mock()
        mock_doc_proc.return_value = mock_document_processor

        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance
        mock_vector_store_instance.get_existing_course_titles.return_value = []

        # Mock course processing
        mock_course1 = Course(title="Course 1", lessons=[])
        mock_course2 = Course(title="Course 2", lessons=[])
        mock_chunks1 = [
            CourseChunk(content="Content 1", course_title="Course 1", chunk_index=0)
        ]
        mock_chunks2 = [
            CourseChunk(content="Content 2", course_title="Course 2", chunk_index=0)
        ]

        mock_document_processor.process_course_document.side_effect = [
            (mock_course1, mock_chunks1),
            (mock_course2, mock_chunks2),
        ]

        # Initialize RAG system
        rag_system = RAGSystem(self.mock_config)

        # Add course folder
        total_courses, total_chunks = rag_system.add_course_folder("test_folder")

        # Verify processing
        self.assertEqual(mock_document_processor.process_course_document.call_count, 2)
        self.assertEqual(total_courses, 2)
        self.assertEqual(total_chunks, 2)

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    def test_get_course_analytics(
        self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc
    ):
        """Test getting course analytics"""
        # Setup mocks
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance
        mock_vector_store_instance.get_course_count.return_value = 3
        mock_vector_store_instance.get_existing_course_titles.return_value = [
            "Course A",
            "Course B",
            "Course C",
        ]

        # Initialize RAG system
        rag_system = RAGSystem(self.mock_config)

        # Get analytics
        analytics = rag_system.get_course_analytics()

        # Verify analytics
        self.assertEqual(analytics["total_courses"], 3)
        self.assertEqual(len(analytics["course_titles"]), 3)
        self.assertIn("Course A", analytics["course_titles"])

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    def test_error_handling_in_query(
        self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc
    ):
        """Test error handling in query processing"""
        # Setup mocks
        mock_ai_generator = Mock()
        mock_ai_gen.return_value = mock_ai_generator
        mock_ai_generator.generate_response.side_effect = Exception("AI API error")

        # Initialize RAG system
        rag_system = RAGSystem(self.mock_config)

        # This should raise an exception as there's no error handling in the query method
        with self.assertRaises(Exception) as context:
            rag_system.query("Test query")

        self.assertIn("AI API error", str(context.exception))

    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.SessionManager")
    def test_sources_reset_after_query(
        self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_proc
    ):
        """Test that sources are properly reset after each query"""
        # Setup mocks
        mock_ai_generator = Mock()
        mock_ai_gen.return_value = mock_ai_generator
        mock_ai_generator.generate_response.return_value = "Response"

        mock_tool_manager = Mock()
        mock_tool_manager.get_last_sources.return_value = [
            {"text": "Source 1", "link": None}
        ]

        # Initialize RAG system
        rag_system = RAGSystem(self.mock_config)
        rag_system.tool_manager = mock_tool_manager

        # Execute query
        response, sources = rag_system.query("Test query")

        # Verify sources were retrieved and then reset
        mock_tool_manager.get_last_sources.assert_called_once()
        mock_tool_manager.reset_sources.assert_called_once()


if __name__ == "__main__":
    unittest.main()
