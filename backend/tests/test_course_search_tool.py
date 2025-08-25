import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from search_tools import CourseSearchTool
from vector_store import SearchResults, VectorStore


class TestCourseSearchTool(unittest.TestCase):
    """Test suite for CourseSearchTool.execute() method"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a mock vector store
        self.mock_vector_store = Mock(spec=VectorStore)
        self.search_tool = CourseSearchTool(self.mock_vector_store)

    def test_execute_basic_query(self):
        """Test basic query execution with successful results"""
        # Mock search results
        mock_results = SearchResults(
            documents=["Course content about MCP"],
            metadata=[{"course_title": "Introduction to MCP", "lesson_number": 1}],
            distances=[0.1],
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = (
            "https://example.com/lesson1"
        )

        # Execute search
        result = self.search_tool.execute("MCP basics")

        # Verify search was called with correct parameters
        self.mock_vector_store.search.assert_called_once_with(
            query="MCP basics", course_name=None, lesson_number=None
        )

        # Verify result formatting
        self.assertIn("[Introduction to MCP - Lesson 1]", result)
        self.assertIn("Course content about MCP", result)

        # Verify sources are tracked
        self.assertEqual(len(self.search_tool.last_sources), 1)
        self.assertEqual(
            self.search_tool.last_sources[0]["text"], "Introduction to MCP - Lesson 1"
        )
        self.assertEqual(
            self.search_tool.last_sources[0]["link"], "https://example.com/lesson1"
        )

    def test_execute_with_course_filter(self):
        """Test query execution with course name filter"""
        mock_results = SearchResults(
            documents=["Specific course content"],
            metadata=[{"course_title": "Advanced MCP", "lesson_number": 2}],
            distances=[0.15],
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None

        # Execute search with course filter
        result = self.search_tool.execute("advanced topics", course_name="Advanced MCP")

        # Verify search called with course filter
        self.mock_vector_store.search.assert_called_once_with(
            query="advanced topics", course_name="Advanced MCP", lesson_number=None
        )

        # Verify result contains course context
        self.assertIn("[Advanced MCP - Lesson 2]", result)
        self.assertIn("Specific course content", result)

    def test_execute_with_lesson_filter(self):
        """Test query execution with lesson number filter"""
        mock_results = SearchResults(
            documents=["Lesson-specific content"],
            metadata=[{"course_title": "MCP Fundamentals", "lesson_number": 3}],
            distances=[0.2],
        )
        self.mock_vector_store.search.return_value = mock_results

        # Execute search with lesson filter
        result = self.search_tool.execute("lesson content", lesson_number=3)

        # Verify search called with lesson filter
        self.mock_vector_store.search.assert_called_once_with(
            query="lesson content", course_name=None, lesson_number=3
        )

    def test_execute_with_both_filters(self):
        """Test query execution with both course and lesson filters"""
        mock_results = SearchResults(
            documents=["Filtered content"],
            metadata=[{"course_title": "Specific Course", "lesson_number": 5}],
            distances=[0.1],
        )
        self.mock_vector_store.search.return_value = mock_results

        # Execute search with both filters
        result = self.search_tool.execute(
            "specific query", course_name="Specific Course", lesson_number=5
        )

        # Verify search called with both filters
        self.mock_vector_store.search.assert_called_once_with(
            query="specific query", course_name="Specific Course", lesson_number=5
        )

    def test_execute_with_error_results(self):
        """Test handling of search errors"""
        # Mock error results
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error="Database connection failed"
        )
        self.mock_vector_store.search.return_value = mock_results

        # Execute search
        result = self.search_tool.execute("any query")

        # Verify error is returned
        self.assertEqual(result, "Database connection failed")

    def test_execute_with_empty_results(self):
        """Test handling of empty search results"""
        # Mock empty results
        mock_results = SearchResults(documents=[], metadata=[], distances=[])
        self.mock_vector_store.search.return_value = mock_results

        # Execute search without filters
        result = self.search_tool.execute("nonexistent content")
        self.assertEqual(result, "No relevant content found.")

        # Execute search with course filter
        result = self.search_tool.execute("nonexistent", course_name="Missing Course")
        self.assertEqual(
            result, "No relevant content found in course 'Missing Course'."
        )

        # Execute search with lesson filter
        result = self.search_tool.execute("nonexistent", lesson_number=99)
        self.assertEqual(result, "No relevant content found in lesson 99.")

        # Execute search with both filters
        result = self.search_tool.execute(
            "nonexistent", course_name="Missing", lesson_number=99
        )
        self.assertEqual(
            result, "No relevant content found in course 'Missing' in lesson 99."
        )

    def test_execute_multiple_results(self):
        """Test execution with multiple search results"""
        mock_results = SearchResults(
            documents=["First result content", "Second result content"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None

        # Execute search
        result = self.search_tool.execute("multiple results")

        # Verify both results are formatted
        self.assertIn("[Course A - Lesson 1]", result)
        self.assertIn("First result content", result)
        self.assertIn("[Course B - Lesson 2]", result)
        self.assertIn("Second result content", result)

        # Verify multiple sources are tracked
        self.assertEqual(len(self.search_tool.last_sources), 2)

    def test_execute_without_lesson_number(self):
        """Test execution with metadata missing lesson number"""
        mock_results = SearchResults(
            documents=["Content without lesson"],
            metadata=[{"course_title": "General Course"}],  # No lesson_number
            distances=[0.1],
        )
        self.mock_vector_store.search.return_value = mock_results

        # Execute search
        result = self.search_tool.execute("general content")

        # Verify result formatting without lesson number
        self.assertIn("[General Course]", result)
        self.assertNotIn("Lesson", result)

    def test_get_tool_definition(self):
        """Test tool definition is correctly formatted"""
        definition = self.search_tool.get_tool_definition()

        # Verify required fields
        self.assertEqual(definition["name"], "search_course_content")
        self.assertIn("description", definition)
        self.assertIn("input_schema", definition)

        # Verify schema structure
        schema = definition["input_schema"]
        self.assertEqual(schema["type"], "object")
        self.assertIn("query", schema["properties"])
        self.assertIn("course_name", schema["properties"])
        self.assertIn("lesson_number", schema["properties"])
        self.assertEqual(schema["required"], ["query"])

    def test_sources_tracking_and_reset(self):
        """Test that sources are properly tracked and can be reset"""
        mock_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
        )
        self.mock_vector_store.search.return_value = mock_results

        # Execute search
        self.search_tool.execute("test query")

        # Verify sources are tracked
        self.assertEqual(len(self.search_tool.last_sources), 1)

        # Reset sources
        self.search_tool.last_sources = []
        self.assertEqual(len(self.search_tool.last_sources), 0)


if __name__ == "__main__":
    unittest.main()
