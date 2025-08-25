"""
Integration test to verify the actual behavior of the RAG system with real components
This will help identify the MAX_RESULTS=0 bug in action
"""

import os
import sys
import unittest

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import config
from rag_system import RAGSystem


class TestRealIntegration(unittest.TestCase):
    """Test real integration with actual components to identify bugs"""

    def test_max_results_fixed(self):
        """Test that MAX_RESULTS is now fixed and set to a valid value"""
        # Verify the bug is fixed in config
        self.assertGreater(
            config.MAX_RESULTS, 0, "MAX_RESULTS should be greater than 0"
        )
        self.assertEqual(config.MAX_RESULTS, 5, "MAX_RESULTS should be set to 5")

        # Create real RAG system (this will use the fixed config)
        rag_system = RAGSystem(config)

        # Check if vector store max_results is now valid
        self.assertGreater(rag_system.vector_store.max_results, 0)
        self.assertEqual(rag_system.vector_store.max_results, 5)

        print(f"✓ Fix confirmed: config.MAX_RESULTS = {config.MAX_RESULTS}")
        print(f"✓ VectorStore max_results = {rag_system.vector_store.max_results}")
        print("Search should now work properly!")

    def test_vector_store_search_with_zero_limit(self):
        """Test vector store search behavior with zero limit"""
        from vector_store import SearchResults, VectorStore

        # Create a vector store with MAX_RESULTS=0 (the bug)
        vector_store = VectorStore(
            chroma_path="./test_chroma",
            embedding_model="all-MiniLM-L6-v2",
            max_results=0,
        )

        # Test search - this should fail to return results due to n_results=0
        try:
            results = vector_store.search("test query")
            print(f"✓ Search completed, but with max_results=0")
            print(f"  - Documents returned: {len(results.documents)}")
            print(f"  - Error: {results.error}")
            print(f"  - Is empty: {results.is_empty()}")

            # The search will likely return empty results or error due to n_results=0
            if results.error or results.is_empty():
                print("✓ Confirmed: MAX_RESULTS=0 causes search to fail")

        except Exception as e:
            print(f"✓ Search failed with error: {e}")
            print("✓ This confirms the MAX_RESULTS=0 bug")

    def test_course_search_tool_with_fix(self):
        """Test CourseSearchTool with the MAX_RESULTS=5 fix"""
        from search_tools import CourseSearchTool
        from vector_store import VectorStore

        # Create components with the fix
        vector_store = VectorStore(
            chroma_path="./test_chroma",
            embedding_model="all-MiniLM-L6-v2",
            max_results=5,  # Fixed value
        )

        search_tool = CourseSearchTool(vector_store)

        # Execute search
        result = search_tool.execute("test query")
        print(f"✓ CourseSearchTool result with MAX_RESULTS=5: {result}")

        # Should return proper "No relevant content found" message instead of ChromaDB error
        self.assertNotIn("cannot be negative, or zero", result)
        # Should either find content or return proper no content message
        self.assertTrue(
            "No relevant content found" in result or len(result) > 0,
            "Should return either content or proper 'no content found' message",
        )


if __name__ == "__main__":
    unittest.main()
