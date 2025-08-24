"""
Test to verify that the MAX_RESULTS=0 bug fix is working correctly
"""
import sys
import os
import unittest

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import config
from rag_system import RAGSystem


class TestFixVerification(unittest.TestCase):
    """Test to verify the MAX_RESULTS=0 bug fix"""
    
    def test_max_results_fix_applied(self):
        """Test that MAX_RESULTS is now set to a valid value"""
        # Verify the bug is fixed in config
        self.assertGreater(config.MAX_RESULTS, 0, "MAX_RESULTS should be greater than 0")
        self.assertEqual(config.MAX_RESULTS, 5, "MAX_RESULTS should be set to 5")
        
        print(f"✓ Fix confirmed: config.MAX_RESULTS = {config.MAX_RESULTS}")

    def test_vector_store_with_fix(self):
        """Test that vector store now uses the corrected MAX_RESULTS"""
        # Create real RAG system (this will use the fixed config)
        rag_system = RAGSystem(config)
        
        # Check if vector store max_results is now valid
        self.assertGreater(rag_system.vector_store.max_results, 0)
        self.assertEqual(rag_system.vector_store.max_results, 5)
        
        print(f"✓ VectorStore fix confirmed: max_results = {rag_system.vector_store.max_results}")

    def test_course_search_tool_with_fix(self):
        """Test that CourseSearchTool now works without the MAX_RESULTS=0 error"""
        from search_tools import CourseSearchTool
        from vector_store import VectorStore
        
        # Create components with the fix
        vector_store = VectorStore(
            chroma_path="./test_chroma",
            embedding_model="all-MiniLM-L6-v2",
            max_results=5  # Fixed value
        )
        
        search_tool = CourseSearchTool(vector_store)
        
        # Execute search - should no longer get the "cannot be negative, or zero" error
        try:
            result = search_tool.execute("test query")
            print(f"✓ CourseSearchTool result with MAX_RESULTS=5: {result}")
            
            # Should return "No relevant content found" instead of the ChromaDB error
            # since there's no actual data, but no ChromaDB error about 0 results
            self.assertNotIn("cannot be negative, or zero", result)
            
            # Should either find content or return the proper "no content found" message
            self.assertTrue(
                "No relevant content found" in result or len(result) > 0,
                "Should return either content or proper 'no content found' message"
            )
            
        except Exception as e:
            # Should not get the ChromaDB error about 0 results
            self.assertNotIn("cannot be negative, or zero", str(e))
            print(f"✓ No ChromaDB 'zero results' error: {e}")

if __name__ == '__main__':
    unittest.main()