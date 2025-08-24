"""
Demonstration of sequential tool calling functionality
This test shows how the new sequential tool calling works with realistic scenarios
"""
import sys
import os
import unittest
from unittest.mock import Mock, patch

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator
from tests.test_ai_generator import MockResponse, MockToolUseBlock


class TestSequentialToolCallingDemo(unittest.TestCase):
    """Demonstration tests for sequential tool calling"""
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_course_comparison_scenario(self, mock_anthropic_class):
        """Demonstrate: 'Compare lesson 3 of Course A with lesson 5 of Course B'"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Round 1: Search for Course A, Lesson 3
        round1_tool = MockToolUseBlock(
            tool_name="search_course_content",
            tool_input={"query": "lesson content", "course_name": "Course A", "lesson_number": 3}
        )
        round1_response = MockResponse(stop_reason="tool_use", tool_calls=[round1_tool])
        
        # Round 2: Search for Course B, Lesson 5  
        round2_tool = MockToolUseBlock(
            tool_name="search_course_content",
            tool_input={"query": "lesson content", "course_name": "Course B", "lesson_number": 5}
        )
        round2_response = MockResponse(stop_reason="tool_use", tool_calls=[round2_tool])
        
        # Final response: Comparison
        final_response = MockResponse("Comparing the two lessons: Course A Lesson 3 focuses on X while Course B Lesson 5 emphasizes Y...")
        
        # Setup API call sequence
        mock_client.messages.create.side_effect = [round1_response, round2_response, final_response]
        
        # Mock tools and tool manager
        generator = AIGenerator(api_key="test_key", model="test_model") 
        mock_tools = [{"name": "search_course_content", "description": "Search course content"}]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course A Lesson 3: Advanced algorithms and data structures...",
            "Course B Lesson 5: Machine learning fundamentals and applications..."
        ]
        
        # Execute the scenario
        response = generator.generate_response(
            query="Compare lesson 3 of Course A with lesson 5 of Course B",
            tools=mock_tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify sequential execution
        print(f"Query: Compare lesson 3 of Course A with lesson 5 of Course B")
        print(f"Round 1: Searched Course A, Lesson 3")
        print(f"Round 2: Searched Course B, Lesson 5") 
        print(f"Final Response: {response}")
        
        # Verify both searches were executed
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        self.assertEqual(mock_client.messages.create.call_count, 3)

    @patch('ai_generator.anthropic.Anthropic')
    def test_topic_exploration_scenario(self, mock_anthropic_class):
        """Demonstrate: 'Find courses that discuss the same topic as lesson 4 of MCP Course'"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Round 1: Get outline to find lesson 4 topic
        round1_tool = MockToolUseBlock(
            tool_name="get_course_outline",
            tool_input={"course_title": "MCP Course"}
        )
        round1_response = MockResponse(stop_reason="tool_use", tool_calls=[round1_tool])
        
        # Round 2: Search for courses with similar topics
        round2_tool = MockToolUseBlock(
            tool_name="search_course_content", 
            tool_input={"query": "advanced MCP patterns"}
        )
        round2_response = MockResponse(stop_reason="tool_use", tool_calls=[round2_tool])
        
        # Final response: Related courses
        final_response = MockResponse("Based on lesson 4's topic 'Advanced MCP Patterns', here are related courses...")
        
        # Setup API call sequence
        mock_client.messages.create.side_effect = [round1_response, round2_response, final_response]
        
        # Mock tools and tool manager
        generator = AIGenerator(api_key="test_key", model="test_model")
        mock_tools = [
            {"name": "get_course_outline", "description": "Get course outline"},
            {"name": "search_course_content", "description": "Search course content"}
        ]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "MCP Course Outline:\n1. Introduction\n2. Basic Concepts\n3. Implementation\n4. Advanced MCP Patterns\n5. Best Practices",
            "Found 3 courses discussing advanced MCP patterns: Advanced Architecture, System Design Patterns, Distributed Systems"
        ]
        
        # Execute the scenario
        response = generator.generate_response(
            query="Find courses that discuss the same topic as lesson 4 of MCP Course",
            tools=mock_tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify sequential execution with different tools
        print(f"\nQuery: Find courses that discuss the same topic as lesson 4 of MCP Course")
        print(f"Round 1: Got course outline to identify lesson 4 topic")
        print(f"Round 2: Searched for courses with similar topics")
        print(f"Final Response: {response}")
        
        # Verify the sequence used different tools
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_title="MCP Course")
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="advanced MCP patterns")

    @patch('ai_generator.anthropic.Anthropic')
    def test_early_completion_scenario(self, mock_anthropic_class):
        """Demonstrate early completion when one tool call is sufficient"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Round 1: Single search provides complete answer
        round1_tool = MockToolUseBlock(
            tool_name="search_course_content",
            tool_input={"query": "basic concepts"}
        )
        round1_response = MockResponse(stop_reason="tool_use", tool_calls=[round1_tool])
        
        # Round 2: AI decides it has enough info, no more tools needed
        final_response = MockResponse("Based on the search, basic concepts include: definitions, examples, and applications.")
        
        # Setup API call sequence (only 2 calls, no second tool round)
        mock_client.messages.create.side_effect = [round1_response, final_response]
        
        # Mock tools and tool manager  
        generator = AIGenerator(api_key="test_key", model="test_model")
        mock_tools = [{"name": "search_course_content", "description": "Search course content"}]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Basic concepts: Fundamental principles, core definitions, practical examples"
        
        # Execute the scenario
        response = generator.generate_response(
            query="What are the basic concepts?",
            tools=mock_tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify early completion
        print(f"\nQuery: What are the basic concepts?")
        print(f"Round 1: Found sufficient information")
        print(f"Early Completion: No second tool round needed")  
        print(f"Final Response: {response}")
        
        # Verify only one tool call was made
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 1)
        self.assertEqual(mock_client.messages.create.call_count, 2)  # 1 tool round + 1 final


if __name__ == '__main__':
    print("=== Sequential Tool Calling Demonstration ===")
    unittest.main(verbosity=2)