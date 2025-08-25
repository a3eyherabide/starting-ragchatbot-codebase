import os
import sys
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ai_generator import AIGenerator


class MockResponse:
    """Mock Anthropic API response"""

    def __init__(self, content_text=None, stop_reason="end_turn", tool_calls=None):
        self.stop_reason = stop_reason
        if tool_calls:
            self.content = tool_calls
        else:
            content_block = Mock()
            content_block.text = content_text or "Mock response"
            self.content = [content_block]


class MockToolUseBlock:
    """Mock tool use content block"""

    def __init__(self, tool_name, tool_input, tool_id="test_id"):
        self.type = "tool_use"
        self.name = tool_name
        self.input = tool_input
        self.id = tool_id


class TestAIGenerator(unittest.TestCase):
    """Test suite for AIGenerator tool calling functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.ai_generator = AIGenerator(api_key="test_key", model="test_model")

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_without_tools(self, mock_anthropic_class):
        """Test response generation without tools"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = MockResponse(
            "Direct response without tools"
        )

        # Create new AIGenerator to use mocked client
        generator = AIGenerator(api_key="test_key", model="test_model")

        # Generate response without tools
        response = generator.generate_response("What is 2+2?")

        # Verify API call
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args

        # Check base parameters
        self.assertEqual(call_args[1]["model"], "test_model")
        self.assertEqual(call_args[1]["temperature"], 0)
        self.assertEqual(call_args[1]["max_tokens"], 800)

        # Check message content
        self.assertEqual(len(call_args[1]["messages"]), 1)
        self.assertEqual(call_args[1]["messages"][0]["role"], "user")
        self.assertEqual(call_args[1]["messages"][0]["content"], "What is 2+2?")

        # Verify no tools parameter
        self.assertNotIn("tools", call_args[1])

        # Verify response
        self.assertEqual(response, "Direct response without tools")

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_with_tools_no_usage(self, mock_anthropic_class):
        """Test response generation with tools available but not used"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = MockResponse(
            "Response without using tools"
        )

        # Create new AIGenerator to use mocked client
        generator = AIGenerator(api_key="test_key", model="test_model")

        # Mock tools and tool manager
        mock_tools = [{"name": "search_course_content", "description": "Search tool"}]
        mock_tool_manager = Mock()

        # Generate response with tools but no usage
        response = generator.generate_response(
            query="General knowledge question",
            tools=mock_tools,
            tool_manager=mock_tool_manager,
        )

        # Verify API call includes tools
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args

        self.assertIn("tools", call_args[1])
        self.assertEqual(call_args[1]["tools"], mock_tools)
        self.assertEqual(call_args[1]["tool_choice"], {"type": "auto"})

        # Verify tool manager not called
        mock_tool_manager.execute_tool.assert_not_called()

        # Verify response
        self.assertEqual(response, "Response without using tools")

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_with_tool_usage(self, mock_anthropic_class):
        """Test response generation with tool usage"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Mock initial response with tool use
        tool_use_block = MockToolUseBlock(
            tool_name="search_course_content",
            tool_input={"query": "MCP concepts"},
            tool_id="test_tool_id",
        )
        initial_response = MockResponse(
            stop_reason="tool_use", tool_calls=[tool_use_block]
        )

        # Mock final response after tool execution
        final_response = MockResponse("Based on search results: MCP stands for...")

        # Setup client to return different responses
        mock_client.messages.create.side_effect = [initial_response, final_response]

        # Create new AIGenerator to use mocked client
        generator = AIGenerator(api_key="test_key", model="test_model")

        # Mock tools and tool manager
        mock_tools = [{"name": "search_course_content", "description": "Search tool"}]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = (
            "Course content about MCP concepts"
        )

        # Generate response with tool usage
        response = generator.generate_response(
            query="What is MCP?", tools=mock_tools, tool_manager=mock_tool_manager
        )

        # Verify initial API call
        self.assertEqual(mock_client.messages.create.call_count, 2)

        # Verify tool execution
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="MCP concepts"
        )

        # Verify final response
        self.assertEqual(response, "Based on search results: MCP stands for...")

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_with_conversation_history(self, mock_anthropic_class):
        """Test response generation with conversation history"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = MockResponse(
            "Response with history context"
        )

        # Create new AIGenerator to use mocked client
        generator = AIGenerator(api_key="test_key", model="test_model")

        # Generate response with conversation history
        conversation_history = "User: Previous question\nAssistant: Previous answer"
        response = generator.generate_response(
            query="Follow-up question", conversation_history=conversation_history
        )

        # Verify system prompt includes history
        call_args = mock_client.messages.create.call_args
        system_content = call_args[1]["system"]

        self.assertIn("Previous conversation:", system_content)
        self.assertIn("User: Previous question", system_content)
        self.assertIn("Assistant: Previous answer", system_content)

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_execution_error_handling(self, mock_anthropic_class):
        """Test handling of tool execution errors"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Mock initial response with tool use
        tool_use_block = MockToolUseBlock(
            tool_name="search_course_content",
            tool_input={"query": "test query"},
            tool_id="test_tool_id",
        )
        initial_response = MockResponse(
            stop_reason="tool_use", tool_calls=[tool_use_block]
        )

        # Mock final response after tool execution
        final_response = MockResponse("I encountered an error while searching")

        # Setup client to return different responses
        mock_client.messages.create.side_effect = [initial_response, final_response]

        # Create new AIGenerator to use mocked client
        generator = AIGenerator(api_key="test_key", model="test_model")

        # Mock tools and tool manager with error
        mock_tools = [{"name": "search_course_content", "description": "Search tool"}]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search error: No results found"

        # Generate response
        response = generator.generate_response(
            query="Search query", tools=mock_tools, tool_manager=mock_tool_manager
        )

        # Verify tool was called and error handled
        mock_tool_manager.execute_tool.assert_called_once()
        self.assertEqual(response, "I encountered an error while searching")

    @patch("ai_generator.anthropic.Anthropic")
    def test_multiple_tool_calls(self, mock_anthropic_class):
        """Test handling of multiple tool calls in one response"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Mock initial response with multiple tool uses
        tool_use_block1 = MockToolUseBlock(
            tool_name="search_course_content",
            tool_input={"query": "first query"},
            tool_id="tool_id_1",
        )
        tool_use_block2 = MockToolUseBlock(
            tool_name="get_course_outline",
            tool_input={"course_title": "MCP Course"},
            tool_id="tool_id_2",
        )
        initial_response = MockResponse(
            stop_reason="tool_use", tool_calls=[tool_use_block1, tool_use_block2]
        )

        # Mock final response
        final_response = MockResponse("Combined response from multiple tools")

        # Setup client to return different responses
        mock_client.messages.create.side_effect = [initial_response, final_response]

        # Create new AIGenerator to use mocked client
        generator = AIGenerator(api_key="test_key", model="test_model")

        # Mock tools and tool manager
        mock_tools = [
            {"name": "search_course_content", "description": "Search tool"},
            {"name": "get_course_outline", "description": "Outline tool"},
        ]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Search results content",
            "Course outline content",
        ]

        # Generate response
        response = generator.generate_response(
            query="Complex query requiring multiple tools",
            tools=mock_tools,
            tool_manager=mock_tool_manager,
        )

        # Verify both tools were called
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content", query="first query"
        )
        mock_tool_manager.execute_tool.assert_any_call(
            "get_course_outline", course_title="MCP Course"
        )

        # Verify final response
        self.assertEqual(response, "Combined response from multiple tools")

    @patch("ai_generator.anthropic.Anthropic")
    def test_sequential_two_round_tool_calling(self, mock_anthropic_class):
        """Test sequential tool calling across two rounds"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Mock round 1: AI uses get_course_outline tool
        round1_tool_block = MockToolUseBlock(
            tool_name="get_course_outline",
            tool_input={"course_title": "MCP Course"},
            tool_id="round1_id",
        )
        round1_response = MockResponse(
            stop_reason="tool_use", tool_calls=[round1_tool_block]
        )

        # Mock round 2: AI uses search_course_content tool based on round 1 results
        round2_tool_block = MockToolUseBlock(
            tool_name="search_course_content",
            tool_input={"query": "lesson 4 content", "course_name": "MCP Course"},
            tool_id="round2_id",
        )
        round2_response = MockResponse(
            stop_reason="tool_use", tool_calls=[round2_tool_block]
        )

        # Mock final response: AI synthesizes information
        final_response = MockResponse(
            "Based on the course outline and lesson content, here's what lesson 4 covers..."
        )

        # Setup client to return responses in sequence
        mock_client.messages.create.side_effect = [
            round1_response,
            round2_response,
            final_response,
        ]

        # Create new AIGenerator to use mocked client
        generator = AIGenerator(api_key="test_key", model="test_model")

        # Mock tools and tool manager
        mock_tools = [
            {"name": "get_course_outline", "description": "Get course outline"},
            {"name": "search_course_content", "description": "Search course content"},
        ]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course outline with 5 lessons including lesson 4: Advanced Topics",
            "Lesson 4 covers advanced MCP concepts including...",
        ]

        # Generate response with sequential tool calling
        response = generator.generate_response(
            query="What does lesson 4 of MCP Course cover?",
            tools=mock_tools,
            tool_manager=mock_tool_manager,
        )

        # Verify 3 API calls were made (2 tool rounds + 1 final)
        self.assertEqual(mock_client.messages.create.call_count, 3)

        # Verify both tools were executed in sequence
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        mock_tool_manager.execute_tool.assert_any_call(
            "get_course_outline", course_title="MCP Course"
        )
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content", query="lesson 4 content", course_name="MCP Course"
        )

        # Verify final response
        self.assertIn("Based on the course outline and lesson content", response)

    @patch("ai_generator.anthropic.Anthropic")
    def test_single_round_completion_no_second_tool(self, mock_anthropic_class):
        """Test that tool calling stops after one round if AI doesn't request more tools"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Mock round 1: AI uses tool but then provides complete response
        round1_tool_block = MockToolUseBlock(
            tool_name="search_course_content",
            tool_input={"query": "basic concepts"},
            tool_id="round1_id",
        )
        round1_response = MockResponse(
            stop_reason="tool_use", tool_calls=[round1_tool_block]
        )

        # Mock final response: AI provides complete answer without requesting more tools
        final_response = MockResponse(
            "Based on the search results, here's what I found about basic concepts..."
        )

        # Setup client to return responses in sequence
        mock_client.messages.create.side_effect = [round1_response, final_response]

        # Create new AIGenerator to use mocked client
        generator = AIGenerator(api_key="test_key", model="test_model")

        # Mock tools and tool manager
        mock_tools = [
            {"name": "search_course_content", "description": "Search course content"}
        ]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = (
            "Search results about basic concepts"
        )

        # Generate response
        response = generator.generate_response(
            query="What are the basic concepts?",
            tools=mock_tools,
            tool_manager=mock_tool_manager,
        )

        # Verify only 2 API calls (1 tool round + 1 final, no second tool round)
        self.assertEqual(mock_client.messages.create.call_count, 2)

        # Verify only one tool was executed
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 1)

        # Verify final response
        self.assertIn("Based on the search results", response)

    @patch("ai_generator.anthropic.Anthropic")
    def test_max_two_rounds_limit(self, mock_anthropic_class):
        """Test that tool calling is limited to maximum 2 rounds"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Mock both rounds requesting tool use
        tool_block1 = MockToolUseBlock(
            "search_course_content", {"query": "first search"}, "id1"
        )
        tool_block2 = MockToolUseBlock(
            "search_course_content", {"query": "second search"}, "id2"
        )

        round1_response = MockResponse(stop_reason="tool_use", tool_calls=[tool_block1])
        round2_response = MockResponse(stop_reason="tool_use", tool_calls=[tool_block2])
        final_response = MockResponse("Final response after 2 rounds")

        # Setup client
        mock_client.messages.create.side_effect = [
            round1_response,
            round2_response,
            final_response,
        ]

        # Create new AIGenerator
        generator = AIGenerator(api_key="test_key", model="test_model")

        # Mock tools and tool manager
        mock_tools = [{"name": "search_course_content", "description": "Search"}]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["First result", "Second result"]

        # Generate response
        response = generator.generate_response(
            query="Complex query requiring multiple searches",
            tools=mock_tools,
            tool_manager=mock_tool_manager,
        )

        # Verify exactly 3 API calls (2 tool rounds + 1 final, not more)
        self.assertEqual(mock_client.messages.create.call_count, 3)

        # Verify exactly 2 tool executions (limited by max_tool_rounds=2)
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_execution_error_handling(self, mock_anthropic_class):
        """Test graceful handling of tool execution errors in sequential rounds"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Mock tool round that will have execution error
        tool_block = MockToolUseBlock("search_course_content", {"query": "test"}, "id1")
        round1_response = MockResponse(stop_reason="tool_use", tool_calls=[tool_block])

        # Setup client
        mock_client.messages.create.return_value = round1_response

        # Create new AIGenerator
        generator = AIGenerator(api_key="test_key", model="test_model")

        # Mock tools and tool manager with error
        mock_tools = [{"name": "search_course_content", "description": "Search"}]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")

        # Generate response
        response = generator.generate_response(
            query="Test query", tools=mock_tools, tool_manager=mock_tool_manager
        )

        # Verify error was handled gracefully
        self.assertEqual(
            response, "I encountered an error while processing your request."
        )

    @patch("ai_generator.anthropic.Anthropic")
    def test_backwards_compatibility_single_tool_call(self, mock_anthropic_class):
        """Test that existing single tool call behavior still works"""
        # This test ensures the old _handle_tool_execution path still works
        # when generate_response is called without the sequential features

        # Setup mock using old pattern
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        tool_block = MockToolUseBlock("search_course_content", {"query": "test"}, "id1")
        initial_response = MockResponse(stop_reason="tool_use", tool_calls=[tool_block])
        final_response = MockResponse("Response after single tool call")

        mock_client.messages.create.side_effect = [initial_response, final_response]

        # Create new AIGenerator
        generator = AIGenerator(api_key="test_key", model="test_model")

        # Test with max_tool_rounds=1 to use backwards compatible behavior
        mock_tools = [{"name": "search_course_content", "description": "Search"}]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        response = generator.generate_response(
            query="Test query",
            tools=mock_tools,
            tool_manager=mock_tool_manager,
            max_tool_rounds=1,
        )

        # Should still work and return proper response
        self.assertIn("Response after single tool call", response)
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 1)

    def test_system_prompt_content(self):
        """Test that system prompt contains expected guidance"""
        system_prompt = self.ai_generator.SYSTEM_PROMPT

        # Check for key instruction elements
        self.assertIn("Course outline queries", system_prompt)
        self.assertIn("Content search queries", system_prompt)
        self.assertIn("get_course_outline", system_prompt)
        self.assertIn("search_course_content", system_prompt)
        self.assertIn("Multi-round tool usage", system_prompt)
        self.assertIn("up to 2 rounds", system_prompt)


if __name__ == "__main__":
    unittest.main()
