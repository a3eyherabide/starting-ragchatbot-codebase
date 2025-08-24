import anthropic
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from copy import deepcopy


@dataclass
class ConversationState:
    """Immutable state container for tracking conversation context across tool rounds"""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    system_prompt: str = ""
    completed_rounds: int = 0
    tool_results_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_assistant_response(self, response):
        """Create new state with assistant response added"""
        new_messages = deepcopy(self.messages)
        new_messages.append({"role": "assistant", "content": response.content})
        
        return ConversationState(
            messages=new_messages,
            system_prompt=self.system_prompt,
            completed_rounds=self.completed_rounds,
            tool_results_history=deepcopy(self.tool_results_history)
        )
    
    def add_tool_results(self, tool_results):
        """Create new state with tool results added"""
        new_messages = deepcopy(self.messages)
        new_messages.append({"role": "user", "content": tool_results})
        
        new_tool_history = deepcopy(self.tool_results_history)
        new_tool_history.extend(tool_results)
        
        return ConversationState(
            messages=new_messages,
            system_prompt=self.system_prompt,
            completed_rounds=self.completed_rounds + 1,
            tool_results_history=new_tool_history
        )
    
    def get_api_params(self, base_params: Dict[str, Any], include_tools: bool = True, tools: Optional[List] = None):
        """Get API parameters for current conversation state"""
        params = {
            **base_params,
            "messages": deepcopy(self.messages),
            "system": self.system_prompt
        }
        
        if include_tools and tools:
            params["tools"] = tools
            params["tool_choice"] = {"type": "auto"}
        
        return params


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search and outline tools for course information.

Tool Usage Guidelines:
- **Multi-round tool usage**: You can make multiple tool calls across up to 2 rounds to gather comprehensive information
- **Round 1 strategy**: Use tools to gather initial information about the user's query
- **Round 2 strategy**: If needed after reviewing round 1 results, use tools again to gather additional context, make comparisons, or clarify findings
- **Course outline queries**: Use get_course_outline tool for questions about course structure, lesson lists, or complete course overviews
- **Content search queries**: Use search_course_content tool for questions about specific course content or detailed educational materials
- **Sequential approach**: Use round 1 results to inform more targeted round 2 tool calls when needed
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Course Outline Responses:
When using get_course_outline, always include in your response:
- Course title
- Course link (if available)
- Complete lesson list with numbers and titles
- Total number of lessons

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline questions**: Use get_course_outline tool first, then provide structured response
- **Course content questions**: Use search_course_content tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the tool results" or "using the outline tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None,
                         max_tool_rounds: int = 2) -> str:
        """
        Generate AI response with optional sequential tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_tool_rounds: Maximum number of sequential tool rounds (default 2)
            
        Returns:
            Generated response as string
        """
        
        # Initialize conversation state
        conversation_state = self._initialize_conversation_state(query, conversation_history)
        
        # Execute tool calling rounds if tools are available
        if tools and tool_manager:
            for round_num in range(1, max_tool_rounds + 1):
                # Execute tool round
                response = self._execute_tool_round(conversation_state, tools, round_num)
                
                # If API call failed
                if response is None:
                    return "I encountered an error while processing your request."
                
                # If no tool use requested, return direct response
                if response.stop_reason != "tool_use":
                    return response.content[0].text
                
                # Execute tools and update conversation state
                conversation_state = self._execute_tools_and_update_state(
                    response, conversation_state, tool_manager
                )
                
                if conversation_state is None:  # Tool execution failed
                    return "I encountered an error while processing your request."
                
                # If this was the last allowed round, break
                if round_num >= max_tool_rounds:
                    break
            
            # Generate final response without tools after tool rounds completed
            return self._generate_final_response(conversation_state)
        else:
            # No tools provided - use simple direct response
            api_params = conversation_state.get_api_params(self.base_params, include_tools=False)
            response = self.client.messages.create(**api_params)
            return response.content[0].text
    
    def _initialize_conversation_state(self, query: str, conversation_history: Optional[str] = None) -> ConversationState:
        """Initialize conversation state with query and optional history"""
        # Build system content with conversation history if provided
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Create initial conversation state
        return ConversationState(
            messages=[{"role": "user", "content": query}],
            system_prompt=system_content,
            completed_rounds=0,
            tool_results_history=[]
        )
    
    def _execute_tool_round(self, conversation_state: ConversationState, tools: List, round_num: int):
        """Execute a single tool calling round"""
        try:
            # Get API parameters with tools
            api_params = conversation_state.get_api_params(self.base_params, include_tools=True, tools=tools)
            
            # Make API call
            response = self.client.messages.create(**api_params)
            return response
            
        except Exception as e:
            # Log error and return None to indicate failure
            print(f"Error in tool round {round_num}: {e}")
            return None
    
    
    def _execute_tools_and_update_state(self, response, conversation_state: ConversationState, tool_manager) -> Optional[ConversationState]:
        """Execute tools from response and update conversation state"""
        try:
            # Add assistant's tool use response to conversation
            conversation_state = conversation_state.add_assistant_response(response)
            
            # Execute all tool calls and collect results
            tool_results = []
            for content_block in response.content:
                if content_block.type == "tool_use":
                    tool_result = tool_manager.execute_tool(
                        content_block.name, 
                        **content_block.input
                    )
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })
            
            # Add tool results to conversation state
            if tool_results:
                conversation_state = conversation_state.add_tool_results(tool_results)
            
            return conversation_state
            
        except Exception as e:
            print(f"Error executing tools: {e}")
            return None  # Indicate tool execution failure
    
    def _generate_final_response(self, conversation_state: ConversationState) -> str:
        """Generate final response without tools"""
        try:
            # Generate final response with accumulated context (no tools)
            api_params = conversation_state.get_api_params(self.base_params, include_tools=False)
            response = self.client.messages.create(**api_params)
            return response.content[0].text
            
        except Exception as e:
            print(f"Error generating final response: {e}")
            return "I encountered an error while generating my response."
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text