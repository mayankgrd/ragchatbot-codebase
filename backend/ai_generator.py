import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Available Tools:
1. **search_course_content**: Search within course content for specific topics, concepts, or details
2. **get_course_outline**: Get course structure including title, instructor, link, and complete lesson list

Tool Selection Strategy:
- Use **get_course_outline** for: course syllabus, lesson lists, what a course covers, course structure
- Use **search_course_content** for: specific content questions, concepts, explanations within lessons
- **Sequential Tool Usage**: You may make multiple tool calls when needed for complex queries
  - Use SINGLE call for: simple lookups, specific content questions
  - Use MULTIPLE calls for: comparisons across courses, gathering comprehensive information
  - Each call should have a distinct purpose (avoid redundant searches)

When to Stop Searching:
- You have sufficient information to answer completely
- Further searches would be redundant
- You've gathered enough sources on the topic

Citation Instructions (for search_course_content only):
- Search results are numbered [1], [2], [3], etc. and accumulate across searches
- Cite sources by including the number in brackets, e.g., "The model uses attention mechanisms [1]."
- Only cite sources you actually use
- Course outlines do not require citations

Response Protocol:
- **General knowledge questions**: Answer without tools
- **Course structure/syllabus questions**: Use get_course_outline
- **Course content questions**: Use search_course_content
- **Comparison questions**: Search each course separately, then synthesize
- **No meta-commentary**: Provide direct answers only

For Course Outlines:
- Present the course title, instructor, and link clearly
- List lessons with their numbers, titles, and links
- Format as a readable list

All responses must be:
1. **Brief and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
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
                         tool_manager=None) -> str:
        """
        Generate AI response with optional sequential tool usage and conversation context.

        Uses a loop-based approach that allows Claude to make 0 to MAX_SEQUENTIAL_TOOL_CALLS
        tool calls per query. Each iteration:
        1. Makes an API call with tools available (if under limit)
        2. If tool_use requested, executes tools and continues loop
        3. If no tool_use or max reached, returns final response

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """
        from config import config

        # Build system content
        system_content = self._build_system_content(conversation_history)

        # Initialize message chain
        messages = [{"role": "user", "content": query}]

        # Track tool call count
        tool_call_count = 0

        # Main execution loop
        while True:
            # Build API parameters
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content
            }

            # Include tools if under limit and available
            can_use_tools = (
                tools is not None
                and tool_manager is not None
                and tool_call_count < config.MAX_SEQUENTIAL_TOOL_CALLS
            )

            if can_use_tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            # Make API call
            response = self.client.messages.create(**api_params)

            # Exit condition 1: Model returned text (no tool use)
            if response.stop_reason != "tool_use":
                # If we made tool calls, ensure citations are included
                if tool_call_count > 0:
                    return self._ensure_citations_in_response(
                        response, messages, system_content
                    )
                return self._extract_text_response(response)

            # Exit condition 2: Max tool calls reached
            if tool_call_count >= config.MAX_SEQUENTIAL_TOOL_CALLS:
                return self._force_final_response(messages, system_content)

            # Execute tools and continue loop
            messages = self._execute_tools_and_update_messages(
                response, messages, tool_manager
            )
            tool_call_count += 1

    def _build_system_content(self, conversation_history: Optional[str]) -> str:
        """Build system prompt with optional conversation history."""
        if conversation_history:
            return f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
        return self.SYSTEM_PROMPT

    def _extract_text_response(self, response) -> str:
        """Extract text content from API response."""
        for block in response.content:
            if hasattr(block, 'text'):
                return block.text
        return ""

    def _ensure_citations_in_response(
        self,
        response,
        messages: List[Dict],
        system_content: str
    ) -> str:
        """
        Check if the response contains citations. If not, request regeneration
        with an explicit reminder to cite sources.

        This handles the case where Claude uses tools but doesn't include
        citations in its final response (common in multi-tool comparison queries).

        Args:
            response: The API response to check
            messages: Current message chain with tool results
            system_content: System prompt content

        Returns:
            Response text with citations
        """
        import re

        text = self._extract_text_response(response)

        # Check if response contains citation patterns like [1], [2], etc.
        has_citations = bool(re.search(r'\[\d+\]', text))

        if has_citations:
            return text

        # No citations found - request regeneration with explicit reminder
        # Build messages including the current response and a reminder
        final_messages = messages.copy()

        # Add the assistant's response that lacked citations
        final_messages.append({
            "role": "assistant",
            "content": text
        })

        # Add reminder to include citations
        final_messages.append({
            "role": "user",
            "content": "Please revise your response to include citations using bracket notation [1], [2], etc. to reference the search results. Each fact from the course materials should cite its source."
        })

        # Make API call without tools to get revised response
        revised_response = self.client.messages.create(
            **self.base_params,
            messages=final_messages,
            system=system_content
        )

        return self._extract_text_response(revised_response)

    def _execute_tools_and_update_messages(
        self,
        response,
        messages: List[Dict],
        tool_manager
    ) -> List[Dict]:
        """
        Execute tools from response and return updated messages.

        Args:
            response: API response containing tool_use blocks
            messages: Current message list
            tool_manager: Manager to execute tools

        Returns:
            Updated messages list with tool use and results
        """
        # Create copy to avoid mutating original
        updated_messages = messages.copy()

        # Add assistant's tool use response
        updated_messages.append({
            "role": "assistant",
            "content": response.content
        })

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

        # Add tool results as user message
        if tool_results:
            updated_messages.append({
                "role": "user",
                "content": tool_results
            })

        return updated_messages

    def _force_final_response(
        self,
        messages: List[Dict],
        system_content: str
    ) -> str:
        """
        Force a final text response when max tool calls reached.
        Makes API call without tools to force text generation.

        Args:
            messages: Current message chain
            system_content: System prompt content

        Returns:
            Final response text
        """
        # Add instruction to generate final response with citation reminder
        final_messages = messages.copy()
        final_messages.append({
            "role": "user",
            "content": "Based on the search results above, please provide your final response. Remember to cite your sources using the bracket notation [1], [2], etc."
        })

        # Make API call without tools
        response = self.client.messages.create(
            **self.base_params,
            messages=final_messages,
            system=system_content
        )

        return self._extract_text_response(response)