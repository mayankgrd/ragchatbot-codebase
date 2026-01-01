import pytest
from unittest.mock import Mock, patch, MagicMock

from ai_generator import AIGenerator


class TestAIGenerator:
    """Tests for AIGenerator"""

    @pytest.fixture
    def ai_generator(self):
        """Create AIGenerator with mocked client"""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            generator = AIGenerator(api_key="test-key", model="test-model")
            return generator, mock_anthropic.return_value

    def test_init_sets_model_and_client(self):
        """AIGenerator should initialize with model and client"""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            generator = AIGenerator(api_key="test-key", model="claude-test")

            assert generator.model == "claude-test"
            mock_anthropic.assert_called_once_with(api_key="test-key")

    def test_system_prompt_includes_tool_guidance(self):
        """System prompt should include guidance for both tools"""
        assert "search_course_content" in AIGenerator.SYSTEM_PROMPT
        assert "get_course_outline" in AIGenerator.SYSTEM_PROMPT
        assert "Tool Selection" in AIGenerator.SYSTEM_PROMPT

    def test_system_prompt_allows_sequential_tool_calls(self):
        """System prompt should allow multiple tool calls"""
        assert "Sequential Tool Usage" in AIGenerator.SYSTEM_PROMPT
        assert "multiple tool calls" in AIGenerator.SYSTEM_PROMPT.lower()

    def test_generate_response_without_tools(self, ai_generator, mock_anthropic_response_text):
        """Generate response should work without tools"""
        generator, mock_client = ai_generator
        mock_client.messages.create.return_value = mock_anthropic_response_text

        result = generator.generate_response(query="What is Python?")

        assert result == "This is a test response."
        mock_client.messages.create.assert_called_once()

    def test_generate_response_includes_conversation_history(self, ai_generator, mock_anthropic_response_text):
        """Generate response should include conversation history in system prompt"""
        generator, mock_client = ai_generator
        mock_client.messages.create.return_value = mock_anthropic_response_text

        generator.generate_response(
            query="Follow-up question",
            conversation_history="User: Hi\nAssistant: Hello"
        )

        call_args = mock_client.messages.create.call_args
        system_content = call_args.kwargs["system"]
        assert "Previous conversation:" in system_content
        assert "User: Hi" in system_content

    def test_generate_response_with_tools_provided(self, ai_generator, mock_anthropic_response_text, mock_tool_manager):
        """Generate response should include tools when provided"""
        generator, mock_client = ai_generator
        mock_client.messages.create.return_value = mock_anthropic_response_text

        tools = [{"name": "test_tool", "description": "A test tool"}]
        generator.generate_response(query="Test", tools=tools, tool_manager=mock_tool_manager)

        call_args = mock_client.messages.create.call_args
        assert call_args.kwargs["tools"] == tools
        assert call_args.kwargs["tool_choice"] == {"type": "auto"}

    def test_generate_response_triggers_tool_execution(self, ai_generator, mock_anthropic_response_tool_use, mock_anthropic_response_text, mock_tool_manager):
        """Generate response should execute tools when model requests them"""
        generator, mock_client = ai_generator

        # First call returns tool use, second call returns final response
        mock_client.messages.create.side_effect = [
            mock_anthropic_response_tool_use,
            mock_anthropic_response_text
        ]

        tools = [{"name": "search_course_content", "description": "Search"}]
        result = generator.generate_response(
            query="Search for MCP",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Tool manager should be called
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="MCP protocol"
        )
        # Final response should be returned
        assert result == "This is a test response."

    def test_base_params_set_correctly(self):
        """Base params should include model and settings"""
        with patch("ai_generator.anthropic.Anthropic"):
            generator = AIGenerator(api_key="test-key", model="claude-test")

            assert generator.base_params["model"] == "claude-test"
            assert generator.base_params["temperature"] == 0
            assert generator.base_params["max_tokens"] == 800


class TestSequentialToolCalling:
    """Tests for sequential/multi-tool execution"""

    @pytest.fixture
    def ai_generator(self):
        """Create AIGenerator with mocked client"""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            generator = AIGenerator(api_key="test-key", model="test-model")
            return generator, mock_anthropic.return_value

    def test_single_tool_call_returns_response(self, ai_generator, mock_anthropic_response_tool_use, mock_anthropic_response_text, mock_tool_manager):
        """Single tool use followed by text should work"""
        generator, mock_client = ai_generator

        mock_client.messages.create.side_effect = [
            mock_anthropic_response_tool_use,
            mock_anthropic_response_text
        ]

        tools = [{"name": "search_course_content", "description": "Search"}]
        result = generator.generate_response(
            query="Search for MCP",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        assert result == "This is a test response."
        assert mock_client.messages.create.call_count == 2

    def test_multiple_tool_calls_accumulate_messages(self, ai_generator, mock_tool_manager):
        """Messages should accumulate across sequential tool calls"""
        generator, mock_client = ai_generator

        # Create responses for: tool1 -> tool2 -> text
        tool_response1 = Mock()
        tool_response1.stop_reason = "tool_use"
        tool1 = Mock()
        tool1.type = "tool_use"
        tool1.id = "tool_1"
        tool1.name = "search_course_content"
        tool1.input = {"query": "MCP"}
        tool_response1.content = [tool1]

        tool_response2 = Mock()
        tool_response2.stop_reason = "tool_use"
        tool2 = Mock()
        tool2.type = "tool_use"
        tool2.id = "tool_2"
        tool2.name = "search_course_content"
        tool2.input = {"query": "Chroma"}
        tool_response2.content = [tool2]

        text_response = Mock()
        text_response.stop_reason = "end_turn"
        text_block = Mock()
        text_block.type = "text"
        text_block.text = "Comparison response."
        text_response.content = [text_block]

        mock_client.messages.create.side_effect = [
            tool_response1,
            tool_response2,
            text_response
        ]

        tools = [{"name": "search_course_content", "description": "Search"}]
        result = generator.generate_response(
            query="Compare MCP and Chroma",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Should have made 3 API calls
        assert mock_client.messages.create.call_count == 3
        # Tool manager should have been called twice
        assert mock_tool_manager.execute_tool.call_count == 2
        # Final response should be returned
        assert result == "Comparison response."

    def test_max_tool_calls_forces_response(self, ai_generator, mock_tool_manager):
        """Should force response after max calls reached"""
        generator, mock_client = ai_generator

        # Create responses that always request tool use
        def create_tool_response(tool_id):
            response = Mock()
            response.stop_reason = "tool_use"
            tool = Mock()
            tool.type = "tool_use"
            tool.id = tool_id
            tool.name = "search_course_content"
            tool.input = {"query": f"query_{tool_id}"}
            response.content = [tool]
            return response

        # Create text response for forced final
        text_response = Mock()
        text_response.stop_reason = "end_turn"
        text_block = Mock()
        text_block.type = "text"
        text_block.text = "Forced final response."
        text_response.content = [text_block]

        # 3 tool uses (hitting max) + 1 forced final (without tools)
        mock_client.messages.create.side_effect = [
            create_tool_response("t1"),
            create_tool_response("t2"),
            create_tool_response("t3"),
            text_response  # Forced final
        ]

        tools = [{"name": "search_course_content", "description": "Search"}]
        result = generator.generate_response(
            query="Complex query",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Should have made 4 API calls (3 tool uses + 1 forced final)
        assert mock_client.messages.create.call_count == 4
        # Tool manager should have been called 3 times (max)
        assert mock_tool_manager.execute_tool.call_count == 3
        assert result == "Forced final response."

    def test_no_tools_provided_returns_direct_response(self, ai_generator, mock_anthropic_response_text):
        """Without tools, should return direct response"""
        generator, mock_client = ai_generator
        mock_client.messages.create.return_value = mock_anthropic_response_text

        result = generator.generate_response(query="General question")

        assert result == "This is a test response."
        assert mock_client.messages.create.call_count == 1
        # Check that tools were not included
        call_args = mock_client.messages.create.call_args
        assert "tools" not in call_args.kwargs

    def test_tool_not_used_when_max_reached(self, ai_generator, mock_tool_manager):
        """Tools should not be included in API call after max reached"""
        generator, mock_client = ai_generator

        # Create 3 tool use responses (reaching max)
        tool_responses = []
        for i in range(3):
            response = Mock()
            response.stop_reason = "tool_use"
            tool = Mock()
            tool.type = "tool_use"
            tool.id = f"tool_{i}"
            tool.name = "search_course_content"
            tool.input = {"query": f"query_{i}"}
            response.content = [tool]
            tool_responses.append(response)

        # Final response (forced)
        text_response = Mock()
        text_response.stop_reason = "end_turn"
        text_block = Mock()
        text_block.type = "text"
        text_block.text = "Final."
        text_response.content = [text_block]

        mock_client.messages.create.side_effect = tool_responses + [text_response]

        tools = [{"name": "search_course_content", "description": "Search"}]
        generator.generate_response(
            query="Complex",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # The 4th call (forced final) should not have tools
        fourth_call = mock_client.messages.create.call_args_list[3]
        assert "tools" not in fourth_call.kwargs


class TestHelperMethods:
    """Tests for helper methods"""

    @pytest.fixture
    def ai_generator(self):
        """Create AIGenerator with mocked client"""
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
            generator = AIGenerator(api_key="test-key", model="test-model")
            return generator, mock_anthropic.return_value

    def test_build_system_content_without_history(self, ai_generator):
        """Should return system prompt when no history"""
        generator, _ = ai_generator

        result = generator._build_system_content(None)

        assert result == generator.SYSTEM_PROMPT

    def test_build_system_content_with_history(self, ai_generator):
        """Should include history in system content"""
        generator, _ = ai_generator

        result = generator._build_system_content("User: Hello\nAssistant: Hi")

        assert "Previous conversation:" in result
        assert "User: Hello" in result
        assert generator.SYSTEM_PROMPT in result

    def test_extract_text_response_finds_text(self, ai_generator, mock_anthropic_response_text):
        """Should extract text from response"""
        generator, _ = ai_generator

        result = generator._extract_text_response(mock_anthropic_response_text)

        assert result == "This is a test response."

    def test_extract_text_response_empty_if_no_text(self, ai_generator):
        """Should return empty string if no text block"""
        generator, _ = ai_generator

        response = Mock()
        response.content = []

        result = generator._extract_text_response(response)

        assert result == ""

    def test_execute_tools_and_update_messages(self, ai_generator, mock_anthropic_response_tool_use, mock_tool_manager):
        """Should execute tools and return updated messages"""
        generator, _ = ai_generator
        messages = [{"role": "user", "content": "test"}]

        result = generator._execute_tools_and_update_messages(
            mock_anthropic_response_tool_use,
            messages,
            mock_tool_manager
        )

        # Should have 3 messages: original, assistant tool use, user tool result
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "user"
        assert result[2]["content"][0]["type"] == "tool_result"

        # Tool should have been executed
        mock_tool_manager.execute_tool.assert_called_once()

    def test_force_final_response(self, ai_generator, mock_anthropic_response_text):
        """Should force a final response without tools"""
        generator, mock_client = ai_generator
        mock_client.messages.create.return_value = mock_anthropic_response_text

        messages = [{"role": "user", "content": "test"}]
        result = generator._force_final_response(messages, "system prompt")

        assert result == "This is a test response."
        # Check that a user message was added asking for final response
        call_args = mock_client.messages.create.call_args
        messages_sent = call_args.kwargs["messages"]
        assert "final response" in messages_sent[-1]["content"].lower()
        # Check no tools were included
        assert "tools" not in call_args.kwargs

    def test_force_final_response_includes_citation_reminder(self, ai_generator, mock_anthropic_response_text):
        """Forced final response prompt should remind AI to cite sources"""
        generator, mock_client = ai_generator
        mock_client.messages.create.return_value = mock_anthropic_response_text

        messages = [{"role": "user", "content": "test"}]
        generator._force_final_response(messages, "system prompt")

        call_args = mock_client.messages.create.call_args
        messages_sent = call_args.kwargs["messages"]
        prompt = messages_sent[-1]["content"].lower()
        # Should remind about citations
        assert "cite" in prompt or "[1]" in prompt or "bracket" in prompt
