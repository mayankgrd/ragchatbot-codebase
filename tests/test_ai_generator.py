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

    def test_generate_response_with_tools_provided(self, ai_generator, mock_anthropic_response_text):
        """Generate response should include tools when provided"""
        generator, mock_client = ai_generator
        mock_client.messages.create.return_value = mock_anthropic_response_text

        tools = [{"name": "test_tool", "description": "A test tool"}]
        generator.generate_response(query="Test", tools=tools)

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

    def test_handle_tool_execution_calls_tool_manager(self, ai_generator, mock_anthropic_response_tool_use, mock_anthropic_response_text, mock_tool_manager):
        """Tool execution should call tool manager with correct parameters"""
        generator, mock_client = ai_generator
        mock_client.messages.create.return_value = mock_anthropic_response_text

        base_params = {
            "messages": [{"role": "user", "content": "test"}],
            "system": "test system prompt"
        }

        generator._handle_tool_execution(
            mock_anthropic_response_tool_use,
            base_params,
            mock_tool_manager
        )

        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="MCP protocol"
        )

    def test_handle_tool_execution_sends_results_back(self, ai_generator, mock_anthropic_response_tool_use, mock_anthropic_response_text, mock_tool_manager):
        """Tool execution should send results back to model"""
        generator, mock_client = ai_generator
        mock_client.messages.create.return_value = mock_anthropic_response_text

        base_params = {
            "messages": [{"role": "user", "content": "test"}],
            "system": "test system prompt"
        }

        generator._handle_tool_execution(
            mock_anthropic_response_tool_use,
            base_params,
            mock_tool_manager
        )

        # Check that the second API call includes tool results
        call_args = mock_client.messages.create.call_args
        messages = call_args.kwargs["messages"]

        # Should have: original user message, assistant tool use, user tool result
        assert len(messages) == 3
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0]["type"] == "tool_result"
        assert messages[2]["content"][0]["tool_use_id"] == "tool_123"

    def test_handle_tool_execution_multiple_tools(self, ai_generator, mock_anthropic_response_text, mock_tool_manager):
        """Tool execution should handle multiple tool calls"""
        generator, mock_client = ai_generator
        mock_client.messages.create.return_value = mock_anthropic_response_text

        # Create response with multiple tool uses
        multi_tool_response = Mock()
        multi_tool_response.stop_reason = "tool_use"

        tool1 = Mock()
        tool1.type = "tool_use"
        tool1.id = "tool_1"
        tool1.name = "search_course_content"
        tool1.input = {"query": "query1"}

        tool2 = Mock()
        tool2.type = "tool_use"
        tool2.id = "tool_2"
        tool2.name = "get_course_outline"
        tool2.input = {"course_title": "MCP"}

        multi_tool_response.content = [tool1, tool2]

        base_params = {
            "messages": [{"role": "user", "content": "test"}],
            "system": "test system prompt"
        }

        generator._handle_tool_execution(
            multi_tool_response,
            base_params,
            mock_tool_manager
        )

        # Tool manager should be called twice
        assert mock_tool_manager.execute_tool.call_count == 2

    def test_base_params_set_correctly(self):
        """Base params should include model and settings"""
        with patch("ai_generator.anthropic.Anthropic"):
            generator = AIGenerator(api_key="test-key", model="claude-test")

            assert generator.base_params["model"] == "claude-test"
            assert generator.base_params["temperature"] == 0
            assert generator.base_params["max_tokens"] == 800
