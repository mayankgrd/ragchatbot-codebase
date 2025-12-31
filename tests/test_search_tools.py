import pytest
from unittest.mock import Mock

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Tests for CourseSearchTool"""

    def test_get_tool_definition_returns_correct_schema(self, mock_vector_store):
        """Tool definition should have correct name and schema"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["query"]

    def test_execute_returns_formatted_results(self, mock_vector_store, sample_search_results):
        """Execute should return formatted search results"""
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="MCP protocol")

        assert "[1]" in result
        assert "[2]" in result
        assert "MCP allows models" in result
        mock_vector_store.search.assert_called_once_with(
            query="MCP protocol",
            course_name=None,
            lesson_number=None
        )

    def test_execute_with_course_filter(self, mock_vector_store):
        """Execute should pass course_name to vector store"""
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="test", course_name="MCP Course")

        mock_vector_store.search.assert_called_once_with(
            query="test",
            course_name="MCP Course",
            lesson_number=None
        )

    def test_execute_with_lesson_filter(self, mock_vector_store):
        """Execute should pass lesson_number to vector store"""
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="test", lesson_number=2)

        mock_vector_store.search.assert_called_once_with(
            query="test",
            course_name=None,
            lesson_number=2
        )

    def test_execute_returns_error_message_when_no_results(self, mock_vector_store_empty):
        """Execute should return message when no results found"""
        tool = CourseSearchTool(mock_vector_store_empty)
        result = tool.execute(query="unknown topic")

        assert "No relevant content found" in result

    def test_execute_returns_error_from_search(self, mock_vector_store):
        """Execute should return error message from search results"""
        mock_vector_store.search.return_value = SearchResults.empty("Search error occurred")
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test")

        assert result == "Search error occurred"

    def test_format_results_includes_citations(self, mock_vector_store, sample_search_results):
        """Formatted results should include numbered citations"""
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test")

        assert "[1]" in result
        assert "[2]" in result

    def test_last_sources_tracked(self, mock_vector_store):
        """Sources should be tracked after execution"""
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="test")

        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["citation_num"] == 1
        assert tool.last_sources[1]["citation_num"] == 2


class TestCourseOutlineTool:
    """Tests for CourseOutlineTool"""

    def test_get_tool_definition_returns_correct_schema(self, mock_vector_store):
        """Tool definition should have correct name and schema"""
        tool = CourseOutlineTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "get_course_outline"
        assert "course_title" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["course_title"]

    def test_execute_returns_formatted_outline(self, mock_vector_store, sample_course_metadata):
        """Execute should return formatted course outline"""
        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_title="MCP")

        assert "Introduction to MCP" in result
        assert "John Doe" in result
        assert "https://example.com/mcp-course" in result
        mock_vector_store.get_course_metadata.assert_called_once_with("MCP")

    def test_execute_returns_error_when_course_not_found(self, mock_vector_store_empty):
        """Execute should return error when course not found"""
        tool = CourseOutlineTool(mock_vector_store_empty)
        result = tool.execute(course_title="Unknown Course")

        assert "No course found matching 'Unknown Course'" in result

    def test_format_outline_includes_all_fields(self, mock_vector_store, sample_course_metadata):
        """Formatted outline should include all course fields"""
        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_title="MCP")

        assert "Course:" in result
        assert "Instructor:" in result
        assert "Course Link:" in result
        assert "Total Lessons:" in result
        assert "Lessons:" in result

    def test_format_outline_with_lesson_links(self, mock_vector_store, sample_course_metadata):
        """Formatted outline should include lesson links"""
        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_title="MCP")

        assert "0. Getting Started" in result
        assert "1. Core Concepts" in result
        assert "2. Advanced Topics" in result
        assert "https://example.com/lesson0" in result


class TestToolManager:
    """Tests for ToolManager"""

    def test_register_tool(self, mock_vector_store):
        """ToolManager should register tools by name"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        assert "search_course_content" in manager.tools

    def test_register_multiple_tools(self, mock_vector_store):
        """ToolManager should register multiple tools"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)

        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)

        assert len(manager.tools) == 2
        assert "search_course_content" in manager.tools
        assert "get_course_outline" in manager.tools

    def test_get_tool_definitions(self, mock_vector_store):
        """ToolManager should return all tool definitions"""
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_vector_store))
        manager.register_tool(CourseOutlineTool(mock_vector_store))

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 2
        names = [d["name"] for d in definitions]
        assert "search_course_content" in names
        assert "get_course_outline" in names

    def test_execute_tool_by_name(self, mock_vector_store):
        """ToolManager should execute tool by name"""
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_vector_store))

        result = manager.execute_tool("search_course_content", query="test")

        assert "[1]" in result  # Formatted results contain citations

    def test_execute_unknown_tool_returns_error(self):
        """ToolManager should return error for unknown tool"""
        manager = ToolManager()
        result = manager.execute_tool("unknown_tool", query="test")

        assert "Tool 'unknown_tool' not found" in result

    def test_get_last_sources(self, mock_vector_store):
        """ToolManager should return sources from last search"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        manager.execute_tool("search_course_content", query="test")
        sources = manager.get_last_sources()

        assert len(sources) == 2

    def test_reset_sources(self, mock_vector_store):
        """ToolManager should reset sources from all tools"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        manager.execute_tool("search_course_content", query="test")
        manager.reset_sources()

        assert tool.last_sources == []
