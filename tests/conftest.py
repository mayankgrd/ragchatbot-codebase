import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from vector_store import SearchResults


@pytest.fixture
def sample_course_metadata():
    """Sample course metadata for testing"""
    return {
        "title": "Introduction to MCP",
        "instructor": "John Doe",
        "course_link": "https://example.com/mcp-course",
        "lesson_count": 3,
        "lessons": [
            {"lesson_number": 0, "lesson_title": "Getting Started", "lesson_link": "https://example.com/lesson0"},
            {"lesson_number": 1, "lesson_title": "Core Concepts", "lesson_link": "https://example.com/lesson1"},
            {"lesson_number": 2, "lesson_title": "Advanced Topics", "lesson_link": "https://example.com/lesson2"},
        ]
    }


@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    return SearchResults(
        documents=[
            "MCP allows models to connect to external tools.",
            "The protocol uses JSON-RPC for communication.",
        ],
        metadata=[
            {"course_title": "Introduction to MCP", "lesson_number": 1},
            {"course_title": "Introduction to MCP", "lesson_number": 2},
        ],
        distances=[0.1, 0.2]
    )


@pytest.fixture
def empty_search_results():
    """Empty search results for testing"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )


@pytest.fixture
def error_search_results():
    """Search results with error for testing"""
    return SearchResults.empty("No course found matching 'Unknown Course'")


@pytest.fixture
def mock_vector_store(sample_search_results, sample_course_metadata):
    """Mock VectorStore for testing tools"""
    store = Mock()
    store.search.return_value = sample_search_results
    store.get_course_metadata.return_value = sample_course_metadata
    store.get_lesson_link.return_value = "https://example.com/lesson1"
    return store


@pytest.fixture
def mock_vector_store_empty(empty_search_results):
    """Mock VectorStore that returns empty results"""
    store = Mock()
    store.search.return_value = empty_search_results
    store.get_course_metadata.return_value = None
    return store


@pytest.fixture
def mock_anthropic_response_text():
    """Mock Anthropic response with text only"""
    response = Mock()
    response.stop_reason = "end_turn"
    text_block = Mock()
    text_block.type = "text"
    text_block.text = "This is a test response."
    response.content = [text_block]
    return response


@pytest.fixture
def mock_anthropic_response_text_with_citations():
    """Mock Anthropic response with text that includes citations"""
    response = Mock()
    response.stop_reason = "end_turn"
    text_block = Mock()
    text_block.type = "text"
    text_block.text = "This is a test response with citations [1]."
    response.content = [text_block]
    return response


@pytest.fixture
def mock_anthropic_response_tool_use():
    """Mock Anthropic response with tool use"""
    response = Mock()
    response.stop_reason = "tool_use"

    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.id = "tool_123"
    tool_block.name = "search_course_content"
    tool_block.input = {"query": "MCP protocol"}

    response.content = [tool_block]
    return response


@pytest.fixture
def mock_tool_manager():
    """Mock ToolManager for testing"""
    manager = Mock()
    manager.execute_tool.return_value = "Tool execution result"
    manager.get_tool_definitions.return_value = [
        {"name": "search_course_content", "description": "Search course content"},
        {"name": "get_course_outline", "description": "Get course outline"}
    ]
    manager.get_all_sources.return_value = []
    manager.reset_sources.return_value = None
    return manager


@pytest.fixture
def mock_anthropic_response_tool_use_mcp():
    """Mock Anthropic response with tool use for MCP search"""
    response = Mock()
    response.stop_reason = "tool_use"

    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.id = "tool_mcp_123"
    tool_block.name = "search_course_content"
    tool_block.input = {"query": "MCP protocol", "course_name": "MCP"}

    response.content = [tool_block]
    return response


@pytest.fixture
def mock_anthropic_response_tool_use_chroma():
    """Mock Anthropic response with tool use for Chroma search"""
    response = Mock()
    response.stop_reason = "tool_use"

    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.id = "tool_chroma_456"
    tool_block.name = "search_course_content"
    tool_block.input = {"query": "Chroma retrieval", "course_name": "Chroma"}

    response.content = [tool_block]
    return response


def create_mock_tool_response(tool_name: str, tool_input: dict, tool_id: str = "tool_123"):
    """Helper to create mock tool use responses"""
    response = Mock()
    response.stop_reason = "tool_use"

    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.id = tool_id
    tool_block.name = tool_name
    tool_block.input = tool_input

    response.content = [tool_block]
    return response


def create_mock_text_response(text: str):
    """Helper to create mock text responses"""
    response = Mock()
    response.stop_reason = "end_turn"

    text_block = Mock()
    text_block.type = "text"
    text_block.text = text

    response.content = [text_block]
    return response
