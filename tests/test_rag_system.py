import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

from rag_system import RAGSystem


class TestRAGSystem:
    """Tests for RAGSystem orchestrator"""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration"""
        config = Mock()
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "/tmp/test_chroma"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.ANTHROPIC_API_KEY = "test-key"
        config.ANTHROPIC_MODEL = "claude-test"
        config.MAX_HISTORY = 2
        return config

    @pytest.fixture
    def rag_system(self, mock_config):
        """Create RAGSystem with mocked components"""
        with patch("rag_system.DocumentProcessor"), \
             patch("rag_system.VectorStore"), \
             patch("rag_system.AIGenerator") as mock_ai, \
             patch("rag_system.SessionManager"):
            system = RAGSystem(mock_config)
            return system, mock_ai.return_value

    def test_both_tools_registered(self, rag_system):
        """RAGSystem should register both search and outline tools"""
        system, _ = rag_system

        tool_names = list(system.tool_manager.tools.keys())
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    def test_query_passes_tools_to_ai_generator(self, rag_system):
        """Query should pass tool definitions to AI generator"""
        system, mock_ai = rag_system
        mock_ai.generate_response.return_value = "Test response"

        system.query("What is MCP?")

        call_args = mock_ai.generate_response.call_args
        assert call_args.kwargs["tools"] is not None
        assert len(call_args.kwargs["tools"]) == 2

    def test_query_passes_tool_manager(self, rag_system):
        """Query should pass tool manager to AI generator"""
        system, mock_ai = rag_system
        mock_ai.generate_response.return_value = "Test response"

        system.query("What is MCP?")

        call_args = mock_ai.generate_response.call_args
        assert call_args.kwargs["tool_manager"] == system.tool_manager

    def test_extract_cited_sources_filters_unused(self, rag_system):
        """Extract cited sources should filter out uncited sources"""
        system, _ = rag_system

        response = "This is about MCP [1] and protocols [3]."
        all_sources = [
            {"citation_num": 1, "title": "Source 1", "url": "url1"},
            {"citation_num": 2, "title": "Source 2", "url": "url2"},
            {"citation_num": 3, "title": "Source 3", "url": "url3"},
        ]

        updated_response, cited = system._extract_cited_sources(response, all_sources)

        # Only sources 1 and 3 should be included
        assert len(cited) == 2
        titles = [s["title"] for s in cited]
        assert "Source 1" in titles
        assert "Source 3" in titles
        assert "Source 2" not in titles

    def test_extract_cited_sources_renumbers_sequentially(self, rag_system):
        """Extract cited sources should renumber citations sequentially"""
        system, _ = rag_system

        response = "First point [3] and second point [5]."
        all_sources = [
            {"citation_num": 3, "title": "Source 3", "url": "url3"},
            {"citation_num": 5, "title": "Source 5", "url": "url5"},
        ]

        updated_response, cited = system._extract_cited_sources(response, all_sources)

        # Citations should be renumbered to [1] and [2]
        assert "[1]" in updated_response
        assert "[2]" in updated_response
        assert "[3]" not in updated_response
        assert "[5]" not in updated_response

        # Cited sources should have new numbers
        assert cited[0]["citation_num"] == 1
        assert cited[1]["citation_num"] == 2

    def test_extract_cited_sources_empty_sources(self, rag_system):
        """Extract cited sources should handle empty sources"""
        system, _ = rag_system

        response = "No citations here."
        all_sources = []

        updated_response, cited = system._extract_cited_sources(response, all_sources)

        assert updated_response == "No citations here."
        assert cited == []

    def test_extract_cited_sources_no_citations_in_response(self, rag_system):
        """Extract cited sources should handle response with no citations"""
        system, _ = rag_system

        response = "This response has no citations."
        all_sources = [
            {"citation_num": 1, "title": "Source 1", "url": "url1"},
        ]

        updated_response, cited = system._extract_cited_sources(response, all_sources)

        assert updated_response == "This response has no citations."
        assert cited == []

    def test_query_resets_sources_after_retrieval(self, rag_system):
        """Query should reset sources after retrieving them"""
        system, mock_ai = rag_system
        mock_ai.generate_response.return_value = "Test response"

        # Mock the tool manager methods
        system.tool_manager.get_last_sources = Mock(return_value=[])
        system.tool_manager.reset_sources = Mock()

        system.query("What is MCP?")

        system.tool_manager.reset_sources.assert_called_once()


class TestToolSelectionIntegration:
    """Integration tests for verifying correct tool selection based on query type"""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration"""
        config = Mock()
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "/tmp/test_chroma"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.ANTHROPIC_API_KEY = "test-key"
        config.ANTHROPIC_MODEL = "claude-test"
        config.MAX_HISTORY = 2
        return config

    def test_tool_definitions_available_for_ai(self, mock_config):
        """Tool definitions should be correctly formatted for AI"""
        with patch("rag_system.DocumentProcessor"), \
             patch("rag_system.VectorStore"), \
             patch("rag_system.AIGenerator"), \
             patch("rag_system.SessionManager"):
            system = RAGSystem(mock_config)

            definitions = system.tool_manager.get_tool_definitions()

            # Verify search tool definition
            search_def = next(d for d in definitions if d["name"] == "search_course_content")
            assert "query" in search_def["input_schema"]["properties"]

            # Verify outline tool definition
            outline_def = next(d for d in definitions if d["name"] == "get_course_outline")
            assert "course_title" in outline_def["input_schema"]["properties"]

    def test_search_tool_can_be_executed(self, mock_config):
        """Search tool should be executable through tool manager"""
        with patch("rag_system.DocumentProcessor"), \
             patch("rag_system.VectorStore") as mock_store_class, \
             patch("rag_system.AIGenerator"), \
             patch("rag_system.SessionManager"):

            # Setup mock vector store
            mock_store = Mock()
            mock_store.search.return_value = Mock(
                documents=["Test content"],
                metadata=[{"course_title": "Test", "lesson_number": 1}],
                distances=[0.1],
                error=None,
                is_empty=lambda: False
            )
            mock_store.get_lesson_link.return_value = "http://test.com"
            mock_store_class.return_value = mock_store

            system = RAGSystem(mock_config)

            result = system.tool_manager.execute_tool(
                "search_course_content",
                query="test query"
            )

            assert "[1]" in result
            mock_store.search.assert_called_once()

    def test_outline_tool_can_be_executed(self, mock_config):
        """Outline tool should be executable through tool manager"""
        with patch("rag_system.DocumentProcessor"), \
             patch("rag_system.VectorStore") as mock_store_class, \
             patch("rag_system.AIGenerator"), \
             patch("rag_system.SessionManager"):

            # Setup mock vector store
            mock_store = Mock()
            mock_store.get_course_metadata.return_value = {
                "title": "Test Course",
                "instructor": "Test Instructor",
                "course_link": "http://test.com",
                "lesson_count": 2,
                "lessons": [
                    {"lesson_number": 0, "lesson_title": "Intro", "lesson_link": "http://l0"},
                    {"lesson_number": 1, "lesson_title": "Basics", "lesson_link": "http://l1"},
                ]
            }
            mock_store_class.return_value = mock_store

            system = RAGSystem(mock_config)

            result = system.tool_manager.execute_tool(
                "get_course_outline",
                course_title="Test Course"
            )

            assert "Test Course" in result
            assert "Test Instructor" in result
            assert "Intro" in result
            mock_store.get_course_metadata.assert_called_once_with("Test Course")
