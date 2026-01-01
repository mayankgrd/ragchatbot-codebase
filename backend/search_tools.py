from typing import Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
from vector_store import VectorStore, SearchResults


class Tool(ABC):
    """Abstract base class for all tools"""
    
    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.all_sources = []      # Accumulated sources across searches
        self._source_counter = 0   # Tracks next citation number for cumulative numbering
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "What to search for in the course content"
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)"
                    }
                },
                "required": ["query"]
            }
        }
    
    def execute(self, query: str, course_name: Optional[str] = None, lesson_number: Optional[int] = None) -> str:
        """
        Execute the search tool with given parameters.
        
        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter
            
        Returns:
            Formatted search results or error message
        """
        
        # Use the vector store's unified search interface
        results = self.store.search(
            query=query,
            course_name=course_name,
            lesson_number=lesson_number
        )
        
        # Handle errors
        if results.error:
            return results.error
        
        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."
        
        # Format and return results
        return self._format_results(results)
    
    def _format_results(self, results: SearchResults) -> str:
        """Format search results with cumulative citation numbers for AI to reference.

        Citation numbers accumulate across multiple tool calls within a query,
        ensuring consistent numbering (e.g., first search: [1][2][3], second: [4][5][6]).
        """
        formatted = []

        for doc, meta in zip(results.documents, results.metadata):
            # Increment counter before using (cumulative across calls)
            self._source_counter += 1
            citation_num = self._source_counter

            course_title = meta.get('course_title', 'unknown')
            lesson_num = meta.get('lesson_number')

            # Build source title
            source_title = course_title
            if lesson_num is not None:
                source_title += f" - Lesson {lesson_num}"

            # Get lesson link from vector store
            lesson_link = None
            if lesson_num is not None:
                lesson_link = self.store.get_lesson_link(course_title, lesson_num)

            # Accumulate source with cumulative citation number
            self.all_sources.append({
                "citation_num": citation_num,
                "title": source_title,
                "url": lesson_link
            })

            # Format with cumulative numbered reference for AI to cite
            header = f"[{citation_num}] {source_title}"
            formatted.append(f"{header}\n{doc}")

        return "\n\n".join(formatted)

    def reset_sources(self):
        """Reset accumulated sources and counter for new query."""
        self.all_sources = []
        self._source_counter = 0


class CourseOutlineTool(Tool):
    """Tool for retrieving course structure and lesson list"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "get_course_outline",
            "description": "Get course structure including title, link, instructor, and complete lesson list with links. Use for questions about course syllabus, what topics a course covers, or listing lessons.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_title": {
                        "type": "string",
                        "description": "Course title to look up (partial matches work, e.g. 'MCP', 'Introduction')"
                    }
                },
                "required": ["course_title"]
            }
        }

    def execute(self, course_title: str) -> str:
        """
        Execute the outline tool to get course structure.

        Args:
            course_title: Course name to look up

        Returns:
            Formatted course outline or error message
        """
        metadata = self.store.get_course_metadata(course_title)

        if not metadata:
            return f"No course found matching '{course_title}'"

        return self._format_outline(metadata)

    def _format_outline(self, metadata: Dict[str, Any]) -> str:
        """Format course metadata as readable outline"""
        lines = [
            f"Course: {metadata.get('title', 'Unknown')}",
            f"Instructor: {metadata.get('instructor', 'Unknown')}",
            f"Course Link: {metadata.get('course_link', 'N/A')}",
            f"Total Lessons: {metadata.get('lesson_count', 0)}",
            "",
            "Lessons:"
        ]

        lessons = metadata.get('lessons', [])
        for lesson in lessons:
            lesson_num = lesson.get('lesson_number', '?')
            lesson_title = lesson.get('lesson_title', 'Untitled')
            lesson_link = lesson.get('lesson_link', '')

            if lesson_link:
                lines.append(f"  {lesson_num}. {lesson_title} - {lesson_link}")
            else:
                lines.append(f"  {lesson_num}. {lesson_title}")

        return "\n".join(lines)


class ToolManager:
    """Manages available tools for the AI"""
    
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    
    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"
        
        return self.tools[tool_name].execute(**kwargs)
    
    def get_all_sources(self) -> list:
        """Get accumulated sources from all search operations.

        Sources accumulate across multiple tool calls within a query,
        with cumulative citation numbering.
        """
        all_sources = []
        for tool in self.tools.values():
            if hasattr(tool, 'all_sources'):
                all_sources.extend(tool.all_sources)
        return all_sources

    def reset_sources(self):
        """Reset sources and counters from all tools that track sources."""
        for tool in self.tools.values():
            if hasattr(tool, 'reset_sources'):
                tool.reset_sources()
            elif hasattr(tool, 'all_sources'):
                tool.all_sources = []