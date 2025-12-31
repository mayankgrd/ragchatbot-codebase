# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Retrieval-Augmented Generation (RAG) system for querying course materials. Built with FastAPI backend, ChromaDB vector store, and Anthropic's Claude AI with tool-based search. The system processes course documents into structured lessons, stores them in ChromaDB with semantic embeddings, and uses Claude with tool calling to intelligently search and answer questions.

## Development Commands

### Environment Setup
```bash
# Install dependencies (using uv package manager)
uv sync

# Set up environment variables (required)
# Create .env file with: ANTHROPIC_API_KEY=your_key_here
```

### Running the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start (from backend directory)
cd backend && uv run uvicorn app:app --reload --port 8000
```

Access points:
- Web UI: http://localhost:8000
- API docs: http://localhost:8000/docs

## Architecture Overview

### Data Flow Architecture

1. **Document Ingestion** (`document_processor.py`)
   - Parses course documents with expected format:
     - Line 1: `Course Title: [title]`
     - Line 2: `Course Link: [url]`
     - Line 3: `Course Instructor: [name]`
     - Following lines: Lesson markers (`Lesson N: Title`) and content
   - Chunks text into overlapping segments (800 chars, 100 char overlap)
   - Creates structured `Course` and `CourseChunk` objects

2. **Vector Storage** (`vector_store.py`)
   - Maintains two ChromaDB collections:
     - `course_catalog`: Course metadata (titles, instructors, lesson lists)
     - `course_content`: Actual lesson content chunks with metadata
   - Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings
   - Supports semantic search with course name resolution and lesson filtering

3. **Query Processing** (`rag_system.py`)
   - Orchestrates all components
   - Uses tool-based approach: AI decides when to search via `search_course_content` tool
   - Maintains conversation history via `SessionManager`
   - Returns answers with source attribution

4. **AI Generation** (`ai_generator.py`)
   - Interfaces with Claude API (claude-sonnet-4-20250514)
   - Handles tool execution flow:
     1. Initial request with tool definitions
     2. Tool execution if model requests it
     3. Final response generation with tool results
   - System prompt emphasizes concise, educational responses

### Key Design Patterns

**Tool-Based Search Architecture**
- AI uses `CourseSearchTool` via Anthropic's tool calling
- Tool handles semantic course name matching (partial matches work)
- Supports filtering by course name and/or lesson number
- Sources tracked and returned to UI separately from response

**Dual Collection Strategy**
- Course catalog enables fuzzy course name matching
- Content collection stores actual searchable material
- Course titles used as unique IDs across both collections

**Chunk Context Enhancement**
- First chunk of each lesson: `"Lesson N content: {chunk}"`
- Last lesson chunks: `"Course {title} Lesson N content: {chunk}"`
- Improves retrieval relevance by adding metadata to content

## Configuration

Key settings in `backend/config.py`:
- `CHUNK_SIZE`: 800 characters (semantic chunking)
- `CHUNK_OVERLAP`: 100 characters (context preservation)
- `MAX_RESULTS`: 5 search results per query
- `MAX_HISTORY`: 2 conversation turns retained
- `ANTHROPIC_MODEL`: claude-sonnet-4-20250514
- `EMBEDDING_MODEL`: all-MiniLM-L6-v2

## Working with Course Documents

Documents must follow this structure:
```
Course Title: [Your Course Title]
Course Link: [URL]
Course Instructor: [Name]

Lesson 0: Introduction
Lesson Link: [URL]
[Lesson content...]

Lesson 1: Next Topic
Lesson Link: [URL]
[Lesson content...]
```

Add documents to `docs/` folder - they're loaded automatically on startup unless already present (deduplication by course title).

## Common Patterns

### Adding New Tools
1. Inherit from `Tool` ABC in `search_tools.py`
2. Implement `get_tool_definition()` returning Anthropic tool schema
3. Implement `execute(**kwargs)` returning string results
4. Register with `ToolManager` in `RAGSystem.__init__`

### Modifying Search Behavior
- Course name resolution: `VectorStore._resolve_course_name()`
- Filter logic: `VectorStore._build_filter()`
- Result formatting: `CourseSearchTool._format_results()`

### Extending Document Processing
- File format handling: `DocumentProcessor.read_file()`
- Parsing logic: `DocumentProcessor.process_course_document()`
- Chunking strategy: `DocumentProcessor.chunk_text()`

## API Endpoints

- `POST /api/query`: Process queries with optional `session_id` for conversation context
- `GET /api/courses`: Get course analytics (total count, titles list)
- Static files served from `frontend/` directory

## Environment

- Python 3.13+ required
- ChromaDB data persisted in `backend/chroma_db/`
- Frontend: Vanilla HTML/CSS/JS (no build step)
- Windows users: Use Git Bash for shell commands
