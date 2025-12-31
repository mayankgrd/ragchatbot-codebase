# Codebase Architecture Overview

This is a **Course Materials RAG (Retrieval-Augmented Generation) System** - a full-stack web application that enables intelligent question-answering about educational course materials using AI.

## Directory Structure

```
starting-ragchatbot-codebase/
├── backend/           # Python FastAPI backend (1,186 LOC)
├── docs/              # Course document storage (4 course transcripts)
├── frontend/          # Static web interface (HTML/CSS/JS)
└── .venv/             # Python virtual environment
```

---

## Core Components & Implementation Files

### **1. RAG System Orchestration**
**File:** `backend/rag_system.py` (146 lines)
- Main coordinator that orchestrates all RAG operations
- Methods: `add_course_document()`, `add_course_folder()`, `query()`, `get_course_analytics()`
- Integrates: document processor, vector store, AI generator, session manager, tool manager

### **2. Document Processing**
**File:** `backend/document_processor.py` (259 lines)
- Parses course documents (TXT, PDF, DOCX)
- Extracts metadata: course title, instructor, lesson markers
- Chunks content using sentence-based splitting (800 chars, 100 char overlap)
- Adds contextual prefixes to chunks for better search

### **3. Vector Search & Storage**
**File:** `backend/vector_store.py` (266 lines)
- Manages ChromaDB vector database with two collections:
  - `course_catalog`: Course metadata (titles, instructors, lessons)
  - `course_content`: Chunked content for semantic search
- Uses embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Semantic course name resolution with fuzzy matching
- Filtered search by course title and/or lesson number

### **4. AI Response Generation**
**File:** `backend/ai_generator.py` (134 lines)
- Interface with Anthropic's Claude API (claude-sonnet-4-20250514)
- Tool calling integration for search capabilities
- Conversation history support
- Two-phase generation: tool request → execution → final response

### **5. Search Tools System**
**File:** `backend/search_tools.py` (153 lines)
- Provides Claude with structured `search_course_content` tool
- Parameters: query, course_name (optional), lesson_number (optional)
- Manages tool registration and execution
- Tracks and returns source citations

### **6. Session Management**
**File:** `backend/session_manager.py` (60 lines)
- In-memory conversation history per session ID
- Keeps last 2 exchanges (4 messages)
- Enables context-aware responses

### **7. Web API Layer**
**File:** `backend/app.py` (118 lines)
- FastAPI application entry point
- **Endpoints:**
  - `POST /api/query`: Process user queries
  - `GET /api/courses`: Retrieve course statistics
  - `GET /`: Serve frontend static files
- CORS enabled, auto-loads documents on startup

### **8. Data Models**
**File:** `backend/models.py` (21 lines)
- Pydantic models: `Lesson`, `Course`, `CourseChunk`
- Data validation and serialization

### **9. Configuration**
**File:** `backend/config.py` (29 lines)
- Environment variable management
- Settings: API keys, model names, chunk sizes, search limits

### **10. Frontend Interface**
**Files:** `frontend/index.html`, `frontend/script.js`, `frontend/style.css` (719 lines CSS)
- Vanilla JavaScript SPA
- Dark theme with course stats sidebar
- Chat interface with Markdown rendering (Marked.js)
- Session-based conversations with source citations

---

## Technology Stack

**Backend:**
- Python 3.13+ with FastAPI & Uvicorn
- Anthropic Claude API (claude-sonnet-4-20250514)
- ChromaDB 1.0.15 (vector database)
- sentence-transformers 5.0.0 (embeddings)

**Frontend:**
- HTML5, CSS3, Vanilla JavaScript
- Marked.js for Markdown rendering

**Tools:**
- uv (package manager)
- pyproject.toml (dependencies)

---

## Execution Flow

### **Startup:**
1. `run.sh` → starts Uvicorn server on port 8000
2. `app.py` startup event → loads documents from `docs/` directory
3. Documents processed through `document_processor.py` → chunked
4. Chunks stored in ChromaDB via `vector_store.py`

### **Query Processing:**
1. User enters question in frontend → `POST /api/query`
2. `rag_system.py` receives query with session ID
3. `ai_generator.py` calls Claude API with search tool definition
4. Claude decides to use `search_course_content` tool
5. `search_tools.py` executes search via `vector_store.py`
6. ChromaDB performs semantic search on chunks
7. Results returned to Claude → generates final response
8. Response + sources returned to frontend
9. Frontend renders Markdown answer with citations

---

## Key Configuration Files

- **`pyproject.toml`**: Python project metadata & dependencies
- **`uv.lock`**: Dependency lock file (245KB)
- **`.env`**: API keys (gitignored)
- **`.env.example`**: Environment template
- **`run.sh`**: Startup script
- **`README.md`**: Project documentation

---

This is a well-architected educational assistant that combines semantic search, AI generation, and a clean web interface to answer questions about course materials intelligently.
