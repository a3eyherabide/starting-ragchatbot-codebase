# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Installation
```bash
# Install dependencies using uv
uv sync

# Create .env file with required environment variables
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Running the Application
```bash
# Quick start using run script (recommended)
chmod +x run.sh
./run.sh

# Manual start 
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Testing
```bash
# Run all tests
cd backend && uv run pytest

# Run specific test files
cd backend && uv run pytest tests/test_rag_system.py
cd backend && uv run pytest tests/test_ai_generator.py
cd backend && uv run pytest tests/test_vector_store.py
```

### Accessing the Application
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Architecture Overview

This is a Retrieval-Augmented Generation (RAG) system for querying course materials. The system consists of:

### Core Components
- **RAGSystem** (`backend/rag_system.py`): Main orchestrator that coordinates all components
- **VectorStore** (`backend/vector_store.py`): ChromaDB integration for semantic search
- **DocumentProcessor** (`backend/document_processor.py`): Processes course documents into chunks
- **AIGenerator** (`backend/ai_generator.py`): Anthropic Claude integration with tool support
- **SessionManager** (`backend/session_manager.py`): Manages conversation history
- **CourseSearchTool** (`backend/search_tools.py`): Tool-based search with semantic course matching
- **Models** (`backend/models.py`): Pydantic data models for Course, Lesson, and CourseChunk
- **Config** (`backend/config.py`): Centralized configuration with environment variables

### Application Structure
- **Backend**: FastAPI application (`backend/app.py`) serving REST API and static files
- **Frontend**: Simple HTML/CSS/JS interface (`frontend/`) served as static files
- **Document Storage**: Course materials in `docs/` folder (auto-loaded on startup)
- **Vector Database**: ChromaDB storage in `./chroma_db` directory

### Key Configuration
- Configuration managed via `backend/config.py` with dataclass pattern
- Environment variables loaded from `.env` file
- ChromaDB path: `./backend/chroma_db` (auto-created)
- Embedding model: `all-MiniLM-L6-v2`
- Claude model: `claude-3-haiku-20240307` (faster, cheaper option)
- Alternative model: `claude-sonnet-4-20250514` (commented out, higher quality but more expensive)
- Document chunking: 800 characters with 100 character overlap
- Max search results: 5
- Conversation history: 2 messages

### Document Processing Flow
1. Documents are processed from `docs/` folder on application startup
2. Text is chunked into 800-character segments with 100-character overlap
3. Embeddings are created using sentence-transformers
4. Course metadata and content chunks are stored in ChromaDB
5. Tool-based search retrieves relevant context for AI responses

### API Endpoints
- `POST /api/query`: Process user queries and return AI responses with sources
  - Request: `{"query": "string", "session_id": "optional_string"}`
  - Response: `{"answer": "string", "sources": [...], "session_id": "string"}`
- `GET /api/courses`: Get course statistics and analytics
  - Response: `{"total_courses": int, "course_titles": [...]}`

### Testing Structure
- **test_rag_system.py**: Integration tests for RAG system
- **test_ai_generator.py**: AI generator and tool integration tests
- **test_vector_store.py**: Vector store functionality tests
- **test_course_search_tool.py**: Course search tool tests
- **test_integration_real.py**: Real API integration tests
- **test_sequential_demo.py**: Sequential tool calling demonstrations

### Development Guidelines
- Always use `uv` to run the server and Python files
- Use `uv` to install all dependencies
- Configuration changes should be made in `backend/config.py`
- New tools should inherit from the `Tool` abstract base class in `search_tools.py`
- All API models are defined in `backend/models.py`