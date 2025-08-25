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
# Quick start using run script
chmod +x run.sh
./run.sh

# Manual start 
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Code Quality Tools
```bash
# Format code with Black and sort imports with isort
chmod +x format.sh
./format.sh

# Run all quality checks (linting, type checking, formatting)
chmod +x lint.sh
./lint.sh

# Individual quality checks
uv run black --check backend/ main.py     # Check formatting
uv run isort --check-only backend/ main.py # Check import sorting
uv run flake8 backend/ main.py            # Linting
uv run mypy backend/ main.py              # Type checking
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
- **AIGenerator** (`backend/ai_generator.py`): Anthropic Claude integration for response generation
- **SessionManager** (`backend/session_manager.py`): Manages conversation history
- **ToolManager/CourseSearchTool** (`backend/search_tools.py`): Tool-based search functionality

### Application Structure
- **Backend**: FastAPI application (`backend/app.py`) serving REST API and static files
- **Frontend**: Simple HTML/CSS/JS interface (`frontend/`) served as static files
- **Document Storage**: Course materials in `docs/` folder (auto-loaded on startup)
- **Vector Database**: ChromaDB storage in `./chroma_db` directory

### Key Configuration
- Configuration managed via `backend/config.py` with dataclass pattern
- Environment variables loaded from `.env` file
- ChromaDB path: `./chroma_db`
- Embedding model: `all-MiniLM-L6-v2`
- Claude model: `claude-sonnet-4-20250514`

### Document Processing Flow
1. Documents are processed from `docs/` folder on application startup
2. Text is chunked into 800-character segments with 100-character overlap
3. Embeddings are created using sentence-transformers
4. Course metadata and content chunks are stored in ChromaDB
5. Tool-based search retrieves relevant context for AI responses

### API Endpoints
- `POST /api/query`: Process user queries and return AI responses with sources
- `GET /api/courses`: Get course statistics and analytics
- Always use uv to run the server
- Use uv to run Python files