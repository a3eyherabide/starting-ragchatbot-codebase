import sys
import os
import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
import tempfile
import shutil
from typing import Generator

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults


class MockConfig:
    """Mock configuration for testing"""
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    CHROMA_PATH = "./test_chroma_db"
    EMBEDDING_MODEL = "test_model"
    MAX_RESULTS = 5
    ANTHROPIC_API_KEY = "test_key"
    ANTHROPIC_MODEL = "test_model"
    MAX_HISTORY = 2


@pytest.fixture
def mock_config():
    """Provide mock configuration for tests"""
    return MockConfig()


@pytest.fixture
def sample_course():
    """Sample course for testing"""
    lessons = [
        Lesson(title="Introduction", content="This is an introduction to the course."),
        Lesson(title="Advanced Topics", content="Advanced concepts and practices.")
    ]
    return Course(title="Test Course", lessons=lessons)


@pytest.fixture
def sample_course_chunks():
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="This is an introduction to the course. It covers basic concepts.",
            course_title="Test Course",
            chunk_index=0,
            lesson_title="Introduction"
        ),
        CourseChunk(
            content="Advanced concepts and practices are discussed here.",
            course_title="Test Course", 
            chunk_index=1,
            lesson_title="Advanced Topics"
        )
    ]


@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    chunks = [
        CourseChunk(
            content="Model Context Protocol (MCP) is a standardized protocol.",
            course_title="MCP Course",
            chunk_index=0,
            lesson_title="Introduction to MCP"
        )
    ]
    return SearchResults(chunks=chunks, query="What is MCP?", total_found=1)


@pytest.fixture
def mock_rag_system():
    """Mock RAG system for API testing"""
    with patch('app.RAGSystem') as mock_rag_class:
        mock_rag = Mock()
        mock_rag_class.return_value = mock_rag
        
        # Default mock behaviors
        mock_rag.query.return_value = ("Test response", [{"text": "Test Course - Lesson 1", "link": None}])
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": ["Course A", "Course B", "Course C"]
        }
        mock_rag.session_manager.create_session.return_value = "test_session_123"
        
        yield mock_rag


@pytest.fixture
def test_app_without_static():
    """Create a test FastAPI app without static file mounting to avoid import issues"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional, Union, Dict, Any
    
    # Create test app
    app = FastAPI(title="Test Course Materials RAG System", root_path="")
    
    # Add middlewares
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Pydantic models for request/response
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None
    
    class QueryResponse(BaseModel):
        answer: str
        sources: Union[List[str], List[Dict[str, Any]]]
        session_id: str
    
    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # Mock RAG system instance
    mock_rag_system = Mock()
    mock_rag_system.query.return_value = ("Test response", [{"text": "Test source", "link": None}])
    mock_rag_system.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Course 1", "Course 2"]
    }
    mock_rag_system.session_manager.create_session.return_value = "test_session_456"
    
    # API Endpoints
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            
            answer, sources = mock_rag_system.query(request.query, session_id)
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System API"}
    
    # Store mock for access in tests
    app.state.mock_rag_system = mock_rag_system
    
    return app


@pytest.fixture
def test_client(test_app_without_static):
    """Create a test client for the FastAPI app"""
    return TestClient(test_app_without_static)


@pytest.fixture
def temp_dir():
    """Create and cleanup temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress warnings during tests"""
    import warnings
    warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning)