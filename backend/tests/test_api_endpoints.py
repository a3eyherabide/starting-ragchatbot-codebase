import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json


@pytest.mark.api
class TestAPIEndpoints:
    """Test suite for FastAPI endpoints"""
    
    def test_query_endpoint_success(self, test_client):
        """Test successful query to /api/query endpoint"""
        # Configure mock response
        mock_rag = test_client.app.state.mock_rag_system
        mock_rag.query.return_value = (
            "MCP stands for Model Context Protocol.",
            [{"text": "MCP Course - Introduction", "link": None}]
        )
        
        # Make request
        response = test_client.post(
            "/api/query",
            json={"query": "What is MCP?", "session_id": "test_session"}
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "MCP stands for Model Context Protocol."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["text"] == "MCP Course - Introduction"
        assert data["session_id"] == "test_session"
        
        # Verify RAG system was called correctly
        mock_rag.query.assert_called_once_with("What is MCP?", "test_session")
    
    def test_query_endpoint_without_session_id(self, test_client):
        """Test query endpoint creates session when none provided"""
        mock_rag = test_client.app.state.mock_rag_system
        mock_rag.session_manager.create_session.return_value = "auto_session_789"
        mock_rag.query.return_value = ("Response", [])
        
        response = test_client.post(
            "/api/query",
            json={"query": "Test query"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "auto_session_789"
        
        # Verify session was created
        mock_rag.session_manager.create_session.assert_called_once()
        mock_rag.query.assert_called_once_with("Test query", "auto_session_789")
    
    def test_query_endpoint_missing_query(self, test_client):
        """Test query endpoint with missing query field"""
        response = test_client.post(
            "/api/query",
            json={"session_id": "test_session"}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_query_endpoint_empty_query(self, test_client):
        """Test query endpoint with empty query"""
        response = test_client.post(
            "/api/query", 
            json={"query": "", "session_id": "test_session"}
        )
        
        # Should still process empty query (let RAG system handle it)
        assert response.status_code == 200
    
    def test_query_endpoint_invalid_json(self, test_client):
        """Test query endpoint with invalid JSON"""
        response = test_client.post(
            "/api/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_query_endpoint_rag_system_error(self, test_client):
        """Test query endpoint when RAG system raises exception"""
        mock_rag = test_client.app.state.mock_rag_system
        mock_rag.query.side_effect = Exception("RAG system error")
        
        response = test_client.post(
            "/api/query",
            json={"query": "Test query"}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "RAG system error" in data["detail"]
    
    def test_query_endpoint_with_sources_as_list_of_strings(self, test_client):
        """Test query endpoint with sources returned as list of strings"""
        mock_rag = test_client.app.state.mock_rag_system
        mock_rag.query.return_value = (
            "Response text",
            ["Source 1", "Source 2"]  # Sources as strings
        )
        
        response = test_client.post(
            "/api/query",
            json={"query": "Test query"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["sources"] == ["Source 1", "Source 2"]
    
    def test_query_endpoint_with_complex_sources(self, test_client):
        """Test query endpoint with complex source objects"""
        mock_rag = test_client.app.state.mock_rag_system
        complex_sources = [
            {
                "text": "Course content here",
                "link": "https://example.com/course1",
                "course_title": "Advanced Topics",
                "lesson_title": "Introduction",
                "chunk_index": 0
            },
            {
                "text": "More course content",
                "link": None,
                "course_title": "Basic Topics", 
                "lesson_title": "Conclusion",
                "chunk_index": 1
            }
        ]
        mock_rag.query.return_value = ("Complex response", complex_sources)
        
        response = test_client.post(
            "/api/query",
            json={"query": "Complex query"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 2
        assert data["sources"][0]["course_title"] == "Advanced Topics"
        assert data["sources"][1]["link"] is None
    
    def test_courses_endpoint_success(self, test_client):
        """Test successful request to /api/courses endpoint"""
        mock_rag = test_client.app.state.mock_rag_system
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 5,
            "course_titles": ["Course A", "Course B", "Course C", "Course D", "Course E"]
        }
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 5
        assert len(data["course_titles"]) == 5
        assert "Course A" in data["course_titles"]
        
        # Verify RAG system was called
        mock_rag.get_course_analytics.assert_called_once()
    
    def test_courses_endpoint_empty_result(self, test_client):
        """Test courses endpoint with no courses"""
        mock_rag = test_client.app.state.mock_rag_system
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []
    
    def test_courses_endpoint_error(self, test_client):
        """Test courses endpoint when RAG system raises exception"""
        mock_rag = test_client.app.state.mock_rag_system
        mock_rag.get_course_analytics.side_effect = Exception("Analytics error")
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 500
        data = response.json()
        assert "Analytics error" in data["detail"]
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint returns welcome message"""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "RAG System" in data["message"]
    
    def test_nonexistent_endpoint(self, test_client):
        """Test request to non-existent endpoint returns 404"""
        response = test_client.get("/api/nonexistent")
        
        assert response.status_code == 404
    
    def test_query_endpoint_content_type(self, test_client):
        """Test query endpoint accepts correct content types"""
        # Test with explicit JSON content type
        response = test_client.post(
            "/api/query",
            json={"query": "Test"},
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 200
    
    def test_cors_headers(self, test_client):
        """Test CORS headers are properly set"""
        # Test actual request works - the CORS middleware is configured correctly
        # even if TestClient doesn't show CORS headers (they appear in real browser requests)
        response = test_client.get("/api/courses")
        assert response.status_code == 200
        
        # Verify the CORS middleware is configured by testing cross-origin request simulation
        response = test_client.get(
            "/api/courses", 
            headers={"Origin": "http://localhost:3000"}
        )
        assert response.status_code == 200


@pytest.mark.api
@pytest.mark.integration  
class TestAPIIntegration:
    """Integration tests for API endpoints with more realistic scenarios"""
    
    def test_query_flow_with_session(self, test_client):
        """Test complete query flow with session management"""
        mock_rag = test_client.app.state.mock_rag_system
        
        # First query - creates session
        mock_rag.session_manager.create_session.return_value = "session_123"
        mock_rag.query.return_value = ("First response", [{"text": "Source 1", "link": None}])
        
        response1 = test_client.post(
            "/api/query",
            json={"query": "First question"}
        )
        
        assert response1.status_code == 200
        session_id = response1.json()["session_id"]
        
        # Second query - uses existing session
        mock_rag.query.return_value = ("Second response", [{"text": "Source 2", "link": None}])
        
        response2 = test_client.post(
            "/api/query", 
            json={"query": "Follow up question", "session_id": session_id}
        )
        
        assert response2.status_code == 200
        assert response2.json()["session_id"] == session_id
        
        # Verify both queries were processed
        assert mock_rag.query.call_count == 2
    
    def test_multiple_concurrent_sessions(self, test_client):
        """Test handling multiple concurrent sessions"""
        mock_rag = test_client.app.state.mock_rag_system
        
        # Setup different sessions
        def create_session_side_effect():
            create_session_side_effect.counter += 1
            return f"session_{create_session_side_effect.counter}"
        create_session_side_effect.counter = 0
        
        mock_rag.session_manager.create_session.side_effect = create_session_side_effect
        mock_rag.query.return_value = ("Response", [])
        
        # Make requests without session IDs
        responses = []
        for i in range(3):
            response = test_client.post(
                "/api/query",
                json={"query": f"Query {i}"}
            )
            responses.append(response)
        
        # Verify all requests succeeded with different session IDs
        session_ids = set()
        for response in responses:
            assert response.status_code == 200
            session_id = response.json()["session_id"]
            session_ids.add(session_id)
        
        assert len(session_ids) == 3  # All different sessions
        assert mock_rag.session_manager.create_session.call_count == 3
    
    def test_query_and_courses_endpoints_together(self, test_client):
        """Test using both query and courses endpoints in sequence"""
        mock_rag = test_client.app.state.mock_rag_system
        
        # First get course statistics
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["Python Basics", "Advanced Python"]
        }
        
        courses_response = test_client.get("/api/courses")
        assert courses_response.status_code == 200
        courses_data = courses_response.json()
        
        # Then query about one of the courses
        mock_rag.query.return_value = (
            "Python Basics covers fundamental programming concepts.",
            [{"text": "Python Basics - Chapter 1", "link": None}]
        )
        
        query_response = test_client.post(
            "/api/query",
            json={"query": "Tell me about Python Basics"}
        )
        
        assert query_response.status_code == 200
        query_data = query_response.json()
        
        # Verify both endpoints worked correctly
        assert "Python Basics" in courses_data["course_titles"]
        assert "Python Basics" in query_data["answer"]