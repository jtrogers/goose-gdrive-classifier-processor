import pytest
import os
import json
import asyncio
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server import GDriveClassifierMCP, ProcessRequest, ValidationRequest, CustomReportRequest

@pytest.fixture
def test_config():
    return {
        "rubric_path": "tests/data/test_rubric.json",
        "confidence_thresholds": {
            "high": 90,
            "medium": 70,
            "low": 0
        },
        "batch_size": 2,
        "processing": {
            "max_retries": 3,
            "timeout_seconds": 300
        },
        "llm": {
            "model": "gpt-4-turbo-preview",
            "temperature": 0.3,
            "max_tokens": 1000
        }
    }

@pytest.fixture
def test_rubric():
    return {
        "categories": [
            {
                "name": "confidential",
                "description": "Contains sensitive information",
                "patterns": ["password", "secret", "private"],
                "keywords": ["confidential", "sensitive", "restricted"]
            },
            {
                "name": "public",
                "description": "Public information",
                "patterns": ["public", "open", "shared"],
                "keywords": ["public", "unrestricted", "general"]
            }
        ]
    }

@pytest.fixture
def mock_credentials():
    return {
        "token": "mock_token",
        "refresh_token": "mock_refresh_token",
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": "mock_client_id",
        "client_secret": "mock_client_secret",
        "scopes": ["https://www.googleapis.com/auth/drive"]
    }

@pytest.fixture
async def mcp_server(test_config, test_rubric, mock_credentials, tmp_path):
    # Create test directories and files
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(test_config, f)
    
    rubric_path = tmp_path / "rubric.json"
    with open(rubric_path, "w") as f:
        json.dump(test_rubric, f)
    
    token_path = tmp_path / "token.json"
    with open(token_path, "w") as f:
        json.dump(mock_credentials, f)
    
    # Set environment variables
    os.environ["CONFIG_PATH"] = str(config_path)
    os.environ["GOOGLE_TOKEN_PATH"] = str(token_path)
    os.environ["OPENAI_API_KEY"] = "mock_openai_key"
    
    # Create and return server instance
    server = GDriveClassifierMCP()
    yield server

@pytest.mark.asyncio
async def test_process_documents(mcp_server, mocker):
    # Mock document content and metadata
    mock_docs = {
        "doc1": {
            "content": "This is a confidential document containing sensitive information.",
            "metadata": {
                "id": "doc1",
                "name": "Confidential Doc",
                "mimeType": "application/vnd.google-apps.document"
            }
        },
        "doc2": {
            "content": "This is a public document for general use.",
            "metadata": {
                "id": "doc2",
                "name": "Public Doc",
                "mimeType": "text/plain"
            }
        }
    }
    
    # Mock Google Drive and OpenAI API calls
    mocker.patch.object(
        mcp_server.drive_service.files(),
        "get",
        side_effect=lambda fileId, **kwargs: mock_docs[fileId]["metadata"]
    )
    mocker.patch.object(
        mcp_server.drive_service.files(),
        "get_media",
        side_effect=lambda fileId: mock_docs[fileId]["content"].encode()
    )
    
    # Mock OpenAI response
    mock_openai_response = mocker.Mock()
    mock_openai_response.choices = [{
        "message": {
            "content": json.dumps({
                "categories": [
                    {
                        "name": "confidential",
                        "confidence": 95,
                        "reasoning": "Contains sensitive information"
                    }
                ],
                "overall_confidence": 95,
                "summary": "Confidential document"
            })
        }
    }]
    mocker.patch.object(
        mcp_server.openai_client.chat.completions,
        "create",
        return_value=mock_openai_response
    )
    
    # Test document processing
    request = ProcessRequest(document_ids=["doc1", "doc2"])
    results = []
    
    async for update in mcp_server.process_documents(request):
        results.append(update)
    
    # Verify results
    assert len(results) > 0
    assert results[-1]["progress"]["processed"] == 2
    assert results[-1]["progress"]["total"] == 2
    assert "batch_results" in results[-1]

@pytest.mark.asyncio
async def test_validate_classification(mcp_server, mocker):
    # Mock classified documents
    mock_files = {
        "files": [
            {
                "id": "doc1",
                "name": "Test Doc 1",
                "properties": {
                    "classified": "true",
                    "classification_date": "2025-03-18T00:00:00Z",
                    "classification_summary": "Confidential document",
                    "overall_confidence": "95",
                    "categories": "confidential"
                }
            },
            {
                "id": "doc2",
                "name": "Test Doc 2",
                "properties": {
                    "classified": "true",
                    "classification_date": "2025-03-18T01:00:00Z",
                    "classification_summary": "Public document",
                    "overall_confidence": "85",
                    "categories": "public"
                }
            }
        ]
    }
    
    # Mock API calls
    mocker.patch.object(
        mcp_server.drive_service.files(),
        "list",
        return_value=mocker.Mock(execute=lambda: mock_files)
    )
    
    # Mock document content retrieval
    mocker.patch.object(
        mcp_server,
        "_get_document_content",
        side_effect=lambda doc_id: "Mock content for " + doc_id
    )
    
    # Test validation
    request = ValidationRequest(sample_size=2, validation_type="random")
    result = await mcp_server.validate_classification(request)
    
    # Verify results
    assert result["sample_size"] == 2
    assert result["total_documents"] == 2
    assert len(result["results"]) == 2
    
    # Check validation details
    for doc_result in result["results"]:
        assert "current_classification" in doc_result
        assert "new_classification" in doc_result
        assert "differences" in doc_result

@pytest.mark.asyncio
async def test_generate_custom_report(mcp_server, mocker):
    # Mock classified documents
    mock_files = {
        "files": [
            {
                "id": "doc1",
                "name": "Confidential Doc",
                "properties": {
                    "classified": "true",
                    "classification_date": "2025-03-18T00:00:00Z",
                    "overall_confidence": "95",
                    "categories": "confidential",
                    "classification_summary": "Contains sensitive data"
                }
            },
            {
                "id": "doc2",
                "name": "Public Doc",
                "properties": {
                    "classified": "true",
                    "classification_date": "2025-03-18T01:00:00Z",
                    "overall_confidence": "85",
                    "categories": "public",
                    "classification_summary": "General information"
                }
            }
        ]
    }
    
    # Mock API calls
    mocker.patch.object(
        mcp_server.drive_service.files(),
        "list",
        return_value=mocker.Mock(execute=lambda: mock_files)
    )
    
    # Test report generation
    request = CustomReportRequest(
        format="markdown",
        filters={
            "confidence_min": 80,
            "categories": ["confidential"]
        },
        grouping=["confidence_level", "category"]
    )
    
    result = await mcp_server.generate_custom_report(request)
    
    # Verify results
    assert result["format"] == "markdown"
    assert "report" in result
    assert "generated_at" in result
    assert "total_documents" in result
    
    # Check report content
    report = result["report"]
    assert "# Document Classification Report" in report
    assert "Confidential Doc" in report
    assert "confidence: 95%" in report.lower()

@pytest.mark.asyncio
async def test_error_handling(mcp_server, mocker):
    # Mock API error
    def mock_api_error(*args, **kwargs):
        raise Exception("API Error")
    
    mocker.patch.object(
        mcp_server.drive_service.files(),
        "list",
        side_effect=mock_api_error
    )
    
    # Test error handling in document processing
    request = ProcessRequest(document_ids=["doc1"])
    results = []
    
    async for update in mcp_server.process_documents(request):
        results.append(update)
    
    # Verify error handling
    assert "error" in results[-1]
    
    # Test error handling in validation
    request = ValidationRequest(sample_size=1)
    with pytest.raises(Exception):
        await mcp_server.validate_classification(request)

@pytest.mark.asyncio
async def test_concurrent_processing(mcp_server, mocker):
    # Create multiple test documents
    doc_ids = [f"doc{i}" for i in range(5)]
    
    # Mock document content and metadata
    mock_docs = {
        doc_id: {
            "content": f"Test content for {doc_id}",
            "metadata": {
                "id": doc_id,
                "name": f"Test Document {doc_id}",
                "mimeType": "text/plain"
            }
        }
        for doc_id in doc_ids
    }
    
    # Mock API calls
    mocker.patch.object(
        mcp_server.drive_service.files(),
        "get",
        side_effect=lambda fileId, **kwargs: mock_docs[fileId]["metadata"]
    )
    mocker.patch.object(
        mcp_server.drive_service.files(),
        "get_media",
        side_effect=lambda fileId: mock_docs[fileId]["content"].encode()
    )
    
    # Mock OpenAI response
    mock_openai_response = mocker.Mock()
    mock_openai_response.choices = [{
        "message": {
            "content": json.dumps({
                "categories": [
                    {
                        "name": "public",
                        "confidence": 85,
                        "reasoning": "General content"
                    }
                ],
                "overall_confidence": 85,
                "summary": "Public document"
            })
        }
    }]
    mocker.patch.object(
        mcp_server.openai_client.chat.completions,
        "create",
        return_value=mock_openai_response
    )
    
    # Test concurrent processing
    request = ProcessRequest(document_ids=doc_ids, batch_size=2)
    results = []
    
    async for update in mcp_server.process_documents(request):
        results.append(update)
        await asyncio.sleep(0.1)  # Simulate processing time
    
    # Verify concurrent processing
    assert len(results) > len(doc_ids) // 2  # At least one update per batch
    assert results[-1]["progress"]["processed"] == len(doc_ids)
    assert results[-1]["progress"]["total"] == len(doc_ids)
    
    # Verify batch processing
    batch_sizes = [len(update["batch_results"]) for update in results if "batch_results" in update]
    assert all(size <= 2 for size in batch_sizes)  # Respect batch size limit

if __name__ == "__main__":
    pytest.main(["-v", __file__])