import pytest
from fastapi.testclient import TestClient
import base64
import os
import json
from api.main import app

client = TestClient(app)

# Test data directory
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def load_test_image(filename):
    """Load a test image and return its base64 encoding"""
    with open(os.path.join(TEST_DATA_DIR, filename), "rb") as f:
        return base64.b64encode(f.read()).decode()

def test_health_endpoint():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_version_endpoint():
    """Test the version endpoint"""
    response = client.get("/version")
    assert response.status_code == 200
    assert "model_version" in response.json()
    assert "config_version" in response.json()
    assert response.json()["model_version"] == "1.0.0"
    assert response.json()["config_version"] == "1.0.0"

def test_extract_endpoint_valid_image():
    """Test the extract endpoint with a valid image"""
    image_data = load_test_image("valid_id.png")
    response = client.post(
        "/extract",
        json={"image": image_data, "threshold": 0.7}
    )
    assert response.status_code == 200
    assert "extracted_fields" in response.json()
    assert "confidence_scores" in response.json()
    assert "overall_confidence" in response.json()

def test_extract_endpoint_invalid_base64():
    """Test the extract endpoint with invalid base64 data"""
    response = client.post(
        "/extract",
        json={"image": "invalid_base64", "threshold": 0.7}
    )
    assert response.status_code == 400
    assert "Invalid base64 image" in response.json()["detail"]

def test_extract_endpoint_missing_image():
    """Test the extract endpoint with missing image data"""
    response = client.post(
        "/extract",
        json={"threshold": 0.7}
    )
    assert response.status_code == 422

def test_extract_file_endpoint_valid_image():
    """Test the file upload endpoint with a valid image"""
    with open(os.path.join(TEST_DATA_DIR, "valid_id.png"), "rb") as f:
        response = client.post(
            "/extract/file",
            files={"file": ("valid_id.png", f, "image/png")},
            params={"threshold": 0.7}
        )
    assert response.status_code == 200
    assert "extracted_fields" in response.json()
    assert "confidence_scores" in response.json()
    assert "overall_confidence" in response.json()

def test_extract_file_endpoint_invalid_file():
    """Test the file upload endpoint with an invalid file"""
    with open(os.path.join(TEST_DATA_DIR, "invalid.txt"), "rb") as f:
        response = client.post(
            "/extract/file",
            files={"file": ("invalid.txt", f, "text/plain")},
            params={"threshold": 0.7}
        )
    assert response.status_code == 500

def test_confidence_thresholds():
    """Test different confidence thresholds"""
    image_data = load_test_image("valid_id.png")
    
    # Test with high threshold
    response_high = client.post(
        "/extract",
        json={"image": image_data, "threshold": 0.9}
    )
    
    # Test with low threshold
    response_low = client.post(
        "/extract",
        json={"image": image_data, "threshold": 0.3}
    )
    
    # High threshold should have fewer fields
    assert len(response_high.json()["extracted_fields"]) <= len(response_low.json()["extracted_fields"]) 