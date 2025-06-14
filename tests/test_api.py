import pytest
import json
import base64
from pathlib import Path
from fastapi.testclient import TestClient
from run_api import app

client = TestClient(app)

def load_test_image(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

def load_expected_output(output_path):
    with open(output_path, 'r') as f:
        return json.load(f)

@pytest.fixture
def test_image():
    return load_test_image('tests/data/images/sample_id.png')

@pytest.fixture
def expected_output():
    return load_expected_output('tests/data/expected_output/sample_output.json')

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_version_endpoint():
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "config_version" in data

def test_extract_endpoint(test_image, expected_output):
    response = client.post(
        "/extract",
        json={"image": test_image, "threshold": 0.7}
    )
    assert response.status_code == 200
    data = response.json()
    
    # Check required fields
    assert "extracted_fields" in data
    assert "confidence_scores" in data
    assert "raw_text" in data
    assert "overall_confidence" in data
    
    # Validate extracted fields
    for field in expected_output["extracted_fields"]:
        assert field in data["extracted_fields"]
        assert data["extracted_fields"][field] == expected_output["extracted_fields"][field]
    
    # Validate confidence scores
    for field in expected_output["confidence_scores"]:
        assert field in data["confidence_scores"]
        assert abs(data["confidence_scores"][field] - expected_output["confidence_scores"][field]) < 0.1

def test_extract_invalid_image():
    response = client.post(
        "/extract",
        json={"image": "invalid_base64", "threshold": 0.7}
    )
    assert response.status_code == 400

def test_extract_invalid_threshold():
    response = client.post(
        "/extract",
        json={"image": "valid_base64", "threshold": 2.0}
    )
    assert response.status_code == 422

def test_extract_missing_image():
    response = client.post(
        "/extract",
        json={"threshold": 0.7}
    )
    assert response.status_code == 422

def test_extract_file_endpoint():
    test_file = Path('tests/data/images/sample_id.png')
    with open(test_file, 'rb') as f:
        response = client.post(
            "/extract/file",
            files={"file": ("sample_id.png", f, "image/png")},
            data={"threshold": 0.7}
        )
    assert response.status_code == 200
    data = response.json()
    assert "extracted_fields" in data
    assert "confidence_scores" in data

def test_extract_file_invalid_type():
    test_file = Path('tests/data/images/invalid.txt')
    with open(test_file, 'rb') as f:
        response = client.post(
            "/extract/file",
            files={"file": ("invalid.txt", f, "text/plain")},
            data={"threshold": 0.7}
        )
    assert response.status_code == 400 