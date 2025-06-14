# ID Card Processing API

This API provides endpoints for processing ID cards using OCR (Optical Character Recognition) and NER (Named Entity Recognition) to extract information from ID card images.

## Constraints

- 100% offline processing
- No GPT/OpenAI/cloud OCR APIs
- Uses only open-source tools:
  - Tesseract for OCR
  - spaCy for NER
  - scikit-learn for ML tasks
- Docker containerized deployment
- Input: base64 image in JSON
- Output: structured JSON

## Setup

1. Build and run using Docker:
```bash
docker build -t idcard-extractor .
docker run -p 8000:8000 idcard-extractor
```

2. Or install dependencies manually:
```bash
pip install -r requirements.txt
```

3. Configure the application:
- Edit `config.json` to set your desired configuration
- Make sure you have the required model files in `trained_models/ner`
- Ensure Tesseract is properly installed and configured

4. Run the API:
```bash
python run_api.py
```

The API will be available at http://localhost:8000 by default.
API documentation will be available at http://localhost:8000/docs

## API Endpoints

### GET /health
Health check endpoint that returns `{"status": "ok"}`.

### GET /version
Returns version information in the format:
```json
{
    "model_version": "1.0.0",
    "config_version": "1.0.0"
}
```

### POST /extract
Process an ID card image provided as base64-encoded string.

Request body:
```json
{
    "image": "base64_encoded_image_string",
    "threshold": 0.7
}
```

### POST /extract/file
Process an ID card image provided as a file upload.

Form data:
- `file`: The image file
- `threshold`: Confidence threshold (optional, default: 0.7)

## Response Format

The extract endpoint returns JSON in the following format:
```json
{
    "extracted_fields": {
        "name": "John Doe",
        "id_number": "123456789",
        "date_of_birth": "1990-01-01",
        "address": "123 Main St",
        "expiry_date": "2025-12-31"
    },
    "confidence_scores": {
        "name": 0.95,
        "id_number": 0.98,
        "date_of_birth": 0.92,
        "address": 0.85,
        "expiry_date": 0.94
    },
    "raw_text": "Full OCR text...",
    "overall_confidence": 0.928
}
```

## Testing

Run the tests using pytest:
```bash
pytest tests/
```

## Configuration

The `config.json` file contains settings for:
- API server (host, port, debug mode)
- OCR processing (Tesseract settings, preprocessing)
- NER model (model path, confidence thresholds)
- Storage (temporary files, logs)

## Directory Structure

```
.
├── api/
│   └── main.py           # FastAPI application
├── Module/
│   ├── ocr_processor.py  # OCR processing
│   └── ner_processor.py  # NER processing
├── tests/
│   ├── data/            # Test data
│   └── test_api.py      # API tests
├── config.json          # Configuration
├── requirements.txt     # Dependencies
└── run_api.py          # Server runner
```

## Error Handling

The API returns appropriate HTTP status codes:
- 200: Successful processing
- 400: Invalid input (e.g., invalid base64)
- 422: Validation error (e.g., missing required fields)
- 500: Server error (e.g., processing failure)

Error responses include a detail message explaining the error.

## Project Timeline (6 Weeks)

1. Week 1: Sample dataset and regex pattern design
2. Week 2: OCR setup + baseline extraction logic
3. Week 3: Train NER model and test on held-out samples
4. Week 4: Integrate with FastAPI and config.json
5. Week 5: Add field-wise confidence, logging, and tests
6. Week 6: Final documentation and packaging

## Docker Image Contents

The Docker image includes:
- All Python dependencies
- Pretrained models
- Tesseract and supporting binaries
- Configuration files
- Application code