from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import base64
import json
import os
from loguru import logger
from datetime import datetime
import sys

# Import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Module.ocr_processor import OCRProcessor
from Module.ner_processor import NERProcessor

# Initialize FastAPI app
app = FastAPI(
    title="ID Card Processing API",
    description="API for processing ID cards using OCR and NER",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)

# Configure logging
logger.remove()
logger.add(
    "logs/api_{time}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# Initialize processors
ocr_processor = OCRProcessor()
ner_processor = NERProcessor(model_path=config.get("ner_model_path", "trained_models/ner"))

class ImageRequest(BaseModel):
    image: str  # base64 encoded image
    threshold: Optional[float] = 0.7

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.get("/version")
async def version():
    """Get API version information"""
    return {
        "model_version": "1.0.0",
        "config_version": "1.0.0"
    }

@app.post("/extract")
async def extract_info(request: ImageRequest):
    """Extract information from ID card image"""
    try:
        # Decode base64 image
        try:
            image_data = base64.b64decode(request.image)
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid base64 image")

        # Save temporary image
        temp_path = f"temp/temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        os.makedirs("temp", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(image_data)

        try:
            # Process with OCR
            logger.info("Starting OCR processing")
            ocr_result = ocr_processor.process_id_card(temp_path)
            
            # Process with NER
            logger.info("Starting NER processing")
            ner_result = ner_processor.process_text(ocr_result["raw_text"])
            
            # Combine results
            combined_result = {
                "extracted_fields": {},
                "confidence_scores": {},
                "raw_text": ocr_result["raw_text"],
                "overall_confidence": 0.0
            }

            # Map fields and calculate confidence
            field_confidences = []
            for field, value in ner_result.items():
                if value["confidence"] >= request.threshold:
                    combined_result["extracted_fields"][field] = value["text"]
                    combined_result["confidence_scores"][field] = value["confidence"]
                    field_confidences.append(value["confidence"])

            # Calculate overall confidence
            if field_confidences:
                combined_result["overall_confidence"] = sum(field_confidences) / len(field_confidences)

            logger.info(f"Processing completed with overall confidence: {combined_result['overall_confidence']}")
            return combined_result

        finally:
            # Cleanup temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract/file")
async def extract_info_from_file(file: UploadFile = File(...), threshold: float = 0.7):
    try:
        # Save temporary file
        temp_path = f"temp/temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        os.makedirs("temp", exist_ok=True)
        
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        try:
            # Process with OCR
            logger.info("Starting OCR processing")
            ocr_result = ocr_processor.process_id_card(temp_path)
            
            # Process with NER
            logger.info("Starting NER processing")
            ner_result = ner_processor.process_text(ocr_result["raw_text"])
            
            # Combine results (same as in /extract endpoint)
            combined_result = {
                "extracted_fields": {},
                "confidence_scores": {},
                "raw_text": ocr_result["raw_text"],
                "overall_confidence": 0.0
            }

            # Map fields and calculate confidence
            field_confidences = []
            for field, value in ner_result.items():
                if value["confidence"] >= threshold:
                    combined_result["extracted_fields"][field] = value["text"]
                    combined_result["confidence_scores"][field] = value["confidence"]
                    field_confidences.append(value["confidence"])

            # Calculate overall confidence
            if field_confidences:
                combined_result["overall_confidence"] = sum(field_confidences) / len(field_confidences)

            logger.info(f"Processing completed with overall confidence: {combined_result['overall_confidence']}")
            return combined_result

        finally:
            # Cleanup temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 