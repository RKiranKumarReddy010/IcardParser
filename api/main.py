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
import cv2
import numpy as np
import pytesseract
from PIL import Image
import io

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

def process_id_card(image_data):
    try:
        # Convert image data to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to preprocess the image
        threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Perform OCR
        text = pytesseract.image_to_string(threshold)
        
        # Process the extracted text to find relevant information
        lines = text.split('\n')
        result = {
            "id_card": {
                "raw_text": text,
                "extracted_fields": {}
            }
        }
        
        # Basic field extraction (you can enhance this based on your needs)
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Look for common ID card patterns
            if "name" in line.lower():
                # Improved name extraction
                # First try to split by colon
                if ":" in line:
                    name_parts = line.split(":", 1)
                    if len(name_parts) > 1:
                        name = name_parts[1].strip()
                else:
                    # If no colon, try to extract name after "name" keyword
                    name_parts = line.lower().split("name", 1)
                    if len(name_parts) > 1:
                        name = name_parts[1].strip()
                
                # Clean up the name
                if 'name' in locals():
                    # Remove any numbers or special characters
                    name = ' '.join([part for part in name.split() if not any(c.isdigit() for c in part)])
                    # Remove any remaining special characters
                    name = ''.join(c for c in name if c.isalpha() or c.isspace())
                    # Remove extra spaces
                    name = ' '.join(name.split())
                    if name:  # Only add if we have a valid name
                        result["id_card"]["extracted_fields"]["name"] = name
            elif "id" in line.lower() and "number" in line.lower():
                result["id_card"]["extracted_fields"]["id_number"] = line.split(":", 1)[-1].strip()
            elif "dob" in line.lower() or "birth" in line.lower():
                result["id_card"]["extracted_fields"]["date_of_birth"] = line.split(":", 1)[-1].strip()
            elif "address" in line.lower():
                result["id_card"]["extracted_fields"]["address"] = line.split(":", 1)[-1].strip()
            elif "college" in line.lower():
                result["id_card"]["extracted_fields"]["college"] = line.split(":", 1)[-1].strip()
            elif "branch" in line.lower():
                result["id_card"]["extracted_fields"]["branch"] = line.split(":", 1)[-1].strip()
            elif "roll" in line.lower() and "number" in line.lower():
                result["id_card"]["extracted_fields"]["roll_number"] = line.split(":", 1)[-1].strip()
            elif "valid" in line.lower() and "upto" in line.lower():
                result["id_card"]["extracted_fields"]["valid_upto"] = line.split(":", 1)[-1].strip()
        
        return result
    except Exception as e:
        return {"error": str(e)}

@app.post("/process-id-card")
async def process_id_card_endpoint(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Process the image
        result = process_id_card(contents)
        
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "ID Card Processing API is running"} 