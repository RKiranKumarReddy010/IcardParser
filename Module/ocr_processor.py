import cv2
import numpy as np
import pytesseract
import json
import os
import re
from PIL import Image
from typing import Dict, Any, Tuple
from .ner_processor import NERProcessor

class OCRProcessor:
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.setup_tesseract()
        self.ner = NERProcessor(model_path="trained_models/ner")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {
            "tesseract": {
                "psm": 4,  # Assume uniform text block
                "oem": 1,  # LSTM only
                "lang": "eng",
                "config_params": "--dpi 300 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-,. "
            },
            "preprocessing": {
                "resize_width": 2400,  # Increased resolution
                "threshold_method": "adaptive",
                "denoise": True,
                "sharpen": True,
                "deskew": True,
                "morph_cleanup": True
            }
        }

    def setup_tesseract(self):
        """Configure Tesseract with optimal parameters"""
        self.tesseract_config = f'-l {self.config["tesseract"]["lang"]} --oem {self.config["tesseract"]["oem"]} --psm {self.config["tesseract"]["psm"]} {self.config["tesseract"]["config_params"]}'

    def deskew(self, image: np.ndarray) -> np.ndarray:
        """Deskew the image if it's rotated"""
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if angle > 0:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return image

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Apply enhanced preprocessing steps to improve OCR accuracy"""
        # Read image
        img = cv2.imread(image_path)
        
        # Get preprocessing config with defaults
        config = self.config.get("preprocessing", {})
        resize_width = config.get("resize_width", 2400)
        threshold_method = config.get("threshold_method", "adaptive")
        use_denoise = config.get("denoise", True)
        use_sharpen = config.get("sharpen", True)
        use_deskew = config.get("deskew", True)
        use_morph_cleanup = config.get("morph_cleanup", True)
        
        # Resize while maintaining aspect ratio
        scale = resize_width / img.shape[1]
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising if configured
        if use_denoise:
            gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Apply sharpening if configured
        if use_sharpen:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            gray = cv2.filter2D(gray, -1, kernel)
        
        # Apply thresholding with better parameters
        if threshold_method == "adaptive":
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 21, 11
            )
        else:
            binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Deskew if configured
        if use_deskew:
            binary = self.deskew(binary)
        
        # Apply morphological operations to clean up the image
        if use_morph_cleanup:
            kernel = np.ones((2,2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary

    def extract_text(self, image_path: str) -> Tuple[str, float]:
        """Extract text from image with improved confidence calculation"""
        # Preprocess image
        processed_img = self.preprocess_image(image_path)
        
        # Convert OpenCV image to PIL Image
        pil_img = Image.fromarray(processed_img)
        
        # Get OCR data including confidence
        ocr_data = pytesseract.image_to_data(
            pil_img, 
            config=self.tesseract_config, 
            output_type=pytesseract.Output.DICT
        )
        
        # Extract text and calculate weighted confidence
        text_parts = []
        confidences = []
        word_lengths = []
        
        for i in range(len(ocr_data["text"])):
            if int(ocr_data["conf"][i]) > 30:  # Increased confidence threshold
                text = ocr_data["text"][i].strip()
                if text:  # Only process non-empty text
                    text_parts.append(text)
                    confidences.append(float(ocr_data["conf"][i]))
                    word_lengths.append(len(text))
        
        if not confidences:
            return "", 0.0
        
        # Calculate length-weighted confidence
        total_length = sum(word_lengths)
        weighted_confidence = sum(conf * length for conf, length in zip(confidences, word_lengths)) / total_length if total_length > 0 else 0.0
        
        # Join text parts with proper spacing
        extracted_text = " ".join(text_parts)
        
        # Clean the extracted text
        cleaned_text = self.clean_text(extracted_text)
        
        return cleaned_text, weighted_confidence

    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning"""
        # Remove non-printable characters
        text = "".join(char for char in text if char.isprintable())
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR mistakes
        text = text.replace('0', 'O').replace('1', 'I').replace('5', 'S')
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text

    def extract_fields(self, text: str) -> Dict[str, str]:
        """Extract specific fields using regex patterns"""
        patterns = {
            "name": r"Name:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            "college": r"College:\s*([A-Za-z\s.,&\-]+)",
            "roll_number": r"Roll Number:\s*([A-Z0-9]{6,15})",
            "branch": r"Branch:\s*([A-Za-z\s]+)"
        }
        
        extracted_fields = {}
        for field, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                extracted_fields[field] = match.group(1).strip()
        
        return extracted_fields

    def process_id_card(self, image_path: str) -> Dict[str, Any]:
        """Process ID card image with improved OCR and NER"""
        # Perform enhanced OCR
        raw_text, confidence = self.extract_text(image_path)
        
        # Process with NER
        extracted_fields = self.ner.process_text(raw_text)
        
        # Post-process extracted fields
        processed_fields = {}
        for field, value in extracted_fields.items():
            # Clean and normalize field values
            cleaned_value = self.clean_text(str(value))
            # Convert to uppercase for consistency
            if field in ['name', 'college', 'branch']:
                cleaned_value = cleaned_value.upper()
            processed_fields[field] = cleaned_value
        
        return {
            "confidence": confidence,
            "raw_text": raw_text,
            "extracted_fields": processed_fields
        } 