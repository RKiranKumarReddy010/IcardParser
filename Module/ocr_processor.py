import cv2
import numpy as np
import pytesseract
import json
import os
import re
from PIL import Image, ImageEnhance
from typing import Dict, Any, Tuple
from .ner_processor import NERProcessor

class OCRProcessor:
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.setup_tesseract()
        self.ner = NERProcessor(model_path=self.config.get("ner", {}).get("model_path", "trained_models/ner"))
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        default_config = {
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
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                # Merge loaded config with default config
                if "tesseract" in loaded_config:
                    default_config["tesseract"].update(loaded_config["tesseract"])
                if "preprocessing" in loaded_config:
                    default_config["preprocessing"].update(loaded_config["preprocessing"])
                return default_config
        return default_config

    def setup_tesseract(self):
        """Configure Tesseract with optimal parameters"""
        tesseract_config = self.config.get("tesseract", {})
        self.tesseract_config = (
            f'-l {tesseract_config.get("lang", "eng")} '
            f'--oem {tesseract_config.get("oem", 1)} '
            f'--psm {tesseract_config.get("psm", 4)} '
            f'{tesseract_config.get("config_params", "--dpi 300")}'
        )

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

    def preprocess_image(self, image: Image) -> Image:
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        image = image.convert('L')
        
        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Apply thresholding
        image = image.point(lambda x: 0 if x < 128 else 255, '1')
        
        # Resize if too small
        if image.width < 1000 or image.height < 600:
            ratio = max(1000/image.width, 600/image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.LANCZOS)
        
        return image

    def extract_text(self, image_path: str) -> Tuple[str, float]:
        """Extract text from image with improved confidence calculation"""
        # Preprocess image
        processed_img = self.preprocess_image(Image.open(image_path))
        
        # Convert OpenCV image to PIL Image
        pil_img = processed_img
        
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

    def process_id_card(self, image_path: str) -> Dict:
        """Process ID card image and extract information"""
        # Load and preprocess image
        image = Image.open(image_path)
        processed_image = self.preprocess_image(image)
        
        # Configure Tesseract
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789:@._- "'
        
        # Extract text
        text = pytesseract.image_to_string(processed_image, config=custom_config)
        
        # Process with NER
        ner_results = self.ner.process_text(text)
        
        # Calculate overall confidence
        confidences = [v.get('confidence', 0) for v in ner_results.values() if isinstance(v, dict)]
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "extracted_fields": ner_results,
            "overall_confidence": overall_confidence,
            "raw_text": text
        } 