import os
from dotenv import load_dotenv
from Module.id_card import IdCard
from Module.ocr_processor import OCRProcessor
import json

load_dotenv()
  
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output_images")
INPUT_DIR = os.getenv("INPUT_DIR", "json_data")
RESULTS_DIR = os.getenv("RESULTS_DIR", "ocr_results")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def normalize_text(text):
    """Normalize text for comparison by fixing common OCR mistakes and case"""
    if not text:
        return ""
    text = str(text).lower()
    # Fix common OCR mistakes
    text = text.replace('o', '0').replace('i', '1').replace('s', '5')
    # Remove spaces and punctuation for comparison
    text = ''.join(c for c in text if c.isalnum())
    return text

def process_and_validate_cards():
    ocr_processor = OCRProcessor()
    results = []
    
    # First create ID cards from JSON
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".json"):
            json_path = os.path.join(INPUT_DIR, filename)
            IdCard.create_id_card(json_path)
    
    # Then process each generated image with OCR
    for filename in os.listdir(OUTPUT_DIR):
        if filename.endswith(".png"):
            image_path = os.path.join(OUTPUT_DIR, filename)
            user_id = os.path.splitext(filename)[0]
            
            # Process the image with OCR
            ocr_result = ocr_processor.process_id_card(image_path)
            
            # Load original JSON for comparison
            json_path = os.path.join(INPUT_DIR, f"{user_id}.json")
            with open(json_path, 'r') as f:
                original_data = json.load(f)
            
            # Compare and calculate accuracy
            original_fields = original_data["extracted_fields"]
            extracted_fields = ocr_result["extracted_fields"]
            field_accuracy = {}
            
            for field in original_fields:
                if field in extracted_fields:
                    original_value = normalize_text(original_fields[field])
                    extracted_value = normalize_text(extracted_fields[field])
                    # Calculate similarity score
                    if original_value and extracted_value:
                        if len(original_value) > len(extracted_value):
                            accuracy = sum(1 for i in range(len(extracted_value)) if i < len(original_value) and extracted_value[i] == original_value[i]) / len(original_value)
                        else:
                            accuracy = sum(1 for i in range(len(original_value)) if i < len(extracted_value) and original_value[i] == extracted_value[i]) / len(extracted_value)
                    else:
                        accuracy = 0.0
                    field_accuracy[field] = accuracy
                else:
                    field_accuracy[field] = 0.0
            
            # Calculate overall accuracy
            overall_accuracy = sum(field_accuracy.values()) / len(field_accuracy)
            
            # Store results
            result = {
                "user_id": user_id,
                "confidence": ocr_result["confidence"],
                "accuracy": overall_accuracy,
                "field_accuracy": field_accuracy,
                "extracted_fields": extracted_fields,
                "original_fields": original_fields,
                "raw_text": ocr_result["raw_text"]
            }
            results.append(result)
    
    # Save results
    results_path = os.path.join(RESULTS_DIR, "ocr_results.json")
    with open(results_path, 'w') as f:
        json.dump({"results": results}, f, indent=2)
    
    print(f"OCR processing complete. Results saved to {results_path}")
    return results

if __name__ == '__main__':
    # Generate ID cards and process with OCR
    results = process_and_validate_cards()
    
    # Calculate and display overall statistics
    total_cards = len(results)
    avg_confidence = sum(r["confidence"] for r in results) / total_cards
    avg_accuracy = sum(r["accuracy"] for r in results) / total_cards
    
    print(f"\nProcessing Summary:")
    print(f"Total cards processed: {total_cards}")
    print(f"Average OCR confidence: {avg_confidence:.2f}%")
    print(f"Average field accuracy: {avg_accuracy:.2%}")