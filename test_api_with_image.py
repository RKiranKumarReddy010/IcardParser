import requests
import base64
import json
from tests.create_test_image import create_sample_id_card

def test_api():
    # Expected data
    expected_data = {
        "name": "Nathan Henry",
        "college": "JNTU Kakinada",
        "roll_number": "22JNT5377",
        "branch": "Computer Science",
        "valid_upto": "2028"
    }
    
    # First create the test image
    image_path = create_sample_id_card()
    
    # Read and encode the image
    with open(image_path, 'rb') as image_file:
        image_data = base64.b64encode(image_file.read()).decode()
    
    # Test health endpoint
    print("\nTesting /health endpoint...")
    response = requests.get('http://localhost:8000/health')
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test version endpoint
    print("\nTesting /version endpoint...")
    response = requests.get('http://localhost:8000/version')
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test extract endpoint
    print("\nTesting /extract endpoint...")
    payload = {
        "image": image_data,
        "threshold": 0.7
    }
    
    response = requests.post('http://localhost:8000/extract', json=payload)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\nExtracted Information:")
        
        # Compare extracted fields with expected data
        print("\nField Comparison:")
        print("=" * 50)
        print(f"{'Field':<15} {'Expected':<20} {'Extracted':<20} {'Match':<10}")
        print("-" * 50)
        
        matches = 0
        for field, expected in expected_data.items():
            extracted = result["extracted_fields"].get(field, "")
            if isinstance(extracted, dict):
                extracted_text = extracted.get("text", "")
                confidence = extracted.get("confidence", 0)
            else:
                extracted_text = extracted
                confidence = 0
                
            is_match = expected.lower() == extracted_text.lower()
            if is_match:
                matches += 1
            
            print(f"{field:<15} {expected:<20} {extracted_text:<20} {'✓' if is_match else '✗'}")
            if confidence:
                print(f"{'Confidence:':<15} {confidence:.2%}")
        
        accuracy = matches / len(expected_data) * 100
        print("\nOverall Results:")
        print(f"Accuracy: {accuracy:.1f}%")
        print(f"Overall Confidence: {result.get('overall_confidence', 0):.2%}")
        
        print("\nRaw Text:")
        print(result.get("raw_text", ""))
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    test_api() 