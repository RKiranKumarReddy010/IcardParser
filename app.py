import requests
import base64

# Read an image file
with open(r'C:\Users\KIRAN\Documents\Multimodel\Test\resource\valid_id.png', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode()

# Send to API
response = requests.post('http://localhost:8000/extract', 
                        json={'image': image_data, 'threshold': 0.7})

# Print results
print(response.json())