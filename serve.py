import http.server
import socketserver
import webbrowser
import os
import json
import cv2
import numpy as np
import pytesseract
from urllib.parse import parse_qs, urlparse
import base64

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
        
        # Basic field extraction
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for common ID card patterns
            if "name" in line.lower():
                if ":" in line:
                    name_parts = line.split(":", 1)
                    if len(name_parts) > 1:
                        name = name_parts[1].strip()
                else:
                    name_parts = line.lower().split("name", 1)
                    if len(name_parts) > 1:
                        name = name_parts[1].strip()
                
                if 'name' in locals():
                    name = ' '.join([part for part in name.split() if not any(c.isdigit() for c in part)])
                    name = ''.join(c for c in name if c.isalpha() or c.isspace())
                    name = ' '.join(name.split())
                    if name:
                        result["id_card"]["extracted_fields"]["name"] = name
            elif "college" in line.lower():
                result["id_card"]["extracted_fields"]["college"] = line.split(":", 1)[-1].strip()
            elif "roll" in line.lower() and "number" in line.lower():
                result["id_card"]["extracted_fields"]["roll_number"] = line.split(":", 1)[-1].strip()
            elif "branch" in line.lower():
                result["id_card"]["extracted_fields"]["branch"] = line.split(":", 1)[-1].strip()
            elif "valid" in line.lower() and "upto" in line.lower():
                result["id_card"]["extracted_fields"]["valid_upto"] = line.split(":", 1)[-1].strip()
        
        return result
    except Exception as e:
        return {"error": str(e)}

class IDCardHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/process-image':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                # Parse the multipart form data
                content_type = self.headers['Content-Type']
                if 'multipart/form-data' in content_type:
                    # Extract the image data from the multipart form
                    boundary = content_type.split('boundary=')[1].encode()
                    parts = post_data.split(boundary)
                    
                    for part in parts:
                        if b'image' in part.lower():
                            # Find the start of the image data
                            start = part.find(b'\r\n\r\n') + 4
                            image_data = part[start:]
                            
                            # Process the image
                            result = process_id_card(image_data)
                            
                            # Send the response
                            self.send_response(200)
                            self.send_header('Content-type', 'application/json')
                            self.end_headers()
                            self.wfile.write(json.dumps(result).encode())
                            return
                
                self.send_error(400, "Invalid request format")
            except Exception as e:
                self.send_error(500, str(e))
        else:
            super().do_POST()

def main():
    # Set the port number
    PORT = 8080
    
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the directory containing the script
    os.chdir(current_dir)
    
    # Create the server with our custom handler
    httpd = socketserver.TCPServer(("", PORT), IDCardHandler)
    
    print(f"Server started at http://localhost:{PORT}")
    print("Opening browser automatically...")
    
    # Open the browser automatically
    webbrowser.open(f'http://localhost:{PORT}/index.html')
         
         
    try:
        # Start the server
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == "__main__":
    main() 