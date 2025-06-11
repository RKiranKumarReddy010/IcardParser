import uvicorn
import json
import os

def main():
    # Create necessary directories
    os.makedirs("temp", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Load configuration
    with open("config.json", "r") as f:
        config = json.load(f)
    
    # Get API settings
    api_config = config.get("api", {})
    host = api_config.get("host", "0.0.0.0")
    port = api_config.get("port", 8000)
    debug = api_config.get("debug", False)
    
    print(f"Starting API server on {host}:{port}")
    print("API Documentation will be available at http://localhost:8000/docs")
    
    # Run the server
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=debug
    )

if __name__ == "__main__":
    main() 