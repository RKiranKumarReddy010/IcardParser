# Use Python base image
FROM python:3.9-slim

# Install system dependencies including Tesseract
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p trained_models/ner logs temp tessdata

# Copy application files
COPY . .

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Set environment variables
ENV PYTHONPATH=/app
ENV TESSDATA_PREFIX=/app/tessdata

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["python", "run_api.py"] 