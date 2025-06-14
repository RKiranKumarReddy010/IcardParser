@echo off
REM Build the Docker image
docker build -t idcard-extractor .

REM Run the container
docker run -p 8080:8080 idcard-extractor 