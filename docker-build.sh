#!/bin/bash

# Build the Docker image
docker build -t idcard-extractor .

# Run the container
docker run -p 8080:8080 idcard-extractor 