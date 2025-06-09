#!/bin/bash

# Build the Docker image
docker build -t cravo-app .

# Run the container with data directory mounted for persistence
docker run --rm -p 8080:8080 \
  -v "$(pwd)/data:/app/data" \
  --env-file .env \
  --name cravo-container \
  cravo-app