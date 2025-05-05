#!/bin/bash

# Load environment variables
set -a
source .env
set +a

# Default values
DB_PATH=${CHROMA_DB_PATH:-"./chromadb"}
COLLECTION_NAME=${CHROMA_COLLECTION_NAME:-"documents"}
IMAGE_NAME="rag-app"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --db_path)
      DB_PATH="$2"
      shift 2
      ;;
    --collection_name)
      COLLECTION_NAME="$2"
      shift 2
      ;;
    --image)
      IMAGE_NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if image exists
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
  echo "Building Docker image..."
  docker build -t $IMAGE_NAME .
fi

# Make sure ChromaDB directory exists
mkdir -p "$DB_PATH"

# Run the application
echo "Starting RAG Chat Assistant in Docker with the following configuration:"
echo "DB Path: $DB_PATH"
echo "Collection Name: $COLLECTION_NAME"
echo "Image: $IMAGE_NAME"

docker run -p 8501:8501 \
  -v "$DB_PATH":/app/chromadb \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e CHROMA_DB_PATH="/app/chromadb" \
  -e CHROMA_COLLECTION_NAME="$COLLECTION_NAME" \
  $IMAGE_NAME