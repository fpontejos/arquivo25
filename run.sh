#!/bin/bash

# Load environment variables
set -a
source .env
set +a

# Default values
DB_PATH=${CHROMA_DB_PATH:-"./data/chroma_db"}
COLLECTION_NAME=${CHROMA_COLLECTION_NAME:-"demodb"}

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
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Print configuration
echo "Starting RAG Chat Assistant with the following configuration:"
echo "DB Path: $DB_PATH"
echo "Collection Name: $COLLECTION_NAME"

# Run the application
streamlit run app.py -- --db_path "$DB_PATH" --collection_name "$COLLECTION_NAME"