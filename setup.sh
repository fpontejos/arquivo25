#!/bin/bash

# Set up RAG application project structure

echo "Setting up RAG application project structure..."

# Create main directories
mkdir -p pages
mkdir -p utils
mkdir -p data
mkdir -p cloud
mkdir -p tests

# Create main application file
touch app.py

# Create page files
touch pages/chat.py
touch pages/about.py

# Create utility files
touch utils/__init__.py
touch utils/config.py
touch utils/embeddings.py
touch utils/retriever.py
touch utils/generator.py
touch utils/create_test_db.py

# Create configuration and data files
touch data/.gitkeep

# Create cloud deployment files
touch cloud/cloudbuild.yaml
touch cloud/service.yaml

# Create test files
touch tests/__init__.py
touch tests/test_retriever.py
touch tests/test_generator.py

# Create Docker and configuration files
touch Dockerfile
touch .dockerignore
touch .env
touch .gitignore
touch requirements.txt
touch README.md
touch run.sh
touch docker-run.sh

# Make shell scripts executable
chmod +x run.sh
chmod +x docker-run.sh

echo "Project structure created successfully!"
echo ""
echo "Directory structure:"
find . -type d -not -path "*/\.*" | sort

echo ""
echo "Files created:"
find . -type f -not -path "*/\.*" | sort

echo ""
echo "Next steps:"
echo "1. Add the code to each file"
echo "2. Install dependencies: pip install -r requirements.txt"
echo "3. Set your OpenAI API key in the .env file"
echo "4. Run the application: ./run.sh"