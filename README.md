# Arquivo Dos Cravos

Team:

Fernando Bação
Farina Pontejos
Yehor Malakhov
Catarina Ferraz

Video: [[Link](https://drive.google.com/file/d/1BcNQRrz1jDN3ksK9fY59QI6X-2c5Pz6l/view?usp=sharing)]

App: [[Link](https://arquivodoscravos.netlify.app/)]

---
## Technical Details

A simple Retrieval-Augmented Generation (RAG) application that connects to an existing ChromaDB vector database. 
The application provides a clean, minimalist chat interface to interact with your documents.

### Features

- Connect to an existing ChromaDB vector database
- Chat interface for asking questions about your documents
- Retrieval of relevant documents using semantic search
- Response generation using OpenAI models
- Configuration management for easy customization
- About page with database statistics and configuration information

### Prerequisites

- Python 3.8+
- OpenAI API key
- Existing ChromaDB vector database

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/rag-app.git
cd rag-app
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI API key:

```bash
# .env
OPENAI_API_KEY=your_openai_api_key_here
```

### Usage

#### Running the Application

Run the application with default settings:

```bash
streamlit run app.py
```

Or specify custom ChromaDB path and collection name:

```bash
streamlit run app.py -- --db_path /path/to/chromadb --collection_name your_collection
```

#### Using the Application

1. Open your browser and navigate to `http://localhost:8501`
2. Use the chat interface to ask questions about your documents
3. View database information and configuration settings in the "About" page

### Configuration

The application can be configured through the `data/config.json` file, which is created automatically on first run with default values:

```json
{
    "embedding_model": "text-embedding-3-large",
    "model": "gpt-4o",
    "temperature": 0.7,
    "top_k": 3,
    "max_tokens": 1000
}
```

### Docker Support

Build and run the application using Docker:

```bash
# Build the Docker image
docker build -t cravo-app .

# Run the container
docker run -p 8501:8501 -v /path/to/chromadb:/app/chromadb -e OPENAI_API_KEY=your_key_here cravo-app
```

## Cloud Deployment

The repository includes configuration files for cloud deployment:

- `cloud/cloudbuild.yaml`: Configuration for Google Cloud Build
- `cloud/service.yaml`: Configuration for Google Cloud Run

## Project Structure

```
rag-app/
├── pages/
│   └── chat.py           # Chat interface
├── utils/
│   ├── config.py           # Configuration management
│   ├── embeddings.py       # OpenAI embeddings
│   ├── retriever.py        # ChromaDB interactions
│   └── generator.py        # Response generation
├── data/                   # App data storage
│   └── config.py                  # Application configuration
├── cloud/                         # Cloud deployment configuration
│   ├── cloudbuild.yaml            # Cloud Build configuration
│   └── service.yaml               # Cloud Run service configuration
├── tests/                         # Placeholder
│   ├── __init__.py
│   └── ...
├── .dockerignore                  # Files to exclude from Docker build
├── .env                           # Example environment variables
├── .gitignore                     # Files to exclude from Git
├── app.py                         # Main application entry point
├── Dockerfile                     # Docker configuration
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## License

MIT