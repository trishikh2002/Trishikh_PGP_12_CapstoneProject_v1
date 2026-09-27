# Climate Compass

Climate Compass is a Streamlit question-answering application for exploring climate-change documents. It uses retrieval-augmented generation (RAG): documents are loaded, split into smaller chunks, converted into local embeddings, stored in ChromaDB, and retrieved as context for a Groq-hosted language model.

## Features

- Ask questions about the climate documents in the knowledge base.
- Retrieve relevant document chunks from ChromaDB.
- Display the source files and document excerpts used for an answer.
- Upload additional PDF or CSV files from the Streamlit sidebar.
- Append uploaded documents to the existing ChromaDB collection without rebuilding the complete database.
- Avoid indexing the same uploaded file repeatedly during a Streamlit session.
- Use local `sentence-transformers/all-MiniLM-L6-v2` embeddings.
- Provide a climate-themed Streamlit interface with a background image and sidebar controls.

## Project Structure

```text
Trishikh_PGP_12_CapstoneProject/
├── data/                       # Source documents used for initial ingestion
│   └── uploads/                # Files uploaded through the Streamlit sidebar
├── chroma_db/                  # Persistent ChromaDB vector store
├── utils/
│   ├── app.py                 # Streamlit application
│   └── Ingest.py              # Document loading, chunking, and indexing logic
├── requirements.txt            # Python dependencies
├── .env                        # Local secrets; do not commit this file
└── README.md
```

## How It Works

1. Files in `data/` are loaded using the appropriate LangChain document loader.
2. Documents are split into chunks of 1,000 characters with a 150-character overlap.
3. Each chunk is embedded using `sentence-transformers/all-MiniLM-L6-v2`.
4. Embeddings are stored in the `climate_docs` collection inside `chroma_db/`.
5. A user question is sent through a history-aware retriever.
6. The most relevant chunks are passed to the Groq chat model as context.
7. The answer and supporting source excerpts are shown in the Streamlit interface.

The ingestion paths are resolved from the project root, so the app can be started from the `utils` directory without accidentally opening a second empty Chroma database.

## Requirements

- Python 3.10 or newer
- A Groq API key
- Internet access for the Groq API and the first download of the local embedding model

## Setup

From the project root, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Create or update `.env` in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Do not commit `.env` or expose the API key in source code.

## Initial Document Ingestion

Place supported source files in `data/`. The batch ingestion script supports:

- `.pdf`
- `.docx`
- `.csv`

Run ingestion from the project root:

```bash
python utils/Ingest.py
```

This creates or updates the persistent vector store at `chroma_db/` and uses the collection name `climate_docs`.

## Run the Application

Start Streamlit from the `utils` directory:

```bash
cd utils
streamlit run app.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`.

## Add Documents from the Sidebar

1. Open the **Knowledge base** section in the sidebar.
2. Upload one PDF or CSV file.
3. The file is saved under `data/uploads/`.
4. Its contents are chunked and appended to the existing `climate_docs` collection.
5. The retrieval chain is refreshed so the new content can be used immediately.

Only PDF and CSV files are accepted by the sidebar uploader. DOCX files can still be included through the initial batch-ingestion workflow.

## Configuration

The main application settings are defined in `utils/Ingest.py` and `utils/app.py`:

- Chroma collection: `climate_docs`
- Vector database: project-root `chroma_db/`
- Chunk size: `1000`
- Chunk overlap: `150`
- Retriever results: `4`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Chat model: `openai/gpt-oss-120b` through Groq

## Troubleshooting

### The app always answers that it does not know

Check that the active database contains embeddings and that the app is using the project-root database:

```bash
sqlite3 chroma_db/chroma.sqlite3 "select count(*) from embeddings;"
```

If the count is zero, run the initial ingestion command again.

### A newly uploaded file is not searchable

Check the Streamlit terminal for the indexing message and verify that the total embedding count increased. Restart Streamlit after code changes so the application uses the latest ingestion logic.

### API key errors

Confirm that `.env` is in the project root and contains `GROQ_API_KEY`. Do not add quotation marks unless they are part of the key value.

## Security Notes

- Keep `.env` private and add it to `.gitignore`.
- Rotate any API key that has been exposed or committed.
- Uploaded documents may contain sensitive information; store and share the project accordingly.
