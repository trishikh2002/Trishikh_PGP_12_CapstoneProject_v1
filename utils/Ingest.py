"""Load PDF / DOCX / CSV files from ./data, chunk them, and store in ChromaDB."""
from pathlib import Path
from uuid import uuid4
 
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
 
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
DB_DIR = str(PROJECT_DIR / "chroma_db")
COLLECTION = "climate_docs"
 
LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".csv": CSVLoader,
}
 
 
def get_embeddings():
    # Free, local embedding model. Swap for OpenAIEmbeddings if you prefer.
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
 
 
def load_documents():
    docs = []
    for path in sorted(DATA_DIR.rglob("*")):
        loader_cls = LOADERS.get(path.suffix.lower())
        if loader_cls is None:
            continue
        try:
            loaded = loader_cls(str(path)).load()
            for d in loaded:
                d.metadata["source"] = path.name
            docs.extend(loaded)
            print(f"Loaded {path.name}: {len(loaded)} pieces")
        except Exception as e:
            print(f"Skipped {path.name}: {e}")
    return docs


def add_uploaded_document(file_path: Path) -> int:
    """Load one PDF or CSV file and append its chunks to the existing collection."""
    loader_cls = LOADERS.get(file_path.suffix.lower())
    if loader_cls is None:
        raise ValueError("Only PDF and CSV files can be added.")

    docs = loader_cls(str(file_path)).load()
    for document in docs:
        document.metadata["source"] = file_path.name

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    ).split_documents(docs)
    if not chunks:
        return 0

    vector_store = Chroma(
        persist_directory=DB_DIR,
        embedding_function=get_embeddings(),
        collection_name=COLLECTION,
    )
    vector_store.add_documents(chunks, ids=[str(uuid4()) for _ in chunks])
    return len(chunks)
 
 
def main():
    docs = load_documents()
    if not docs:
        raise SystemExit("No supported files found in ./data")
 
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
 
    Chroma.from_documents(
        chunks,
        embedding=get_embeddings(),
        persist_directory=DB_DIR,
        collection_name=COLLECTION,
    )
    print(f"Saved to {DB_DIR}/")
 
 
if __name__ == "__main__":
    main()