import os
import chromadb
import chromadb.errors

from llama_index.core import SimpleDirectoryReader, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

load_dotenv()

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG) # Or logging.INFO
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

VECTOR_INDEX_STORAGE_DIR = "./app/storage_local"
FILES_DIR = "./app/files"
NODE_CHUNK_SIZE = 2096
NODE_CHUNK_OVERLAP = 800
EMBED_MODEL = OllamaEmbedding(model_name="nomic-embed-text:v1.5") # Default embedding model
LLM_MODEL = Ollama(model="mistral:7b", temperature=0.2, seed=334, request_timeout=90.0) # Default LLM model
# EMBED_MODEL = OpenAIEmbedding(model="text-embedding-3-small") # Default embedding model
# LLM_MODEL = OpenAI(model="gpt-4o-mini", temperature=0.2, seed=334, request_timeout=90.0) # Default LLM model

def llm_settings():
    """Function to set up LLM settings."""
    Settings.embed_model = EMBED_MODEL
    Settings.llm = LLM_MODEL

    # Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    # Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2, seed=334, request_timeout=90.0)
    # Uncomment the line below to use a different model
    # Settings.llm = Ollama(model="deepseek-r1:7b", request_timeout=90.0)

def load_documents_with_metadata(directory_path):
    print(f"Loading documents from directory: {directory_path}")
    reader = SimpleDirectoryReader(directory_path, recursive=True)
    loaded_docs = reader.load_data()
    
    docs_with_metadata = []
    for doc in loaded_docs:
        file_name = os.path.basename(doc.metadata.get('file_path', ''))

        # Extract aircraft model from filename (e.g., "poh_162_1.pdf" -> "162")
        aircraft_model = None
        document_type = None
        if "poh_" in file_name:
            aircraft_model = "Cessna 162"
            document_type = "Pilot Operating Handbook"
        elif "phak" in file_name:
            document_type = "Pilot Handbook of Aeronautical Knowledge"
        elif "acs" in file_name:
            document_type = "Airman Certification Standards"
        elif "afh" in file_name:
            document_type = "Airplane Flying Handbook"

        # Add metadata to the document
        new_metadata = doc.metadata.copy()
        if aircraft_model:
            new_metadata['aircraft_model'] = aircraft_model

        new_metadata['document_type'] = document_type
        
        docs_with_metadata.append(Document(text=doc.text, metadata=new_metadata))

    print(f"Loaded {len(docs_with_metadata)} documents with metadata.")
    assert len(docs_with_metadata) > 0, "No documents loaded with metadata."

    print("Sample metadata:", docs_with_metadata[0].metadata)
    return docs_with_metadata

def main():
    """Main function to run the script."""
    llm_settings()

    chroma_client = chromadb.PersistentClient(VECTOR_INDEX_STORAGE_DIR)
    collection_name = "faa_documents"

    try:
        chroma_client.get_collection(collection_name)
    except chromadb.errors.NotFoundError:
        pass
    else:
        chroma_client.delete_collection(collection_name)  # Clear existing collection

    chroma_collection = chroma_client.create_collection(collection_name, embedding_function=OllamaEmbeddingFunction(url="http://localhost:11434/api/embeddings", model_name="nomic-embed-text:v1.5"))

    documents = load_documents_with_metadata(FILES_DIR)
    node_parser = SentenceSplitter(chunk_size=NODE_CHUNK_SIZE, chunk_overlap=NODE_CHUNK_OVERLAP)
    nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
    chroma_collection.upsert(
        documents=[doc.text for doc in nodes],
        metadatas=[doc.metadata for doc in nodes],
        ids=[str(i) for i in range(len(nodes))],
    )
    print("\nVectorStoreIndex created for all documents.")
    print(f"\nVerifying data directly in ChromaDB collection '{collection_name}'...")
    print(f"Number of items in ChromaDB collection: {chroma_collection.count()}")


if __name__ == "__main__":
    main()