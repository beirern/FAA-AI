from langchain_chroma import Chroma
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import (
    LLMImageBlobParser,
    PyMuPDFParser,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .common import CHROMA_DIRECTORY, COLLECTION_NAME, EMBEDDING_MODEL, PDF_LOADER_MODEL


def load_pdfs(file_path: str, model: str = "gemma3:4b"):
    loader = GenericLoader(
        blob_loader=FileSystemBlobLoader(
            path=file_path,
            glob="**/*.pdf",
            show_progress=True,
        ),
        blob_parser=PyMuPDFParser(
            mode="page",
            images_inner_format="markdown-img",
            images_parser=LLMImageBlobParser(model=PDF_LOADER_MODEL),
            extract_tables="markdown",
        ),
    )
    docs = loader.load()

    assert len(docs) > 0, "No documents loaded. Check the file path and glob pattern."
    print(f"Total number of documents: {len(docs)}")

    return docs


def split_docs(docs: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"Split blog post into {len(all_splits)} sub-documents.")

    return all_splits


def add_metadata_to_documents(docs: list[Document]):
    for doc in docs:
        file_name = doc.metadata.get("source")

        # Extract aircraft model from filename (e.g., "poh_162_1.pdf" -> "162")
        if "poh_" in file_name:
            doc.metadata["aircraft_model"] = "Cessna 162"
            doc.metadata["document_type"] = "Pilot Operating Handbook"
        elif "phak" in file_name:
            doc.metadata["document_type"] = "Pilot Handbook of Aeronautical Knowledge"
        elif "acs" in file_name:
            doc.metadata["document_type"] = "Airman Certification Standards"
        elif "afh" in file_name:
            doc.metadata["document_type"] = "Airplane Flying Handbook"

    print("Sample metadata:", docs[0].metadata)
    return docs


def setup_vector_store(all_splits: list[Document]):
    print(
        f"Storing embeddings in collection '{COLLECTION_NAME}' at '{CHROMA_DIRECTORY}'"
    )
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=EMBEDDING_MODEL,
        persist_directory=CHROMA_DIRECTORY,
    )
    vector_store.reset_collection()
    document_ids = vector_store.add_documents(documents=all_splits)

    print("Sample metadata:", docs_with_metadata[0].metadata)
    return docs_with_metadata

def main():
    """Main function to run the script."""
    llm_settings()

    chroma_client = chromadb.HttpClient(host='chromadb', port=CHROMA_PORT)
    collection_name = "faa_documents"

    try:
        chroma_client.get_collection(collection_name)
    except chromadb.errors.NotFoundError:
        pass
    else:
        chroma_client.delete_collection(collection_name)  # Clear existing collection

    chroma_collection = chroma_client.create_collection(collection_name, embedding_function=EMBED_FUNCTION)

    documents = load_documents_with_metadata(FILES_DIR)
    node_parser = SentenceSplitter(chunk_size=NODE_CHUNK_SIZE, chunk_overlap=NODE_CHUNK_OVERLAP)
    print(f"Total documents loaded: {len(documents)}")
    nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)

    print(f"Total nodes created: {len(nodes)}")
    # Upsert nodes in batches since embedding function uses API calls
    # if it's a cloud LLM
    length = len(nodes)
    for i in range(0, length // 100):
        print(f"Upserting chunk {i + 1}/{length // 100}...")
        sub_nodes = nodes[i * 100:(i + 1) * 100]
        for node in sub_nodes:
            if not node.text:
                node.text += "empty string"
            chroma_collection.upsert(
                documents=[doc.text for doc in sub_nodes],
                metadatas=[doc.metadata for doc in sub_nodes],
                ids=[str((i * 100) + j) for j in range(len(sub_nodes))],
            )

    print("\nVectorStoreIndex created for all documents.")
    print(f"\nVerifying data directly in ChromaDB collection '{collection_name}'...")
    print(f"Number of items in ChromaDB collection: {chroma_collection.count()}")


if __name__ == "__main__":
    docs = load_pdfs("./app/files/")
    docs_with_metadata = add_metadata_to_documents(docs)
    all_splits = split_docs(docs_with_metadata)
    setup_vector_store(all_splits)
