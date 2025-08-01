import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import (
    LLMImageBlobParser,
    PyMuPDFParser,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .common import (
    CHROMA_DIRECTORY,
    CHROMA_HOST,
    CHROMA_PORT,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    PDF_LOADER_MODEL,
)


def load_pdfs(file_path: str):
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

    vector_store = None
    if (
        os.environ.get("ENVIRONMENT") == "production"
        or os.environ.get("ENVIRONMENT") == "development"
    ):
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=EMBEDDING_MODEL,
            host=CHROMA_HOST,
            port=CHROMA_PORT,
        )
    else:
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=EMBEDDING_MODEL,
            persist_directory=CHROMA_DIRECTORY,
        )
    vector_store.reset_collection()
    document_ids = vector_store.add_documents(documents=all_splits)

    assert (
        len(document_ids) > 0
    ), "No documents were added to the vector store. Check the embeddings model and documents."
    return vector_store


if __name__ == "__main__":
    docs = load_pdfs("./app/files/")
    docs_with_metadata = add_metadata_to_documents(docs)
    all_splits = split_docs(docs_with_metadata)
    setup_vector_store(all_splits)
