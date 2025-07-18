import os
from llama_index.core import VectorStoreIndex, DocumentSummaryIndex, SimpleDirectoryReader, get_response_synthesizer, Settings, StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

def chunk():
    pass

def llm_settings():
    """Function to set up LLM settings."""
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text:v1.5")
    Settings.llm = Ollama(model="mistral:7b", request_timeout=90.0)
    # Uncomment the line below to use a different model
    # Settings.llm = Ollama(model="deepseek-r1:7b", request_timeout=90.0)

def index(type="vector"):
    """Function to create or load an index from the specified directory."""
    # Create the index from your documents
    directory_path = f"{type}_storage"
    file_path = "files"

    if os.path.isdir(directory_path):
        print(f"Loading index from {directory_path}...")
        storage_context = StorageContext.from_defaults(persist_dir=directory_path)
        index = load_index_from_storage(storage_context)

    else:
        print(f"Creating new index in {directory_path} from files in {file_path}...")
        # This helps LlamaIndex find your PDF file
        reader = SimpleDirectoryReader(file_path)
        documents = reader.load_data()

        print(f"Loaded {len(documents)} document(s).")
        match type:
            case "vector":
                index = VectorStoreIndex.from_documents(documents, show_progress=True)
            case "summary":
                index = DocumentSummaryIndex.from_documents(documents, show_progress=True)
            case _:
                raise ValueError(f"Unknown index type: {type}")
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        index.storage_context.persist(directory_path)
    return index

def retriever_prompt():
    """Function to create a retriever prompt."""
    if "POH" in query or "Pilot Operating Handbook" in query:
        query += " Please ensure the answer comes from any files that start with 162_poh."
    elif "ACS or Airman Certification Standards" in query:
        query += " Please ensure the answer comes from private_airplane_acs_6.pdf."
    elif "PHAK or Pilot's Handbook of Aeronautical Knowledge" in query:
        query += " Please ensure the answer comes from any files that start with phak_."
    elif "AFH or Airplane Flying Handbook" in query:
        query += " Please ensure the answer comes from any files that start with afh_."
    return "You are an FAA exam preparation assistant. Answer the question based on the provided documents."

def query_llm(query: str, query_engine) -> str:
    """Function to ask a question to the query engine."""
    query += " If you are not sure then say you are not sure."
    query += " Please list your sources at the end of your answer."
    response = query_engine.query(query)
    return response

def print_response(response):
    """Function to print the response from the query engine."""
    print("Response:", response.response)
    print("\nSources (Metadata and Content Snippet):")
    for i, node in enumerate(response.source_nodes):
        print(f"--- Source {i+1} ---")
        print(f"Node ID: {node.node.id_}")
        print(f"Similarity Score: {node.score}")
        print(f"Metadata: {node.node.metadata}")
        print(f"Content (snippet): {node.node.text[:200]}...")  # Print first 200 chars
        print("-" * 20)

if __name__ == "__main__":
    llm_settings();
    index = index("summary")

    # configure retriever
    # retriever = VectorIndexRetriever(
    #     index=index(),
    #     similarity_top_k=3,
    # )

    retriever = index.as_retriever(
        retriever_mode="llm",
        choice_top_k=3
    )

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer()

    # This is just to ensure the script runs when executed directly
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        # node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
    )

    queries = [
        "You are an FAA exam preparation assistant. Generate one multiple-choice question with four possible answers about the engine of a Cessna 162 based on the POH of the 162. The question should test a key concept from the text. Clearly indicate the correct answer.",
        "Summarize Chapter 1 Introduction to Flying for me from the PHAK.",
        "According to the AFH what is the protocol for Turbulent Air Approach and Landing?",
        "According to the Airman Certification Standards for the Private Pilots license what skills when maneuvering during slow flight do I need to exhibit and what are their codes?"
    ]

    for query in queries:
        response = query_llm(query, query_engine)
        print_response(response)