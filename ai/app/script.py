import os
import logging
import sys
import json
from llama_index.core import VectorStoreIndex, DocumentSummaryIndex, SimpleDirectoryReader, get_response_synthesizer, Settings, StorageContext, load_index_from_storage, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.core.tools import QueryEngineTool
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response_synthesizers import TreeSummarize
from dotenv import load_dotenv

load_dotenv()

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG) # Or logging.INFO
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

VECTOR_INDEX_STORAGE_DIR = "./app/storage"
FILES_DIR = "./app/files"
NODE_CHUNK_SIZE = 2096
NODE_CHUNK_OVERLAP = 800

def llm_settings():
    """Function to set up LLM settings."""
    # Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text:v1.5")
    # Settings.llm = Ollama(model="mistral:7b", temperature=0.2, seed=334, request_timeout=90.0)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2, seed=334, request_timeout=90.0)
    # Uncomment the line below to use a different model
    # Settings.llm = Ollama(model="deepseek-r1:7b", request_timeout=90.0)

def load_documents_with_metadata(directory_path):
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

def load_index(vector_index_storage_dir: str):
    """Function to load an index from the specified directory."""
        
    storage_context = StorageContext.from_defaults(persist_dir=vector_index_storage_dir)
    index = load_index_from_storage(storage_context)
    print(f"Loaded index from {vector_index_storage_dir}.")

    return index

# 3.1 Create a generic query engine that can filter by metadata
def get_filtered_query_engine(index, filters=None):
    # This retriever will apply filters before fetching nodes
    retriever = index.as_retriever(
        similarity_top_k=5, # You can adjust top_k
        filters=filters # Pass the filters here
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=get_response_synthesizer(),
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
    )
    return query_engine

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

def lambda_handler(event, context):
    """Main function to run the script."""
    llm_settings()

    index = None
    if os.path.exists(VECTOR_INDEX_STORAGE_DIR):
        index = load_index(VECTOR_INDEX_STORAGE_DIR)
    else:
        documents = load_documents_with_metadata(FILES_DIR)
        node_parser = SentenceSplitter(chunk_size=NODE_CHUNK_SIZE, chunk_overlap=NODE_CHUNK_OVERLAP)
        nodes = node_parser.get_nodes_from_documents(documents)
        index = VectorStoreIndex(nodes)
        index.storage_context.persist(VECTOR_INDEX_STORAGE_DIR)
        print("\nVectorStoreIndex created for all documents.")


    # --- Custom QA Prompt Template ---
    # This is the prompt that tells the LLM how to answer based on the context.
    # The {{context_str}} and {{query_str}} are placeholders LlamaIndex will fill.
    # The key is to add strong instructions about using *only* the context.
    qa_prompt_template = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n" # This is where the retrieved nodes' content will be inserted
        "---------------------\n"
        "Given the context information and NOT prior knowledge, "
        "answer the query. If the answer is not in the context, clearly state 'I do not have enough information to answer this question from the provided documents.'\n"
        "Query: {query_str}\n" # This is the user's question
        "Answer: "
    )
    qa_prompt = PromptTemplate(qa_prompt_template)

    # 3.2 Define individual QueryEngineTools for each document type/model
    # We'll create "dummy" filter objects that the Router can use for description,
    # but the actual filtering will happen dynamically.

    # Query engine for Cessna 162 POH
    poh_162_filters = MetadataFilters(
        filters=[MetadataFilter(key="aircraft_model", value="Cessna 162", operator=FilterOperator.EQ)]
    )
    poh_162_qe_tool = QueryEngineTool.from_defaults(
        query_engine=get_filtered_query_engine(index, poh_162_filters),
        description=(
            "Useful for questions specifically about the Cessna 162 aircraft, "
            "including its engine, performance, and operational procedures. "
            "This uses the Pilot Operating Handbook (POH) for Cessna 162."
        ),
        name="poh_cessna_162_tool",
    )

    # Query engine for general aviation regulations
    phak_filters = MetadataFilters(
        filters=[MetadataFilter(key="document_type", value="Pilot Handbook of Aeronautical Knowledge", operator=FilterOperator.EQ)]
    )
    phak_qe_tool = QueryEngineTool.from_defaults(
        query_engine=get_filtered_query_engine(index, phak_filters),
        description=(
            "Useful for general aviation regulations, VFR flight rules, "
            "air traffic control procedures, and other non-aircraft-specific aviation topics."
        ),
        name="phak_tool",
    )

    afh_filters = MetadataFilters(
        filters=[MetadataFilter(key="document_type", value="Airplane Flying Handbook", operator=FilterOperator.EQ)]
    )
    afh_qe_tool = QueryEngineTool.from_defaults(
        query_engine=get_filtered_query_engine(index, afh_filters),
        description=(
            "Useful for techniques for flying aircraft of all type."
        ),
        name="afh_tool"
    )

    acs_filters = MetadataFilters(
        filters=[MetadataFilter(key="document_type", value="Airman Certification Standards", operator=FilterOperator.EQ)]
    )
    acs_qe_tool = QueryEngineTool.from_defaults(
        query_engine=get_filtered_query_engine(index, acs_filters),
        description=(
            "The standards and expectations of Private Pilots. "
            "Useful to understand what requirements they must achieve to pass the final test (Checkride) "
            "and expectations of a Private Pilots Licesnse (PPL)"
        ),
        name="acs_tool"
    )

    # Default/General query engine (if no specific tool is chosen)
    # This one doesn't have specific metadata filters at the tool level, it searches everything.
    general_qe_tool = QueryEngineTool.from_defaults(
        query_engine=get_filtered_query_engine(index),
        description=(
            "Use if the other tools are not suitable."
        ),
        name="general_aviation_tool",
    )

    # 3.3 Create the RouterQueryEngine
    # The LLMSingleSelector will use the descriptions of the tools to decide which one to use.
    router_query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[
            poh_162_qe_tool,
            phak_qe_tool,
            afh_qe_tool,
            acs_qe_tool,
            general_qe_tool, # Include a general tool as a fallback
        ],
        summarizer=TreeSummarize(verbose=True, summary_template=qa_prompt),
        verbose=True # Set to True to see which tool the router selects
    )

    print("\nRouterQueryEngine configured.")

    ### --- Step 4: Querying with Chain Reasoning ---

    print("\n--- Querying Examples ---")

    queries = [
        "What kind of engine does the Cessna 162 use?",
        "Describe the electrical system of the Cessna 172.",
        "What are the rules for VFR flight?",
        "According to the Airman Certification Standards for the Private Pilots license what skills when maneuvering during slow flight do I need to exhibit and what are their codes?",
        "Summarize Chapter 1 Introduction to Flying for me from the PHAK.",
        "According to the AFH what is the protocol for Turbulent Air Approach and Landing?",
        "You are an FAA exam preparation assistant. Generate one multiple-choice question with four possible answers about the engine of a Cessna 162 based on the POH of the 162. The question should test a key concept from the text. Clearly indicate the correct answer."
    ]

    full_response = ""

    for i, query in enumerate(queries):
        print(f"\n--- Query {i+1}: {query} ---")
        response = router_query_engine.query(query)
        print(response)
        print("\nSource Document(s) and Page Number(s):")

        full_response += f"\nQuery {i+1}: {query}\nResponse: {response.response}\nSources:\n"
        
        # Iterate through the source_nodes to get file name and page label
        # Each node in response.source_nodes is a NodeWithScore object
        for node_with_score in response.source_nodes:
            # Access the underlying TextNode (or other Node type)
            node = node_with_score.node
            
            # Get metadata from the node
            file_name = node.metadata.get('file_name', 'N/A')
            page_label = node.metadata.get('page_label', 'N/A') # 'page_label' is the common key for page number
            
            print(f"- File: {file_name}, Page: {page_label}")
            full_response += f"- File: {file_name}, Page: {page_label}\n"
            # Optionally, you can print a snippet of the content to verify
            # print(f"  Content Snippet: {node.get_content()[:150]}...")
    return json.dumps({
        'statusCode': 200,
        'body': full_response
    })

if __name__ == "__main__":
    lambda_handler(None, None)