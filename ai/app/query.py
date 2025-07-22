import os
import logging
import sys
import json
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from llama_index.core import VectorStoreIndex, DocumentSummaryIndex, SimpleDirectoryReader, get_response_synthesizer, Settings, StorageContext, load_index_from_storage, Document
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
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

VECTOR_INDEX_STORAGE_DIR = "./app/storage_local"
FILES_DIR = "./app/files"
NODE_CHUNK_SIZE = 512
NODE_CHUNK_OVERLAP = 100
EMBED_MODEL = OllamaEmbedding(url="http://localhost:11434/api/embeddings", model_name="nomic-embed-text:v1.5") # Default embedding model
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

def load_index(chroma_collection):
    """Function to load an index from the specified directory."""
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=EMBED_MODEL)
    print(f"Loaded indeces from {VECTOR_INDEX_STORAGE_DIR}")

    return index

# 3.1 Create a generic query engine that can filter by metadata
def get_filtered_query_engine(index, filters=None):
    # This retriever will apply filters before fetching nodes
    retriever = index.as_retriever(
        similarity_top_k=5, # You can adjust top_k
        filters=filters, # Pass the filters here
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=get_response_synthesizer(),
        # node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
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

def full_query(query):
    """Main function to run the script."""
    llm_settings()
    
    chroma_client = chromadb.PersistentClient(VECTOR_INDEX_STORAGE_DIR)
    collection_name = "faa_documents"
    chroma_collection = chroma_client.get_collection(collection_name, embedding_function=OllamaEmbeddingFunction(model_name="nomic-embed-text:v1.5"))

    index = load_index(chroma_collection)
    print("\nVectorStoreIndex loaded.")
    print(f"\nVerifying data directly in ChromaDB collection '{collection_name}'...")
    print(f"Number of items in ChromaDB collection: {chroma_collection.count()}")

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
        summarizer=TreeSummarize(verbose=True, summary_template=qa_prompt.format()),
        verbose=True # Set to True to see which tool the router selects
    )

    print("\nRouterQueryEngine configured.")

    response = router_query_engine.query(query)
    return response
