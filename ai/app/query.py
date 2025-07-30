from langchain_chroma import Chroma

from langchain import hub

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

from langchain_core.documents import Document
from typing_extensions import List, TypedDict, Annotated

from langgraph.graph import START, StateGraph

from typing import Literal

from dotenv import load_dotenv

from functools import cache

load_dotenv()  # take environment variables

# Define schema for search
class Search(TypedDict):
    """Search query."""

    query: Annotated[str, ..., "Search query to run."]
    document_type: Annotated[
        Literal["Pilot Operating Handbook", "Pilot Handbook of Aeronautical Knowledge", "Airman Certification Standards", "Airplane Flying Handbook"],
        ...,
        "Document to query.",
    ]

class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str

@cache
def load_vector_store():
    return Chroma(
        collection_name="faa_documents",
        embedding_function=OllamaEmbeddings(model="nomic-embed-text:v1.5"),
        persist_directory="./chroma_langchain_db",
    )

def analyze_query(state: State):
    structured_llm = ChatOllama(model="gemma3:4b", max_tokens=2048, temperature=0.0).with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}

def retrieve(state: State):
    query = state["query"]
    retrieved_docs = load_vector_store().similarity_search(
        query["query"],
        filter={
            "document_type": query["document_type"],
        },
    )
    return {"context": retrieved_docs}

def generate(state: State):
    prompt = hub.pull("rlm/rag-prompt")
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = ChatOllama(model="gemma3:4b", max_tokens=2048, temperature=0.0).invoke(messages)
    return {"answer": response.content}

def query(question: str) -> str:
    graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
    graph_builder.add_edge(START, "analyze_query")
    graph = graph_builder.compile()

    result = graph.invoke({"question": question})

    return result['answer']

if __name__ == "__main__":
    vector_store = load_vector_store()

    question = "What should I do prior to takeoff?"

    graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
    graph_builder.add_edge(START, "analyze_query")
    graph = graph_builder.compile()

    result = graph.invoke({"question": question})

    print(f"Context: {result['context']}\n\n")
    print(f"Answer: {result['answer']}")
