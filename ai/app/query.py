from dataclasses import dataclass
from functools import cache

from langchain import hub
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import Annotated, List, TypedDict

from .common import CHAT_MODEL, CHROMA_DIRECTORY, COLLECTION_NAME, EMBEDDING_MODEL


# Define schema for search
class Search(TypedDict):
    """Search query."""

    query: Annotated[str, ..., "Search query to run."]
    # document_type: Annotated[
    #     Literal["Pilot Operating Handbook", "Pilot Handbook of Aeronautical Knowledge", "Airman Certification Standards", "Airplane Flying Handbook"],
    #     ...,
    #     "Document to query.",
    # ]


class State(TypedDict):
    question: str
    query: Search
    document_name: str
    context: List[Document]
    answer: str


@dataclass
class ContextSchema:
    document_name: str


@cache
def load_vector_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=EMBEDDING_MODEL,
        persist_directory=CHROMA_DIRECTORY,
    )


def analyze_query(state: State):
    structured_llm = CHAT_MODEL.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}


def retrieve(state: State):
    query = state["query"]
    retrieved_docs = load_vector_store().similarity_search(
        query["query"],
        filter={
            "document_type": state["document_name"],
        },
    )
    return {"context": retrieved_docs}


def generate(state: State):
    prompt = hub.pull("rlm/rag-prompt")
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = CHAT_MODEL.invoke(messages)
    return {"answer": response.content}


def query(question: str, document_name: str) -> str:
    graph_builder = StateGraph(State, context_schema=ContextSchema).add_sequence(
        [analyze_query, retrieve, generate]
    )
    graph_builder.add_edge(START, "analyze_query")
    graph = graph_builder.compile()

    result = graph.invoke({"question": question, "document_name": document_name})

    return result["answer"]


if __name__ == "__main__":
    # To run this script, use `python -m ai.app.query` from the project root directory.
    # This ensures that relative imports work correctly.
    vector_store = load_vector_store()

    question = "What should I do prior to takeoff?"

    graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
    graph_builder.add_edge(START, "analyze_query")
    graph = graph_builder.compile()

    result = graph.invoke(
        {
            "question": question,
            "document_name": "Pilot Handbook of Aeronautical Knowledge",
        }
    )

    print(f"Context: {result['context']}\n\n")
    print(f"Answer: {result['answer']}")
