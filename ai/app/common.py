import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()  # take environment variables


environment = os.environ.get("ENVIRONMENT", "development")

CHAT_MODEL = ""
EMBEDDING_MODEL = ""
COLLECTION_NAME = "faa-documents"
PDF_LOADER_MODEL = ""
CHROMA_DIRECTORY = ""
CHROMA_HOST = "chromadb"  # Container network
CHROMA_PORT = 5005  # In Config file and compose

if environment == "production":
    CHAT_MODEL = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    PDF_LOADER_MODEL = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    CHROMA_DIRECTORY = "/tmp/chroma_langchain_db"
else:
    CHAT_MODEL = ChatOllama(
        model="gemma3:4b",
        max_tokens=2048,
        temperature=0.0,
        base_url="http://host.docker.internal:11434",
    )
    EMBEDDING_MODEL = OllamaEmbeddings(
        model="nomic-embed-text:v1.5", base_url="http://host.docker.internal:11434"
    )
    PDF_LOADER_MODEL = ChatOllama(
        model="gemma3:4b",
        max_tokens=2048,
        temperature=0.0,
        base_url="http://host.docker.internal:11434",
    )
    CHROMA_DIRECTORY = "./chroma_langchain_db"
