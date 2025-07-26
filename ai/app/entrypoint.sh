#!/bin/bash
set -e

echo "Running embed_docs.py to populate ChromaDB..."
python ./app/embed_docs.py

echo "Starting FastAPI app..."
uvicorn main:app --host 0.0.0.0 --port 5000