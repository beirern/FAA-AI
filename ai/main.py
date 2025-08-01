import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

if (
    os.environ.get("ENVIRONMENT") == "production"
    or os.environ.get("ENVIRONMENT") == "development"
):
    from app.query import query as llm_query  # Docker containers like this
else:
    from ai.app.query import query as llm_query  # FastAPI likes this


class Query(BaseModel):
    query: str
    selectedFile: str


app = FastAPI()


@app.get("/test")
async def root():
    return {"message": "Hello World"}


@app.post("/submit")
async def submit(query: Query):
    response: str = llm_query(query.query, query.selectedFile)

    return {"message": response}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
