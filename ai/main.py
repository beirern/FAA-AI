from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

from ai.app.query import full_query

class Query(BaseModel):
    query: str

app = FastAPI()

@app.get("/test")
async def root():
    return {"message": "Hello World"}

@app.post("/submit")
async def submit(query: Query):

    response = full_query(query.query)

    return { "message": response.response }

app.mount("/", StaticFiles(directory="static", html=True), name="static")
