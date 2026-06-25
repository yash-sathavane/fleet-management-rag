from fastapi import FastAPI
from pydantic import BaseModel

from graph.workflow import graph

app = FastAPI(
    title="Fleet Management RAG API",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {
        "message": "Fleet Management RAG API is running."
    }


@app.post("/ask")
def ask(request: QueryRequest):

    result = graph.invoke(
        {
            "query": request.query,
            "intent": "",
            "response": "",
            "error": "",
            "manual_context": "",
            "retrieved_chunks": [],
            "truck_data": {},
            "prompt": "",
        }
    )

    return {
        "query": request.query,
        "intent": result.get("intent", ""),
        "manual_context": result.get("manual_context", ""),
        "retrieved_chunks": result.get("retrieved_chunks", []),
        "truck_data": result.get("truck_data", {}),
        "prompt": result.get("prompt", ""),
        "response": result.get("response", ""),
        "error": result.get("error", "")
    }