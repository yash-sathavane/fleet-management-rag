from fastapi import FastAPI

from graph.workflow import graph

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Hello"}


@app.post("/ask")
def ask(query: str):
    result = graph.invoke(
        {
            "query": query,
            "intent": "",
            "response": "",
            "error": "",
        }
    )

    return result
