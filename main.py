from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from backend.agents.workflow import graph

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserQuery(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "Fleet Management AI Backend Running"
    }


@app.post("/ask")
def ask_question(query: UserQuery):

    try:

        response = graph.invoke(
            {
                "question": query.question
            }
        )

        return {
            "success": True,
            "question": query.question,
            "answer": response["answer"]
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }