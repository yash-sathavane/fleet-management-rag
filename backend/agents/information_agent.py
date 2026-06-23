import json
import chromadb
import google.generativeai as genai

from langchain_google_genai import GoogleGenerativeAIEmbeddings

#taking api key
from dotenv import load_dotenv
import os

from graph.state import FleetState

# Load variables from .env
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=API_KEY
)

client = chromadb.PersistentClient(
    path=os.path.join(os.path.dirname(__file__), "../chromadb")
)

try:
    collection = client.get_collection("fleet_manual")
except Exception:
    collection = client.get_or_create_collection("fleet_manual")


def information_agent(question_or_state):

    # Support both State dict (from my graph) and string (direct call/their graph)
    if isinstance(question_or_state, dict):
        question = question_or_state.get("query", question_or_state.get("question", ""))
        intent = question_or_state.get("intent", "hybrid")
    else:
        question = question_or_state
        intent = "hybrid"

    # IoT Data
    truck_data = "{}"
    if intent in ["iot", "hybrid"]:
        try:
            data_path = os.path.join(os.path.dirname(__file__), "../../iot_data/trucks.json")
            with open(data_path) as file:
                truck_data = json.load(file)
        except FileNotFoundError:
            try:
                data_path = os.path.join(os.path.dirname(__file__), "../iot_data/trucks.json")
                with open(data_path) as file:
                    truck_data = json.load(file)
            except Exception:
                truck_data = "{}"

    # RAG Search
    manual_context = "No relevant manual data needed."
    if intent in ["manual", "hybrid"]:
        try:
            query_embedding = embedding_model.embed_query(question)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=2
            )
            documents = results.get("documents", [])
            if documents and documents[0]:
                manual_context = "\n".join(documents[0])
        except Exception as exc:
            manual_context = f"No relevant manual data found. ({exc})"

    # Build Context
    full_context = f"""
    Manual Information:

    {manual_context}

    Truck Data:

    {truck_data}
    """

    prompt = f"""
    Answer using provided information.

    Context:

    {full_context}

    Question:

    {question}
    """

    try:
        response = model.generate_content(prompt)
        response_text = response.text
    except Exception as exc:
        response_text = (
            "Unable to reach the Gemini API right now. "
            f"Falling back to local context only. ({exc})"
        )

    if isinstance(question_or_state, dict):
        return {"response": response_text, "answer": response_text}

    return response_text
