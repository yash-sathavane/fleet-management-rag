import json
import os

import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from graph.state import FleetState

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=API_KEY)

BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../.."))

CHROMA_PATH = os.path.join(ROOT_DIR, "backend", "chromadb")
IOT_DATA_PATH = os.path.join(ROOT_DIR, "iot_data", "trucks.json")

MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"
COLLECTION_NAME = "fleet_manual"

# ------------------------------------------------------------
# Models
# ------------------------------------------------------------

model = genai.GenerativeModel(MODEL_NAME)

embedding_model = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=API_KEY,
)

client = chromadb.PersistentClient(path=CHROMA_PATH)

try:
    collection = client.get_collection(COLLECTION_NAME)
except Exception:
    collection = client.get_or_create_collection(COLLECTION_NAME)

# ------------------------------------------------------------
# Load IoT Data Once
# ------------------------------------------------------------

def load_iot_data():
    try:
        with open(IOT_DATA_PATH, "r") as file:
            return json.load(file)
    except Exception:
        return {}


TRUCK_DATA = load_iot_data()

# ------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------

def retrieve_manual_context(question: str):

    try:

        query_embedding = embedding_model.embed_query(question)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=6,
        )

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        if documents and documents[0]:

            retrieved_chunks = documents[0]
            

            manual_context = "\n\n".join(retrieved_chunks)

            return (
                manual_context,
                retrieved_chunks,
                
            )

        return (
            "No relevant manual data found.",
            [],
            [],
        )

    except Exception as exc:

        return (
            f"No relevant manual data found. ({exc})",
            [],
            [],
        )

# ------------------------------------------------------------
# Prompt Builder
# ------------------------------------------------------------

def build_prompt(
    question: str,
    manual_context: str,
    truck_data,
):

    context = f"""
Manual Information:

{manual_context}

Truck Data:

{truck_data}
"""

    return f"""
Answer using ONLY the provided information.

Context:

{context}

Question:

{question}
"""

# ------------------------------------------------------------
# Gemini
# ------------------------------------------------------------

def generate_answer(prompt: str):

    try:
        response = model.generate_content(prompt)
        return response.text

    except Exception as exc:

        return (
            "Unable to reach Gemini.\n"
            f"{exc}"
        )

# ------------------------------------------------------------
# Main Agent
# ------------------------------------------------------------

def information_agent(question_or_state):

    if isinstance(question_or_state, dict):
        question = question_or_state.get("query", "")
        intent = question_or_state.get("intent", "hybrid")
    else:
        question = question_or_state
        intent = "hybrid"

    manual_context = "No relevant manual data needed."
    retrieved_chunks = []
    
    truck_data = {}

    if intent == "manual":

        (
            manual_context,
            retrieved_chunks,
        
        ) = retrieve_manual_context(question)

    elif intent == "iot":

        truck_data = TRUCK_DATA

    elif intent == "hybrid":

        (
            manual_context,
            retrieved_chunks,
         
        ) = retrieve_manual_context(question)

        truck_data = TRUCK_DATA

    prompt = build_prompt(
        question=question,
        manual_context=manual_context,
        truck_data=truck_data,
    )

    response_text = generate_answer(prompt)

    if isinstance(question_or_state, dict):

        return {

            "response": response_text,
            "answer": response_text,

            "manual_context": manual_context,
            "retrieved_chunks": retrieved_chunks,
           
            "truck_data": truck_data,
            "prompt": prompt,

        }

    return response_text