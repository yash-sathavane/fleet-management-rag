import json
import chromadb
import google.generativeai as genai

from langchain_google_genai import GoogleGenerativeAIEmbeddings

#taking api key
from dotenv import load_dotenv
import os

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
    path="../chromadb"
)

collection = client.get_collection(
    "fleet_manual"
)



def information_agent(question):

    # IoT Data

    with open("../../iot_data/trucks.json") as file:
        truck_data = json.load(file)

    # RAG Search

    query_embedding = embedding_model.embed_query(
        question
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )

    manual_context = "\n".join(
        results["documents"][0]
    )

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

    response = model.generate_content(
        prompt
    )

    return response.text

