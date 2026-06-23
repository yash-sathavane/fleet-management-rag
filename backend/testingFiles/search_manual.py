import chromadb

from langchain_google_genai import GoogleGenerativeAIEmbeddings

client = chromadb.PersistentClient(
    path="./chromadb"
)

collection = client.get_collection(
    "fleet_manual"
)

#taking api key
from dotenv import load_dotenv
import os

# Load variables from .env
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=API_KEY
)

question = "What is engine temperature?"

query_embedding = embedding_model.embed_query(
    question
)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=1
)

print(results["documents"][0][0])