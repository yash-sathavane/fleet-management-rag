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

client = chromadb.PersistentClient(
    path="./chromadb"
)

collection = client.get_collection(
    "fleet_manual"
)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=API_KEY
)

question = input("Ask Question: ")

query_embedding = embedding_model.embed_query(
    question
)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=1
)

context = results["documents"][0][0]

prompt = f"""
Answer only from context.

Context:
{context}

Question:
{question}
"""

response = model.generate_content(prompt)

print("\nAnswer:")
print(response.text)            