from langchain_google_genai import GoogleGenerativeAIEmbeddings

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

vector = embedding_model.embed_query(
    "What is engine temperature?"
)

print("Vector Length:", len(vector))
print(vector[:5])
