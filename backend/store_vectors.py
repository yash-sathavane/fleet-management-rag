from pypdf import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb

#taking api key
from dotenv import load_dotenv
import os

# Load variables from .env
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")


# Read PDF

reader = PdfReader("../manuals/manual.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text

# Create chunks

chunk_size = 500

chunks = []

for i in range(0, len(text), chunk_size):
    chunks.append(text[i:i + chunk_size])

# Create embedding model

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=API_KEY
)

# ChromaDB

client = chromadb.PersistentClient(path="./chromadb")

collection = client.get_or_create_collection(
    name="fleet_manual"
)

# Store chunks

for index, chunk in enumerate(chunks):

    embedding = embedding_model.embed_query(chunk)

    collection.add(
        ids=[str(index)],
        embeddings=[embedding],
        documents=[chunk]
    )

print("All chunks stored successfully.")