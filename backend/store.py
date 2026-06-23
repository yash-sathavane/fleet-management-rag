from pypdf import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb
from dotenv import load_dotenv
import os

# Extra libraries for parsing
import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# Load API key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Step 1: Extract text with PdfReader ---
reader = PdfReader("../manuals/blazo-brochure.pdf")
text = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text

# --- Step 2: Extract text + tables with pdfplumber ---
table_texts = []
with pdfplumber.open("../manuals/manual.pdf") as pdf:
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += "\n" + page_text  # merge with PdfReader text
        tables = page.extract_tables()
        for table in tables:
            # Convert table rows into a string
            table_str = "\n".join([", ".join(row) for row in table if row])
            table_texts.append(table_str)

# --- Step 3: Extract images ---
doc = fitz.open("../manuals/manual.pdf")
image_files = []
for page_index in range(len(doc)):
    page = doc[page_index]
    for img_index, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        filename = f"image_{page_index}_{img_index}.png"
        pix.save(filename)
        image_files.append(filename)

# --- Step 4: OCR images ---
ocr_texts = []
for img_file in image_files:
    img = Image.open(img_file)
    text_from_img = pytesseract.image_to_string(img)
    if text_from_img.strip():
        ocr_texts.append(text_from_img)

# --- Step 5: Merge everything ---
all_text = text + "\n".join(table_texts) + "\n".join(ocr_texts)

chunk_size = 500
chunks = [all_text[i:i+chunk_size] for i in range(0, len(all_text), chunk_size)]

# --- Step 6: Embedding model ---
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=API_KEY
)

# --- Step 7: ChromaDB setup ---
client = chromadb.PersistentClient(path="./chromadb")
collection = client.get_or_create_collection(name="fleet_manual1")

# --- Step 8: Store chunks ---
for index, chunk in enumerate(chunks):
    embedding = embedding_model.embed_query(chunk)
    collection.add(
        ids=[str(index)],
        embeddings=[embedding],
        documents=[chunk]
    )

print("All text, tables, and OCR content stored successfully.")
