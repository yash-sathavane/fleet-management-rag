import google.generativeai as genai

#taking api key
from dotenv import load_dotenv
import os

# Load variables from .env
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

response = model.generate_content(
    "What is AI?"
)

print(response.text)

# import google.generativeai as genai

# genai.configure(api_key="YOUR_API_KEY")

models = genai.list_models()
for m in models:
    print(m.name)
