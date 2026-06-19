import google.generativeai as genai

genai.configure(api_key="AIzaSyDCbzVthVyzDy8Liu6qcJIhLFa1FLoFN3U")

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
