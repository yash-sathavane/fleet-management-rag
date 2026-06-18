import google.generativeai as genai

genai.configure(api_key="AIzaSyDYvxv8BW2LsLi8io-hUReZrLwTTc6jB78")

model = genai.GenerativeModel("gemini-2.5-flash")

response = model.generate_content(
    "What is AI?"
)

print(response.text)