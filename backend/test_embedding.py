from langchain_google_genai import GoogleGenerativeAIEmbeddings

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key="AIzaSyDCbzVthVyzDy8Liu6qcJIhLFa1FLoFN3U"
)

vector = embedding_model.embed_query(
    "What is engine temperature?"
)

print("Vector Length:", len(vector))
print(vector[:5])
