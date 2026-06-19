from pypdf import PdfReader

reader = PdfReader("../manuals/manual.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text

chunk_size = 500

chunks = []

for i in range(0, len(text), chunk_size):
    chunks.append(text[i:i + chunk_size])

print("Total Chunks:", len(chunks))

for index, chunk in enumerate(chunks):
    print("\n")
    print("=" * 50)
    print(f"CHUNK {index + 1}")
    print("=" * 50)
    print(chunk)