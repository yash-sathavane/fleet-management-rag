from pypdf import PdfReader

reader = PdfReader("../manuals/manual.pdf")

for page in reader.pages:
    print(page.extract_text())