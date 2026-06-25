from __future__ import annotations

import hashlib
import io
import json
import os
import re
import socket
from pathlib import Path
from typing import Any, Literal, TypedDict

import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

try:
    import pdfplumber
except ImportError:  # pragma: no cover - optional dependency
    pdfplumber = None

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - optional dependency
    fitz = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None

try:
    import pytesseract
except ImportError:  # pragma: no cover - optional dependency
    pytesseract = None

if pytesseract is not None:
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

class ParsedBlock(TypedDict, total=False):
    block_id: str
    source_file: str
    page_number: int
    block_type: Literal["text", "table", "ocr_text", "llm_text"]
    content: str
    order_index: int
    confidence: float | None
    metadata: dict[str, Any]


class SemanticChunk(TypedDict, total=False):
    chunk_id: str
    source_block_ids: list[str]
    source_file: str
    page_numbers: list[int]
    chunk_text: str
    chunk_type: Literal["text", "table", "mixed"]
    metadata: dict[str, Any]


BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
MANUALS_DIR = ROOT_DIR / "manuals"
CHROMA_DIR = BACKEND_DIR / "chromadb"
COLLECTION_NAME = "fleet_manual"
GOOGLE_EMBED_MODEL = "models/gemini-embedding-001"
GOOGLE_LLM_MODEL = "gemini-2.5-flash"
SEMANTIC_SIMILARITY_THRESHOLD = 0.72
MAX_CHUNK_WORDS = 420
EMBEDDING_DIMENSION = 3072

SOURCE_PDFS = [
    MANUALS_DIR / "manual.pdf",
    MANUALS_DIR / "blazo-brochure.pdf",
]


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def stable_hash(*parts: str) -> str:
    digest = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def ensure_required_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Source PDF not found: {path}")


def make_block(
    *,
    source_file: str,
    page_number: int,
    block_type: Literal["text", "table", "ocr_text", "llm_text"],
    content: str,
    order_index: int,
    parser: str,
    confidence: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> ParsedBlock:
    cleaned_content = clean_text(content)
    block_id = stable_hash(source_file, str(page_number), block_type, str(order_index), cleaned_content)
    block: ParsedBlock = {
        "block_id": block_id,
        "source_file": source_file,
        "page_number": page_number,
        "block_type": block_type,
        "content": cleaned_content,
        "order_index": order_index,
        "confidence": confidence,
        "metadata": {
            "parser": parser,
            "source_file": source_file,
            "page_number": page_number,
            "block_type": block_type,
            **(metadata or {}),
        },
    }
    return block


def table_to_markdown(table: list[list[Any]]) -> str:
    rows: list[list[str]] = []
    for row in table:
        cleaned_row = [clean_text("" if cell is None else str(cell)) for cell in row]
        if any(cleaned_row):
            rows.append(cleaned_row)

    if not rows:
        return ""

    column_count = max(len(row) for row in rows)
    normalized_rows: list[list[str]] = []
    for row in rows:
        padded_row = row + [""] * (column_count - len(row))
        normalized_rows.append(padded_row)

    header = normalized_rows[0]
    separator = ["---"] * column_count
    body = normalized_rows[1:]

    lines = [
        " | ".join(header),
        " | ".join(separator),
    ]
    for row in body:
        lines.append(" | ".join(row))

    return "\n".join(lines)


def group_blocks_by_page(blocks: list[ParsedBlock]) -> dict[tuple[str, int], list[ParsedBlock]]:
    grouped: dict[tuple[str, int], list[ParsedBlock]] = {}
    for block in blocks:
        key = (block["source_file"], block["page_number"])
        grouped.setdefault(key, []).append(block)
    for key in grouped:
        grouped[key].sort(key=lambda item: item["order_index"])
    return grouped


def extract_text_and_table_blocks(source_path: Path) -> tuple[list[ParsedBlock], int, int]:
    text_blocks: list[ParsedBlock] = []
    table_count = 0
    page_count = 0
    reader = PdfReader(str(source_path))

    if pdfplumber is not None:
        try:
            with pdfplumber.open(str(source_path)) as pdf:
                page_count = len(pdf.pages)
                for page_index, page in enumerate(pdf.pages, start=1):
                    plumber_text = clean_text(page.extract_text() or "")
                    if plumber_text:
                        page_text = plumber_text
                        parser_name = "pdfplumber"
                    else:
                        parser_name = ""
                        page_text = ""

                    if not page_text:
                        fallback_text = clean_text(reader.pages[page_index - 1].extract_text() or "")
                        if fallback_text:
                            page_text = fallback_text
                            parser_name = "pypdf"

                    if page_text:
                        text_blocks.append(
                            make_block(
                                source_file=source_path.name,
                                page_number=page_index,
                                block_type="text",
                                content=page_text,
                                order_index=0,
                                parser=parser_name or "pdfplumber",
                                metadata={
                                    "page_count": page_count,
                                },
                            )
                        )

                    tables = page.extract_tables() or []
                    for table_index, table in enumerate(tables):
                        markdown = table_to_markdown(table)
                        if markdown:
                            table_count += 1
                            text_blocks.append(
                                make_block(
                                    source_file=source_path.name,
                                    page_number=page_index,
                                    block_type="table",
                                    content=markdown,
                                    order_index=100 + table_index,
                                    parser="pdfplumber",
                                    confidence=1.0,
                                    metadata={
                                        "page_count": page_count,
                                        "table_index": table_index,
                                        "row_count": len(table),
                                        "raw_table": table,
                                    },
                                )
                            )
        except Exception as exc:
            print(f"pdfplumber parsing failed for {source_path.name}: {exc}")

    if text_blocks:
        return text_blocks, table_count, page_count

    reader = PdfReader(str(source_path))
    page_count = len(reader.pages)
    for page_index, page in enumerate(reader.pages, start=1):
        page_text = clean_text(page.extract_text() or "")
        if page_text:
            text_blocks.append(
                make_block(
                    source_file=source_path.name,
                    page_number=page_index,
                    block_type="text",
                    content=page_text,
                    order_index=0,
                    parser="pypdf",
                    metadata={
                        "page_count": page_count,
                    },
                )
            )

    return text_blocks, table_count, page_count


def extract_ocr_blocks(source_path: Path) -> list[ParsedBlock]:
    if fitz is None or pytesseract is None or Image is None:
        raise RuntimeError(
            "OCR parsing requires PyMuPDF (fitz), Pillow, and pytesseract to be installed."
        )

    ocr_blocks: list[ParsedBlock] = []
    doc = fitz.open(str(source_path))
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            images = page.get_images(full=True)
            for image_index, image_info in enumerate(images):
                xref = image_info[0]
                pixmap = fitz.Pixmap(doc, xref)
                try:
                    if pixmap.n >= 5:
                        pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
                    png_bytes = pixmap.tobytes("png")
                    pil_image = Image.open(io.BytesIO(png_bytes))
                    ocr_text = clean_text(pytesseract.image_to_string(pil_image))
                finally:
                    pixmap = None

                if ocr_text:
                    ocr_blocks.append(
                        make_block(
                            source_file=source_path.name,
                            page_number=page_index + 1,
                            block_type="ocr_text",
                            content=ocr_text,
                            order_index=200 + image_index,
                            parser="pymupdf+tesseract",
                            confidence=0.8,
                            metadata={
                                "image_index": image_index,
                                "image_count_on_page": len(images),
                            },
                        )
                    )
    finally:
        doc.close()

    return ocr_blocks


def is_gemini_reachable() -> bool:
    try:
        socket.getaddrinfo("generativelanguage.googleapis.com", 443)
        return True
    except Exception:
        return False


class EmbeddingProvider:
    def __init__(self, kind: str, google_model: Any | None = None, local_model: Any | None = None):
        self.kind = kind
        self.google_model = google_model
        self.local_model = local_model

    def embed_query(self, text: str) -> list[float]:
        if self.kind == "google" and self.google_model is not None:
            return self.google_model.embed_query(text)

        if self.local_model is None:
            raise RuntimeError("No embedding model available.")

        vector = self.local_model.encode(text)
        vector_list = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        if len(vector_list) >= EMBEDDING_DIMENSION:
            return [float(v) for v in vector_list[:EMBEDDING_DIMENSION]]
        padded = [float(v) for v in vector_list]
        padded.extend([0.0] * (EMBEDDING_DIMENSION - len(padded)))
        return padded

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.kind == "google" and self.google_model is not None:
            return self.google_model.embed_documents(texts)
            
        if self.local_model is None:
            raise RuntimeError("No embedding model available.")

        vectors = self.local_model.encode(texts)
        results = []
        for vector in vectors:
            vector_list = vector.tolist() if hasattr(vector, "tolist") else list(vector)
            if len(vector_list) >= EMBEDDING_DIMENSION:
                results.append([float(v) for v in vector_list[:EMBEDDING_DIMENSION]])
            else:
                padded = [float(v) for v in vector_list]
                padded.extend([0.0] * (EMBEDDING_DIMENSION - len(padded)))
                results.append(padded)
        return results


def fallback_llm_normalization_doc(blocks: list[ParsedBlock], source_name: str) -> str:
    sections: list[str] = [f"# {source_name}"]
    for block in blocks:
        label = block["block_type"].upper()
        sections.append(f"## {label}")
        sections.append(block["content"])
    return clean_text("\n\n".join(sections))


def normalize_document_blocks_with_llm(
    source_name: str,
    document_blocks: list[ParsedBlock],
    llm_model: Any,
) -> list[ParsedBlock]:
    """Normalizes ALL blocks from the entire document in a single LLM API call."""
    blocks_by_page = group_blocks_by_page(document_blocks)
    
    if not is_gemini_reachable():
        normalized_blocks = []
        for (src, page), blocks in blocks_by_page.items():
            text = fallback_llm_normalization_doc(blocks, source_name)
            normalized_blocks.append(
                make_block(
                    source_file=source_name,
                    page_number=page,
                    block_type="llm_text",
                    content=text,
                    order_index=900,
                    parser="gemini_fallback_offline",
                    confidence=1.0,
                    metadata={"offline_fallback": True}
                )
            )
        return normalized_blocks

    payload_pages = []
    for (src, page), blocks in blocks_by_page.items():
        payload_pages.append({
            "page_number": page,
            "blocks": [{"type": b["block_type"], "content": b["content"]} for b in blocks]
        })

    prompt = f"""You are normalizing a full PDF extraction output for semantic chunking.
The input contains text, table, and OCR blocks extracted from multiple pages of a fleet management document.

You MUST output ONLY a valid JSON array. Each object in the array must correspond to a page and contain the normalized markdown text for that page.
Format exactly like this:
[
  {{
    "page_number": 1,
    "normalized_text": "Clean markdown for page 1..."
  }}
]

Rules for normalized_text:
- Preserve tables as markdown tables.
- Merge duplicate fragments.
- Combine text, table, and OCR fragments into coherent, structured text.
- Keep section headings if present.
- Do not answer the document content or add explanations.

Input Data:
{json.dumps(payload_pages, ensure_ascii=False)}
"""

    try:
        response = llm_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            ),
            request_options={"timeout": 120},  # Extended timeout for larger docs
        )
        json_output = json.loads(response.text)
        
        normalized_blocks = []
        for item in json_output:
            page_num = item.get("page_number", 1)
            text = clean_text(item.get("normalized_text", ""))
            if not text:
                continue
                
            orig_blocks = blocks_by_page.get((source_name, page_num), [])
            normalized_blocks.append(
                make_block(
                    source_file=source_name,
                    page_number=page_num,
                    block_type="llm_text",
                    content=text,
                    order_index=900,
                    parser="gemini",
                    confidence=1.0,
                    metadata={
                        "normalized_from_block_ids": [b["block_id"] for b in orig_blocks],
                        "raw_block_count": len(orig_blocks),
                    }
                )
            )
        if normalized_blocks:
            return normalized_blocks
            
    except Exception as exc:
        print(f"LLM normalization failed for entire document {source_name}: {exc}")

    # Fallback if Gemini fails the JSON structure or times out
    fallback_blocks = []
    for (src, page), blocks in blocks_by_page.items():
        text = fallback_llm_normalization_doc(blocks, source_name)
        fallback_blocks.append(
            make_block(
                source_file=source_name,
                page_number=page,
                block_type="llm_text",
                content=text,
                order_index=900,
                parser="gemini_fallback",
                confidence=1.0,
                metadata={"offline_fallback": True}
            )
        )
    return fallback_blocks


def split_structured_block(block: ParsedBlock) -> list[ParsedBlock]:
    content = clean_text(block["content"])
    if not content:
        return []

    if block["block_type"] == "table":
        return [block]

    chunks: list[str] = []
    for section in re.split(r"\n{2,}", content):
        cleaned_section = clean_text(section)
        if not cleaned_section:
            continue
        if word_count(cleaned_section) <= 120:
            chunks.append(cleaned_section)
            continue

        sentence_buffer: list[str] = []
        current_text: list[str] = []
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned_section):
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_buffer.append(sentence)
            if word_count(" ".join(sentence_buffer)) >= 90:
                current_text.append(" ".join(sentence_buffer))
                sentence_buffer = []
        if sentence_buffer:
            current_text.append(" ".join(sentence_buffer))
        chunks.extend(current_text or [cleaned_section])

    split_blocks: list[ParsedBlock] = []
    for index, chunk_text in enumerate(chunks):
        split_blocks.append(
            make_block(
                source_file=block["source_file"],
                page_number=block["page_number"],
                block_type=block["block_type"],
                content=chunk_text,
                order_index=block["order_index"] * 1000 + index,
                parser=block["metadata"].get("parser", "unknown"),
                confidence=block.get("confidence"),
                metadata={
                    **block["metadata"],
                    "split_index": index,
                    "split_from_block_id": block["block_id"],
                },
            )
        )

    return split_blocks


def build_semantic_chunks(
    normalized_blocks: list[ParsedBlock],
    local_embedder: EmbeddingProvider,
) -> list[SemanticChunk]:
    units: list[ParsedBlock] = []
    for block in normalized_blocks:
        units.extend(split_structured_block(block))

    semantic_chunks: list[SemanticChunk] = []
    current_units: list[ParsedBlock] = []
    current_last_embedding: list[float] | None = None

    def flush_current_chunk() -> None:
        nonlocal current_units, current_last_embedding
        if not current_units:
            return

        chunk_text_parts = [unit["content"] for unit in current_units if unit["content"]]
        chunk_text = clean_text("\n\n".join(chunk_text_parts))
        source_block_ids = [unit["block_id"] for unit in current_units]
        page_numbers = sorted({unit["page_number"] for unit in current_units})
        chunk_types = {unit["block_type"] for unit in current_units}
        chunk_type: Literal["text", "table", "mixed"]
        if chunk_types == {"table"}:
            chunk_type = "table"
        elif "table" in chunk_types:
            chunk_type = "mixed"
        else:
            chunk_type = "text"

        semantic_chunks.append(
            {
                "chunk_id": stable_hash(
                    normalized_blocks[0]["source_file"] if normalized_blocks else "unknown",
                    str(len(semantic_chunks)),
                    "|".join(source_block_ids),
                    chunk_text,
                ),
                "source_block_ids": source_block_ids,
                "source_file": current_units[0]["source_file"],
                "page_numbers": page_numbers,
                "chunk_text": chunk_text,
                "chunk_type": chunk_type,
                "metadata": {
                    "source_file": current_units[0]["source_file"],
                    "page_numbers": page_numbers,
                    "source_block_ids": source_block_ids,
                    "chunk_type": chunk_type,
                    "unit_count": len(current_units),
                    "semantic_similarity_threshold": SEMANTIC_SIMILARITY_THRESHOLD,
                },
            }
        )

        current_units = []
        current_last_embedding = None

    for unit in units:
        unit_text = unit["content"]
        if not unit_text:
            continue

        # Use Local Embedder for semantic similarity logic (No Gemini Calls here)
        unit_embedding = local_embedder.embed_query(unit_text)

        if not current_units:
            current_units.append(unit)
            current_last_embedding = unit_embedding
            continue

        similarity = 0.0
        if current_last_embedding is not None:
            dot_product = sum(a * b for a, b in zip(unit_embedding, current_last_embedding))
            magnitude_a = sum(a * a for a in unit_embedding) ** 0.5
            magnitude_b = sum(b * b for b in current_last_embedding) ** 0.5
            if magnitude_a and magnitude_b:
                similarity = dot_product / (magnitude_a * magnitude_b)

        current_word_total = word_count("\n\n".join(unit_item["content"] for unit_item in current_units))
        unit_word_total = word_count(unit_text)
        should_split = (
            similarity < SEMANTIC_SIMILARITY_THRESHOLD
            or current_word_total + unit_word_total > MAX_CHUNK_WORDS
        )

        if should_split:
            flush_current_chunk()
            current_units.append(unit)
            current_last_embedding = unit_embedding
            continue

        current_units.append(unit)
        current_last_embedding = unit_embedding

    flush_current_chunk()
    return semantic_chunks


def serialize_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, ensure_ascii=False)


def upsert_chunks_to_chroma(
    collection: Any,
    semantic_chunks: list[SemanticChunk],
    google_embedder: EmbeddingProvider,
) -> None:
    if not semantic_chunks:
        return

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for chunk in semantic_chunks:
        ids.append(chunk["chunk_id"])
        documents.append(chunk["chunk_text"])
        metadatas.append(
            {
                "source_file": chunk["source_file"],
                "page_numbers": serialize_value(chunk["page_numbers"]),
                "source_block_ids": serialize_value(chunk["source_block_ids"]),
                "chunk_type": chunk["chunk_type"],
                "metadata": serialize_value(chunk["metadata"]),
            }
        )

    # ONE Gemini Embedding API Call for all documents batched together
    embeddings = google_embedder.embed_documents(documents)

    if hasattr(collection, "upsert"):
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    else:
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )


def process_source_pdf(
    source_path: Path,
    llm_model: Any,
    local_embedder: EmbeddingProvider,
) -> dict[str, Any]:
    ensure_required_file(source_path)

    text_and_table_blocks, table_count, page_count = extract_text_and_table_blocks(source_path)
    ocr_blocks = extract_ocr_blocks(source_path)

    all_blocks = text_and_table_blocks + ocr_blocks
    # Ensure blocks are strictly sorted chronologically before batch normalization
    all_blocks.sort(key=lambda b: (b["page_number"], b["order_index"]))

    normalized_blocks: list[ParsedBlock] = []
    if all_blocks:
        # 1. ONE API Call per PDF handles all blocks
        normalized_blocks = normalize_document_blocks_with_llm(
            source_name=source_path.name,
            document_blocks=all_blocks,
            llm_model=llm_model,
        )

    # 2. Local chunking ensures NO Gemini calls are made for chunk similarity calculations
    semantic_chunks = build_semantic_chunks(normalized_blocks, local_embedder)

    print(f"\nProcessing: {source_path.name}")
    print(f"Pages scanned: {page_count}")
    print(f"Text blocks extracted: {sum(1 for block in text_and_table_blocks if block['block_type'] == 'text')}")
    print(f"Tables extracted: {table_count}")
    print(f"OCR blocks extracted: {len(ocr_blocks)}")
    print(f"LLM normalized pages generated: {len(normalized_blocks)}")
    print(f"Semantic chunks generated: {len(semantic_chunks)}")

    return {
        "text_and_table_blocks": text_and_table_blocks,
        "table_count": table_count,
        "ocr_blocks": ocr_blocks,
        "normalized_blocks": normalized_blocks,
        "semantic_chunks": semantic_chunks,
        "page_count": page_count,
    }


def run_ingestion_pipeline() -> None:
    load_dotenv(BACKEND_DIR / ".env")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing from backend/.env")

    genai.configure(api_key=api_key)

    llm_model = genai.GenerativeModel(GOOGLE_LLM_MODEL)
    
    # Instantiate Providers individually
    local_embedder = EmbeddingProvider(
        kind="local", 
        local_model=SentenceTransformer("all-MiniLM-L6-v2")
    )
    
    google_embedder = EmbeddingProvider(
        kind="google",
        google_model=GoogleGenerativeAIEmbeddings(
            model=GOOGLE_EMBED_MODEL,
            google_api_key=api_key,
        )
    )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        collection = client.get_or_create_collection(name=COLLECTION_NAME)

    count_before = collection.count()
    print(f"Chroma document count before ingestion: {count_before}")

    total_text_blocks = 0
    total_table_blocks = 0
    total_ocr_blocks = 0
    total_llm_blocks = 0
    total_semantic_chunks = 0

    for source_path in SOURCE_PDFS:
        result = process_source_pdf(
            source_path=source_path,
            llm_model=llm_model,
            local_embedder=local_embedder,
        )

        source_text_blocks = result["text_and_table_blocks"]
        total_text_blocks += sum(1 for block in source_text_blocks if block["block_type"] == "text")
        total_table_blocks += result["table_count"]
        total_ocr_blocks += len(result["ocr_blocks"])
        total_llm_blocks += len(result["normalized_blocks"])
        total_semantic_chunks += len(result["semantic_chunks"])

        # 3. Batch insert final chunks (ONE Google Embed request per document)
        upsert_chunks_to_chroma(
            collection=collection,
            semantic_chunks=result["semantic_chunks"],
            google_embedder=google_embedder,
        )

    count_after = collection.count()
    print(f"\nTotal text blocks extracted: {total_text_blocks}")
    print(f"Total tables extracted: {total_table_blocks}")
    print(f"Total OCR blocks extracted: {total_ocr_blocks}")
    print(f"Total LLM normalized pages generated: {total_llm_blocks}")
    print(f"Total semantic chunks generated: {total_semantic_chunks}")
    print(f"Chroma document count after ingestion: {count_after}")
    print(f"Chroma document count delta: {count_after - count_before}")

    try:
        peek = collection.peek()
        embeddings = peek.get("embeddings", [])
        if embeddings:
            print(f"Persisted embedding dimension: {len(embeddings[0])}")
        print(f"Final storing embedding provider used: {google_embedder.kind}")
    except Exception as exc:
        print(f"Chroma peek validation failed: {exc}")

    print("Ingestion pipeline completed successfully.")


def main() -> None:
    run_ingestion_pipeline()


if __name__ == "__main__":
    main()