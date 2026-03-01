# Build RAG Index from Medical Documents

import os
import re
import pickle
from tqdm import tqdm
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

from config import DOCS_DIR, RAG_INDEX, RAG_CHUNKS, EMBEDDING_MODEL, CHUNK_SIZE

def clean_text(text):
    """Clean extracted PDF text"""
    text = re.sub(r'\[[0-9, ]+\]', '', text)  # Remove citations
    text = re.sub(r'Pathogens\s+\d{4}.*?\d+', '', text)  # Remove headers
    text = re.sub(r'Figure\s+\d+\.?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+\s+of\s+\d+\b', '', text)  # Remove page numbers
    text = re.sub(r'\n\s*\d+\.\s+[A-Z].*', '', text)  # Remove references
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF"""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text

def chunk_text(text, chunk_size=400):
    """Split text into chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) > 100:  # Skip tiny chunks
            chunks.append(chunk)
    return chunks

def build_rag_index():
    """Build RAG index from PDF documents"""
    
    all_chunks = []
    
    print("📄 Extracting and cleaning PDFs...\n")
    
    for file in os.listdir(DOCS_DIR):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(DOCS_DIR, file)
            print(f"Processing: {file}")
            
            raw_text = extract_text_from_pdf(pdf_path)
            cleaned = clean_text(raw_text)
            chunks = chunk_text(cleaned, CHUNK_SIZE)
            
            print(f" → {len(chunks)} chunks created")
            all_chunks.extend(chunks)
    
    print(f"\n✅ Total chunks: {len(all_chunks)}")
    
    print("\n🔎 Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("🔢 Generating embeddings...")
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    
    print("📦 Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    print("💾 Saving index and chunks...")
    faiss.write_index(index, str(RAG_INDEX))
    
    with open(RAG_CHUNKS, "wb") as f:
        pickle.dump(all_chunks, f)
    
    print("\n🎉 RAG index built successfully!")
    print(f"Index: {RAG_INDEX}")
    print(f"Chunks: {RAG_CHUNKS}")

if __name__ == "__main__":
    build_rag_index()
