# Qdrant-based RAG System with Mistral Embeddings

import os
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from mistralai import Mistral
import numpy as np
from tqdm import tqdm
import json

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
QDRANT_PATH = BASE_DIR / "qdrant_db"
CHUNKS_PATH = BASE_DIR / "rag_chunks.json"

# Mistral config - load .env first
_env_path = BASE_DIR / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
                if _k and _k not in os.environ:
                    os.environ[_k] = _v

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
EMBEDDING_MODEL = "mistral-embed"
COLLECTION_NAME = "tb_medical_knowledge"

class QdrantRAG:
    """RAG system using Qdrant and Mistral"""
    
    def __init__(self):
        self.client = QdrantClient(path=str(QDRANT_PATH))
        self.mistral = Mistral(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None
        
        if not self.mistral:
            print("⚠️  MISTRAL_API_KEY not set. RAG will not work.")
    
    def embed_text(self, texts):
        """Generate embeddings using Mistral"""
        if not self.mistral:
            raise ValueError("Mistral API key not configured")
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings_batch_response = self.mistral.embeddings.create(
            model=EMBEDDING_MODEL,
            inputs=texts
        )
        
        embeddings = [item.embedding for item in embeddings_batch_response.data]
        return embeddings
    
    def create_collection(self, vector_size=1024):
        """Create Qdrant collection"""
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except:
            pass
        
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"✅ Collection '{COLLECTION_NAME}' created")
    
    def index_documents(self, chunks, metadata_list):
        """Index document chunks into Qdrant"""
        print("🔢 Generating embeddings...")
        
        # Batch embedding
        batch_size = 32
        all_embeddings = []
        
        for i in tqdm(range(0, len(chunks), batch_size)):
            batch = chunks[i:i+batch_size]
            embeddings = self.embed_text(batch)
            all_embeddings.extend(embeddings)
        
        # Get vector size from first embedding
        vector_size = len(all_embeddings[0])
        self.create_collection(vector_size)
        
        # Create points
        print("📦 Indexing into Qdrant...")
        points = []
        for idx, (chunk, embedding, metadata) in enumerate(zip(chunks, all_embeddings, metadata_list)):
            point = PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    "text": chunk,
                    "source": metadata.get("source", "unknown"),
                    "page": metadata.get("page", 0),
                    "chunk_id": idx
                }
            )
            points.append(point)
        
        # Upload in batches
        batch_size = 100
        for i in tqdm(range(0, len(points), batch_size)):
            batch = points[i:i+batch_size]
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch
            )
        
        print(f"✅ Indexed {len(points)} chunks")
    
    def query(self, query_text, top_k=5, filter_source=None):
        """Query the RAG system"""
        if not self.mistral:
            raise ValueError("Mistral API key not configured")
        
        # Generate query embedding
        query_embedding = self.embed_text(query_text)[0]
        
        # Search - handle both old and new qdrant_client API
        try:
            # New API (qdrant_client >= 1.7)
            results = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_embedding,
                limit=top_k,
            ).points
        except (AttributeError, TypeError):
            # Old API fallback
            results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding,
                limit=top_k,
            )
        
        # Format results
        retrieved = []
        for result in results:
            retrieved.append({
                "text": result.payload["text"],
                "source": result.payload["source"],
                "page": result.payload.get("page", 0),
                "score": result.score
            })
        
        return retrieved

def build_rag_index():
    """Build RAG index from medical documents"""
    from pypdf import PdfReader
    import re
    
    print("="*60)
    print("Building Qdrant RAG Index")
    print("="*60)
    
    if not MISTRAL_API_KEY:
        print("❌ MISTRAL_API_KEY not set in environment")
        return
    
    # Extract text from PDFs
    all_chunks = []
    all_metadata = []
    
    print("\n📄 Processing PDFs...")
    for pdf_file in DOCS_DIR.glob("*.pdf"):
        print(f"\nProcessing: {pdf_file.name}")
        
        reader = PdfReader(pdf_file)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            
            if not text:
                continue
            
            # Clean text
            text = re.sub(r'\[[0-9, ]+\]', '', text)
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            # Chunk text
            words = text.split()
            chunk_size = 400
            
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i+chunk_size])
                if len(chunk.split()) > 50:
                    all_chunks.append(chunk)
                    all_metadata.append({
                        "source": pdf_file.name,
                        "page": page_num + 1
                    })
        
        print(f"  Extracted {len([m for m in all_metadata if m['source'] == pdf_file.name])} chunks")
    
    # Also process .txt files
    print("\n📄 Processing text files...")
    for txt_file in DOCS_DIR.glob("*.txt"):
        print(f"\nProcessing: {txt_file.name}")
        
        text = txt_file.read_text(encoding='utf-8', errors='ignore')
        
        # Split into sections by separator lines
        sections = re.split(r'={5,}', text)
        
        for section_num, section in enumerate(sections):
            section = section.strip()
            if not section or len(section.split()) < 30:
                continue
            
            # Clean text
            section = re.sub(r'\s+', ' ', section)
            
            # Chunk text
            words = section.split()
            chunk_size = 400
            overlap = 50
            
            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i:i+chunk_size])
                if len(chunk.split()) > 50:
                    all_chunks.append(chunk)
                    all_metadata.append({
                        "source": txt_file.name,
                        "page": section_num + 1
                    })
        
        print(f"  Extracted {len([m for m in all_metadata if m['source'] == txt_file.name])} chunks")
    
    print(f"\n✅ Total chunks: {len(all_chunks)}")
    
    # Save chunks
    with open(CHUNKS_PATH, 'w') as f:
        json.dump({
            "chunks": all_chunks,
            "metadata": all_metadata
        }, f)
    
    # Index into Qdrant
    rag = QdrantRAG()
    rag.index_documents(all_chunks, all_metadata)
    
    print("\n✅ RAG index built successfully!")
    print(f"📁 Qdrant DB: {QDRANT_PATH}")
    print(f"📄 Chunks: {CHUNKS_PATH}")

def query_rag(query_text, top_k=5):
    """Query the RAG system"""
    rag = QdrantRAG()
    results = rag.query(query_text, top_k=top_k)
    
    print(f"\n🔎 Query: {query_text}")
    print("="*60)
    
    for i, result in enumerate(results, 1):
        print(f"\n📄 Result {i} (Score: {result['score']:.3f})")
        print(f"Source: {result['source']} (Page {result['page']})")
        print(f"Text: {result['text'][:300]}...")
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python qdrant_rag.py build          # Build RAG index")
        print("  python qdrant_rag.py query <text>   # Query RAG")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "build":
        build_rag_index()
    elif command == "query":
        if len(sys.argv) < 3:
            print("Please provide query text")
            sys.exit(1)
        query_text = " ".join(sys.argv[2:])
        query_rag(query_text)
    else:
        print(f"Unknown command: {command}")
