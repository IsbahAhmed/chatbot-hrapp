# app/retriever.py
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os
from typing import List


EMBED_MODEL = os.getenv("CHROMA_EMBEDDING_MODEL", "all-MiniLM-L6-v2")


class Retriever:
    def __init__(self):
# Embedded chroma instance (in-process). For production, use server or managed instance inside VPC.
        self.client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))
        self.collection = None
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self._ensure_collection()


    def _ensure_collection(self):
        try:
            self.collection = self.client.get_collection(name="hr_policies")
        except Exception:
            self.collection = self.client.create_collection(name="hr_policies")


    def index_documents(self, docs: List[dict]):
# docs: list of {"id": str, "text": str}
        texts = [d["text"] for d in docs]
        ids = [d["id"] for d in docs]
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        self.collection.add(ids=ids, documents=texts, embeddings=embeddings)
        self.client.persist()


    def query(self, query_text: str, n_results: int = 3):
        q_emb = self.embedder.encode([query_text])[0]
        results = self.collection.query(query_embeddings=[q_emb], n_results=n_results, include=['documents','distances'])
# results format: dict with 'documents' and 'distances'
        docs = results['documents'][0]
        distances = results['distances'][0]
# chroma uses distance; higher means less similar depending on implementation; convert to similarity
        similarities = [1 - d for d in distances]
        return list(zip(docs, similarities))