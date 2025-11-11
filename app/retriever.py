# app/retriever.py
from sentence_transformers import SentenceTransformer
import chromadb
import os
from typing import List


EMBED_MODEL = os.getenv("CHROMA_EMBEDDING_MODEL", "all-MiniLM-L6-v2")


class Retriever:
    def __init__(self):
# Embedded chroma instance (in-process). For production, use server or managed instance inside VPC.
        self.client = chromadb.PersistentClient(path="./chroma_db")
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
        if self.collection:
            self.collection.add(ids=ids, documents=texts, embeddings=embeddings)


    def query(self, query_text: str, n_results: int = 3):
        q_emb = self.embedder.encode([query_text])[0]
        if not self.collection:
            return []
        results = self.collection.query(query_embeddings=[q_emb], n_results=n_results, include=['documents','distances'])
        if not results:
            return []
# results format: dict with 'documents' and 'distances'
        docs_list = results.get('documents')
        distances_list = results.get('distances')
        if not docs_list or not distances_list or len(docs_list) == 0 or len(distances_list) == 0:
            return []
        docs = docs_list[0]
        distances = distances_list[0]
# chroma uses distance; higher means less similar depending on implementation; convert to similarity
        similarities = [1 - d for d in distances]
        return list(zip(docs, similarities))