# app/retriever.py
import os
from typing import List
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

EMBED_MODEL = os.getenv("CHROMA_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "hr_policies")


class Retriever:
    def __init__(self):
        # Persist to a stable path regardless of CWD
        persist_directory = os.getenv(
            "CHROMA_PERSIST_DIR",
            os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "chroma_db")),
        )

        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self.vstore = Chroma(
            collection_name=CHROMA_COLLECTION,
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
        )
        self.retriever = self.vstore.as_retriever(
             search_type="similarity",
            search_kwargs={"k": 4}
        )


    def index_documents(self, docs: List[Document]):
        if not docs:
            return

        ids = [str(uuid4()) for _ in range(len(docs))]

        self.vstore.add_documents(documents=docs, ids=ids)


    def query(self, query_text: str, n_results: int = 3) -> list: 
        if not query_text.strip():
            return []

        # Prefer LangChain-native relevance scores (0..1 when available)
        try:
            results = self.vstore.similarity_search_with_relevance_scores(
                query_text, k=n_results
            )
            return [(doc.page_content, float(score)) for doc, score in results]
        except Exception:
            # Fallback to distance scores; convert to a bounded similarity proxy
            results = self.vstore.similarity_search_with_score(query_text, k=n_results)
            return [(doc.page_content, 1.0 / (1.0 + float(dist))) for doc, dist in results]