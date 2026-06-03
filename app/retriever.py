# app/retriever.py
import os
from typing import List
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

EMBED_MODEL = os.getenv("CHROMA_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "hr_policies")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def _build_embeddings() -> Embeddings:
    """Embeddings for Chroma. Use EMBEDDING_PROVIDER=ollama to run on Ollama (AMD GPU via Ollama)."""
    if EMBEDDING_PROVIDER == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError as e:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=ollama requires langchain-ollama. "
                "Install it and run: ollama pull " + OLLAMA_EMBED_MODEL
            ) from e
        return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_URL)

    from langchain_huggingface import HuggingFaceEmbeddings

    model_kwargs = {}
    encode_kwargs = {"normalize_embeddings": True}
    device_mode = os.getenv("EMBED_DEVICE", "cpu").lower()
    if device_mode not in ("cpu", "auto"):
        if device_mode in ("dml", "directml", "gpu"):
            try:
                import torch_directml  # type: ignore[import-not-found]

                model_kwargs["device"] = torch_directml.device()
            except ImportError:
                print("torch-directml not installed; using CPU for embeddings.")
        elif device_mode == "cuda":
            import torch

            if torch.cuda.is_available():
                model_kwargs["device"] = "cuda"

    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )


class Retriever:
    def __init__(self):
        persist_directory = os.getenv(
            "CHROMA_PERSIST_DIR",
            os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "chroma_db")),
        )

        self.embeddings = _build_embeddings()
        self.vstore = Chroma(
            collection_name=CHROMA_COLLECTION,
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
        )
        self.retriever = self.vstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

        if EMBEDDING_PROVIDER != "ollama":
            try:
                self.embeddings.embed_query("warmup")
            except Exception as e:
                print(f"Embedding warmup skipped: {e}")

    def index_documents(self, docs: List[Document]):
        if not docs:
            return

        ids = [str(uuid4()) for _ in range(len(docs))]
        self.vstore.add_documents(documents=docs, ids=ids)

    def query(self, query_text: str, n_results: int = 3) -> list:
        if not query_text.strip():
            return []

        try:
            results = self.vstore.similarity_search_with_relevance_scores(
                query_text, k=n_results
            )
            return [(doc.page_content, float(score)) for doc, score in results]
        except Exception:
            results = self.vstore.similarity_search_with_score(query_text, k=n_results)
            return [(doc.page_content, 1.0 / (1.0 + float(dist))) for doc, dist in results]
