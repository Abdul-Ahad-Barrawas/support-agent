"""
Vector store wrapper around ChromaDB.

Owner: Strong technical member (setup)
        + Non-technical member (loading KB documents)
"""

import re
from pathlib import Path
from typing import Callable, Optional

import chromadb

COLLECTION_NAME = "support_kb"
KB_EXTENSIONS = {".md", ".txt"}

# Takes a batch of texts, returns one embedding vector per text
EmbeddingFn = Callable[[list[str]], list[list[float]]]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split a document into overlapping chunks for embedding.

    Markdown docs are split on "## " headings first, so each FAQ entry stays
    one coherent chunk. Sections longer than chunk_size are then cut into
    windows of chunk_size chars that overlap by `overlap` chars.
    """
    sections = [s.strip() for s in re.split(r"\n(?=## )", text) if s.strip()]

    # A doc title / intro before the first "## " joins the first section
    # instead of becoming a near-empty chunk of its own
    if len(sections) > 1 and not sections[0].startswith("## "):
        sections[1] = f"{sections[0]}\n\n{sections[1]}"
        sections.pop(0)

    chunks = []
    for section in sections:
        start = 0
        while True:
            chunks.append(section[start:start + chunk_size].strip())
            if start + chunk_size >= len(section):
                break
            start += chunk_size - overlap

    return [chunk for chunk in chunks if chunk]


def _sentence_transformer_embedder(model_name: str) -> EmbeddingFn:
    # Imported lazily so tests with a fake embedder never load torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return lambda texts: model.encode(texts, normalize_embeddings=True).tolist()


class KnowledgeStore:
    def __init__(
        self,
        persist_dir: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_fn: Optional[EmbeddingFn] = None,
    ):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embed = embedding_fn or _sentence_transformer_embedder(embedding_model)
        self.collection = self._get_collection()

    def _get_collection(self):
        # Cosine space so that score = 1 - distance is a similarity in [0, 1]
        return self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Drop every indexed chunk — use before re-indexing an edited KB."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self._get_collection()

    def add_documents(self, docs: list[dict]) -> None:
        """Embed and upsert docs of the form {"id", "text", "metadata"}."""
        if not docs:
            return

        texts = [doc["text"] for doc in docs]
        self.collection.upsert(
            ids=[doc["id"] for doc in docs],
            documents=texts,
            embeddings=self.embed(texts),
            metadatas=[doc.get("metadata") or None for doc in docs],
        )

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Return the top_k most similar chunks, best match first."""
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=self.embed([query]),
            n_results=min(top_k, self.collection.count()),
        )

        return [
            {"text": text, "score": 1.0 - distance, "metadata": metadata or {}}
            for text, distance, metadata in zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0],
            )
        ]

    def load_from_directory(self, kb_dir: str) -> int:
        """Chunk and index every .md/.txt file in kb_dir. Returns chunk count."""
        docs = []
        for path in sorted(Path(kb_dir).iterdir()):
            if path.suffix not in KB_EXTENSIONS:
                continue

            # billing_faq.md → category "billing"
            category = path.stem.split("_")[0]
            for i, chunk in enumerate(chunk_text(path.read_text(encoding="utf-8"))):
                docs.append({
                    "id": f"{path.name}::{i}",
                    "text": chunk,
                    "metadata": {"source": path.name, "category": category},
                })

        self.add_documents(docs)
        return len(docs)


if __name__ == "__main__":
    # Rebuild the index after editing data/knowledge_base/:
    #   python -m src.knowledge.vector_store
    from config.settings import CHROMA_PERSIST_DIR, KNOWLEDGE_BASE_DIR

    store = KnowledgeStore(CHROMA_PERSIST_DIR)
    store.reset()
    print(f"Indexed {store.load_from_directory(KNOWLEDGE_BASE_DIR)} chunks from {KNOWLEDGE_BASE_DIR}")
