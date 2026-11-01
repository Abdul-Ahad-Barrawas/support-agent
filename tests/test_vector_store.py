import pytest

from src.knowledge.vector_store import KnowledgeStore, chunk_text
from tests.conftest import fake_embed


# --- chunk_text ---

def test_chunk_short_text_is_single_chunk():
    assert chunk_text("Refunds take 3-5 business days.") == ["Refunds take 3-5 business days."]


def test_chunk_empty_text_returns_nothing():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_chunks_respect_size_limit_and_are_non_empty():
    text = "word " * 400  # 2000 chars
    chunks = chunk_text(text, chunk_size=500, overlap=50)

    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 500 for chunk in chunks)


def test_consecutive_chunks_overlap():
    text = "".join(f"{i:04d} " for i in range(300))  # unique 5-char tokens, 1500 chars
    chunks = chunk_text(text, chunk_size=500, overlap=50)

    for prev, nxt in zip(chunks, chunks[1:]):
        # Some text from the end of one chunk should reappear at the start of the next
        tail = prev[-40:].strip().split()[-1]
        assert tail in nxt[:100]


def test_chunks_cover_entire_document():
    text = "".join(f"{i:04d} " for i in range(300))
    joined = " ".join(chunk_text(text, chunk_size=500, overlap=50))

    for i in range(300):
        assert f"{i:04d}" in joined


def test_markdown_split_on_headings():
    text = "# Billing FAQ\n\n## Refunds\nRefunds take 5 days.\n\n## Payments\nWe accept Visa."
    chunks = chunk_text(text)

    assert len(chunks) == 2
    assert chunks[0].startswith("# Billing FAQ") and "Refunds take 5 days." in chunks[0]
    assert chunks[1] == "## Payments\nWe accept Visa."


# --- KnowledgeStore ---

def test_search_empty_store_returns_nothing(store):
    assert store.search("refund") == []


def test_search_ranks_relevant_doc_first(store):
    store.add_documents([
        {"id": "refund", "text": "Refunds are processed within 5 business days",
         "metadata": {"source": "billing_faq.md", "category": "billing"}},
        {"id": "password", "text": "Reset your password from the login page",
         "metadata": {"source": "account_faq.md", "category": "account"}},
    ])

    results = store.search("how long do refunds take to be processed", top_k=2)

    assert results[0]["text"].startswith("Refunds")
    assert results[0]["metadata"]["category"] == "billing"
    assert results[0]["score"] > results[1]["score"]


def test_search_scores_are_similarities(store):
    store.add_documents([{"id": "a", "text": "duplicate charge refund", "metadata": {"source": "x"}}])

    [result] = store.search("duplicate charge refund")

    # Identical text → cosine distance 0 → similarity 1
    assert result["score"] == pytest.approx(1.0, abs=1e-4)


def test_search_top_k_capped_by_collection_size(store):
    store.add_documents([{"id": "only", "text": "the only doc", "metadata": {"source": "x"}}])

    assert len(store.search("doc", top_k=5)) == 1


def test_add_documents_upserts_by_id(store):
    store.add_documents([{"id": "doc", "text": "old text", "metadata": {"source": "x"}}])
    store.add_documents([{"id": "doc", "text": "new text", "metadata": {"source": "x"}}])

    assert store.collection.count() == 1
    assert store.search("text")[0]["text"] == "new text"


def test_load_from_directory_indexes_md_and_txt_only(store, tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "billing_faq.md").write_text("Refunds are processed within 5 business days.")
    (kb / "account_notes.txt").write_text("Reset your password from the login page.")
    (kb / "ignore.csv").write_text("should,not,be,indexed")

    count = store.load_from_directory(str(kb))

    assert count == 2
    sources = {r["metadata"]["source"] for r in store.search("anything", top_k=10)}
    assert sources == {"billing_faq.md", "account_notes.txt"}
    assert store.search("refunds processed")[0]["metadata"]["category"] == "billing"


def test_load_from_directory_is_idempotent(store, tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "billing_faq.md").write_text("Refunds are processed within 5 business days.")

    store.load_from_directory(str(kb))
    store.load_from_directory(str(kb))

    assert store.collection.count() == 1


def test_store_persists_across_instances(tmp_path):
    persist_dir = str(tmp_path / "chroma")
    KnowledgeStore(persist_dir, embedding_fn=fake_embed).add_documents(
        [{"id": "a", "text": "persisted refund policy", "metadata": {"source": "x"}}]
    )

    reopened = KnowledgeStore(persist_dir, embedding_fn=fake_embed)

    assert reopened.search("refund policy")[0]["text"] == "persisted refund policy"


def test_reset_clears_collection(store):
    store.add_documents([{"id": "a", "text": "stale chunk", "metadata": {"source": "x"}}])

    store.reset()

    assert store.collection.count() == 0
    assert store.search("stale") == []
