import pytest

from src.agent.state import new_ticket_state
from src.tools import kb_retriever
from src.tools.kb_retriever import retrieve_from_kb


@pytest.fixture
def kb_store(store, monkeypatch):
    store.add_documents([
        {"id": "refund", "text": "Refunds for a duplicate charge are processed within 5 business days",
         "metadata": {"source": "billing_faq.md", "category": "billing"}},
        {"id": "password", "text": "Reset your password from the login page",
         "metadata": {"source": "account_faq.md", "category": "account"}},
        {"id": "darkmode", "text": "Feature requests are reviewed by the product team monthly",
         "metadata": {"source": "product_faq.md", "category": "product"}},
        {"id": "outage", "text": "Check the status page during an outage",
         "metadata": {"source": "tech_faq.md", "category": "tech"}},
    ])
    monkeypatch.setattr(kb_retriever, "_store", store)
    return store


def ticket(text, intent="billing", trace=None):
    state = new_ticket_state(text)
    state["intent"] = intent
    state["reasoning_trace"] = trace or []
    return state


def test_returns_top_k_docs_best_first(kb_store):
    out = retrieve_from_kb(ticket("I got a duplicate charge, when is my refund processed?"))

    assert len(out["retrieved_docs"]) == kb_retriever.TOP_K
    assert out["retrieved_docs"][0].startswith("Refunds")
    assert out["retrieval_scores"] == sorted(out["retrieval_scores"], reverse=True)


def test_docs_and_scores_line_up(kb_store):
    out = retrieve_from_kb(ticket("reset password"))

    assert len(out["retrieved_docs"]) == len(out["retrieval_scores"])


def test_appends_to_reasoning_trace(kb_store):
    out = retrieve_from_kb(ticket("refund", trace=["Classified intent: billing (0.90)"]))

    assert out["reasoning_trace"] == [
        "Classified intent: billing (0.90)",
        f"Retrieved {kb_retriever.TOP_K} docs from KB",
    ]


def test_store_failure_returns_empty_and_logs_reason(monkeypatch):
    def broken_store():
        raise RuntimeError("chroma unavailable")

    monkeypatch.setattr(kb_retriever, "get_store", broken_store)

    out = retrieve_from_kb(ticket("refund"))

    assert out["retrieved_docs"] == [] and out["retrieval_scores"] == []
    assert "chroma unavailable" in out["reasoning_trace"][-1]


def test_get_store_indexes_kb_on_first_use(tmp_path, monkeypatch):
    from tests.conftest import fake_embed
    from src.knowledge.vector_store import KnowledgeStore

    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "billing_faq.md").write_text("## Refunds\nRefunds take 5 days.")

    monkeypatch.setattr(kb_retriever, "_store", None)
    monkeypatch.setattr(kb_retriever, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(kb_retriever, "KNOWLEDGE_BASE_DIR", str(kb))
    monkeypatch.setattr(
        kb_retriever, "KnowledgeStore", lambda d: KnowledgeStore(d, embedding_fn=fake_embed)
    )

    store = kb_retriever.get_store()

    assert store.collection.count() == 1
    assert kb_retriever.get_store() is store  # cached for the process


def test_get_store_is_built_once_under_concurrent_first_calls(monkeypatch):
    import threading
    import time

    built = []

    class SlowStore:
        def __init__(self, persist_dir):
            time.sleep(0.05)  # widen the race window
            built.append(self)
            self.collection = type("C", (), {"count": staticmethod(lambda: 1)})()

    monkeypatch.setattr(kb_retriever, "_store", None)
    monkeypatch.setattr(kb_retriever, "KnowledgeStore", SlowStore)

    results = []
    threads = [threading.Thread(target=lambda: results.append(kb_retriever.get_store()))
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(built) == 1
    assert all(store is built[0] for store in results)
