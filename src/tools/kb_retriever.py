"""
Tool 3: Knowledge Base Retriever
Searches the vector store for relevant FAQ and policy documents.

Owner: Strong technical member (retrieval logic)
        + Non-technical member (KB content curation)
"""

import threading

from config.settings import CHROMA_PERSIST_DIR, KNOWLEDGE_BASE_DIR
from src.agent.state import TicketState
from src.knowledge.vector_store import KnowledgeStore

TOP_K = 3

_store = None
_store_lock = threading.Lock()


def get_store() -> KnowledgeStore:
    """Open the store once per process, indexing the KB on first ever run."""
    global _store
    # The evaluator runs tickets in threads — without the lock, the first few
    # would each load the embedding model and race to index the KB
    with _store_lock:
        if _store is None:
            store = KnowledgeStore(CHROMA_PERSIST_DIR)
            if store.collection.count() == 0:
                store.load_from_directory(KNOWLEDGE_BASE_DIR)
            _store = store
    return _store


def retrieve_from_kb(state: TicketState) -> dict:
    """Retrieve the top-k KB chunks most similar to the ticket + its intent."""
    ticket_text = state["ticket_text"]
    intent = state["intent"]

    # The intent label nudges the embedding toward the right KB section
    query = f"{intent.replace('_', ' ')}: {ticket_text}" if intent else ticket_text

    try:
        results = get_store().search(query, top_k=TOP_K)
        trace = f"Retrieved {len(results)} docs from KB"
    except Exception as e:
        # Drafting still runs; with no context its confidence drops → flag_for_review
        results = []
        trace = f"KB retrieval failed ({type(e).__name__}: {e}) — drafting without context"

    return {
        "retrieved_docs": [r["text"] for r in results],
        "retrieval_scores": [r["score"] for r in results],
        "reasoning_trace": state.get("reasoning_trace", []) + [trace],
    }
