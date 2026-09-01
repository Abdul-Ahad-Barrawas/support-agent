"""
Tool 3: Knowledge Base Retriever
Searches the vector store for relevant FAQ and policy documents.

Owner: Strong technical member (retrieval logic)
        + Non-technical member (KB content curation)
"""

from src.agent.state import TicketState


def retrieve_from_kb(state: TicketState) -> dict:
    """
    Retrieve relevant documents from ChromaDB based on ticket + intent.

    TODO: Implement this tool
    - Build query from ticket_text + intent
    - Query ChromaDB collection
    - Return top-k documents with similarity scores
    - k=3 is a good default, tune during ablation

    Uses: src/knowledge/vector_store.py for the ChromaDB wrapper
    """
    ticket_text = state["ticket_text"]
    intent = state["intent"]

    # ---- PLACEHOLDER — replace with vector search ----
    retrieved_docs = []
    retrieval_scores = []
    # ---------------------------------------------------

    return {
        "retrieved_docs": retrieved_docs,
        "retrieval_scores": retrieval_scores,
        "reasoning_trace": state.get("reasoning_trace", [])
        + [f"Retrieved {len(retrieved_docs)} docs from KB"],
    }
