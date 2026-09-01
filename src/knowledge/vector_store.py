"""
Vector store wrapper around ChromaDB.

Owner: Strong technical member (setup)
        + Non-technical member (loading KB documents)
"""

# TODO: Implement these functions
#
# class KnowledgeStore:
#     def __init__(self, persist_dir, embedding_model="all-MiniLM-L6-v2"):
#         - Initialize ChromaDB client with persist_dir
#         - Load sentence-transformer embedding model
#         - Create or get collection "support_kb"
#
#     def add_documents(self, docs: list[dict]):
#         - Each doc has: {"id", "text", "metadata": {"source", "category"}}
#         - Embed text using sentence-transformer
#         - Upsert into ChromaDB collection
#
#     def search(self, query: str, top_k: int = 3) -> list[dict]:
#         - Embed query
#         - Query collection
#         - Return [{"text": ..., "score": ..., "metadata": ...}]
#
#     def load_from_directory(self, kb_dir: str):
#         - Read all .txt/.md files from kb_dir
#         - Chunk them (500 chars, 50 char overlap)
#         - Call add_documents
