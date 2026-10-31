import re
import zlib

import pytest

from src.knowledge.vector_store import KnowledgeStore


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic bag-of-words embedder — texts sharing words land close together."""
    vectors = []
    for text in texts:
        vec = [0.0] * 64
        for word in re.findall(r"\w+", text.lower()):
            vec[zlib.crc32(word.encode()) % 64] += 1.0
        vectors.append(vec if any(vec) else [1.0] + [0.0] * 63)
    return vectors


@pytest.fixture
def store(tmp_path):
    return KnowledgeStore(persist_dir=str(tmp_path / "chroma"), embedding_fn=fake_embed)
