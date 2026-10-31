import os
import re
import zlib

# The tools build their LLM client at import time, which needs a key present.
# Tests never call the API — fill in fakes only where no real key is set.
for _name in ("OPENROUTER_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"):
    os.environ.setdefault(_name, "test-key-not-real")

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
