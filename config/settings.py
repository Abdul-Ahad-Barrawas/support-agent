import os
from dotenv import load_dotenv

load_dotenv()

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Agent
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))

# Paths
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
KNOWLEDGE_BASE_DIR = "./data/knowledge_base"
GOLDEN_SET_PATH = "./data/golden_set/golden_qa.csv"

# Intent categories
INTENT_LABELS = [
    "billing",
    "technical_issue",
    "account_access",
    "feature_request",
    "complaint",
    "general_inquiry",
]

# Priority levels
PRIORITY_LEVELS = ["low", "medium", "high", "critical"]
