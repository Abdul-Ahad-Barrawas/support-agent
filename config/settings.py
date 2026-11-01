import os
from dotenv import load_dotenv

# Real environment variables (e.g. exported in your shell) win over .env
load_dotenv(override=False)

# LLM — API keys are deliberately NOT read here; src/llm.py reads them from
# os.environ at the moment a client is built, so they never sit in settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")  # used by groq / openai
# Total seconds per LLM request attempt (enforced by src/llm.DeadlineTransport),
# so one stalled upstream provider can't freeze an evaluation run
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

# Agent
COMPANY_NAME = os.getenv("COMPANY_NAME", "Acme Cloud")
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
