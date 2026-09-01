# Support Ticket Triage & Response Agent

A multi-step AI agent that automatically classifies, prioritizes, and responds to customer support tickets using LangGraph orchestration, RAG-based knowledge retrieval, and LLM-powered response generation.

## Architecture

```
Incoming Ticket
      │
      ▼
┌─────────────────┐
│ Intent Classifier│ ← Tool 1: Categorize (billing, technical, etc.)
└────────┬────────┘
         ▼
┌─────────────────┐
│ Priority Scorer  │ ← Tool 2: Sentiment + keyword urgency
└────────┬────────┘
         ▼
   ┌──────────┐     YES
   │ Critical?├──────────► Escalate to Human
   └─────┬────┘
      NO ▼
┌─────────────────┐
│  KB Retriever   │ ← Tool 3: ChromaDB vector search
└────────┬────────┘
         ▼
┌─────────────────┐
│ Response Drafter │ ← Tool 4: LLM generates grounded reply
└────────┬────────┘
         ▼
   ┌───────────┐    NO
   │ Confident?├──────────► Flag for Human Review
   └─────┬─────┘
      YES ▼
  Send Drafted Response
```

## Tech Stack

| Layer             | Tool                        | Why                                      |
|-------------------|-----------------------------|------------------------------------------|
| Agent framework   | LangGraph                   | State machine with conditional routing    |
| LLM orchestration | LangChain                   | Prompts, tool definitions, output parsing |
| LLM provider      | Groq (free) / OpenAI (paid) | Groq gives free Llama 3.1 70B inference   |
| Vector store      | ChromaDB                    | Local, no infra, persistent               |
| Embeddings        | sentence-transformers       | Free local embeddings (MiniLM)            |
| UI                | Streamlit                   | Fast prototyping, easy for non-coders     |
| Evaluation        | scikit-learn + custom       | Precision, recall, F1 + ablation table    |
| Language          | Python 3.10+                | Only language needed                      |

## Project Structure

```
support-agent/
├── config/
│   └── settings.py              # All config in one place
├── data/
│   ├── raw/                     # Raw ticket datasets (Bitext, Kaggle)
│   ├── knowledge_base/          # FAQ and policy markdown files
│   └── golden_set/              # Ground-truth evaluation set (CSV)
├── src/
│   ├── agent/
│   │   ├── graph.py             # LangGraph pipeline definition
│   │   └── state.py             # TicketState TypedDict
│   ├── tools/
│   │   ├── intent_classifier.py # Tool 1
│   │   ├── priority_scorer.py   # Tool 2
│   │   ├── kb_retriever.py      # Tool 3
│   │   └── response_drafter.py  # Tool 4
│   ├── knowledge/
│   │   └── vector_store.py      # ChromaDB wrapper
│   └── evaluation/
│       └── evaluator.py         # Golden set eval + ablation runner
├── app/
│   └── streamlit_app.py         # Streamlit frontend
├── notebooks/
│   └── exploration.ipynb        # Data exploration
├── tests/
│   └── (unit tests)
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
# 1. Clone and enter
git clone <repo-url>
cd support-agent

# 2. Create virtual env
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env → add your Groq API key (free at console.groq.com)

# 5. Run the app
streamlit run app/streamlit_app.py
```

## Team Assignments

### Strong Technical Members (2-3 people)
- **Agent pipeline**: `src/agent/graph.py` — LangGraph state machine
- **Tool implementations**: All 4 files in `src/tools/`
- **Vector store**: `src/knowledge/vector_store.py` — ChromaDB + embeddings
- **Evaluation code**: `src/evaluation/evaluator.py` — metrics + ablation runner

### Other Members (2 people)
- **Knowledge base curation**: Write FAQ/policy docs in `data/knowledge_base/`
- **Golden test set**: Build 50-100 rows in `data/golden_set/golden_qa.csv`
- **Streamlit UI**: `app/streamlit_app.py` — layout and display logic
- **Prompt ablation execution**: Run experiments, record results, build comparison table
- **Documentation**: README, demo slides, final report

## Evaluation Strategy (CV Differentiator)

### Golden Test Set
50-100 support tickets with ground-truth labels for intent, priority, action, and reference response. This is what separates a real project from a toy demo.

### Ablation Table
Run 4 agent variants against the golden set and compare:

| Variant          | Intent Acc | Priority Acc | Action Acc | Response Quality |
|------------------|-----------|-------------|-----------|-----------------|
| Full agent       | —         | —           | —         | —               |
| No KB retrieval  | —         | —           | —         | —               |
| No priority gate | —         | —           | —         | —               |
| Single-shot LLM  | —         | —           | —         | —               |

This table proves each component adds value — that's what hiring managers look for.

## Data Sources
- **Ticket datasets**: Bitext customer support dataset, Kaggle support ticket classification
- **KB content**: Write your own (simulated company FAQ/policies)
