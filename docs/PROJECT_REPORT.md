# Support Ticket Triage & Response Agent — Project Report

| | |
|---|---|
| **Project** | Multi-step AI agent that classifies, prioritises and drafts replies to customer support tickets |
| **Repository** | `github.com/Abdul-Ahad-Barrawas/support-agent` (branch `feature/rag-drafter-evaluation`) |
| **Language / runtime** | Python 3.14 (3.10+ supported) |
| **Core stack** | LangGraph · LangChain · ChromaDB · sentence-transformers · Streamlit · OpenRouter (`openai/gpt-oss-120b`) |
| **Codebase size** | ~1,300 lines of application code · 146 automated tests + 10 live end-to-end tests |
| **Status** | Both work streams committed and merged (`ac70000`); results confirmed by a post-merge ablation and live-test run (§8.4, §8.5) |
| **Report date** | November 2026 |

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [System Architecture](#3-system-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Implementation](#5-implementation)
6. [Data](#6-data)
7. [Evaluation Methodology](#7-evaluation-methodology)
8. [Results](#8-results)
9. [Testing & Quality Assurance](#9-testing--quality-assurance)
10. [Engineering Challenges & Solutions](#10-engineering-challenges--solutions)
11. [Security & Privacy](#11-security--privacy)
12. [Limitations & Future Work](#12-limitations--future-work)
13. [Running the Project](#13-running-the-project)
14. [Project Structure](#14-project-structure)
15. [Team & Process](#15-team--process)
16. [Conclusion](#16-conclusion)
17. [References](#17-references)
18. [Appendices](#appendices)

---

## 1. Abstract

Customer support teams spend much of their time on repetitive triage: working out what a ticket is about, how urgent it is, and which policy answers it. This project builds an **AI agent that automates that triage end to end**. Each incoming ticket passes through four tools: an **intent classifier**, a **priority scorer**, a **knowledge-base retriever** (retrieval-augmented generation over company FAQ/policy documents) and a **response drafter**. Two **decision gates** route the ticket to one of three outcomes: *send* the drafted reply, *escalate* to a human (critical tickets), or *flag for review* (low-confidence drafts).

The pipeline is orchestrated as a **LangGraph state machine**, uses **ChromaDB** with local **MiniLM** embeddings for retrieval, and calls `openai/gpt-oss-120b` through **OpenRouter** with schema-constrained structured outputs. A **Streamlit** UI exposes the full reasoning trace for each ticket.

The project is evaluated rigorously against a **73-ticket golden test set**: 5 hand-written tickets plus 68 sampled from a public Kaggle dataset using reproducible, rule-based labelling. An **ablation study** compares the full agent against three reduced variants (no retrieval, no priority gate, single LLM call) to measure what each component contributes. The baseline ablation exposed two defects: a priority gate that escalated **none** of the 9 critical tickets, and confusion between intent labels. An **improvement round**, tuned on a separate non-overlapping dev set and measured once on the golden set, fixed both. Critical-ticket escalation went from **0/9 to 9/9**, intent accuracy from **0.726 to 0.767**, priority accuracy from 0.329 to 0.384, and action accuracy from 0.123 to 0.205, all with 0 degraded tickets. The same round **mitigated the prompt-injection finding** from the security review with role separation and a retrieval floor on auto-send, which now holds back 21 confident but ungrounded drafts. The remaining gap is knowledge-base coverage: the KB covers billing only, and expanding it is the next lever.

Quality is backed by **146 automated tests** with the LLM stubbed and **10 live end-to-end tests** against the real model.

---

## 2. Introduction

### 2.1 Problem statement

Support tickets arrive as unstructured free text. Before a human can reply, someone must:

1. **Classify** what the customer wants (billing, technical issue, account access, …).
2. **Prioritise** it: an angry customer threatening legal action cannot wait in the same queue as a feature suggestion.
3. **Find the relevant policy**: refund windows, cancellation rules, payment methods.
4. **Write a reply** that is accurate to policy, not invented.

Doing this by hand is slow and inconsistent. Fully automated replies, on the other hand, are risky: a confidently wrong answer or a mishandled critical complaint does real damage.

### 2.2 Objectives

| # | Objective | How it is met |
|---|---|---|
| O1 | Automatically classify ticket intent into 6 categories | Tool 1, LLM with structured output |
| O2 | Score urgency from sentiment and urgency signals | Tool 2, LLM sentiment + keyword rules |
| O3 | Ground replies in company policy, not model memory | Tool 3, RAG over ChromaDB |
| O4 | Draft an empathetic, policy-accurate reply with a confidence estimate | Tool 4, LLM with structured output |
| O5 | Keep humans in the loop for risky cases | Two decision gates: escalate critical tickets; flag low-confidence drafts |
| O6 | Prove each component adds value | Golden test set + 4-variant ablation study |
| O7 | Make the agent's reasoning transparent | Step-by-step `reasoning_trace` shown in the UI |

### 2.3 Scope

**In scope:** English-language tickets; six intent labels; four priority levels; a single company knowledge base; three terminal actions; offline batch evaluation; a demo UI.

**Out of scope:** sending real emails, ticketing-system integration (Zendesk etc.), multi-turn conversations, multilingual support, fine-tuning models.

---

## 3. System Architecture

### 3.1 Pipeline overview

![Agent pipeline architecture](architecture.svg)

```
incoming ticket
   │
   ▼
[Tool 1] classify_intent ──► intent, intent_confidence
   │
   ▼
[Tool 2] score_priority ───► priority, sentiment_score, should_escalate
   │
   ▼
◆ Gate 1: critical priority? ── yes ──► ESCALATE to human  (END)
   │ no
   ▼
[Tool 3] retrieve_from_kb ─► retrieved_docs, retrieval_scores
   │
   ▼
[Tool 4] draft_response ───► drafted_response, response_confidence
   │
   ▼
◆ Gate 2: confidence ≥ threshold (0.7)?
   ├── yes ──► SEND drafted reply          (END)
   └── no ───► FLAG FOR REVIEW by a human  (END)
```

### 3.2 Component-to-code map

| Diagram node | Kind | File | Function |
|---|---|---|---|
| Incoming ticket | input | `src/agent/state.py` | `TicketState`, `new_ticket_state()` |
| Intent classifier | Tool 1 | `src/tools/intent_classifier.py` | `classify_intent()` |
| Priority scorer | Tool 2 | `src/tools/priority_scorer.py` | `score_priority()` |
| Critical priority? | Gate 1 | `src/agent/graph.py` | `route_after_priority()` |
| Escalate to human | terminal | `src/agent/graph.py` | `escalate_node()` |
| KB retrieval | Tool 3 | `src/tools/kb_retriever.py` + `src/knowledge/vector_store.py` | `retrieve_from_kb()` |
| Response drafter | Tool 4 | `src/tools/response_drafter.py` | `draft_response()` |
| Confidence ≥ threshold? | Gate 2 | `src/agent/graph.py` | `route_after_draft()` |
| Flag for review | terminal | `src/agent/graph.py` | `flag_node()` |
| Send drafted response | terminal | `src/agent/graph.py` | `send_node()` |
| LLM client | infrastructure | `src/llm.py` | `get_llm()` |

### 3.3 Shared state

Every node reads from and writes to one shared `TicketState` (a `TypedDict`), which makes the data flow explicit and easy to inspect:

| Field | Type | Written by | Meaning |
|---|---|---|---|
| `ticket_text`, `ticket_id` | `str` | input | Raw customer message and tracking ID |
| `intent`, `intent_confidence` | `str`, `float` | Tool 1 | Predicted category and confidence 0–1 |
| `priority`, `sentiment_score` | `str`, `float` | Tool 2 | low/medium/high/critical; sentiment −1 (angry) to +1 (happy) |
| `should_escalate` | `bool` | Tool 2 | True → Gate 1 routes to a human |
| `retrieved_docs`, `retrieval_scores` | `list` | Tool 3 | Top-k policy chunks and their cosine similarities |
| `drafted_response`, `response_confidence` | `str`, `float` | Tool 4 | Draft reply and self-rated confidence |
| `final_action` | `send` / `escalate` / `flag_for_review` | terminal node | Outcome |
| `reasoning_trace` | `list[str]` | every node | Human-readable step log, also used to detect degraded runs |

`new_ticket_state()` initialises every field, so no node can fail on a missing key regardless of which route the graph takes.

### 3.4 Design principles

- **One tool, one responsibility.** Each tool is a pure function `state → partial state update`, testable in isolation.
- **Fail safe, never fail silent.** Every tool catches its own errors and falls back to a safe default. The fallback is always written to the reasoning trace, and a failed draft gets confidence 0.0, so it can never be auto-sent.
- **Humans for the hard cases.** The agent automates only what it is confident about; critical and uncertain tickets go to people.
- **Configurable for experimentation.** The confidence threshold and component switches are parameters of `build_agent_graph()`, which makes the ablation study a configuration change, not a code fork.

---

## 4. Technology Stack

| Layer | Technology | Why it was chosen |
|---|---|---|
| Agent framework | **LangGraph** 1.2 | Explicit state machine with conditional routing, a natural fit for decision gates |
| LLM orchestration | **LangChain** (core 1.6) | Prompting and `with_structured_output()` for Pydantic-validated responses |
| LLM provider | **OpenRouter** (default) · Groq · OpenAI | One switch in `.env`; OpenRouter serves the same `gpt-oss-120b` without Groq's free-tier quota |
| LLM | **`openai/gpt-oss-120b`** | Strong open-weight reasoning model; consistent across providers |
| Vector store | **ChromaDB** 1.5 | Local, persistent, no infrastructure |
| Embeddings | **sentence-transformers** `all-MiniLM-L6-v2` | Free, fast, runs locally |
| UI | **Streamlit** 1.63 | Rapid prototyping; non-specialists can work on it |
| Evaluation | **pandas** + **scikit-learn** + custom ROUGE-L | Accuracy, macro-F1, response quality, ablation tables |
| Data acquisition | **kagglehub** | Reproducible download of the public ticket dataset |
| Config | **python-dotenv** | `.env` support; shell variables take precedence |
| Testing | **pytest** + Streamlit `AppTest` | Unit, integration, UI and live end-to-end tests |

---

## 5. Implementation

### 5.1 LLM client: `src/llm.py`

A single factory, `get_llm(temperature)`, builds the chat model used by every tool, so the provider is a one-line `.env` change.

| `LLM_PROVIDER` | Client | Model setting | Key variable |
|---|---|---|---|
| `openrouter` *(default)* | `ChatOpenAI` → `https://openrouter.ai/api/v1` | `OPENROUTER_MODEL` (default `openai/gpt-oss-120b`) | `OPENROUTER_API_KEY` |
| `groq` | `ChatGroq` | `LLM_MODEL` | `GROQ_API_KEY` |
| `openai` | `ChatOpenAI` | `LLM_MODEL` | `OPENAI_API_KEY` |

Key properties:
- **Keys are read from `os.environ` when the client is built.** They are never stored in config or logged. A missing or blank key raises a clear `RuntimeError` at startup, not a 401 on the first ticket.
- **OpenRouter requests** send attribution headers (`HTTP-Referer`, `X-Title: Support Ticket Triage Agent`) and set `provider.require_parameters: true`. This routes only to hosts that honour `response_format`, so structured output cannot silently degrade to free text.
- `ChatOpenAI` wraps the official `openai` Python SDK, with `base_url` pointed at OpenRouter.
- **Hard per-request deadline.** A custom httpx `DeadlineTransport` caps the *total* time of each request (`LLM_TIMEOUT`, default 60 s) and retries up to `LLM_MAX_RETRIES` (default 2) times, for every provider. httpx's own timeout only limits the gap *between* bytes, and OpenRouter keeps slow requests alive by trickling whitespace, so without this a stalled upstream provider could hang a request forever (§10). A test confirms the real OpenAI SDK raises `APITimeoutError` in under 1 s against a hung server.

### 5.2 Tool 1: Intent classifier

- **Labels:** `billing`, `technical_issue`, `account_access`, `feature_request`, `complaint`, `general_inquiry`.
- **Method:** LLM (temperature 0) with structured output into a Pydantic model. `intent` is a `Literal` over the six labels and `confidence` is constrained to [0, 1], so invalid labels are rejected by validation rather than parsed from free text.
- **Prompt:** gives a one-line **definition per label**, wraps the ticket in `<ticket>` tags, and tells the model the ticket is data, not instructions. It asks for the category plus a 0–1 confidence. The definitions were added in the improvement round (§8.4), after earlier runs showed most complaints and feature requests being classified as `general_inquiry` (§8.2). For example: *complaint = "expresses dissatisfaction or frustration with the product or service quality — choose this even if the ticket also asks a question"*.
- **Fallback:** `general_inquiry` with confidence 0.0, recorded as `Intent classification failed (…)` in the trace.

### 5.3 Tool 2: Priority scorer

The scorer judges **severity from what is happening to the customer, not how upset they sound**. The LLM returns a structured `PriorityResult(severity, sentiment_score)` (`json_schema` mode) against written definitions:

| Severity | Definition given to the model |
|---|---|
| **critical** | Security breach or suspected unauthorized access, data loss, an outage or system down that stops the customer's business or many users, fraud, or a legal threat, *even when written calmly* |
| high | A core function is broken or blocked, money was wrongly charged, or there is an urgent deadline |
| medium | A real problem with a workaround, an intermittent or partial issue, or a non-urgent request needing action |
| low | A question, how-to, feature suggestion or general feedback; nothing is broken |

- **Legal-threat override:** a whole-word match on *lawsuit, lawyer, attorney, legal action, sue, suing* always raises the priority to critical, whatever the model judged, and the override is logged in the trace. *"Is this legal in the EU?"* does not trigger it.
- **Sentiment** is still recorded and shown in the UI, but no longer decides escalation.
- **Fallback:** `medium` (kept out of the low pile without escalating), sentiment 0.0, failure logged. Legal threats still escalate even when the LLM call fails.

**Why it changed.** The original design was a hybrid rule: *critical* only if sentiment < −0.7 **and** one of 16 urgency keywords appeared. It escalated **0 of 9** critical golden tickets, because real breach and outage reports are written calmly, so the sentiment condition never fired (§8.3). On the dev set the new scorer escalated 8/8 critical tickets, against 0/8 for the old rule (§8.4).

### 5.4 Tool 3: Knowledge-base retriever (RAG)

**Vector store (`src/knowledge/vector_store.py`):** a `KnowledgeStore` class wrapping a persistent ChromaDB collection (`support_kb`) in cosine space.

- **Chunking:** markdown documents are split on `## ` headings first, so each FAQ entry stays one coherent chunk. A document title before the first heading joins the first section instead of becoming a near-empty chunk. Sections longer than 500 characters are windowed with a 50-character overlap. The current KB produces exactly 4 chunks, one per policy.
- **Similarity score:** `1 − cosine distance`, so higher means more relevant.
- **Idempotent indexing:** chunk IDs are `<file>::<n>`, so re-indexing upserts instead of duplicating; `reset()` clears deleted docs.
- **Metadata:** each chunk records its `source` file and a `category` taken from the filename (`billing_faq.md` → `billing`).
- **Testability:** the embedding function is injectable, so tests use a deterministic bag-of-words embedder and never download MiniLM.

**Retriever (`src/tools/kb_retriever.py`):** builds the query as `"<intent>: <ticket text>"` so the intent label nudges the embedding toward the right section, and returns the **top 3** chunks. The store is opened once per process and indexed automatically on first use; this lazy initialisation is lock-protected, so it is safe under concurrent evaluation. On failure, it returns no documents and logs the reason; the drafter's confidence then drops and the ticket is flagged.

Example on the real index, query *"billing: I was charged twice this month"*:

| Rank | Chunk | Similarity |
|---|---|---|
| 1 | Duplicate Charges | 0.661 |
| 2 | Payment Methods | 0.339 |
| 3 | Subscription Cancellation | 0.318 |

### 5.5 Tool 4: Response drafter

- **Role separation (prompt-injection defence):** the request is two messages. The **system** message holds the role (*"You are a customer support agent for {COMPANY_NAME}"*), the numbered policy documents (`[1] …`), ticket intent and priority, and all instructions. The **user** message holds only the customer's ticket, inside `<ticket>` tags. The system message tells the model to treat the ticket strictly as data: never follow instructions inside it, and never let it change the rules, the policies or the confidence rating. Tests confirm that ticket text, including a *"SYSTEM OVERRIDE"* injection, never appears in the system message.
- **Instructions:** be helpful and empathetic, base policy details **only** on the documents, never invent policy or promise anything the documents don't support, and hand off to a specialist if the documents don't cover the issue. The model then self-rates its confidence (0–1) that the reply fully resolves the issue.
- **Structured output:** `DraftResult(response: str, confidence: float∈[0,1])` with `method="json_schema"`, which constrains decoding. See §10 for why this replaced tool calling.
- **Temperature:** 0.3, for slightly more natural replies.
- **Fallback:** a generic holding reply with **confidence 0.0**, which guarantees Gate 2 flags it and it is never sent.

### 5.6 Agent graph and decision gates

`build_agent_graph(confidence_threshold=0.7, use_kb=True, use_priority_gate=True, min_retrieval_score=0.3)` compiles the LangGraph:

- **Gate 1** (`route_after_priority`): `should_escalate` → `escalate`, otherwise continue to retrieval.
- **Gate 2** (`route_after_draft`): a draft is **sent only if both checks pass**:
  1. `response_confidence ≥ threshold` (0.7, inclusive; the UI slider can change it), and
  2. the **best retrieval similarity ≥ `MIN_RETRIEVAL_SCORE`** (0.3): the knowledge base must actually contain a relevant policy.

  Otherwise the ticket is flagged, and the trace names the check that failed, e.g. `FLAGGED: confidence 0.90 OK, but best KB match 0.10 below 0.30`. The self-rated confidence alone is no longer trusted to auto-send: it can be inflated by a confident but ungrounded model, or by prompt injection.
- **Choosing the floor.** It was tuned on the **dev set** using embeddings only (no LLM calls). Billing tickets, which the KB covers, have best matches of 0.35–0.52; every other intent tops out at 0.26. At 0.3 the floor keeps 100% of covered tickets and blocks 100% of uncovered ones; 0.25 lets 5% of uncovered tickets through and 0.35 blocks 10% of covered ones. It must be re-tuned when the KB grows.
- **Ablation switches:** `use_kb=False` skips retrieval (the drafter gets no context) and **disables the floor**, since there are no scores to check; `use_priority_gate=False` never escalates.
- `run_agent(ticket_text)` is the single entry point used by the UI, the evaluator and the tests.

### 5.7 Single-shot baseline: `src/agent/single_shot.py`

For the ablation, a `SingleShotAgent` performs the whole task in **one** LLM call. It returns intent, priority, reply and confidence in one structured response, then applies the same routing rules (critical → escalate; confidence ≥ threshold → send; else flag). It exposes the same `.invoke(state)` interface as the compiled graph, so the evaluator treats every variant identically.

### 5.8 Evaluation harness: `src/evaluation/evaluator.py`

| Function | Purpose |
|---|---|
| `load_golden_set(path)` | Loads the CSV (path or uploaded file); fails loudly if required columns are missing or the file is empty |
| `predict(agent, golden_set)` | Runs every ticket; records predictions, errors, and whether any tool degraded |
| `score_predictions(df)` | Computes all metrics (§7.2) |
| `run_evaluation(agent, golden_set)` | `predict` + `score` for one variant |
| `default_variants(threshold)` | The four ablation variants |
| `run_ablation(variants, golden_set)` | One metrics row per variant |
| CLI | `python -m src.evaluation.evaluator [--ablation] [--golden-set PATH] [--workers N]` |

**Concurrency and progress.** Tickets are independent, so `predict()` runs them through a thread pool (`DEFAULT_WORKERS = 8`). The ablation runs all four variants side by side, each with a progress bar showing a live `degraded=` count, and results keep golden-set order. On 10 live tickets this took 31 s in total, against about 11 s per call when run one at a time.

Two integrity rules were built in after real failures (§10):
1. **Crashed tickets count as wrong**, never dropped, so a flaky API cannot inflate accuracy.
2. **Degraded tickets are counted** (`n_degraded`): any ticket where a tool fell back to a default, detected from the reasoning trace. A warning is printed whenever `n_degraded > 0`.

### 5.9 Golden-set builder: `src/evaluation/build_golden_set.py`

Turns the raw Kaggle dataset into labelled golden rows with deterministic rules and seeded sampling (§6.3). Hand-written rows (`source=hand`) are always preserved, and a hand-corrected Kaggle ticket takes precedence over a fresh copy of the same ticket. CLI: `python -m src.evaluation.build_golden_set [--n-per-intent 10] [--n-critical 8] [--seed 42]`.

With `--dev` it writes a separate **dev (tuning) set** instead (`data/golden_set/dev_qa.csv`, §6.4), drawn the same way but never overlapping the golden set.

### 5.10 Streamlit UI: `app/streamlit_app.py`

| Area | Content |
|---|---|
| Main panel | Ticket text area → **Run Agent** → colour-coded action banner (green *send* / red *escalate* / yellow *flag*), metric tiles (intent + confidence, priority, sentiment, response confidence), drafted reply, expandable retrieved KB docs with similarity scores, numbered reasoning trace |
| Sidebar: Settings | Confidence-threshold slider; active provider/model |
| Sidebar: Evaluation | Upload a golden CSV (defaults to the repo's); **Run evaluation** (metrics table); **Run ablation** (comparison table) |

Implementation notes: the compiled graph is cached per threshold value (`st.cache_resource`); LLM output and trace lines are rendered as **plain text** for security (§11); the Run button stays enabled because Streamlit commits a text area only on blur, and a disabled button would swallow the first click after pasting.

---

## 6. Data

### 6.1 Knowledge base

`data/knowledge_base/billing_faq.md` contains four policies the retriever searches:

| Section | Key policy |
|---|---|
| Refund Policy | Refunds processed in 3–5 business days; full refund within 30 days; prorated after, at management discretion |
| Duplicate Charges | Verify, then refund immediately without manager approval |
| Subscription Cancellation | Cancel any time; active until end of cycle; no partial refunds |
| Payment Methods | Visa, Mastercard, American Express, UPI; update in Account Settings → Billing |

The KB currently covers **billing only**. Tickets about other topics correctly receive low drafter confidence and are flagged; expanding the KB is the most direct lever on action accuracy (§12).

### 6.2 Source dataset

| Property | Value |
|---|---|
| Name | *Customer IT Support – Ticket Dataset* (Tobias Bueck), Kaggle `tobiasbueck/multilingual-customer-support-tickets` |
| Licence | CC BY-NC 4.0 (non-commercial; suitable for academic/portfolio use) |
| Main file | `aa_dataset-tickets-multi-lang-5-2-50-version.csv`, 28,587 tickets (16,338 English, 12,249 German) |
| Fields | `subject`, `body`, `answer` (reference agent reply), `type`, `queue`, `priority`, `language`, `tag_1…tag_8` |
| Types | Incident 11,466 · Request 8,187 · Problem 6,012 · Change 2,922 |
| Priorities | medium 11,515 · high 11,178 · low 5,894 (**no critical level**) |
| Queues | 10, e.g. Technical Support 8,362; Product Support 5,252; Customer Service 4,268; Billing and Payments 2,788 |
| Nature | Synthetic, anonymised (placeholders such as `<name>`, `<tel_num>`) |
| Storage | Not committed (`data/raw/` is gitignored); download command in §13 |

**Why this dataset:** it was the only candidate with ticket text, a reference answer **and** a priority. The Bitext dataset has clean intents and responses but no priority, so it could not test escalation.

### 6.3 Golden test set construction

The dataset's labels do not match the project's schema, so labels are **derived by deterministic rules**, applied in priority order:

| Intent | Rule |
|---|---|
| `billing` | Queue is *Billing and Payments* or *Returns and Exchanges* **and** the text has billing wording (refund, invoice, charge, payment, subscription, price…) |
| `account_access` | User-level access phrases: *forgot/reset password, locked out, can't log in, login error, 2FA* |
| `complaint` | Dissatisfaction wording: *unacceptable, disappointed, frustrated, dissatisfied, fed up, terrible, worst* |
| `feature_request` | `Feature` tag **and** type Request/Change **and** feature wording (*feature, enhancement, functionality, integration…*) |
| `technical_issue` | Technical/Product/IT Support or Outage queues with type Incident/Problem |
| `general_inquiry` | General Inquiry / Customer Service / Sales queues with type Request |
| *(dropped)* | Anything else, e.g. Human Resources |

- **Priority:** the dataset's value, except a `high` **Incident/Problem** mentioning a breach, hack, ransomware, outage, data loss, fraud or legal threat is promoted to **critical**.
- **Action:** `critical` → `escalate`; everything else → `send`, matching the hand-written convention.
- **Sampling:** 8 critical tickets plus 10 non-critical per intent, seeded (42) and therefore reproducible. Regenerating produces a byte-identical file.

**Label-quality iteration.** Rules were refined over three rounds by spot-checking real rows, not just label counts:

| Round | Problem found in spot check | Fix |
|---|---|---|
| 1 | "Unauthorized access to patient data" labelled `account_access` | Require user-level access phrases, not any mention of login |
| 1 | "We received several complaints about payroll" labelled `complaint` | Drop `complain*`; keep dissatisfaction words only |
| 1 | A feature *request* to improve breach protection promoted to `critical` | Only Incident/Problem tickets can become critical |
| 1 | Marketing questions labelled `feature_request` | Also require feature wording in the text |
| 1 | Raw `<br>` HTML in ticket text | Converted to newlines |
| 2 | Dataset queue noise: marketing requests filed under *Billing and Payments* | Billing also requires billing wording (944 of 952 billing-worded tickets are in that queue) |
| 2 | `Account` tag pulling in security incidents | Removed the tag route |

**Final golden set: 73 tickets.**

| Breakdown | Counts |
|---|---|
| Source | 68 Kaggle (rule-labelled) · 5 hand-written |
| Intent | technical_issue 19 · billing 11 · account_access 11 · complaint 11 · feature_request 11 · general_inquiry 10 |
| Priority | high 25 · medium 23 · low 16 · critical 9 |
| Action | send 64 · escalate 9 |

### 6.4 Dev (tuning) set

Tuning prompts and thresholds on the same tickets you report results on inflates the reported scores. To avoid this, a separate **dev set** of 68 tickets (`data/golden_set/dev_qa.csv`) was drawn from the remaining ~11,000 labelled Kaggle tickets. It uses the same rules and stratification as the golden set (8 critical + 10 per intent) with a different seed (7), and has **zero overlap** with the golden set, which is enforced in code and tested.

All design choices in the improvement round (§8.4) were checked on the dev set first: the intent definitions, the severity-based priority scorer, and the retrieval-floor value. The golden set was then used **once** to measure the result.

---

## 7. Evaluation Methodology

### 7.1 Ablation study design

An **ablation study** removes one component at a time and measures the drop in performance, showing what each part contributes.

| Variant | Component removed | Question it answers |
|---|---|---|
| **Full agent** | none (baseline) | How well does the complete pipeline perform? |
| **No KB retrieval** | Tool 3 | Does grounding in policy documents improve outcomes? |
| **No priority gate** | Gate 1 | Does early escalation route critical tickets correctly? |
| **Single-shot LLM** | the whole pipeline (one prompt) | Does the multi-step architecture beat a single LLM call? |

All variants run on the same 73 tickets with the same model and a threshold of 0.7.

### 7.2 Metrics

| Metric | Definition |
|---|---|
| **Intent accuracy** | Exact-match rate of predicted vs expected intent |
| **Intent macro-F1** | scikit-learn macro-averaged F1 across intents (crashes count as an "error" class) |
| **Priority accuracy** | Exact match over low/medium/high/critical |
| **Action accuracy** | Exact match over send/escalate/flag_for_review |
| **Response quality** | Mean **ROUGE-L F1** (longest-common-subsequence of words) between draft and reference answer, over tickets whose expected action is not `escalate` (escalated tickets have no reference) |
| `n_errors` | Tickets where the agent raised an exception (scored as wrong) |
| `n_degraded` | Tickets where any tool fell back to a default; **a run with `n_degraded > 0` is not a valid measurement** |

---

## 8. Results

### 8.1 Preliminary run: 5 hand-written tickets (Groq)

| Variant | Intent | Priority | Action | Response quality |
|---|---|---|---|---|
| Full agent | 1.00 | 0.60 | 0.40 | 0.16 |
| No KB retrieval | 1.00 | 0.60 | **0.00** | 0.14 |
| No priority gate | 1.00 | 0.60 | 0.40 | 0.12 |
| Single-shot LLM | 1.00 | 0.80 | 1.00 | 0.13 |

Already informative at this size:
- Removing retrieval collapsed action accuracy from 0.40 to 0.00, early evidence that the KB matters.
- The single-shot baseline caught the *"contacting my lawyer"* ticket that the full agent missed, which exposed that **"lawyer"/"attorney" are absent from the escalation keyword list**.

### 8.2 Full agent on the 73-ticket golden set (Groq)

| Intent acc. | Intent macro-F1 | Priority acc. | Action acc. | Response quality |
|---|---|---|---|---|
| 0.685 | 0.664 | 0.397 | 0.301 | 0.143 |

> **Caveat:** this run predates two fixes. 11 of 73 drafts (15%) were silent fallbacks caused by a structured-output failure, later fixed by switching to `json_schema` (§10), and the `n_degraded` counter did not yet exist. Action accuracy is therefore understated.

Confusion analysis from this run:
- **Escalation recall was low:** of 9 critical tickets, only 2 were predicted critical; 7 were scored `high`. The priority scorer under-escalates.
- **Most `send` tickets were flagged** (44 of 64), mainly because the KB covers billing only, so the drafter correctly reports low confidence elsewhere.
- **Intent errors** clustered in `complaint` and `feature_request` being predicted as `general_inquiry`, consistent with the dataset's neutral B2B tone.

### 8.3 Baseline ablation on the 73-ticket golden set (OpenRouter, `openai/gpt-oss-120b`)

*This is the agent **before** the improvement round. §8.4 reports the improved agent.*

**Setup:** 73 golden tickets, confidence threshold 0.7, concurrent evaluator (`--workers 8`). The whole four-variant ablation ran in **3 min 36 s**.

| Variant | Intent acc. | Intent F1 | Priority acc. | Action acc. | Response quality (ROUGE-L) | n_errors | n_degraded |
|---|---|---|---|---|---|---|---|
| Full agent | **0.726** | **0.713** | 0.329 | 0.123 | 0.135 | 0 | 0 |
| No KB retrieval | 0.699 | 0.689 | 0.342 | 0.096 | 0.106 | 0 | 0 |
| No priority gate | 0.712 | 0.700 | 0.356 | 0.096 | **0.136** | 0 | 0 |
| Single-shot LLM | 0.712 | 0.712 | **0.452** | **0.836** | 0.095 | 0 | 0 |

**Validity:** all four variants completed with **0 errors and 0 degraded tickets**, so every prediction is a genuine model answer. This supersedes two earlier attempts. A Groq run was **discarded** after the free-tier daily token quota ran out mid-run; `n_degraded` caught 44–73 degraded tickets per variant before any invalid numbers were published. A first OpenRouter run lost its *No priority gate* variant to a request that hung for more than 40 minutes, which led to the request-deadline fix in §5.1.

> **Reading the numbers.** With 73 tickets, one ticket moves a metric by ≈ 0.014. Differences of one or two tickets (≤ 0.03) are within run-to-run LLM sampling variation. A repeat single-shot run, for example, gave action 0.767 instead of 0.836. Only larger gaps should be treated as real effects.

**Findings:**

1. **Retrieval improves reply quality.** Removing the KB lowers response quality from 0.135 to 0.106, about **27% worse**, the clearest ablation effect within the pipeline. Action accuracy also falls (0.123 → 0.096), but that gap is only two tickets. The intent gap (0.726 vs 0.699) is noise: intent is classified *before* retrieval, so the KB cannot influence it.

2. **Low action accuracy is a knowledge-base coverage problem, not a pipeline fault.** The KB covers billing only, and the per-ticket data shows the agent behaving as designed:

   | Signal | Billing tickets | All other intents |
   |---|---|---|
   | Mean top retrieval similarity | 0.49 | 0.09–0.17 |
   | Mean draft confidence | 0.54 | 0.17–0.31 |

   Only **11%** of drafts reach the 0.7 send threshold. The full agent's action confusion (expected → predicted) is *send → flag* 56, *escalate → flag* 9, *send → send* 8, and nothing is escalated. The agent **correctly flags tickets it cannot ground in policy**, but the golden set expects 64 of 73 tickets to be sent, so this safe behaviour scores as wrong.

3. **Single-shot's high action accuracy (0.836) comes from sending ungrounded answers.** With no documents and no grounding check, it sends confidently whether or not policy supports the reply. It also has the **lowest response quality (0.095)**. The action metric rewards sending, not grounded correctness, and under the current billing-only KB it cannot fairly compare a grounded agent with an ungrounded one.

4. **The priority gate currently contributes nothing: 0 of 9 critical tickets were escalated.** All nine were scored `high`. The scorer requires sentiment < −0.7 **and** an escalation keyword, but the dataset's breach and outage reports are written calmly, so sentiment never crosses the threshold. That is why removing the gate barely changes anything (action 0.123 → 0.096, two tickets). The single-shot LLM judges priority clearly better (**0.452** vs 0.329). *Caveat:* the golden `critical` labels were produced by a keyword rule (§6.3), so a purely keyword-based fix would partly match the labels by construction and overstate the improvement. Escalation fixes should be validated on hand-labelled critical tickets.

**What the ablation tells us to do next:** expand the KB beyond billing (the main lever on action accuracy and the precondition for a fair grounded-vs-ungrounded comparison), and rework priority scoring, for example by letting the LLM judge severity directly as single-shot does, rather than gating on angry sentiment.

### 8.4 Improvement round and re-run

Guided by §8.3, four changes were made. Each was checked on the dev set (§6.4) before the golden set was used once to measure the result:

| # | Change | Targets |
|---|---|---|
| 1 | **Severity-based priority scoring**: the LLM judges severity against written definitions, plus a legal-threat override (§5.3) | 0/9 escalation recall |
| 2 | **Intent label definitions** in the classifier prompt (§5.2) | complaints and feature requests misread as `general_inquiry` |
| 3 | **Role separation** in the drafter: instructions in the system message, ticket in a delimited user message (§5.5) | prompt injection |
| 4 | **Retrieval floor on the send gate**: send only if the best KB match ≥ 0.3 (§5.6) | auto-sending confident but ungrounded drafts; prompt injection |

**Dev-set check (old vs new prompts, 68 tickets, scored like-for-like):**

| Dev metric | Old | New |
|---|---|---|
| Intent accuracy | 0.662 | **0.765** |
| Priority accuracy | 0.250 | **0.309** |
| Critical tickets escalated | 0/8 | **8/8** |
| Non-critical tickets wrongly escalated | 1 | 4 |

**Golden-set re-run** (73 tickets; 292 agent runs, 0 timed out; same model and threshold):

| Variant | Intent acc. | Intent F1 | Priority acc. | Action acc. | Response quality | n_degraded |
|---|---|---|---|---|---|---|
| **Full agent (improved)** | **0.767** | **0.761** | 0.384 | 0.205 | 0.149 | 0 |
| No KB retrieval | 0.767 | 0.765 | 0.397 | 0.315 | 0.141 | 0 |
| No priority gate | 0.726 | 0.707 | 0.397 | 0.068 | **0.160** | 0 |
| Single-shot LLM | 0.740 | 0.635 | **0.493** | **0.740** | 0.081 | 1 |

**Full agent, before → after:**

| Metric | Baseline (§8.3) | Improved | Change |
|---|---|---|---|
| Intent accuracy | 0.726 | **0.767** | +0.041 (+3 tickets) |
| Intent macro-F1 | 0.713 | **0.761** | +0.048 |
| Priority accuracy | 0.329 | **0.384** | +0.055 (+4 tickets) |
| Action accuracy | 0.123 | **0.205** | +0.082 (+6 tickets) |
| Response quality (ROUGE-L) | 0.135 | **0.149** | +0.014 |
| **Critical tickets escalated** | **0 / 9** | **9 / 9** | escalation now works |
| Non-critical tickets wrongly escalated | 0 | 4 | the cost of catching all 9 |
| Ungrounded drafts blocked by the retrieval floor | — | 21 | new safety check |

**Findings:**

1. **Escalation went from broken to complete.** All 9 critical tickets now reach a human (previously none). The cost is 4 false escalations, a sensible trade for a support desk, since a missed breach is far worse than a human glancing at a non-urgent ticket. *Caveat:* the golden `critical` labels are keyword-derived (§6.3). The new scorer uses an LLM judgment against written definitions, not keyword matching (apart from explicit legal threats), so the effect is not purely by construction, but hand-labelled critical tickets would make this claim stronger.

2. **The priority gate now earns its place.** Removing it drops action accuracy from 0.205 to 0.068, a 10-ticket gap. In the baseline the gap was 2 tickets, because the gate never fired. This is the clearest ablation effect in the improved agent.

3. **Intent accuracy improved by 3 tickets**, consistent with the larger dev-set gain (+7 tickets). The remaining weak labels are `complaint` and `feature_request`; the dataset's neutral B2B tone makes these hard (§12).

4. **The retrieval floor works as designed, and the action metric penalises it.** The full agent sent only billing tickets, which are what the KB covers. **21 drafts** had confidence ≥ 0.7 but no supporting policy and were flagged, each with a trace line such as `FLAGGED: confidence 0.90 OK, but best KB match 0.10 below 0.30`. The *No KB* variant has no floor, so it sent 14 such ungrounded answers (feature requests, complaints, technical issues). The golden set labels them `send`, which is why that variant's action accuracy (0.315) is *higher* than the full agent's. This is the same effect that inflates single-shot's 0.740: **the metric rewards sending, not grounded correctness.** Had the floor been off, roughly 21 more tickets would have been sent unchecked.

5. **Reply quality held up** (0.135 → 0.149) despite the stricter prompt, and the full agent's replies still beat single-shot's (0.081).

> **Method note.** Variants, scoring (`score_predictions`) and degraded-detection (`is_degraded`) are the repository's own. The tickets above were run through a temporary 16-worker concurrent runner with an overall deadline, because the concurrent evaluator (§5.8) had not yet been merged into this branch.

**Confirmation run after merge** (`python -m src.evaluation.evaluator --ablation`, merge commit `ac70000`, 0 degraded rows):

| Variant | Intent acc. | Intent F1 | Priority acc. | Action acc. | Response quality | n_degraded |
|---|---|---|---|---|---|---|
| **Full agent** | **0.767** | 0.753 | 0.411 | 0.247 | 0.154 | 0 |
| No KB retrieval | 0.767 | 0.760 | 0.425 | 0.192 | 0.141 | 0 |
| No priority gate | 0.753 | 0.749 | 0.425 | 0.082 | 0.160 | 0 |
| Single-shot LLM | 0.740 | 0.739 | 0.479 | 0.767 | 0.082 | 0 |

A separate run of the full agent alone gave intent 0.740, priority 0.425 and action 0.205, with **9/9 critical tickets escalated and 2 false escalations** (a service-interruption report and a medical-data security concern).

What the confirmation run does and does not reproduce:
- **Reproduced:** escalation 9/9, the gate effect (action 0.247 → 0.082 without it, a 12-ticket gap), the intent gain, and 0 degraded rows.
- **Not reproduced: No-KB beating the full agent on action.** Here No-KB scored 0.192, *below* the full agent's 0.247. In the temporary run it scored 0.315. That 9-ticket swing on one variant shows that action accuracy is dominated by run-to-run variation in which borderline drafts the model rates ≥ 0.7. **Finding 4's mechanism (the floor holds back ungrounded drafts) still stands, but the ordering of the full and No-KB rows on action is not a robust result.**

**Bottom line:** the improvement round fixed the two defects the baseline exposed: escalation recall (0/9 → 9/9) and intent confusion (+3 tickets). It also made auto-sending safer. The remaining gap in action accuracy is knowledge-base coverage. **Expanding the KB is now the single biggest lever**, and it is what will let the full agent send the ~21 confident drafts it currently, correctly, holds back.

### 8.5 Live end-to-end test results (OpenRouter)

*Re-run against the improved agent after the merge: **9 passed, 1 xpassed, ≈ 57 s**. The prompt-injection probe now passes because the retrieval floor gates sending independently of self-rated confidence, so its `xfail` marker was removed and it is a strict regression test. The results below are from the first run.*

| Check | Result |
|---|---|
| Intent classifier on a duplicate-charge ticket | `billing`, confidence > 0.5 ✅ |
| Priority scorer on a legal-threat ticket | sentiment < −0.7 → `critical`, escalate ✅ |
| Priority scorer on a calm ticket | low/medium, no escalation ✅ |
| Retriever | top chunk = *Duplicate Charges*, scores sorted ✅ |
| Drafter grounded in a policy chunk | reply mentions refund, confidence valid ✅ |
| Full graph, covered billing ticket | **send** ✅ (e.g. billing / high / confidence 0.96) |
| Full graph, legal threat | **escalate** before retrieval or drafting ✅ |
| Full graph, ticket the KB cannot answer | **flag_for_review** ✅ |
| Single-shot validity | labels within schema ✅ |
| Prompt-injection probe | resisted on this sample (XPASS); see §11 |

**9 passed, 1 xpassed, no tool fallbacks, ≈ 49 s.**

---

## 9. Testing & Quality Assurance

### 9.1 Automated test suite (≈ 3 s, no network)

The LLM is stubbed with fakes; ChromaDB runs for real in temporary directories with a deterministic embedder. After merging both work streams the suite has **146 tests** (plus 10 opt-in live tests).

| Test file | Tests cover |
|---|---|
| `test_vector_store.py` | Chunking contract (size, overlap, coverage, headings, title merge), search ranking, similarity scores, upsert, directory loading, idempotency, persistence, reset |
| `test_kb_retriever.py` | Top-k ordering, doc/score alignment, trace, failure handling, auto-indexing on first use |
| `test_response_drafter.py` | Grounded system prompt, **ticket isolated in a delimited user message (incl. an injection payload)**, no-docs path, fallback with confidence 0, trace, schema validation |
| `test_classifier_and_scorer.py` | Classifier success/fallback, label definitions and ticket delimiting; priority follows LLM severity, **calm critical incidents escalate**, legal-threat override and detection (whole words, *"legal"* alone is not a threat), fallback to medium, legal threats escalate even on LLM failure |
| `test_graph.py` | Real LangGraph wiring with stub tools: full route, flagging, configurable threshold, escalation before retrieval, both ablation variants, **retrieval floor** (confident but ungrounded → flagged, floor off in the no-KB variant), flag reasons in the trace |
| `test_single_shot.py` | Send / flag / escalate routing, failure handling, label validation |
| `test_evaluator.py` | Golden-set validation, ROUGE-L maths, scoring, crashes counted as wrong, degraded detection, ablation table, concurrent execution preserving order |
| `test_build_golden_set.py` | Every labelling rule including regressions from the spot checks, HTML cleaning, stratified and reproducible sampling, hand-row preservation, **dev set never overlaps the golden set** |
| `test_llm.py` | Provider selection, OpenRouter config/headers, runtime key reading, fail-fast on missing key, no key in error messages, per-request deadline against a hung server |
| `test_streamlit_app.py` | Headless UI via `AppTest`: page load, send/escalate banners, empty-ticket warning, evaluation buttons, **injected Markdown not rendered** |

### 9.2 Live end-to-end tests (opt-in)

`tests/test_live.py` (10 tests, marker `live` registered in `pytest.ini`) exercises the real provider and the real MiniLM index. They are skipped unless `RUN_LIVE_TESTS=1`. Results are in §8.5. The prompt-injection probe is a non-strict xfail; its occasional XPASS is not evidence that the issue is fixed.

### 9.3 Entry-point verification

| Entry point | Verified |
|---|---|
| `python -m src.knowledge.vector_store` | Indexes 4 chunks; repeatable |
| `python -m src.evaluation.build_golden_set` | Deterministic: regenerated file is byte-identical to the committed one |
| `python -m src.evaluation.evaluator` | Runs end to end with 0 degraded rows; rejects malformed CSVs with a clear error |
| `streamlit run app/streamlit_app.py` | Server healthy (HTTP 200), no log errors; a real ticket produced SEND at 96% confidence |
| Startup without an API key | Clear "OPENROUTER_API_KEY is not set…" message instead of a crash |

---

## 10. Engineering Challenges & Solutions

| # | Challenge | Root cause | Solution |
|---|---|---|---|
| 1 | Default Llama model returned 404s that a `try/except` hid | Model deprecated on Groq | Switched to `openai/gpt-oss-120b`; added trace-level failure logging |
| 2 | FAQ title became a useless near-empty chunk | Heading split left the preamble alone | Merge the preamble into the first section; regression test |
| 3 | First click after pasting a ticket did nothing | Streamlit commits a text area on blur; the button was disabled until then | Button always enabled; warn on empty input; test |
| 4 | 15% of drafts were silent fallbacks | gpt-oss sometimes answered in prose instead of calling the structured-output tool (Groq 400 `tool_use_failed`) | Switched drafter and single-shot to `method="json_schema"`: 11/11 previously failing tickets succeeded, vs 8/11 |
| 5 | Evaluator reported `n_errors: 0` on degraded runs | Tools catch their own errors, so a flaky API looked like a model answering wrong | Tools log fallbacks to the trace; evaluator counts `n_degraded` |
| 6 | Ablation on Groq produced meaningless numbers | Free tier: 200k tokens/day; a full ablation needs over 1M | Added provider abstraction; moved to OpenRouter (same model, pay per token) |
| 7 | Kaggle labels didn't match the schema; dataset queues were noisy | Different taxonomy; no critical level | Rule-based labelling refined over 3 spot-check rounds (§6.3) |
| 8 | Ablation runs were slow (≈ 35–40 min per variant) | ~200 sequential calls per variant to a reasoning model with high latency variance | Thread-pool concurrency (8 workers), variants side by side, progress bars: the full four-variant ablation now runs in **3 min 36 s** |
| 9 | One ablation variant hung for 40+ minutes | httpx timeouts only limit the gap between bytes; OpenRouter keeps slow requests alive with whitespace, so a stuck upstream never timed out | `DeadlineTransport` caps total time per request (60 s) with 2 retries; tested against a hung server |
| 10 | Concurrent evaluation could build the vector store twice | Lazy `get_store()` initialisation was not thread-safe | Initialisation guarded by a lock |
| 11 | Escalation never fired (0/9 critical tickets) | Rule required angry sentiment, but breach and outage reports are written calmly | Severity judged by the LLM against written definitions + legal-threat override → 9/9 (§5.3, §8.4) |
| 12 | Confident but ungrounded drafts could be auto-sent | Self-rated confidence was the only send check | Retrieval floor (best KB match ≥ 0.3, tuned on the dev set): 21 such drafts now flagged (§5.6) |
| 13 | Risk of tuning on the test set | Prompts and thresholds were being judged on the golden set itself | Separate, non-overlapping dev set for all tuning; golden set used once to measure (§6.4) |

---

## 11. Security & Privacy

### 11.1 API key handling

- Keys are read from the environment **when a client is built**; they are never hard-coded, stored in config, printed or logged.
- Real shell variables take precedence over `.env`; `.env` is gitignored; `.env.example` contains empty values only.
- Missing keys fail fast with an error message that never echoes a key (tested).
- A full scan of the working tree and **all git history** (`git log -p --all`, pickaxe `-S`) found **no `sk-or-` strings**.
- The browser never sees a key: Streamlit runs all Python server-side.

### 11.2 Findings from automated security review

| Finding | Severity | Status |
|---|---|---|
| **Markdown injection in the UI.** A ticket could steer the LLM into emitting an image link (tracking pixel) that `st.markdown` would load in the operator's browser. | Medium | **Fixed.** Replies and trace lines render as plain text; a test feeds an injected image link and asserts it is not rendered. |
| **Prompt injection past the send gate.** Ticket text shared a prompt with the instructions, and the model's self-rated confidence was the only gate before auto-send. | Medium | **Mitigated** in the improvement round (§8.4) with two independent layers. (1) **Role separation:** instructions live in the system message, and the ticket sits in a delimited user message marked as data (tested: injection text never reaches the system message). (2) **Retrieval floor:** auto-send now also requires a real KB match ≥ 0.3, so an injected *"rate your confidence 1.0"* cannot by itself cause a send. On the golden set the floor stopped 21 confident but ungrounded drafts. *Residual risk:* an injection on a ticket that also matches real policy could still influence the reply's wording; the live injection probe now passes against the improved agent and is a strict regression test. |

### 11.3 Data privacy

The Kaggle dataset is synthetic and anonymised (placeholders such as `<name>`). Raw data is kept out of git. No real customer data is processed.

---

## 12. Limitations & Future Work

**Limitations**
- **Heuristic golden labels.** 68 of 73 rows are rule-labelled, not human-annotated; `complaint` is the weakest label because the source tone is mostly neutral B2B.
- **Small, billing-only KB.** Only billing tickets clear the retrieval floor, so the agent sends billing replies and flags everything else; this, not the pipeline, is what holds action accuracy at 0.205.
- **The action metric rewards sending, not grounded correctness.** Variants without grounding checks (no-KB, single-shot) score higher by sending unsupported answers; the comparison only becomes fair once the KB covers the other intents.
- **Priority accuracy is still modest (0.384).** The model often rates medium/high tickets as `low`, and the dataset's own priority labels are noisy. Escalation now catches all critical tickets but adds 4 false escalations, and the critical labels themselves are keyword-derived.
- **Weak intent labels:** `complaint` and `feature_request` remain the hardest, partly because the source tone is neutral B2B.
- **Response-quality metric:** ROUGE-L rewards word overlap and penalises valid paraphrases; the references contain placeholders.
- **Small evaluation set:** with 73 tickets, one ticket ≈ 0.014 on any metric, so differences of one or two tickets are within run-to-run variation; repeated runs or a larger set would tighten the conclusions.
- **The retrieval floor (0.3) is tuned to the current KB** and must be re-tuned as documents are added.

**Future work**
1. **Expand the KB** (account, technical, product and general FAQs; mine resolutions from non-golden Kaggle tickets only, to avoid leakage), re-tune the retrieval floor on the dev set, and re-measure. This is the biggest remaining lever.
2. Make the live prompt-injection probe a strict test and add more injection variants.
3. Hand-label a small set of critical tickets to validate escalation independently of the keyword-derived labels.
4. Human-review pass over the Kaggle rows (mark corrected rows `source=hand`; the builder preserves them).
5. Grounded-correctness metric: score actions as correct only when a sent reply is supported by policy, or add an LLM-as-judge alongside ROUGE-L.
6. Confidence calibration study (reliability of self-rated confidence vs actual correctness).
7. Few-shot examples (from the dev pool) for the weakest intent labels.
8. Grow the golden set to 100+ rows with more critical and complaint examples; repeat runs to report variance.

---

## 13. Running the Project

### 13.1 Setup

```bash
git clone <repo-url> && cd support-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# API key: export it in your shell, or put it in .env (shell wins; .env is gitignored)
export OPENROUTER_API_KEY=...
cp .env.example .env

streamlit run app/streamlit_app.py        # indexes the KB on first run
```

### 13.2 Commands

| Command | Purpose |
|---|---|
| `pytest` | Unit/integration tests, no API calls |
| `RUN_LIVE_TESTS=1 pytest tests/test_live.py -v` | Live end-to-end tests (a few cents of tokens) |
| `python -m src.knowledge.vector_store` | Rebuild the KB index after editing `data/knowledge_base/` |
| `python -m src.evaluation.build_golden_set` | Regenerate the Kaggle rows of the golden set |
| `python -m src.evaluation.build_golden_set --dev` | Write the non-overlapping dev (tuning) set |
| `python -m src.evaluation.evaluator --golden-set data/golden_set/dev_qa.csv` | Evaluate on the dev set while tuning |
| `python -m src.evaluation.evaluator` | Score the full agent on the golden set |
| `python -m src.evaluation.evaluator --ablation` | Run all four ablation variants side by side |
| `python -m src.evaluation.evaluator --workers N` | Concurrent tickets per variant (default 8) |

Download the raw dataset:

```bash
python -c "import kagglehub; kagglehub.dataset_download('tobiasbueck/multilingual-customer-support-tickets', output_dir='data/raw/customer-support-tickets')"
```

### 13.3 Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `openrouter` | `openrouter`, `groq` or `openai` |
| `OPENROUTER_API_KEY` | — | Required for OpenRouter |
| `OPENROUTER_MODEL` | `openai/gpt-oss-120b` | Model on OpenRouter |
| `GROQ_API_KEY` / `OPENAI_API_KEY` | — | Only for those providers |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Model for Groq/OpenAI |
| `COMPANY_NAME` | `Acme Cloud` | Name used in drafted replies |
| `CONFIDENCE_THRESHOLD` | `0.7` | Gate 2 send threshold |
| `MIN_RETRIEVAL_SCORE` | `0.3` | Gate 2 also requires the best KB match to reach this (re-tune when the KB grows) |
| `LLM_TIMEOUT` | `60` | Hard total deadline per LLM request (seconds) |
| `LLM_MAX_RETRIES` | `2` | Retries per LLM request |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | Vector index location |

---

## 14. Project Structure

```
support-agent/
├── app/
│   └── streamlit_app.py           # Streamlit UI
├── config/
│   └── settings.py                # All configuration (env-driven)
├── data/
│   ├── golden_set/golden_qa.csv   # 73-ticket evaluation (test) set
│   ├── golden_set/dev_qa.csv      # 68-ticket tuning set, no overlap with the golden set
│   ├── knowledge_base/billing_faq.md
│   └── raw/                       # Kaggle dataset (gitignored)
├── docs/
│   ├── architecture.svg           # Pipeline diagram
│   └── PROJECT_REPORT.md          # This report
├── src/
│   ├── llm.py                     # Provider-agnostic LLM client factory
│   ├── agent/
│   │   ├── state.py               # TicketState + new_ticket_state()
│   │   ├── graph.py               # LangGraph pipeline + decision gates
│   │   └── single_shot.py         # One-call ablation baseline
│   ├── tools/
│   │   ├── intent_classifier.py   # Tool 1
│   │   ├── priority_scorer.py     # Tool 2
│   │   ├── kb_retriever.py        # Tool 3
│   │   └── response_drafter.py    # Tool 4
│   ├── knowledge/
│   │   └── vector_store.py        # ChromaDB wrapper + chunking
│   └── evaluation/
│       ├── evaluator.py           # Metrics + ablation runner
│       └── build_golden_set.py    # Kaggle → golden set labelling
├── tests/                         # Automated tests + opt-in live suite
├── pytest.ini                     # Registers the `live` marker
├── requirements.txt
├── .env.example
└── README.md
```

---

## 15. Team & Process

### 15.1 Roles

| Role | Responsibilities |
|---|---|
| **Technical members (2–3)** | Agent pipeline (`graph.py`); the four tools; vector store; evaluation code |
| **Other members (2)** | Knowledge-base curation; golden test set; Streamlit UI layout; running ablation experiments; documentation, demo slides and final report |

### 15.2 Process

- **Sprint plan:** a 7-day Gantt chart maps every task to a day and an owner (linked from the README).
- **Version control:** feature work on `feature/rag-drafter-evaluation`; commits pushed to GitHub with descriptive messages.
- **Quality gates:** tests accompany every module; live runs verify behaviour against the real model; automated security review on commits.

### 15.3 Commit history (feature branch)

| Commit | Summary |
|---|---|
| `b4edf70` | Intent classifier and priority scorer with structured output |
| `8294b38` | KB retrieval, response drafter, evaluation harness, Kaggle golden set, UI |
| `a7d2633` | Render LLM output in the UI as plain text (security fix) |
| `f679988` | OpenRouter as the default LLM provider |
| `42faf04` | Concurrent evaluator, per-request deadlines, thread-safe store init, live test suite |
| `031080c` | Improvement round: severity-based priority scoring, intent definitions, drafter role separation, retrieval floor, dev set, this report |
| `ac70000` | Merge of the two work streams; confirmation ablation (§8.4) |

---

## 16. Conclusion

The project delivers a working, tested, end-to-end support-triage agent. Tickets are classified, prioritised, grounded in company policy and either answered, escalated or flagged, with every step visible in a reasoning trace. The live tests confirm all three routes behave correctly on the real model.

Beyond the agent itself, the project's main contribution is **measurement discipline**: a reproducible golden set, an ablation framework that isolates each component's value, and integrity checks (`n_degraded`) that caught two invalid runs before their numbers could mislead. The project also shows the full loop that discipline makes possible. The baseline ablation diagnosed two concrete defects: a priority gate that escalated none of the 9 critical tickets because it waited for angry sentiment, and intent-label confusion. Fixes were developed on a separate dev set and measured once on the golden set. They lifted escalation from **0/9 to 9/9**, intent accuracy from 0.726 to 0.767 and action accuracy from 0.123 to 0.205. The same round turned the open prompt-injection finding into a two-layer mitigation. The ablation also kept the analysis honest: variants that send ungrounded answers score higher on action accuracy, so the full agent's caution is a feature of the design, not a fault. With escalation, intent handling and grounding checks in place, **expanding the knowledge base beyond billing** is now the clearest path to a production-ready system.

---

## 17. References

1. LangGraph documentation, LangChain, Inc. — https://langchain-ai.github.io/langgraph/
2. LangChain structured output — https://python.langchain.com/
3. ChromaDB — https://www.trychroma.com/
4. Reimers & Gurevych (2019), *Sentence-BERT*; model `sentence-transformers/all-MiniLM-L6-v2`
5. OpenRouter API and structured outputs — https://openrouter.ai/docs/features/structured-outputs
6. OpenAI, `gpt-oss-120b` open-weight model — https://openrouter.ai/openai/gpt-oss-120b
7. Bueck, T., *Customer IT Support – Ticket Dataset*, Kaggle (CC BY-NC 4.0) — https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets
8. Bitext, *Customer Support LLM Chatbot Training Dataset* — https://github.com/bitext/customer-support-llm-chatbot-training-dataset
9. Lin, C.-Y. (2004), *ROUGE: A Package for Automatic Evaluation of Summaries*, ACL Workshop
10. Streamlit documentation — https://docs.streamlit.io/

---

## Appendices

### Appendix A: Golden-set schema

| Column | Values |
|---|---|
| `ticket_text` | Customer message (subject + body) |
| `expected_intent` | One of the 6 intent labels |
| `expected_priority` | low / medium / high / critical |
| `expected_action` | send / escalate / flag_for_review |
| `reference_response` | Reference agent reply (`N/A - escalated to human agent` for escalations) |
| `source` | `hand` (human-written) or `kaggle` (rule-labelled) |

### Appendix B: Sample golden rows

| Ticket (abridged) | Intent | Priority | Action |
|---|---|---|---|
| "I was charged twice for my subscription this month…" | billing | high | send |
| "How do I reset my password? I forgot it and can't log in." | account_access | low | send |
| "Your product is absolute garbage. I'm contacting my lawyer…" | complaint | critical | escalate |
| "Would be great if you could add dark mode to the mobile app." | feature_request | low | send |
| "My dashboard has been showing a loading error for 3 days…" | technical_issue | high | send |

### Appendix C: Example reasoning trace (live run)

```
1. Classified intent: billing (0.98)
2. Priority: medium (sentiment: -0.40)
3. Retrieved 3 docs from KB
4. Drafted response (confidence: 0.95)
5. SENT: confidence 0.95 at or above threshold
```

### Appendix D: Response-drafter prompt template

```
You are a customer support agent for {COMPANY_NAME}.

Company policy documents:
[1] {chunk 1}
[2] {chunk 2}
[3] {chunk 3}

Ticket intent: {intent}
Ticket priority: {priority}
Customer message: "{ticket_text}"

Write a reply to the customer. Be helpful and empathetic, and base any policy
details (timelines, eligibility, steps) ONLY on the documents above. Never invent
policy. If the documents don't cover the issue, say a specialist will follow up.

Then rate your confidence from 0.0 to 1.0 that this reply fully resolves the
customer's issue using only the given documents.
```
