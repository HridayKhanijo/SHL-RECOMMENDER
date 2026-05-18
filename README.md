\---

title: SHL Recommender

emoji: 🎯

colorFrom: blue

colorTo: green

sdk: docker

pinned: false

\---



\# SHL Assessment Recommender

SHL Assessment Recommender

A production-quality conversational agent that recommends SHL Individual Test Solutions
through multi-turn dialogue. Built for the SHL Labs AI Intern take-home assignment.

## Architecture

```
POST /chat
    │
    ▼
Intent Classifier (LLM, fast call)
    │
    ├─ clarify\\\_needed  → Ask ONE clarifying question
    ├─ recommend       → Retrieve + Recommend (FAISS + BM25 → RRF → LLM)
    ├─ refine          → Re-retrieve + Update shortlist
    ├─ compare         → Fetch two catalog items → LLM comparison
    ├─ off\\\_topic       → Polite refusal
    └─ injection       → Rejection

    ▼
Hallucination Firewall (URL allowlist validation)
    ▼
Pydantic Schema Enforcement
    ▼
Response
```

## Quick Start

### 1\. Install dependencies

```bash
python -m venv venv \\\&\\\& source venv/bin/activate
pip install -r requirements.txt
```

### 2\. Configure environment

```bash
cp .env.example .env
# Edit .env — add your GROQ\\\_API\\\_KEY (free at console.groq.com)
```

### 3\. Scrape the SHL catalog

```bash
python scripts/scrape\\\_catalog.py
# Produces data/shl\\\_catalog.json
```

### 4\. Run locally

```bash
uvicorn app.main:app --reload --port 8000
```

### 5\. Test

```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \\\\
  -H "Content-Type: application/json" \\\\
  -d '{"messages": \\\[{"role": "user", "content": "I need to hire a Java developer"}]}'
```

### 6\. Evaluate

```bash
python tests/evaluate.py --base-url http://localhost:8000
```

## Deployment (Render)

1. Push to GitHub
2. Create new **Web Service** on render.com → connect repo
3. Set environment variables (GROQ\_API\_KEY, etc.) in Render dashboard
4. Set **Start Command**: `sh -c "python scripts/scrape\\\_catalog.py \\\&\\\& uvicorn app.main:app --host 0.0.0.0 --port $PORT"`
5. Set **Disk**: mount at `/app/data`, 1 GB (for FAISS index persistence)

Alternatively, use the Dockerfile for Railway or Fly.io.

## API Reference

### GET /health

```json
{"status": "ok"}
```

### POST /chat

**Request:**

```json
{
  "messages": \\\[
    {"role": "user", "content": "Hiring a Java developer, 4 years exp"},
    {"role": "assistant", "content": "What level of seniority?"},
    {"role": "user", "content": "Mid-level"}
  ]
}
```

**Response:**

```json
{
  "reply": "Here are 3 assessments for a mid-level Java developer:",
  "recommendations": \\\[
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test\\\_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test\\\_type": "P"}
  ],
  "end\\\_of\\\_conversation": false
}
```

**Rules:**

* `recommendations` is `\\\[]` while clarifying or refusing
* `recommendations` has 1–10 items when the agent recommends
* `end\\\_of\\\_conversation: true` only after a complete shortlist is given

## Project Structure

```
shl-recommender/
├── app/
│   ├── main.py              # FastAPI app + lifespan
│   ├── api/
│   │   ├── routes.py        # /health + /chat endpoints
│   │   └── schemas.py       # Pydantic request/response models
│   ├── core/
│   │   └── index.py         # FAISS index build/load
│   ├── retrieval/
│   │   └── retriever.py     # Hybrid BM25 + semantic + RRF
│   └── agent/
│       ├── chat\\\_agent.py    # Intent routing + orchestration
│       ├── llm\\\_client.py    # LLM API wrapper (Groq/Gemini/OpenAI)
│       └── prompts.py       # All prompt templates
├── scripts/
│   └── scrape\\\_catalog.py    # SHL catalog scraper
├── tests/
│   └── evaluate.py          # Recall@10 + behavior probes harness
├── data/                    # Generated: shl\\\_catalog.json, faiss\\\_index.bin
├── docs/
│   └── approach\\\_document.md
├── requirements.txt
├── Dockerfile
└── .env.example
```

## Evaluation Metrics

|Metric|Target|
|-|-|
|Schema compliance|100%|
|Mean Recall@10|≥ 0.70|
|Behavior probe pass rate|≥ 0.80|
|Avg response time|< 10s|
|Max turns honored|Always|

Run `python tests/evaluate.py` to check all metrics locally before submission.

