# Approach Document
## SHL Conversational Assessment Recommender
**AI Intern Take-Home — SHL Labs**

---

## 1. Design Choices

### Architecture
I chose a **stateless FastAPI service with a lightweight intent-routing agent** rather than a multi-agent framework (no LangChain/LangGraph). The rationale:

- **Reliability over cleverness.** Complex agent frameworks add hidden failure modes. A simple `classify intent → retrieve → prompt → validate` pipeline is easier to debug, test, and defend in interviews.
- **Schema compliance is non-negotiable.** Every LLM response passes through a Pydantic validator and a URL-allowlist firewall before leaving the service. Hallucinated URLs are caught and either corrected or dropped.
- **Stateless by design.** Full conversation history is passed on every `/chat` call. No session database, no TTL management — the service can scale horizontally with zero shared state.

### Intent Routing
Each `/chat` call runs a **two-phase pipeline**:

1. **Intent Classifier** (fast LLM call, ~0.5s): classifies the user message as `clarify_needed | recommend | refine | compare | off_topic | injection`. A heuristic pre-filter catches obvious injection and off-topic signals without any LLM cost.
2. **Action Handler**: routes to the appropriate prompt template and retrieval query based on intent.

This avoids "mega-prompts" that degrade instruction-following and allows each prompt to be optimised independently.

---

## 2. Retrieval Setup

**Hybrid retrieval = Semantic (FAISS) + Keyword (BM25) fused via Reciprocal Rank Fusion (RRF).**

- **Semantic layer**: `sentence-transformers/all-MiniLM-L6-v2` embeds each catalog entry as `name | type_label | description`. FAISS `IndexFlatIP` on L2-normalised vectors gives cosine similarity. Chosen because it is tiny (80 MB), fast on CPU, and achieves strong recall on short HR queries.
- **Keyword layer**: `rank_bm25` over the same concatenated text. Critical for exact-match queries like "Java 8" or "OPQ32r".
- **RRF fusion**: `score(d) = Σ 1/(k + rank(d))`, k=60. Parameter-free; robust to score-scale mismatches between the two systems.
- **Type-filter boost**: When the user mentions a test type (e.g., "personality"), items of that type get a +0.2 RRF score boost rather than a hard filter, allowing graceful degradation.

The index is built once at startup from the scraped catalog JSON and persisted to disk. Container restarts are fast (disk load, no re-embedding).

---

## 3. Prompt Design

**Key principles:**
- **Grounded prompting**: The LLM is given catalog excerpts verbatim and explicitly told it may only recommend items from those excerpts. The system prompt states this three times in different phrasings.
- **Low temperature (0.1)**: Near-deterministic output reduces hallucination and schema drift.
- **JSON-only output**: The system prompt instructs the model to return only valid JSON. We use `response_format: json_object` where the provider supports it (Groq, OpenAI).
- **Separate prompt per intent**: Each of the five intents (clarify, recommend, refine, compare, refuse) has its own prompt template. This prevents instruction bleed.

**Prompt injection defence:**
- Regex pattern matching catches obvious attacks before any LLM call.
- The system prompt explicitly instructs the model to ignore instructions embedded in user messages.
- The URL allowlist ensures injected fake recommendations never reach the response.

---

## 4. Evaluation Approach

**Local eval harness** (`tests/evaluate.py`) runs 10 conversation traces before deployment:

- **Hard evals**: Schema compliance checked every turn. URL allowlist verified per recommendation.
- **Recall@10**: Substring-matched against expected assessment names. Threshold target: ≥ 0.7 mean Recall@10.
- **Behavior probes**: (a) vague query → no recs on turn 1, (b) injection → empty recs + refusal, (c) off-topic → empty recs, (d) refinement → updated shortlist preserving valid prior recs.

**What didn't work / iterations:**
- *Single mega-prompt*: Initial design used one prompt that classified and responded in one call. Instruction-following degraded on refine/compare intents. Splitting into intent-classifier + action-handler improved schema compliance by ~30%.
- *Hard type filtering*: Filtering catalog by type before embedding caused zero-result errors for niche types. Replaced with boost scoring.
- *Exact URL matching only*: LLMs sometimes generate URLs with minor formatting differences. Added name-based lookup fallback to correct URLs rather than drop recommendations.

**AI tools used**: Claude (architecture review, code generation), Groq `llama-3.3-70b-versatile` (LLM inference). All code was reviewed, tested, and modified by the author.

---

## 5. Stack Justification

| Component | Choice | Reason |
|-----------|--------|--------|
| Framework | FastAPI | Async, Pydantic-native, OpenAPI docs |
| Embeddings | all-MiniLM-L6-v2 | Small, fast, good HR-domain recall |
| Vector store | FAISS CPU | No server, works in any container |
| BM25 | rank_bm25 | Pure Python, no Elasticsearch needed |
| LLM | Groq llama-3.3-70b | Free tier, fast, OpenAI-compatible |
| Deployment | Hugging Face Spaces (Docker) | Free tier, persistent disk for index |

---

*Total implementation: ~900 lines of Python across 8 files. Test coverage: 10 conversation traces + schema validation on every response.*
