# Reliable RAG API

A small, production-style **knowledge assistant** that answers questions from your documents,
**shows its evidence**, and **says "I don't know"** instead of inventing facts.

**Stack:** FastAPI · LlamaIndex · ChromaDB · Groq (free LLM API) · local FastEmbed embeddings · pytest · Docker

---

## Table of contents
1. [Quick start](#1-quick-start)
2. [API reference](#2-api-reference)
3. [Architecture](#3-architecture)
4. [How reliability is enforced](#4-how-reliability-is-enforced)
5. [Architecture decisions & trade-offs](#5-architecture-decisions--trade-offs)
6. [Evaluation](#6-evaluation)
7. [Tests](#7-tests)
8. [Limitations](#8-limitations)
9. [Configuration](#9-configuration)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Quick start

### Prerequisites
* A **free Groq API key** → <https://console.groq.com/keys>
* Either **Docker** *or* **Python 3.11+**

### Option A – Docker (recommended)
```bash
cp .env.example .env            # Windows: copy .env.example .env
# edit .env and paste your key:  GROQ_API_KEY=gsk_...
docker compose up --build
```
The first build downloads the embedding model (~90 MB) into the image. When the log shows
`Application startup complete`, open **<http://localhost:8000/docs>** (interactive Swagger UI).
The 4 sample documents are indexed automatically (`INGEST_SAMPLES_ON_STARTUP=true`); vector data
persists in a Docker volume.

### Option B – Local Python
```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cp .env.example .env             # then put your GROQ_API_KEY in it
uvicorn app.main:app --reload
```
Open <http://localhost:8000/docs>. (First start downloads the embedding model once.)

### Try it
```bash
# health + what's indexed
curl localhost:8000/health
curl localhost:8000/documents

# upload your own documents (PDF / TXT / MD, several at once)
curl -X POST localhost:8000/documents -F "files=@my_notes.pdf" -F "files=@policy.md"

# answerable question
curl -s -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "How many days of paid annual leave do employees get?"}' | python -m json.tool

# unanswerable question -> honest refusal
curl -s -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "Who is the CEO of Helio Systems?"}' | python -m json.tool
```

**Windows PowerShell** (the `curl` lines above do not work there because of quoting):
```powershell
Invoke-RestMethod -Uri http://localhost:8000/ask -Method Post -ContentType "application/json" `
  -Body '{"question": "How many days of paid annual leave do employees get?"}' | ConvertTo-Json -Depth 5
```

Example answerable response:
```json
{
  "question": "How many days of paid annual leave do employees get?",
  "answer": "Employees receive 24 days of paid annual leave per calendar year. [1]",
  "answerable": true,
  "reason": "answered",
  "sources": [
    {
      "document": "employee_handbook.md",
      "page": null,
      "excerpt": "Every full-time employee receives 24 days of paid annual leave per calendar year.",
      "similarity": 0.5429,
      "chunk_id": "employee_handbook.md::0000",
      "quote_verified": true
    }
  ],
  "top_similarity": 0.5429
}
```
Example unanswerable response: `"answerable": false`, `"reason": "no_relevant_context"` or `"not_in_documents"`,
`"sources": []`, and an answer saying the documents don't contain the information.

### Run the evaluation and tests
```bash
python -m evaluation.run_eval --ingest        # needs the API running + GROQ_API_KEY (see §6)
python -m pytest -q                           # offline, no API key needed (see §7)
# in Docker:
docker compose exec api python -m evaluation.run_eval --ingest
docker compose exec api python -m pytest -q
```

---

## 2. API reference

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness + number of indexed docs/chunks, whether the LLM key is configured |
| `POST` | `/documents` | Upload one or more `.pdf` / `.txt` / `.md` files (multipart field `files`). Per-file results returned |
| `GET` | `/documents` | List indexed documents and chunk counts |
| `DELETE` | `/documents/{file_name}` | Remove a document and all its chunks |
| `POST` | `/ask` | `{"question": "...", "top_k": 5 (optional)}` → grounded answer + sources |

**Answer fields:** `answerable` (bool) · `reason` (`answered`, `no_documents`, `no_relevant_context`,
`not_in_documents`, `ungrounded_answer`) · `sources[]` (`document`, `page`, `excerpt`, `similarity`,
`chunk_id`, `quote_verified`).

**Error format** (all non-2xx from the app): `{"error": "<code>", "message": "<human text>"}`

| Status | Code | When |
|---|---|---|
| 400 | – (`results[].error_code`) | every uploaded file failed (unsupported type, empty, corrupt, too large) |
| 404 | `document_not_found` | deleting an unknown document |
| 422 | validation | empty/short/too-long question, bad `top_k`, no files |
| 502 | `llm_unavailable` / `llm_invalid_response` | Groq down / rate-limited / model not available / returned unusable output |
| 503 | `llm_not_configured` | `GROQ_API_KEY` missing |

---

## 3. Architecture

```
                 ┌──────────────────────── Ingestion ────────────────────────┐
 PDF/TXT/MD ───► │ loader.py ─► chunker.py ─► embeddings.py ─► ChromaDB       │
 POST /documents │ (pypdf,      (LlamaIndex    (FastEmbed       (cosine HNSW,  │
                 │  per page)    SentenceSplit) MiniLM-L6)       persistent)   │
                 └─────────────────────────────────────────────────────────────┘

                 ┌───────────────────────── Question answering ─────────────────────────┐
 POST /ask ────► │ embed question ─► top-k search ─► similarity gate ─► Groq LLM (JSON)  │
                 │                                   (no LLM call if    ─► citation check │
                 │                                    nothing relevant)  ─► quote check   │
                 └──────────────────────────────────────────► answer + sources / "unavailable"
```

```
reliable-rag-api/
├── app/
│   ├── main.py                  # app factory, lifespan, error handlers, request logging
│   ├── config.py                # all settings from env vars (pydantic-settings)
│   ├── dependencies.py          # singleton providers (swapped out in tests)
│   ├── api/routes.py            # HTTP layer only – no business logic
│   ├── core/exceptions.py       # typed domain errors -> HTTP status + code
│   ├── ingestion/
│   │   ├── loader.py            # PDF/TXT/MD parsing + validation
│   │   └── chunker.py           # sentence-aware chunking, deterministic chunk ids
│   ├── retrieval/
│   │   ├── embeddings.py        # FastEmbed (real) / hash (offline test) embeddings
│   │   └── vector_store.py      # ChromaDB wrapper: add / query / list / delete
│   ├── generation/
│   │   ├── llm.py               # Groq client factory
│   │   └── prompts.py           # system + user prompt templates
│   ├── services/
│   │   ├── ingestion_service.py # load -> chunk -> embed -> store (idempotent)
│   │   └── qa_service.py        # retrieve -> gate -> LLM -> validate -> respond
│   └── schemas/{requests,responses}.py
├── data/sample_documents/       # employee_handbook.md, it_security_policy.txt, product_faq.md, onboarding_guide.pdf
├── evaluation/{questions.json, run_eval.py}
├── scripts/ingest_samples.py
├── tests/                       # test_ingestion / test_retrieval / test_qa / test_api (+ conftest)
├── Dockerfile · docker-compose.yml · requirements.txt · .env.example
```
Compared with the minimal layout in the brief, I added `core/` (typed errors), `generation/` (LLM + prompts kept
out of the service), `dependencies.py` (dependency injection so tests can run with fakes), `schemas/responses.py`,
and `scripts/`. Layers only depend downward: `routes → services → (retrieval, generation, ingestion)`.

---

## 4. How reliability is enforced

Hallucination is handled by **independent layers**, so no single failure produces an invented answer:

1. **Similarity gate.** Chunks below `MIN_SIMILARITY` (cosine) are dropped before prompting. If none remain, the
   API replies "unavailable" **without calling the LLM** (`reason: no_relevant_context`) – off-topic questions
   like "capital of France?" can't be hallucinated because the model never sees them.
2. **Grounding prompt.** The LLM may use only the numbered passages, must treat them as untrusted data
   (prompt-injection hardening), must say `answerable: false` when unsure, and must answer partially-supported
   questions by stating what is missing. Temperature is 0.
3. **Structured output.** The model returns JSON (`answerable`, `answer`, `citations[{source, quote}]`).
   Invalid JSON triggers one corrective retry, then a clean `502` – never a silent guess.
4. **Citation validation.** Citations must point to passages that were actually retrieved. An "answer" with no
   valid citation is **rejected** (`reason: ungrounded_answer`) and converted into "unavailable".
5. **Quote verification.** Each quoted excerpt is checked to be a verbatim substring of the cited chunk
   (case/punctuation-insensitive). Verified → shown as the excerpt with `quote_verified: true`; otherwise the
   raw chunk text is shown as evidence instead and `quote_verified: false` – the user can see the difference.
6. **Fail loudly on infrastructure problems.** LLM timeouts, rate limits, bad keys, unavailable models →
   `502/503` with a message, not a fake answer. Bad uploads are rejected before they can modify the index.

---

## 5. Architecture decisions & trade-offs

| Decision | Why | Trade-off |
|---|---|---|
| **Groq `openai/gpt-oss-120b`** | Free tier, very fast, strong instruction/JSON following | Free tier is rate limited (the eval runner has `--delay`); it is a reasoning model, so it needs a larger `LLM_MAX_TOKENS`; model is changeable via `GROQ_MODEL` |
| **Local FastEmbed (`all-MiniLM-L6-v2`) for embeddings** | Groq has no embeddings API; local ONNX = no extra key, no torch (small image), free | Smaller model than commercial embedders; English-centric. Swap via `EMBEDDING_MODEL` |
| **ChromaDB, persistent, cosine HNSW** | Zero-ops, embedded, persistent on disk/volume | Single-node; for multi-instance scale use a server vector DB (Qdrant/pgvector) |
| **LlamaIndex for loading/splitting/embedding/indexing/LLM; direct Chroma query** | LlamaIndex's Chroma adapter returns `exp(-distance)` as "score"; I query Chroma directly to get a real cosine similarity that can be thresholded and explained | Slightly less "framework-pure", more transparent |
| **Chunk size 200 tokens / overlap 30** | MiniLM only reads ~256 tokens – bigger chunks would be silently truncated when embedded. Small chunks also give precise excerpts | More chunks; multi-hop answers rely on `top_k` retrieving several of them |
| **Per-page PDF documents** | Gives page numbers in citations | Scanned (image-only) PDFs have no text → rejected with a clear error (no OCR) |
| **Deterministic chunk ids + content hash** | Re-uploading an unchanged file is a no-op; a changed file *replaces* its old chunks – ingestion is idempotent, no duplicates | File name is the document identity |
| **LLM judges "answerable" + similarity gate** | Gate is cheap and catches clearly off-topic questions; the LLM handles subtle "related but not stated" cases | Threshold needs tuning per embedding model/corpus (see §6) |
| **Sync endpoints (`def`)** | Embedding/LLM calls are blocking; FastAPI runs them in a threadpool without blocking the event loop | Not async-native; fine at this scale |
| **Dependency injection + `hash` embedding + fake LLM** | Whole test-suite runs offline in seconds, deterministically, with no API key | Tests prove the pipeline logic, not Groq's answer quality – that's the job of `evaluation/` |

---

## 6. Evaluation

`evaluation/questions.json` holds **16 questions** over the sample documents:

| Category | Count | Purpose |
|---|---|---|
| `answerable` | 6 | single-fact lookups across the TXT / MD / PDF files |
| `multi_evidence` | 5 | need several chunks and/or several documents (e.g. training deadline from the PDF **and** frequency from the security policy) |
| `unanswerable` | 5 | not in the documents (stock options, on-prem deployment, CEO, dental insurance) or off-topic (capital of France) |

Each question records `expected_sources` and `expected_keywords` (`"a|b"` = either accepted).

**Run it** against a live API (needs your Groq key):
```bash
python -m evaluation.run_eval --ingest --delay 3
```
Reported metrics (also saved to `evaluation/results.json`):
* **decision_accuracy** – answered when it should, abstained when it should
* **keyword_pass** – answers contain the expected facts
* **source_recall** – all expected documents are cited
* **hallucination** – unanswerable questions that were nevertheless answered (target 0%)

**Tuning `MIN_SIMILARITY`:** if answerable questions get `no_relevant_context`, lower it (e.g. 0.20); if off-topic
questions reach the LLM, raise it (e.g. 0.30). The LLM's own `answerable` check is the second line of defence.

### Results

Run on the 16-question set with `GROQ_MODEL=openai/gpt-oss-120b` (`LLM_MAX_TOKENS=4096`), local FastEmbed
`all-MiniLM-L6-v2` embeddings, `TOP_K=5` and `MIN_SIMILARITY=0.25`:

| Metric | Result |
|---|---|
| decision_accuracy | 16/16 (100%) |
| keyword_pass | 11/11 (100%) |
| source_recall | 11/11 (100%) |
| hallucination | 0/5 (0%) |
| request_errors | 0 |

Off-topic questions (e.g. "capital of France") were stopped by the similarity gate (`no_relevant_context`) without
an LLM call. Related-but-missing questions (e.g. stock options, CEO, on-premise deployment) were refused by the LLM
(`not_in_documents`). Per-question answers and sources are in `evaluation/results.json`.

> **Caveat:** the question set is small (16) and was written together with the sample documents, so treat this as a
> regression baseline, not a general accuracy estimate.

An offline **retrieval-recall** check (is every expected document in the top-5 chunks?) *is* part of the pytest suite
and runs for all 11 answerable questions.

---

## 7. Tests

```bash
python -m pytest -q
```
Fully offline: deterministic hash embeddings, a scripted fake LLM and a temporary ChromaDB directory.

| File | Covers |
|---|---|
| `test_ingestion.py` | TXT/MD/PDF loading (page numbers), unsupported/empty/corrupt/oversized files, path-traversal names, chunk ids/overlap, idempotent re-ingest, replace-on-change, failed ingest leaves index untouched |
| `test_retrieval.py` | ranking, cosine scores sorted, `top_k`, delete, persistence across restarts, unrelated-query scores, embedding determinism, **retrieval recall on the evaluation set** |
| `test_qa.py` | grounded answers + quote verification, multi-source answers, LLM-says-unanswerable, similarity gate (**LLM not called**), empty KB, no-citation rejection, hallucinated citation index, fenced JSON, retry-on-bad-JSON, persistent bad JSON, provider failure |
| `test_api.py` | every endpoint, multi-file upload, partial failures, 400/404/422/502/503 paths |

**Result of the last run (Python 3.12):**
```
65 passed (about 6 s natively, about 50 s inside Docker Desktop on Windows)
```
A harmless Starlette deprecation warning about `httpx` in the test client may be shown; it does not affect results.

---

## 8. Limitations

* **Small evaluation set.** Results (§6) come from 16 questions written together with the sample documents and a
  single run; LLM output can vary between runs. Good for regression testing, not a general accuracy estimate.
  The automated tests use a fake LLM, so they verify pipeline logic, not Groq's answer quality.
* **Threshold sensitivity.** `MIN_SIMILARITY=0.25` is a sensible starting point for MiniLM but was not tuned on a
  large corpus.
* **No OCR.** Image-only PDFs are rejected. Tables and multi-column PDFs may extract in a jumbled order.
* **Dense retrieval only.** No keyword (BM25) hybrid search or re-ranker, so exact identifiers/codes can be missed
  and wide "list everything" questions are limited by `top_k`.
* **Single-turn.** No conversation memory; each question is independent.
* **Documents are a flat namespace**, keyed by file name; no per-user isolation, auth or rate limiting on the API.
* **Groq free tier** has request/token limits; bursts return `502 llm_unavailable`. Available model names can also
  change over time.
* **Quote check is textual**, not semantic: it proves the quote exists in the chunk, not that it fully supports every
  claim in the answer.
* **Embedded Chroma** is single-process; run one API worker per data volume.

Natural next steps: hybrid BM25 + dense retrieval, a cross-encoder re-ranker, an LLM-as-judge faithfulness metric,
API-key auth, streaming responses, OCR fallback.

---

## 9. Configuration

All via environment variables / `.env` (see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | – | **Required** for `/ask` |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Any Groq chat model available to your key |
| `LLM_TEMPERATURE` / `LLM_TIMEOUT_SECONDS` / `LLM_MAX_RETRIES` | `0` / `60` / `2` | LLM call behaviour |
| `LLM_MAX_TOKENS` | `4096` | Output token budget. Reasoning models like `gpt-oss-120b` need headroom or the JSON answer can be cut off |
| `EMBEDDING_PROVIDER` | `fastembed` | `fastembed` (real) or `hash` (offline/tests) |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | FastEmbed model name |
| `CHROMA_PERSIST_DIR` / `CHROMA_COLLECTION` | `./storage/chroma` / `documents` | Vector DB location |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `200` / `30` | Tokens per chunk / overlap |
| `TOP_K` | `5` | Chunks retrieved per question (max 20 via `MAX_TOP_K`) |
| `MIN_SIMILARITY` | `0.25` | Similarity gate |
| `MAX_UPLOAD_MB` | `20` | Per-file upload limit |
| `INGEST_SAMPLES_ON_STARTUP` | `false` (`true` in `.env.example`) | Auto-index `data/sample_documents` |

> **Changing the embedding model/provider?** Delete the vector store first (`docker compose down -v`, or remove
> `./storage/chroma`) – vectors from different models are not comparable.
>
> **Changing `.env`?** Settings are read at startup – restart the server (`docker compose up -d --force-recreate`).

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `/ask` → 503 `llm_not_configured` | Set `GROQ_API_KEY` in `.env` and restart |
| `/ask` → 502 `llm_unavailable` | Invalid key, no internet, or Groq rate limit – check the server log; retry shortly |
| `/ask` → 502 with `model_not_found` | The model isn't available to your key. List your models (`GET https://api.groq.com/openai/v1/models` with your key) and set `GROQ_MODEL` |
| `/ask` → 502 `llm_invalid_response` | Output was cut off or not valid JSON – increase `LLM_MAX_TOKENS` (e.g. 8192) |
| Everything answers "no_relevant_context" | Lower `MIN_SIMILARITY`; confirm documents are indexed (`GET /documents`) |
| First start is slow | One-time embedding-model download (pre-baked in the Docker image) |
| `docker compose` complains about `.env` | Run `cp .env.example .env` first (Windows: `copy .env.example .env`) |
| Switched embedding model, results are garbage | Wipe the vector store (see note in §9) |
| `pytest` / `httpx` "not recognized" or "No module named ..." | Activate your virtualenv, run `python -m pip install -r requirements.txt`, then `python -m pytest -q` |
| PowerShell: `curl` examples fail with JSON errors | Use `Invoke-RestMethod` (see §1) – PowerShell strips the quotes from `curl` bodies |