# How I built Know-It

I built Know-It as a small technical documentation assistant. I chose five pages from the official FastAPI documentation so I could demonstrate retrieval, citations, and honest answers when the corpus lacks evidence. I used Gemini for chat and embeddings, local ChromaDB for vector search, LangGraph for the question workflow, FastAPI for the API, and SQLite for feedback and short conversation history. I put the setup commands, API examples, and full architecture diagram in the [README](README.md).

## The architecture I chose

I kept each responsibility in one place: ingestion prepares documents, Chroma stores and retrieves chunks, LangGraph decides what to do with those chunks, and FastAPI handles HTTP input and output. I construct provider and storage clients in `app/bootstrap.py` and pass them into the services. I can therefore test the graph with fake clients without calling Gemini.

I designed the main path as:

```text
Question → Analyze → Retrieve → Grade
                                ├─ relevant → Generate → Verify → Answer
                                └─ none → Rewrite → Retrieve (up to twice)
                                           └─ budget exhausted → optional Tavily → Grade
                                                                    └─ Generate or Abstain
```

I keep the graph's state in memory for one question. I store only completed answer records and a small history window in SQLite; I do not persist the graph's internal steps.

## Phase 0 — I set the foundation

I defined dependencies and a lockfile in `pyproject.toml` and `uv.lock`. I put settings and limits in `app/config.py`, example values in `.env.example`, and request and response models in `app/schemas.py`. I keep credentials in an ignored `.env` file. I allow one Gemini key or a comma-separated pool, which my provider adapters use in round-robin order.

I started with these contracts so the API, graph, and storage code would agree on what a question, source, answer, and feedback record contain.

## Phase 1 — I ingested the documentation

I listed the five FastAPI pages in `corpus/seed_urls.json`. My ingestion path loads a file or public URL, extracts useful text, and preserves headings and nearby code. I split by heading first, then target 700-token chunks with 80 tokens of overlap when a section is too long. I chose this size to keep each passage focused while leaving enough room for an explanation and its code example. I use overlap to protect context when an answer spans two chunks.

I use `gemini-embedding-001` for both document chunks and search queries, with its document and query retrieval task types. I store the resulting 3,072-dimensional vectors in local Chroma. I use one embedding model to keep documents and queries in a comparable vector space. I use stable document IDs and content hashes so an unchanged page is not embedded again. I record the model and dimension with the collection because changing either requires a new index. I use the same ingestion service for the seed command and `POST /ingest`.

## Phase 2 — I built the LangGraph workflow

I implemented the graph in `app/query_workflow.py`. I use typed state to hold the original question, current search query, retrieved chunks, relevance grades, approved chunks, rewrite count, draft, verification result, and final answer. I replace retrieval results on every pass so a failed search cannot leak old evidence into generation.

I made query analysis produce a standalone question and search query. I retrieve the top five Chroma chunks, then ask Gemini to grade each one. If at least one chunk is relevant, I generate only from those approved chunks. If none is relevant, I route through a conditional edge to rewrite the query and retrieve again. I allow two rewrites, so a question has at most three local retrieval passes. If evidence is still missing, I abstain or try the optional web fallback.

I separate analysis from retrieval so a follow-up question can stand on its own. I grade after vector search because semantic similarity alone does not prove that a chunk answers the question. I keep generation separate so it receives only chunks that passed grading.

I validate citation IDs and inline markers against the approved chunks before returning an answer. I also check the draft against its evidence and allow one revision when the verifier finds unsupported claims. I treat malformed model output as an error, while missing evidence produces `insufficient_context`.

## Phase 3 — I exposed the API

I added the four required routes in `app/api.py`: `POST /query`, `POST /ingest`, `GET /documents`, and `POST /feedback`. I use Pydantic models for validation and map application errors to meaningful HTTP status codes. I store answer IDs and feedback in SQLite so feedback refers to an answer that was actually returned. I also added `GET /health` for local checks.

## Phase 4 — I bounded failures and work

I separated two kinds of retry. I rewrite a query after irrelevant results and increment `rewrite_count`. I use LangGraph node retries for temporary provider or network failures without changing that count. I set limits for provider attempts and timeouts, question and context size, uploads, URL downloads, and graph recursion. I surface permanent permission or validation errors as service errors rather than treating them as missing evidence.

## Phase 5 — I tested the behavior

I used pytest with fake model and storage adapters to test ingestion, graph routing, retries, citations, API responses, memory, and web fallback without spending API credits. I added a separate live smoke test for Gemini ingestion and a cited answer. I also created 15 evaluation questions in `evaluation/questions.json`, including questions that should be answered and questions that should cause abstention.

I report the evaluation pass rate and latency through `app/evaluation.py`. I check expected status, sources, URLs, answer terms, and citation availability. I do not present these checks as retrieval precision, recall, MRR, nDCG, or independently measured faithfulness; I would need labeled chunks or claim-level judgments for those metrics.

## Phase 6 — I documented the submission

I made the README the main guide for reviewers. There I explain the stack and graph, setup and API keys, local Chroma storage, seed and run commands, example requests and responses, decisions, tradeoffs, limitations, and future improvements. I kept this file as a short account of why I built the pieces in this order.

## Bonus features I added after the core workflow

1. I added a groundedness check after generation. I use it to accept an answer, request one revision, or make the graph abstain.
2. I added follow-up questions with an optional `session_id`. I pass at most three completed turns to query analysis, then retrieve fresh evidence for every answer.
3. I added an optional Tavily search after local rewrites fail. I restrict results to configured official domains, grade the excerpts, and label any resulting source as `web`.
4. I added a small Streamlit interface that calls the FastAPI routes for questions, citations, sessions, and feedback. I keep Gemini, Chroma, and SQLite access in the backend.

## Why I made these choices

I chose an explicit graph over a general agent because the assignment has a known path and one important decision loop. I chose local Chroma and SQLite because they are easy to run and inspect for a single-instance demonstration. I accept extra model calls for chunk grading and answer verification because they make the evidence decisions visible. I kept the corpus to five pages so it is easy to evaluate, although I need Tavily to cover questions outside those pages.

## Assumptions I made

I assume a small English-language documentation corpus and one local API instance. I use Gemini credentials supplied through `.env`, keep Tavily disabled unless configured, and treat uploaded or fetched documentation as untrusted data. I do not assume that a valid citation automatically means every claim is true; that is why I added answer verification.

## What I would improve with more time

I would label relevant chunks and measure how often the retriever finds them and ranks them near the top. I would review answers claim by claim against the source passages instead of relying only on an LLM verifier. I would also combine keyword and vector search for exact API names and add stronger controls for multi-user ingestion.

## What I have verified

I have indexed the five pages locally. I passed the deterministic tests, lint checks, and a one-document live Gemini smoke test. I have not recorded a full 15-case live evaluation or completed live Tavily and UI checks.
