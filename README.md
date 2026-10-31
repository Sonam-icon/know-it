# Know-It

Know-It is a self-corrective retrieval-augmented generation (RAG) assistant for technical documentation. It resolves a question into a standalone search query, retrieves and grades evidence, generates a cited draft, and verifies the draft against that evidence before returning it. It supports bounded follow-up memory, an optional allowlisted Tavily fallback, and a small Streamlit client. If neither indexed nor web evidence can support an answer, it returns an explicit insufficient-context response.

The required phases and all four bonus features are implemented. Live corpus seeding and end-to-end Gemini/Tavily evaluation still require credentials; deterministic tests cover the graph, API, persistence, web adapter, and UI client without spending model or search credits. The rationale and acceptance criteria are in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

## What the project does

The application can:

- ingest Markdown, text, or HTML documentation from an uploaded file or URL;
- split technical content into heading-aware chunks and index Gemini embeddings in ChromaDB;
- analyze and rewrite questions for better retrieval;
- grade every retrieved chunk before allowing it into the generation context;
- retry weak retrieval with a bounded query-rewrite loop;
- generate answers with validated references to indexed source chunks;
- verify generated claims and revise a failed draft once before abstaining;
- resolve follow-up questions from a bounded, persisted session history;
- search allowlisted official documentation through Tavily after corpus retries are exhausted;
- abstain when the available documents do not contain enough evidence;
- expose query, ingestion, document-listing, and feedback operations through FastAPI;
- provide an optional Streamlit chat and feedback interface.

The initial corpus will contain five focused pages from the official FastAPI documentation. This deliberately narrow scope makes retrieval quality and abstention behavior easy to evaluate.

## Architecture

~~~mermaid
flowchart LR
    Client[API client or Streamlit] --> API[FastAPI API]

    API --> QueryService[Query service]
    API --> IngestionService[Ingestion service]
    API --> Repository[(SQLite answers, sessions, and feedback)]

    QueryService --> Graph[LangGraph StateGraph]
    Graph --> GeminiChat[Gemini chat model]
    Graph --> VectorStore[(ChromaDB)]
    Graph -. optional fallback .-> Tavily[Tavily official-doc search]

    IngestionService --> Loader[File and URL loaders]
    Loader --> Chunker[Heading-aware chunker]
    Chunker --> GeminiEmbeddings[Gemini embeddings]
    GeminiEmbeddings --> VectorStore
~~~

The HTTP layer owns validation and status codes. Application services coordinate use cases. LangGraph owns query state and routing. Provider and storage clients are injected at application startup so each component can be tested independently.

### Query workflow

~~~mermaid
flowchart TD
    Start([Question plus bounded history]) --> Analyze[Resolve standalone question and search query]
    Analyze --> Retrieve[Retrieve top-k chunks]
    Retrieve --> Grade[Grade every chunk]
    Grade --> Decision{Relevant chunks?}
    Decision -->|Yes| Generate[Generate cited draft]
    Generate --> Cite[Validate source references]
    Cite -->|Valid| Verify[Verify claims against evidence]
    Cite -->|Invalid, one repair remains| Generate
    Verify -->|Supported| Done([Answer with sources])
    Verify -->|Unsupported, revision remains| Revise[Revise draft once]
    Revise --> Cite
    Verify -->|Still unsupported| Abstain
    Decision -->|No, retry budget remains| Rewrite[Rewrite search query]
    Rewrite --> Retrieve
    Decision -->|No, retry budget exhausted and web enabled| Web[Search Tavily once]
    Web --> WebGrade[Grade web excerpts]
    WebGrade -->|Relevant| Generate
    WebGrade -->|Irrelevant| Abstain[Return insufficient context]
    Decision -->|No web fallback| Abstain
    Abstain --> Done
~~~

Semantic rewrites, answer revision, and operational retries have independent counters. Every loop and external fallback is bounded.

## Technology choices

| Area | Tool | Role |
| --- | --- | --- |
| API | FastAPI and Pydantic | HTTP endpoints, validation, and OpenAPI |
| Workflow | LangGraph | Typed state, nodes, conditional routing, and retries |
| Generation and grading | Google Gemini | Query analysis, relevance grading, and answers |
| Embeddings | Gemini Embeddings | Vector representations for documents and queries |
| Vector search | ChromaDB | Persistent local similarity search |
| Web fallback | Tavily | Optional allowlisted official-documentation search |
| Operational data | SQLite | Answers, bounded session history, and feedback |
| UI | Streamlit | Optional API-only chat client |
| Tests and quality | pytest and Ruff | Deterministic tests, linting, and formatting |

Persistent local Chroma and SQLite keep the take-home straightforward to run on one machine.

## Implementation status

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Project foundation, settings, API contracts, health endpoint | Complete |
| 1 | Corpus, safe loaders, chunking, Gemini embeddings, Chroma ingestion | Implemented; live seed pending API key |
| 2 | LangGraph query analysis, retrieval, grading, rewrite, generation | Implemented; live run pending API key |
| 3 | Required FastAPI routes and SQLite answer/feedback persistence | Complete |
| 4 | Bounded retries, fresh graph state, and local limits | Complete |
| 5 | Automated tests and retrieval/answer evaluation | Harness complete; live run pending API key |
| 6 | Final setup guide, API examples, and submission review | Complete; live completion gate pending |
| Bonus | Groundedness check, session memory, Tavily fallback, Streamlit UI | Implemented; live provider check pending credentials |

## Current project structure

~~~text
app/
  api.py          # required HTTP routes and error translation
  bootstrap.py    # dependency construction
  chunking.py     # heading-aware, token-bounded splitting
  cli.py          # corpus seed, inspect, and list commands
  config.py       # validated environment configuration
  documents.py    # ingestion domain models
  errors.py       # application-level errors
  evaluation.py   # versioned evaluation runner and report models
  ingestion.py    # ingestion use-case coordinator
  loaders.py      # bounded file and public URL loading
  main.py         # FastAPI application factory and health endpoint
  prompts.py      # versioned prompts for each model responsibility
  providers.py    # Gemini embedding and structured chat adapters
  query_workflow.py # typed LangGraph nodes, edges, and query service
  rag.py          # query workflow domain and structured-output contracts
  reliability.py  # transient failure classification and retry policy
  repository.py   # SQLite answer, session-history, and feedback persistence
  schemas.py      # public API contracts
  ui.py           # optional Streamlit interface
  ui_client.py    # small HTTP client used by Streamlit
  vector_store.py # versioned Chroma document index
  web_search.py   # bounded, allowlisted Tavily adapter
corpus/
  seed_urls.json  # five official FastAPI documentation pages
evaluation/
  questions.json  # 15 supported and abstention evaluation cases
tests/            # deterministic unit and integration tests
.env.example      # documented configuration
pyproject.toml    # dependencies and development tools
~~~

Each module represents a step or boundary that can be explained independently. The graph itself lives in one file.

## Development setup

Requirements:

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- a Gemini API key for live ingestion and query phases
- optionally, a Tavily key to enable official-documentation fallback

Install the locked dependencies:

~~~bash
uv sync --locked
~~~

Create local configuration:

~~~bash
cp .env.example .env
~~~

On PowerShell:

~~~powershell
Copy-Item .env.example .env
~~~

Set `GOOGLE_API_KEY` in `.env`. To enable web fallback, also set `TAVILY_API_KEY` and change `WEB_FALLBACK_ENABLED=true`. The application can start without either key for local contract tests; live ingestion and queries require Gemini. Chroma runs locally at `CHROMA_PATH`, so this design does not require a Chroma API key.

Run the current API:

~~~bash
uv run uvicorn app.main:app --reload
~~~

Install and launch the optional UI in a second terminal:

~~~bash
uv sync --locked --extra ui
uv run --extra ui streamlit run app/ui.py
~~~

The UI calls `http://127.0.0.1:8000` by default. Override it with `KNOW_IT_API_URL`; Gemini and Tavily credentials remain only in the API process.

Check application health:

~~~bash
curl http://127.0.0.1:8000/health
~~~

Expected response:

~~~json
{
  "status": "ok",
  "version": "0.1.0",
  "provider_configured": true
}
~~~

Run quality checks:

~~~bash
uv run ruff check .
uv run pytest
~~~

## Corpus ingestion

Inspect the real corpus without calling Gemini or writing to Chroma:

~~~bash
uv run python -m app.cli inspect
~~~

Once GOOGLE_API_KEY is configured, load and embed all five seed pages:

~~~bash
uv run python -m app.cli seed
~~~

The seed operation is idempotent. An unchanged document reuses its indexed chunks without another embedding call. A changed document is embedded under versioned chunk IDs before stale chunks are removed.

List the documents currently stored in Chroma:

~~~bash
uv run python -m app.cli list
~~~

Phase 1 accepts UTF-8 Markdown, plain text, and HTML. URL loading permits public HTTP and HTTPS destinations, validates every redirect, caps download size and redirects, removes page navigation, and retains headings, lists, tables, paragraphs, and code blocks. Chunks target 700 tokens with an 80-token overlap for oversized sections.

## Query workflow

After seeding the corpus, run a question through the LangGraph workflow:

~~~bash
uv run python -m app.cli query "How do FastAPI dependencies work?"
~~~

The graph performs these steps:

1. Gemini resolves the current message and up to three prior completed turns into a standalone question and retrieval query.
2. The query is embedded and Chroma returns the top five chunks.
3. Gemini grades each chunk independently; irrelevant chunks are removed.
4. If nothing is relevant, Gemini creates a materially different query. The graph permits two rewrites and stops early when a rewrite repeats an earlier query.
5. Relevant chunks are bounded to the configured context limit and passed to generation.
6. The graph accepts the draft only when inline source markers and structured citation IDs match approved chunks. A malformed structured draft or citation contract gets one corrected generation attempt; a second failure returns 502.
7. Gemini verifies the valid draft against the same evidence. One unsupported draft may be revised and verified again; a second verification failure abstains.
8. After corpus rewrites are exhausted, an enabled Tavily adapter searches the configured official domains once. Web excerpts are graded and follow the same generation, citation, and verification path.
9. If no path has relevant, supporting evidence, the graph returns an insufficient-context response with no sources.

The original question remains unchanged throughout the graph. Retrieval results, grades, and relevant chunks are replaced on every loop, preventing evidence from an earlier failed search from leaking into generation.

## Reliability and state management

The harness has four separate bounded recovery mechanisms:

- A semantic retry rewrites a query only when every retrieved chunk is irrelevant. Two rewrites allow at most three retrieval passes.
- A generation repair retries once with the exact structured-output or citation-validation error; the invalid draft is never published.
- A groundedness correction revises an unsupported generated draft once; it does not consume retrieval retries.
- LangGraph retries external nodes only for network failures, timeouts, rate limits, temporary provider failures, and transient SQLite lock contention. Invalid input, authentication failures, schema errors, and programming errors are not retried.
- Gemini calls use provider-level request timeouts. The installed LangGraph version supports graph node timeouts only for asynchronous nodes, while this workflow intentionally uses synchronous nodes behind FastAPI's thread pool.

LangGraph keeps state in memory for one query invocation and compiles without a checkpointer. SQLite supplies only the last three completed turns, under a character budget, to the analysis node. History helps resolve a follow-up but is never generation evidence; every turn must retrieve and grade current documentation. This keeps graph execution stateless while still supporting local single-instance sessions.

Simple logs record graph steps, result counts, rewrite count, and citation count. Questions, document contents, credentials, and API keys are not written to operational logs.

Resource limits cover question length, retrieval count, generation context, upload bytes, fetched bytes, redirects, provider attempts, and provider timeout. Grading runs sequentially over the small top-k set.

## Evaluation

The versioned dataset contains 15 questions:

- direct API and conceptual questions from all five corpus pages;
- how-to and troubleshooting questions;
- questions that combine multiple parameter concepts;
- WebSocket, OAuth2, and background-task questions that should abstain.

After seeding the corpus, run:

~~~bash
uv run python -m app.cli evaluate
~~~

The command writes data/evaluation.json and reports pass rate and average latency. Each case checks the expected answer status, source presence, expected documentation URL, required answer terms, and citation availability.

Routine tests use deterministic model and vector-store fakes:

~~~bash
uv run pytest
~~~

The live smoke test is opt-in because it calls Gemini and consumes quota:

~~~bash
RUN_LIVE_TESTS=1 uv run pytest -m live
~~~

On PowerShell:

~~~powershell
$env:RUN_LIVE_TESTS = "1"
uv run pytest -m live
~~~

It ingests a temporary Markdown document with Gemini embeddings and asks a question that must return a cited answer.

## API

All required routes are registered in FastAPI and appear in the generated documentation at http://127.0.0.1:8000/docs.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| POST | /query | Submit a question and receive an answer with sources |
| POST | /ingest | Ingest a file or URL |
| GET | /documents | List indexed documents |
| POST | /feedback | Record thumbs up/down feedback and an optional comment |

GET /health is also available as an operational endpoint. Without GOOGLE_API_KEY, health and document listing remain available while query and ingestion return HTTP 503.

### Submit a question

~~~bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"How do FastAPI dependencies work?"}'
~~~

Example response:

~~~json
{
  "answer_id": "93b9c7a6-aa54-4e98-bf00-b3877bc9f185",
  "session_id": "4ef42a80-e0ce-419c-98e5-991b48d092a6",
  "answer": "Declare a dependency with Depends and a callable. [S1]",
  "status": "answered",
  "sources": [
    {
      "citation_id": "S1",
      "chunk_id": "chunk-hash",
      "document_id": "document-hash",
      "title": "Dependencies",
      "heading": "Dependencies / Import Depends",
      "source": "https://fastapi.tiangolo.com/tutorial/dependencies/",
      "source_kind": "url",
      "origin": "corpus",
      "excerpt": "FastAPI provides the Depends function..."
    }
  ],
  "rewrite_count": 0,
  "web_attempted": false
}
~~~

The server stores the answer before returning its IDs. Send the returned `session_id` with the next question to enable follow-up resolution:

~~~bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is its default behavior?","session_id":"4ef42a80-e0ce-419c-98e5-991b48d092a6"}'
~~~

Omitting `session_id` starts a new session; an unknown supplied session returns 404. A source has `origin: "web"` only when the optional Tavily path supplied its evidence.

### Ingest a URL or file

The endpoint accepts multipart form data with exactly one source:

~~~bash
curl -X POST http://127.0.0.1:8000/ingest \
  -F "url=https://fastapi.tiangolo.com/tutorial/dependencies/"
~~~

~~~bash
curl -X POST http://127.0.0.1:8000/ingest \
  -F "file=@docs/guide.md;type=text/markdown"
~~~

Example response:

~~~json
{
  "document_id": "document-hash",
  "title": "Dependencies",
  "source": "https://fastapi.tiangolo.com/tutorial/dependencies/",
  "source_kind": "url",
  "chunk_count": 6,
  "status": "created"
}
~~~

### List documents

~~~bash
curl http://127.0.0.1:8000/documents
~~~

### Submit feedback

~~~bash
curl -X POST http://127.0.0.1:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"answer_id":"93b9c7a6-aa54-4e98-bf00-b3877bc9f185","rating":"up","comment":"Useful answer"}'
~~~

Feedback for an unknown answer returns HTTP 404. Invalid request bodies return 422, oversized uploads return 413, and unsupported file formats return 415.

## Design decisions and tradeoffs

### Explicit graph instead of a general agent

The assignment has a known sequence and one controlled loop. Explicit nodes and conditional edges make state transitions, retry counts, and abstention behavior visible and testable. A free-form agent would add tool-selection behavior that is unnecessary for this use case.

### LLM grading after vector retrieval

Similarity search provides candidate selection, while Gemini performs the semantic relevance decision required by the assignment. Grading every top-k chunk costs more than score-threshold filtering, so top-k is bounded. Sequential grading keeps the flow simple to inspect.

### Heading-aware chunks

Technical documentation depends on section hierarchy and nearby code. The splitter retains heading paths, keeps ordinary code blocks intact, and applies token overlap when a long section requires multiple chunks. The default target is 700 tokens with an 80-token overlap.

### Deterministic document versions

Document IDs derive from canonical sources, and chunk IDs include the document content hash. Re-ingesting unchanged content skips embedding. Changed content is embedded under new IDs before old chunks are removed, reducing the chance of losing a working version during replacement.

### Local persistence

Chroma stores document vectors, while SQLite stores answers and feedback. LangGraph state is temporary and scoped to one query. This is easy to run and explain for a single-instance take-home project.

### Citation validation in application code

The generator receives only graded chunks labeled S1, S2, and so on. Application code verifies that inline markers and structured citation IDs match and belong to that exact context. Unknown or inconsistent citations fail instead of being returned to the user.

### Groundedness as a graph gate

Citation syntax alone does not prove a claim is supported. A separate verifier sees the draft and the exact approved evidence. It can finalize, request one focused revision, or force abstention. The revision cap prevents an open-ended model loop and makes the additional cost visible.

### Bounded memory without LangGraph checkpoints

The API persists completed answer records in the existing SQLite database and passes a small history window only to query analysis. This is easier to explain than durable graph checkpoints and avoids treating previous model answers as factual evidence. It is intentionally a single-instance conversation feature rather than a distributed chat system.

### Optional, restricted web fallback

Web search occurs only after the local corpus exhausts its rewrite budget. Tavily is limited by result count, response bytes, excerpt size, and an official-domain allowlist. Returned hosts are checked again locally, results are not written into Chroma, and web provenance is exposed in every source.

## Assumptions and current limitations

- The initial knowledge base is intentionally limited to five English FastAPI pages.
- Uploaded documents must be UTF-8 Markdown, plain text, or HTML.
- Local Chroma and SQLite target a single application instance.
- Changing the embedding model or dimension requires rebuilding the collection because vector spaces are incompatible.
- Retrieval is dense semantic search only; keyword or hybrid retrieval could improve exact error-code and symbol searches.
- Feedback is stored for later analysis but does not automatically change prompts or ranking.
- Session ordering is designed for a local single-instance demo, not concurrent requests to the same session.
- Tavily snippets are less reproducible than indexed documents, so fallback is disabled by default and web sources are labeled.
- Live quality and latency numbers remain pending until the corpus is seeded with a configured Gemini key.

## Improvements with more time

The completed [bonus implementation](IMPLEMENTATION_PLAN.md#10-bonus-implementation) covers groundedness verification, bounded session memory, Tavily fallback, and Streamlit. Further improvements outside this take-home scope include hybrid retrieval, answer-quality telemetry, a dedicated bonus evaluation set, authenticated multi-user ingestion, and transactional ordering for concurrent turns.

## Submission checklist

- Install: uv sync --locked
- Configure: copy .env.example to .env and set GOOGLE_API_KEY; Tavily is optional
- Inspect corpus: uv run python -m app.cli inspect
- Seed corpus: uv run python -m app.cli seed
- Start API: uv run uvicorn app.main:app --reload
- Run deterministic tests: uv run pytest
- Run live smoke test: set RUN_LIVE_TESTS=1, then uv run pytest -m live
- Run evaluation: uv run python -m app.cli evaluate
- Review interactive API docs: http://127.0.0.1:8000/docs
- Optional UI: uv run --extra ui streamlit run app/ui.py
