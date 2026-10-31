# Know-It

Know-It is a self-corrective retrieval-augmented generation (RAG) assistant for technical documentation. A user asks a natural-language question, the system improves the search query, retrieves relevant documentation chunks, grades each chunk for relevance, and generates an answer grounded in the accepted evidence. Answers include source references. If the indexed corpus cannot support an answer, the system returns an explicit insufficient-context response.

The required application phases are implemented: ingestion, the self-corrective LangGraph workflow, FastAPI routes, persistence, retries, checkpoints, observability, and an evaluation harness. Live corpus seeding and end-to-end Gemini evaluation require GOOGLE_API_KEY; deterministic tests cover the graph and API paths without using model credits. The complete rationale and acceptance criteria are in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

## What the project does

The finished baseline will:

- ingest Markdown, text, or HTML documentation from an uploaded file or URL;
- split technical content into heading-aware chunks and index Gemini embeddings in ChromaDB;
- analyze and rewrite questions for better retrieval;
- grade every retrieved chunk before allowing it into the generation context;
- retry weak retrieval with a bounded query-rewrite loop;
- generate answers with validated references to indexed source chunks;
- abstain when the available documents do not contain enough evidence;
- expose query, ingestion, document-listing, and feedback operations through FastAPI.

The initial corpus will contain five focused pages from the official FastAPI documentation. This deliberately narrow scope makes retrieval quality and abstention behavior easy to evaluate.

## Architecture

~~~mermaid
flowchart LR
    Client[API client] --> API[FastAPI API]

    API --> QueryService[Query service]
    API --> IngestionService[Ingestion service]
    API --> FeedbackRepository[(SQLite answers and feedback)]

    QueryService --> Graph[LangGraph StateGraph]
    Graph --> GeminiChat[Gemini chat model]
    Graph --> VectorStore[(ChromaDB)]

    IngestionService --> Loader[File and URL loaders]
    Loader --> Chunker[Heading-aware chunker]
    Chunker --> GeminiEmbeddings[Gemini embeddings]
    GeminiEmbeddings --> VectorStore
~~~

The HTTP layer owns validation and status codes. Application services coordinate use cases. LangGraph owns query state and routing. Provider and storage clients are injected at application startup so each component can be tested independently.

### Query workflow

~~~mermaid
flowchart TD
    Start([Question]) --> Analyze[Analyze and expand query]
    Analyze --> Retrieve[Retrieve top-k chunks]
    Retrieve --> Grade[Grade every chunk]
    Grade --> Decision{Relevant chunks?}
    Decision -->|Yes| Generate[Generate grounded answer]
    Generate --> Cite[Validate source references]
    Cite --> Done([Answer with sources])
    Decision -->|No, retry budget remains| Rewrite[Rewrite search query]
    Rewrite --> Retrieve
    Decision -->|No, retry budget exhausted| Abstain[Return insufficient context]
    Abstain --> Done
~~~

Semantic retries and operational retries are separate. A semantic retry rewrites an unproductive query and is limited to two rewrites. Temporary provider or storage failures use their own bounded retry policy and do not consume the semantic rewrite budget.

## Technology choices

| Area | Tool | Role |
| --- | --- | --- |
| API | FastAPI and Pydantic | HTTP endpoints, validation, and OpenAPI |
| Workflow | LangGraph | Typed state, nodes, conditional routing, and retries |
| Generation and grading | Google Gemini | Query analysis, relevance grading, and answers |
| Embeddings | Gemini Embeddings | Vector representations for documents and queries |
| Vector search | ChromaDB | Persistent local similarity search |
| Operational data | SQLite | Answer IDs, feedback, and graph checkpoints |
| Tests and quality | pytest and Ruff | Deterministic tests, linting, and formatting |

Persistent local Chroma and SQLite keep the take-home straightforward to run. Their boundaries are isolated so a server-backed vector store and database can replace them for a multi-instance deployment.

## Implementation status

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Project foundation, settings, API contracts, health endpoint | Complete |
| 1 | Corpus, safe loaders, chunking, Gemini embeddings, Chroma ingestion | Implemented; live seed pending API key |
| 2 | LangGraph query analysis, retrieval, grading, rewrite, generation | Implemented; live run pending API key |
| 3 | Required FastAPI routes and SQLite answer/feedback persistence | Complete |
| 4 | Retry policies, checkpoints, observability, operating limits | Complete |
| 5 | Automated tests and retrieval/answer evaluation | Harness complete; live run pending API key |
| 6 | Final setup guide, API examples, and submission review | Complete; live completion gate pending |
| Bonus | Groundedness check, memory, web fallback, UI | Deferred |

## Current project structure

~~~text
app/
  api.py          # required HTTP routes and error translation
  bootstrap.py    # dependency construction
  checkpoints.py  # LangGraph SQLite lifecycle and retention
  chunking.py     # heading-aware, token-bounded splitting
  cli.py          # corpus seed, inspect, and list commands
  config.py       # validated environment configuration
  documents.py    # ingestion domain models
  errors.py       # application-level errors
  evaluation.py   # versioned evaluation runner and report models
  ingestion.py    # ingestion use-case coordinator
  loaders.py      # bounded file and public URL loading
  main.py         # FastAPI application factory and health endpoint
  observability.py # request IDs and operational logging
  prompts.py      # versioned prompts for each model responsibility
  providers.py    # Gemini embedding and structured chat adapters
  query_workflow.py # typed LangGraph nodes, edges, and query service
  rag.py          # query workflow domain and structured-output contracts
  reliability.py  # transient failure classification and retry policy
  repository.py   # SQLite answer and feedback persistence
  schemas.py      # public API contracts
  vector_store.py # versioned Chroma document index
corpus/
  seed_urls.json  # five official FastAPI documentation pages
evaluation/
  questions.json  # 15 supported and abstention evaluation cases
tests/            # deterministic unit and integration tests
.env.example      # documented configuration
pyproject.toml    # dependencies and development tools
~~~

The structure will grow only as each implementation phase introduces a new responsibility.

## Development setup

Requirements:

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- a Gemini API key for live ingestion and query phases

Install the locked dependencies:

~~~bash
uv sync
~~~

Create local configuration:

~~~bash
cp .env.example .env
~~~

On PowerShell:

~~~powershell
Copy-Item .env.example .env
~~~

Set GOOGLE_API_KEY in .env. The application can start without a key for local contract tests, but live model and embedding operations will require it.

Run the current API:

~~~bash
uv run uvicorn app.main:app --reload
~~~

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

1. Gemini produces a structured retrieval query and optional query type.
2. The query is embedded and Chroma returns the top five chunks.
3. Gemini grades each chunk independently; irrelevant chunks are removed.
4. If nothing is relevant, Gemini creates a materially different query. The graph permits two rewrites and stops early when a rewrite repeats an earlier query.
5. Relevant chunks are bounded to the configured context limit and passed to generation.
6. The graph accepts the answer only when inline source markers and structured citation IDs match approved chunks.
7. If the rewrite budget is exhausted, it returns an insufficient-context response with no sources.

The original question remains unchanged throughout the graph. Retrieval results, grades, and relevant chunks are replaced on every loop, preventing evidence from an earlier failed search from leaking into generation.

## Reliability and state management

Semantic retries and operational retries are tracked separately:

- A semantic retry rewrites a query only when every retrieved chunk is irrelevant. Two rewrites allow at most three retrieval passes.
- LangGraph retries external nodes only for network failures, timeouts, rate limits, temporary provider failures, and transient SQLite lock contention. Invalid input, authentication failures, schema errors, and programming errors are not retried.
- Gemini calls use provider-level request timeouts. The installed LangGraph version supports graph node timeouts only for asynchronous nodes, while this workflow intentionally uses synchronous nodes behind FastAPI's thread pool.

Every API query uses a new LangGraph thread ID. Intermediate states are stored in a dedicated SQLite checkpoint database, separate from answers and feedback. Completed threads are registered for retention cleanup and removed after seven days by default. Checkpoints support inspection and failure analysis; they are not reused as conversation memory.

Each HTTP response includes an X-Request-ID header. Logs contain the request ID, method, path, status, latency, graph node, result counts, rewrite count, and citation count. Questions, document contents, credentials, and API keys are not written to operational logs.

Resource limits cover question length, retrieval count, grading concurrency, generation context, upload bytes, fetched bytes, redirects, provider attempts, and provider timeout. All values are configurable through the environment.

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
      "excerpt": "FastAPI provides the Depends function..."
    }
  ],
  "rewrite_count": 0
}
~~~

The server stores the answer before returning its ID, allowing later feedback to reference a real response.

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

Similarity search provides inexpensive candidate selection, while Gemini performs the semantic relevance decision required by the assignment. Grading every top-k chunk costs more than score-threshold filtering, so top-k and grading concurrency are bounded.

### Heading-aware chunks

Technical documentation depends on section hierarchy and nearby code. The splitter retains heading paths, keeps ordinary code blocks intact, and applies token overlap when a long section requires multiple chunks. The default target is 700 tokens with an 80-token overlap.

### Deterministic document versions

Document IDs derive from canonical sources, and chunk IDs include the document content hash. Re-ingesting unchanged content skips embedding. Changed content is embedded under new IDs before old chunks are removed, reducing the chance of losing a working version during replacement.

### Local persistence behind narrow interfaces

Chroma stores document vectors, one SQLite database stores answers and feedback, and another stores graph checkpoints. This is easy to run and explain for the take-home. A multi-instance deployment would replace them with server-backed services while retaining the service and graph boundaries.

### Citation validation in application code

The generator receives only graded chunks labeled S1, S2, and so on. Application code verifies that inline markers and structured citation IDs match and belong to that exact context. Unknown or inconsistent citations fail instead of being returned to the user.

## Assumptions and current limitations

- The initial knowledge base is intentionally limited to five English FastAPI pages.
- Uploaded documents must be UTF-8 Markdown, plain text, or HTML.
- Local Chroma and SQLite target a single application instance.
- Changing the embedding model or dimension requires rebuilding the collection because vector spaces are incompatible.
- Retrieval is dense semantic search only; keyword or hybrid retrieval could improve exact error-code and symbol searches.
- Feedback is stored for later analysis but does not automatically change prompts or ranking.
- Live quality and latency numbers remain pending until the corpus is seeded with a configured Gemini key.

## Improvements with more time

After the required live completion gate, the next improvements would be:

1. add a groundedness verifier that checks answer claims against cited chunks;
2. evaluate hybrid lexical and vector retrieval plus reranking;
3. move Chroma, checkpoints, and feedback to shared production services;
4. add authenticated ingestion and per-corpus access controls;
5. add conversation memory with explicit session retention;
6. add a small UI after the API behavior remains stable.

Web fallback and conversation memory remain bonus work and are intentionally outside the required baseline.

## Submission checklist

- Install: uv sync --locked
- Configure: copy .env.example to .env and set GOOGLE_API_KEY
- Inspect corpus: uv run python -m app.cli inspect
- Seed corpus: uv run python -m app.cli seed
- Start API: uv run uvicorn app.main:app --reload
- Run deterministic tests: uv run pytest
- Run live smoke test: set RUN_LIVE_TESTS=1, then uv run pytest -m live
- Run evaluation: uv run python -m app.cli evaluate
- Review interactive API docs: http://127.0.0.1:8000/docs
