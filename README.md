# Know-It

Know-It is a self-corrective retrieval-augmented generation (RAG) assistant for technical documentation. A user asks a natural-language question, the system improves the search query, retrieves relevant documentation chunks, grades each chunk for relevance, and generates an answer grounded in the accepted evidence. Answers include source references. If the indexed corpus cannot support an answer, the system returns an explicit insufficient-context response.

The project is being implemented phase by phase. The application foundation and Phase 1 ingestion pipeline are implemented. Live corpus seeding requires GOOGLE_API_KEY; the loaders and chunker have already been validated against all five source pages without using model credits. The complete rationale and acceptance criteria are in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

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
| 2 | LangGraph query analysis, retrieval, grading, rewrite, generation | Next |
| 3 | Required FastAPI routes and feedback persistence | Planned |
| 4 | Retry policies, checkpoints, observability, operating limits | Planned |
| 5 | Automated tests and retrieval/answer evaluation | Planned |
| 6 | Final setup guide, API examples, and submission review | Planned |
| Bonus | Groundedness check, memory, web fallback, UI | Deferred |

## Current project structure

~~~text
app/
  bootstrap.py    # dependency construction
  chunking.py     # heading-aware, token-bounded splitting
  cli.py          # corpus seed, inspect, and list commands
  config.py       # validated environment configuration
  documents.py    # ingestion domain models
  errors.py       # application-level errors
  ingestion.py    # ingestion use-case coordinator
  loaders.py      # bounded file and public URL loading
  main.py         # FastAPI application factory and health endpoint
  providers.py    # Gemini embedding adapter
  schemas.py      # public API contracts
  vector_store.py # versioned Chroma document index
corpus/
  seed_urls.json  # five official FastAPI documentation pages
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

## Required API surface

These routes will be implemented in Phase 3 after ingestion and graph services exist:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| POST | /query | Submit a question and receive an answer with sources |
| POST | /ingest | Ingest a file or URL |
| GET | /documents | List indexed documents |
| POST | /feedback | Record thumbs up/down feedback and an optional comment |

GET /health is already available as an operational endpoint.
