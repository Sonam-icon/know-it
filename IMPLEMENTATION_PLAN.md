# Implementation plan: technical documentation assistant

## Implementation status

- **Phase 0 — Complete:** project metadata, locked dependencies, validated settings, public API schemas, application factory, health endpoint, and baseline tests.
- **Phase 1 — Implemented:** reproducible corpus, safe loaders, heading-aware chunks, Gemini embedding adapter, versioned Chroma storage, and corpus CLI. Live seeding awaits GOOGLE_API_KEY in the runtime environment.
- **Phase 2 — Implemented:** typed LangGraph state, analysis, vector retrieval, concurrent per-chunk grading, bounded rewrites, grounded generation, citation validation, and abstention. A live Gemini run awaits GOOGLE_API_KEY and a seeded index.
- **Phase 3 — Complete:** required query, ingestion, document-listing, and feedback routes; typed responses; HTTP error mapping; SQLite answer and feedback persistence; and OpenAPI contract tests.
- **Phase 4 — Complete:** narrow transient retries, provider timeouts, unique checkpoint threads, SQLite checkpoint retention, request correlation, structured graph metrics, and configurable resource limits.
- **Phase 5 — Harness complete:** deterministic tests, a 15-case evaluation dataset, report generation, and an opt-in live ingestion/query smoke test. Live results await GOOGLE_API_KEY.
- **Phase 6 — Complete:** final setup, architecture, API examples, design tradeoffs, assumptions, limitations, improvement path, and submission checklist. The live completion gate remains pending.
- **Bonus work — Deferred:** begins only after the required completion gate passes.

## 1. Outcome and scope

Build a locally runnable FastAPI service that answers questions using a small, explicit corpus of technical documentation. Every substantive answer must be grounded in indexed chunks and return references to those chunks. When the corpus cannot support an answer, the service must say so. The required submission includes ingestion, a LangGraph StateGraph, the four specified API endpoints, tests, and a complete README. Bonus features begin only after these requirements pass.

This plan uses the choices already made for the project: **Google Gemini** for chat and embeddings, **ChromaDB** for vector search, and **FastAPI's official documentation** as the initial five-page corpus. The user will supply GOOGLE_API_KEY through an environment variable for live validation; credentials will never be committed.

### Guiding design principles

1. **One responsibility per component.** The API validates HTTP input; ingestion prepares documents; the vector store indexes and searches; graph nodes make decisions and generate answers. This makes each part understandable and testable on its own.
2. **Explicit state and bounded work.** Graph state records the question, current query, results, grades, and retry count. Every loop has a termination condition; external calls have timeouts.
3. **Dependency injection at the edges.** Construct Gemini and Chroma clients once in application setup and pass them into services and the graph builder. Tests substitute fakes without network calls. Avoid a broad plugin framework or unnecessary base classes.
4. **Grounded output by construction.** The generator sees only relevant chunks with stable IDs. The server maps citation IDs to stored source metadata and rejects unknown IDs. An unsupported question produces an explicit abstention.
5. **Local simplicity with honest scaling boundaries.** Persistent Chroma and SQLite make the take-home easy to run. The storage interfaces and configuration permit server-backed replacements later; local files are not presented as a multi-instance production setup.

## 2. Planned architecture

The API delegates queries to a query service and ingestion requests to an ingestion service. The query service invokes the graph. The graph uses Gemini for analysis, grading, and generation, and Chroma for retrieval. The ingestion service uses the same Chroma collection and Gemini embeddings. A small SQLite repository stores answer IDs and feedback.

Graph path:

    START -> Analyze -> Retrieve -> Grade
                                | relevant chunks -> Generate -> END
                                | none, budget left -> Rewrite -> Retrieve
                                | none, budget used -> Abstain -> END

The graph handles one question per invocation. Conversation memory is a bonus feature, so a new request starts with fresh question state. The query service converts final graph state to the public API response; HTTP types do not leak into graph nodes.

## 3. Phase 0 — Define contracts and baseline decisions

### Work

- Establish Python version, package management, dependency ranges, formatting, and test commands in pyproject.toml. Record exact resolved versions in a lock file or pinned requirements before submission.
- Define environment settings in one place: GOOGLE_API_KEY, chat model, embedding model, Chroma path, top-k, maximum query rewrites, provider timeout, and ingestion limits. Provide .env.example and ignore .env, local databases, caches, and keys.
- Define public Pydantic request/response schemas before implementing routes. Decide the exact shape of an answer, source reference, document summary, ingest result, and feedback receipt.
- Select a configurable stable Gemini Flash model for analysis, grading, and generation; start with gemini-3.5-flash-lite and verify it with the supplied key. Use gemini-embedding-001 at its default 3,072 dimensions for the text corpus. This avoids custom normalization for reduced dimensions. Keep model identifiers configurable and record them with the index so a model change triggers a full rebuild.

### Design reason

Contracts first prevent graph nodes, storage code, and HTTP routes from inventing slightly different representations of the same data. One model for the three chat tasks keeps setup and cost explanations simple. The corpus is small enough that 3,072-dimensional vectors are acceptable; the main optimization target is answer quality and clarity.

**Exit criterion:** A reviewer can read the schemas and configuration and know what goes into and out of each component without reading implementation code.

## 4. Phase 1 — Build a reproducible corpus and ingestion pipeline

**Initial corpus:** five official FastAPI pages: [path parameters](https://fastapi.tiangolo.com/tutorial/path-params/), [query parameters](https://fastapi.tiangolo.com/tutorial/query-params/), [request bodies](https://fastapi.tiangolo.com/tutorial/body/), [dependencies](https://fastapi.tiangolo.com/tutorial/dependencies/), and [error handling](https://fastapi.tiangolo.com/tutorial/handling-errors/). Keep these URLs in a versioned manifest. A seed command uses the same ingestion service as the API, so there is only one indexing path.

### Pipeline

1. Load a local Markdown or text file, or fetch a permitted HTTP or HTTPS URL. For HTML, select the main article and retain headings, paragraphs, lists, and code while dropping navigation and page chrome. Store the canonical source URL and title.
2. Split first on heading boundaries, then group paragraphs and whole code blocks into chunks targeting roughly 600–800 tokens, with about 80 tokens of overlap when a section needs multiple chunks. Keep a heading path in each chunk. Hard-limit oversized blocks to the embedding model's input limit and record when a block was split.
3. Compute a document ID from the canonical source and a content hash for change detection. Give every chunk a deterministic ID and metadata: document ID, title, URL or filename, heading, chunk number, and content hash.
4. Batch-embed chunks with Gemini and upsert them into a persistent Chroma collection. Skip identical documents. When a source changes, embed its new chunks before replacing the old set; clean up stale chunk IDs after the replacement succeeds.
5. Expose indexed document summaries by grouping Chroma metadata by document ID. Store the embedding model name and dimension alongside the collection and fail clearly if the configured embedding model differs from the indexed one.

### Robustness and tradeoff

Limit upload size, accepted types, fetch time, redirect count, and downloaded bytes. For URL ingestion, validate each destination and redirect against private or local network addresses. Reject an empty extraction. An ingestion failure must be visible to the caller; it must not report a partially indexed document as successful. Local Chroma does not provide a cross-service transaction, so recovery from an interrupted replacement should be documented and tested.

Headings preserve the subject of a technical passage, while keeping examples and nearby explanation together makes retrieval more useful. Overlap reduces context loss at long-section boundaries. Chunk size remains comfortably under the chosen embedding model's input limit.

**Exit criterion:** The seed command indexes five documents; repeating it does not increase chunk counts; GET /documents reports the expected sources; retrieval returns chunks with usable citation metadata.

## 5. Phase 2 — Implement the LangGraph workflow

### State schema

Use a typed state with these fields:

| Field | Purpose | Update rule |
| --- | --- | --- |
| question | Original user wording, including API symbols | Set once; never rewritten |
| search_query | Query for the current retrieval pass | Replaced by analysis or rewrite |
| query_type | Optional conceptual, how-to, troubleshooting, or API reference type | Set by analysis |
| retrieved_chunks | Current top-k chunks and similarity metadata | Replaced on every retrieval |
| grades | Per-chunk relevance decisions | Replaced on every grading pass |
| relevant_chunks | Only chunks approved for generation | Replaced on every grading pass |
| rewrite_count | Number of semantic rewrites already used | Incremented only by rewrite node |
| answer, sources, status | Final result or abstention | Set by terminal node |

The maximum rewrite count is configuration, not mutable state. Nodes return partial state updates and do not mutate the input object. In particular, previous search results must not accumulate across loop iterations. A request ID can travel as graph metadata for logs and checkpoints without changing answer logic.

### Node responsibilities

- **Analyze query:** produce a concise retrieval query and optional type from the user's question. Preserve exact class names, function names, flags, and error text. Use structured model output; if analysis fails with a non-retryable formatting error, the original question is a safe retrieval query.
- **Retrieve:** embed the current query and request the top five Chroma results. Return text, metadata, IDs, and distance; do not call the generator here. Empty results are a normal graph outcome.
- **Grade:** ask Gemini whether each chunk contains information useful to answering the original question. Use a small structured relevance result per chunk, with bounded concurrency. Filter chunks based on the grades. A provider failure must not silently turn into an irrelevant grade.
- **Rewrite:** when no chunks pass grading, use the original question, failed search queries, and grading outcome to make a materially different query. Increment rewrite_count exactly once. If the proposed query repeats an earlier one, terminate through abstention rather than loop pointlessly.
- **Generate:** pass only relevant chunks with assigned citation IDs and an instruction to use the documents as data, not as instructions. Ask for structured answer parts with cited chunk IDs. Render the final answer with inline references and a sources array containing the corresponding URL, title, heading, and excerpt. Reject references to chunks outside the approved set. If the evidence supports only part of the question, answer that part and state the gap.
- **Abstain:** return a consistent insufficient_context status, a short explanation, and no invented citations. This is a successful query result, not a server error.

**Routing:** a pure conditional function after grading sends state to generation if relevant_chunks is nonempty, rewrite if the rewrite budget remains, or abstention otherwise. Two semantic rewrites mean no more than three retrieval passes. Set a LangGraph recursion limit above the expected path length as an additional guard.

**Exit criterion:** With fake models and a fake store, tests demonstrate both routes, partial filtering, the retry limit, and a clean state on every new request.

## 6. Phase 3 — Build the API and feedback path

| Endpoint | Input | Successful result | Important errors |
| --- | --- | --- | --- |
| POST /query | JSON question | answer_id, answer, status, sources, rewrite_count | 422 invalid input; 503 exhausted provider availability |
| POST /ingest | Multipart form with exactly one file or URL | Document ID, title, chunk count, created/updated/unchanged | 413 oversize; 415 unsupported type; 422 invalid source |
| GET /documents | None | Indexed document summaries and count | 503 storage unavailable |
| POST /feedback | JSON answer_id, rating (up/down), optional comment | Feedback ID and receipt | 404 unknown answer; 422 invalid rating/comment |

Use a small SQLite repository to persist returned answer IDs and feedback records. Store only query/result data needed to associate feedback and evaluate the system. GET /health may be added for local diagnostics, but it is not a substitute for the four required routes.

FastAPI should own validation, status codes, and error translation. Services should raise typed application errors rather than HTTPException. The route layer maps those errors to HTTP responses. One POST /ingest endpoint can accept either a form file or form URL; documenting the multipart contract avoids ambiguity between JSON and upload payloads.

**Exit criterion:** All four routes work through FastAPI's generated OpenAPI schema, and example curl requests in the README match actual responses.

## 7. Phase 4 — Reliability, state persistence, and operating limits

### Two distinct retries

1. **Semantic retry:** the graph rewrites a weak query and searches again, at most twice. This increments rewrite_count.
2. **Transient operation retry:** provider calls and vector search retry only errors likely to clear, such as rate limits or temporary network failures. Use LangGraph node retry policies with bounded attempts, backoff, and per-attempt timeouts. These retries do not increment rewrite_count. Do not retry invalid input, bad credentials, or a persistent model/schema mismatch.

An exhausted external dependency returns a service error; an exhausted relevance loop returns insufficient_context. This distinction prevents downtime from being mistaken for an answerability judgment.

**Persistence:** typed graph state is the source of truth while a request runs. In this phase, add a local SQLite LangGraph checkpointer with a unique thread ID per query to make intermediate state inspectable and recoverable. Keep its tables separate from Chroma documents and answer/feedback records, and document checkpoint retention and clean-up. Checkpointing is not conversation memory: the next independent question still starts with fresh state. A server-backed checkpointer and vector store would be needed for multiple API instances.

**Observability and resource control:** log request ID, graph node, elapsed time, retrieval count, grade count, rewrite count, status, and provider error class; never log API keys. Cap question length, chunks passed to generation, total context size, upload bytes, URL fetch bytes, and grading concurrency. Use configuration for these limits so they can be tuned after measurement.

**Exit criterion:** Injected transient failures retry within bounds, permanent failures surface correctly, repeated weak retrieval terminates, and concurrent independent requests do not share graph state.

## 8. Phase 5 — Tests and evidence of answer quality

Use pytest and FastAPI's test client with fake Gemini and Chroma adapters for deterministic tests. Add one opt-in live smoke test when GOOGLE_API_KEY is present; routine tests must not spend API credits or depend on network availability.

### Essential tests

- Corpus extraction preserves headings and code while excluding navigation.
- Chunk IDs and counts are stable across identical ingestion; changed content replaces old chunks.
- Retrieval returns the right source metadata; an empty index is handled.
- Grading removes irrelevant chunks; all-irrelevant results trigger rewrites and eventually abstain.
- Query rewrites and transient node retries have separate counters and limits.
- Generated citations refer only to graded, indexed chunks; partial answers state their limits.
- The four required routes validate inputs and return the documented status codes.
- New graph invocations do not inherit earlier questions or result lists.

Create a small evaluation sheet of roughly 12–15 questions: direct facts, how-to questions, exact API names, questions requiring two passages, and out-of-corpus questions. For each, record expected source pages, whether abstention is expected, answer correctness, citation validity, and observed latency. Use this to tune chunk size and top-k rather than adding complexity by intuition.

**Exit criterion:** The automated suite passes, the live smoke test shows end-to-end ingestion and a cited answer, and the evaluation includes successful abstentions.

## 9. Phase 6 — Submission documentation and review

Replace the planning README with the final project guide: overview, architecture diagram, source corpus, environment setup, dependency installation, seed ingestion command, server command, all API examples and responses, testing commands, design decisions, assumptions, and limitations. Include the evaluation summary and a short section on what would improve with more time. Run all documented commands from a clean environment before considering the required work complete.

### Proposed code layout

    app/
      main.py          # application creation and dependency wiring
      api.py           # HTTP routes and error translation
      config.py        # validated environment settings
      schemas.py       # public request and response models
      graph.py         # state, nodes, edges, and routing
      prompts.py       # versioned prompt text and structured LLM outputs
      ingestion.py     # loading, extraction, chunking, and indexing
      vector_store.py  # Chroma adapter and source metadata
      providers.py     # Gemini chat and embedding construction
      repository.py    # answer IDs and feedback persistence
    corpus/
      seed_urls.json
    tests/
    pyproject.toml
    .env.example
    README.md

Keep related code in these files while each remains short. Split modules only when their responsibilities genuinely diverge. The graph builder accepts its dependencies as arguments; there is no global mutable query state.

### Required completion gate

The baseline is complete when a fresh clone can install dependencies, set GOOGLE_API_KEY, seed the five-page corpus, start FastAPI, receive a supported answer with valid sources, receive an honest abstention for an unsupported question, use all four required routes, and pass the documented tests.

## 10. Bonus phase — only after the completion gate

Consider bonuses in this order, each behind its own testable change:

1. **Groundedness check:** a separate verification node inspects answer claims against cited chunks and either requests one correction or abstains.
2. **Conversation memory:** introduce a session ID and checkpoint policy for follow-ups; define how context is summarized and when it expires.
3. **Web fallback:** add an explicitly labeled external-source path with separate citations and source trust rules.
4. **UI:** a minimal client only after the API behavior is stable.

## 11. Key tradeoffs and assumptions

| Decision | Benefit | Cost or limitation |
| --- | --- | --- |
| Gemini for chat and embeddings | One credential and one provider setup | Provider latency, quotas, and costs affect both ingestion and queries |
| One stable text embedding model per collection | Consistent vector space and straightforward index validation | Changing models requires a full re-embedding |
| Persistent local Chroma | Easy setup and inspectable corpus | Replace with a server-backed store for multi-instance deployment |
| LLM grading per chunk | Explicit self-correction and clear evidence trail | More model calls; cap top-k and grading concurrency |
| SQLite for answer/feedback records | Small, reliable local persistence | Shared database needed when running multiple instances |
| Two query rewrites | Bounded recovery from poor retrieval | Some answerable questions may still abstain |
| Five-page FastAPI corpus | Small, coherent, easy to evaluate | Answers outside those pages must abstain |

## 12. Reference documentation

- [LangGraph Graph API: state, conditional edges, retries, and timeouts](https://docs.langchain.com/oss/python/langgraph/use-graph-api)
- [LangGraph persistence and checkpointers](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangChain Gemini chat integration and structured output](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai)
- [LangChain Gemini embedding integration](https://docs.langchain.com/oss/python/integrations/embeddings/google_generative_ai)
- [Google Gemini model identifiers](https://ai.google.dev/gemini-api/docs/models)
- [Google Gemini embedding models and dimensions](https://ai.google.dev/gemini-api/docs/embeddings)
- [Chroma Python client and persistence](https://docs.trychroma.com/reference/python/client)
- [FastAPI file uploads](https://fastapi.tiangolo.com/tutorial/request-files/)
