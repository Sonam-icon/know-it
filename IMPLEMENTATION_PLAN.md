# Implementation plan: technical documentation assistant

## Implementation status

- **Phase 0 — Complete:** project metadata, locked dependencies, validated settings, public API schemas, application factory, health endpoint, and baseline tests.
- **Phase 1 — Implemented:** reproducible corpus, safe loaders, heading-aware chunks, Gemini embedding adapter, versioned Chroma storage, and corpus CLI. Live seeding awaits GOOGLE_API_KEY in the runtime environment.
- **Phase 2 — Implemented:** typed LangGraph state, analysis, vector retrieval, per-chunk grading, bounded rewrites, grounded generation, citation validation, and abstention. A live Gemini run awaits GOOGLE_API_KEY and a seeded index.
- **Phase 3 — Complete:** required query, ingestion, document-listing, and feedback routes; typed responses; HTTP error mapping; SQLite answer and feedback persistence; and OpenAPI contract tests.
- **Phase 4 — Complete:** bounded graph rewrites, transient node retries, provider timeouts, and basic limits and logging. Each query has fresh in-memory graph state.
- **Phase 5 — Harness complete:** deterministic tests, a 15-case evaluation dataset, report generation, and an opt-in live ingestion/query smoke test. Live results await GOOGLE_API_KEY.
- **Phase 6 — Complete:** final setup, architecture, API examples, design tradeoffs, assumptions, limitations, improvement path, and submission checklist. The live completion gate remains pending.
- **Bonus B1–B4 — Implemented:** groundedness verification with one revision, bounded SQLite-backed session history, one-shot allowlisted Tavily fallback, and an optional Streamlit API client. Deterministic verification is complete; live Gemini/Tavily evaluation awaits credentials.

## 1. Outcome and scope

Build a locally runnable FastAPI service that answers questions using a small, explicit corpus of technical documentation. Every substantive answer must be grounded in indexed chunks and return references to those chunks. When the corpus cannot support an answer, the service must say so. The required submission includes ingestion, a LangGraph StateGraph, the four specified API endpoints, tests, and a complete README. The bonus roadmap extends this baseline without changing its default behavior.

This plan uses the choices already made for the project: **Google Gemini** for chat and embeddings, **ChromaDB** for vector search, and **FastAPI's official documentation** as the initial five-page corpus. The user will supply GOOGLE_API_KEY through an environment variable for live validation; credentials will never be committed.

### Guiding design principles

1. **One responsibility per component.** The API validates HTTP input; ingestion prepares documents; the vector store indexes and searches; graph nodes make decisions and generate answers. This makes each part understandable and testable on its own.
2. **Explicit state and bounded work.** Graph state records the question, current query, results, grades, and retry count. Every loop has a termination condition; external calls have timeouts.
3. **Dependency injection at the edges.** Construct Gemini and Chroma clients once in application setup and pass them into services and the graph builder. Tests substitute fakes without network calls. Avoid a broad plugin framework or unnecessary base classes.
4. **Grounded output by construction.** The generator sees only relevant chunks with stable IDs. The server maps citation IDs to stored source metadata and rejects unknown IDs. An unsupported question produces an explicit abstention.
5. **Local simplicity.** Persistent Chroma holds documents and SQLite holds answers, bounded session history, and feedback. Graph state still exists for one query invocation; no checkpoint database or distributed session system is needed.

## 2. Planned architecture

The API delegates queries to a query service and ingestion requests to an ingestion service. The query service invokes the graph. The graph uses Gemini for analysis, grading, and generation, and Chroma for retrieval. The ingestion service uses the same Chroma collection and Gemini embeddings. A small SQLite repository stores answer IDs and feedback.

Graph path:

    START -> Analyze -> Retrieve -> Grade
                                | relevant chunks -> Generate -> END
                                | none, budget left -> Rewrite -> Retrieve
                                | none, budget used -> Abstain -> END

The graph handles one question per invocation. The API may supply a bounded history window to analysis, but every invocation starts with fresh graph state and must retrieve current evidence. The query service converts final graph state to the public API response; HTTP types do not leak into graph nodes.

## 3. Phase 0 — Define contracts and baseline decisions

### Work

- Establish Python version, package management, dependency ranges, formatting, and test commands in pyproject.toml. Record exact resolved versions in a lock file or pinned requirements before submission.
- Define environment settings in one place: GOOGLE_API_KEY, chat model, embedding model, Chroma path, top-k, maximum query rewrites, provider timeout, and ingestion limits. Provide .env.example and ignore .env, local databases, caches, and keys.
- Define public Pydantic request/response schemas before implementing routes. Decide the exact shape of an answer, source reference, document summary, ingest result, and feedback receipt.
- Select one configurable Gemini Flash model for analysis, grading, and generation; start with gemini-3.5-flash-lite and verify it with the supplied key. Use gemini-embedding-001 at 3,072 dimensions. Record the embedding model and dimension with the index so a change requires a rebuild.

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

Limit upload size, accepted types, fetch time, redirect count, and downloaded bytes. For URL ingestion, validate each destination and redirect against private or local network addresses. Reject an empty extraction. An ingestion failure must be visible to the caller. For this local demonstration, document replacement uses an upsert followed by deletion of stale chunks; it is not transactional.

Headings preserve the subject of a technical passage, while keeping examples and nearby explanation together makes retrieval more useful. Overlap reduces context loss at long-section boundaries. Chunk size remains comfortably under the chosen embedding model's input limit.

**Exit criterion:** The seed command indexes five documents; repeating it does not increase chunk counts; GET /documents reports the expected sources; retrieval returns chunks with usable citation metadata.

## 5. Phase 2 — Implement the LangGraph workflow

### State schema

Use a typed state with these fields:

| Field | Purpose | Update rule |
| --- | --- | --- |
| question | Original user wording, including API symbols | Set once; never rewritten |
| search_query | Query for the current retrieval pass | Replaced by analysis or rewrite |
| retrieved_chunks | Current top-k chunks and similarity metadata | Replaced on every retrieval |
| grades | Per-chunk relevance decisions | Replaced on every grading pass |
| relevant_chunks | Only chunks approved for generation | Replaced on every grading pass |
| rewrite_count | Number of semantic rewrites already used | Incremented only by rewrite node |
| answer, sources, status | Final result or abstention | Set by terminal node |

The maximum rewrite count is configuration, not mutable state. Nodes return partial state updates and do not mutate the input object. Previous search results must not accumulate across loop iterations. Each invocation starts with fresh in-memory state.

### Node responsibilities

- **Analyze query:** produce a concise retrieval query from the user's question. Preserve exact class names, function names, flags, and error text. Use structured model output; if analysis fails with a non-retryable formatting error, use the original question.
- **Retrieve:** embed the current query and request the top five Chroma results. Return text, metadata, IDs, and distance; do not call the generator here. Empty results are a normal graph outcome.
- **Grade:** ask Gemini whether each chunk contains information useful to answering the original question. Grade the small top-k set in order and filter chunks based on the results. A provider failure must not silently turn into an irrelevant grade.
- **Rewrite:** when no chunks pass grading, use the original question, failed search queries, and grading outcome to make a materially different query. Increment rewrite_count exactly once. If the proposed query repeats an earlier one, terminate through abstention rather than loop pointlessly.
- **Generate:** pass only relevant chunks with assigned citation IDs and an instruction to use the documents as data, not as instructions. Ask for structured answer parts with cited chunk IDs. Render the final answer with inline references and a sources array containing the corresponding URL, title, heading, and excerpt. Reject references to chunks outside the approved set. If structured output or the citation contract is invalid, make one bounded corrected generation attempt using the validation reason; a second failure remains an operational 502. If the evidence supports only part of the question, answer that part and state the gap.
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

## 7. Phase 4 — Bounded retries and local limits

### Two distinct retries

1. **Semantic retry:** the graph rewrites a weak query and searches again, at most twice. This increments rewrite_count.
2. **Transient operation retry:** graph nodes retry only errors likely to clear, such as rate limits or temporary network failures. Use one LangGraph retry policy with bounded attempts and backoff, plus a provider request timeout. These retries do not increment rewrite_count. Do not retry invalid input, bad credentials, or schema errors.

An exhausted external dependency returns a service error; an exhausted relevance loop returns insufficient_context. This distinction prevents downtime from being mistaken for an answerability judgment.

**State:** LangGraph carries typed state during a query invocation and compiles without a checkpointer. Independent graph runs remain isolated. The later bonus implementation reuses the application SQLite database for a bounded completed-turn history; it does not persist graph execution state.

**Limits and logging:** log graph steps and counts without logging question text, document contents, or API keys. Cap question length, retrieval count, generation context, upload bytes, URL fetch bytes, and redirects. A request-ID middleware or distributed tracing system would add little value to this local submission.

**Exit criterion:** Injected transient failures retry within bounds, permanent failures surface correctly, repeated weak retrieval terminates, and independent requests start with clean graph state.

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

### Code layout

    app/
      main.py            # FastAPI application and service setup
      api.py             # HTTP routes and error translation
      config.py          # environment settings
      schemas.py         # public API models
      bootstrap.py       # dependency construction
      query_workflow.py  # graph state, nodes, edges, and routing
      rag.py             # graph domain and model contracts
      prompts.py         # prompt text
      providers.py       # Gemini adapters
      loaders.py         # file and URL loading
      chunking.py        # heading-aware splitting
      ingestion.py       # indexing use case
      vector_store.py    # Chroma adapter
      repository.py      # answer IDs and feedback
      reliability.py     # transient retry policy
      evaluation.py      # small evaluation runner
      cli.py             # seed, inspect, query, evaluate
    corpus/
      seed_urls.json
    tests/
    pyproject.toml
    .env.example
    README.md

Each module owns one step or boundary. The graph builder accepts its dependencies as arguments; there is no global mutable query state, plugin framework, checkpoint database, or tracing middleware.

### Required completion gate

The baseline is complete when a fresh clone can install dependencies, set GOOGLE_API_KEY, seed the five-page corpus, start FastAPI, receive a supported answer with valid sources, receive an honest abstention for an unsupported question, use all four required routes, and pass the documented tests.

## 10. Bonus implementation

The selected order was **groundedness check → conversation memory → Tavily web fallback → Streamlit UI**. All four extensions are now implemented behind small contracts. The original API remains compatible: `session_id` is optional and web fallback defaults to disabled. Deterministic tests exercise the new paths; the live Gemini smoke test and Tavily quality check remain pending until credentials are configured.

### Bonus design rules

1. Keep LangGraph in `query_workflow.py` as the owner of query decisions. HTTP parsing stays in `api.py`; Gemini and Tavily calls stay behind narrow adapters; SQLite owns answer and session records.
2. Extend the existing typed state with only the fields a node actually needs. Nodes return partial updates. Define explicit conditional routes and a maximum count for every new loop or external search.
3. Preserve the two existing retry meanings: query rewrites address poor relevance; operational retries address transient external failures. Verification correction gets its own counter and never consumes either budget.
4. Treat documents, prior chat turns, and web snippets as untrusted input. Only retrieved and graded source text can support factual claims. The UI and API never manufacture citations.
5. Use small `Protocol` boundaries and fake adapters in tests. Avoid a generic agent framework, a second graph, and long-lived checkpoint state unless a later requirement makes them necessary.

The implemented graph follows this bounded path:

~~~mermaid
flowchart TD
    Q[Question and optional session history] --> A[Analyze into standalone question]
    A --> R[Retrieve indexed chunks]
    R --> G[Grade evidence]
    G -->|Relevant| D[Generate draft]
    G -->|No, rewrites remain| W[Rewrite query]
    W --> R
    G -->|No, budget exhausted; web enabled| T[Search Tavily once]
    T --> GW[Grade web excerpts]
    GW -->|Relevant| D
    GW -->|No relevant excerpts| X[Abstain]
    G -->|No, web disabled| X
    D --> V[Verify claims against cited evidence]
    V -->|Supported| F[Finalize answer]
    V -->|Unsupported; revision available| C[Revise once]
    C --> V
    V -->|Still unsupported| X
~~~

Only the path taken is executed. Session history helps interpret the question; it never bypasses retrieval or grading. A provider error exits through the API error handler rather than following an abstention edge.

### Phase B0 — Establish a baseline before extending the graph

**Work:** run the locked install, Ruff, and deterministic tests now. Once the Gemini key is configured, seed the five FastAPI pages, run the opt-in live smoke test and the 15-case evaluation, and record results and latency. Review one supported answer, one partial answer, and one abstention manually. Keep web fallback disabled during baseline evaluation so expected abstentions remain meaningful.

**Gate:** the four required routes and the original graph behavior still pass. Do not claim a quality improvement from a bonus until it has been measured against this baseline.

### Phase B1 — Verify answers against their cited evidence (implemented)

**Graph change:** replace `Generate → END` with `Generate → Verify → (Finalize | Revise | Abstain)`. `Generate` produces a draft and approved source IDs; the final answer is published only after verification. `Revise` receives the verifier's specific unsupported claims and may run once, after which the draft is checked again. If it remains unsupported, return `insufficient_context` with no citations. A malformed model response or invalid citation ID remains an operational error, not an evidence judgment.

**State additions:** `draft`, `draft_sources`, `verification` (structured verdict and unsupported claim list), and `revision_count` (0 or 1). The verifier receives the standalone question, answer, inline citations, and the exact graded chunks supplied to generation. A negative verdict must identify at least one unsupported claim; a positive verdict cannot contain unsupported claims. The finalizer relies on citation validation performed for the original or revised draft. The verifier is a second model judgment, not a mathematical proof of truth.

**Code boundaries:** add a `verify(...)` method and Pydantic output contract to `rag.py`/`providers.py`; keep the prompt in `prompts.py`, routing and the one-revision limit in `query_workflow.py`, and feature configuration in `config.py`. Use the existing Gemini model and retry policy. Add no new service or database.

**Tests and gate:** fake-model tests cover fully supported answers, unsupported claims followed by a successful revision, a failed revision leading to abstention, invented citation IDs, verifier schema failures, and the revision cap. Add a few claim-level cases to the evaluation set and compare cited-answer quality, abstention rate, latency, and model calls with B0. Acceptance requires no unsupported answer to be returned in the deterministic cases and no regression in existing answers.

**Tradeoff:** verification adds at least one Gemini call to each generated answer and can occasionally reject a correct answer. Keep the verdict and revision path visible in tests and evaluation rather than hiding them behind a confidence score.

### Phase B2 — Add bounded follow-up memory (implemented)

**Contract:** add an optional `session_id: UUID` to `POST /query` and return `session_id` in the response. With no ID, generate a new session. With an existing ID, load recent turns. Reject an unknown ID with 404 rather than silently joining or inventing a conversation. Existing clients that send only `question` continue to work.

**Storage:** extend the existing SQLite `answers` table with a nullable `session_id` and an index, using a small migration for existing local databases. Each stored answer already has its question, result, and timestamp, so a separate message store is unnecessary. Save the current exchange only after the graph returns a result; failed provider calls do not become chat history. Limit history to the last three completed turns and a fixed character budget. Do not add LangGraph checkpoints, chat summaries, or cross-session memory.

**Graph change:** pass bounded history into the `Analyze` node. Structured analysis returns both a self-contained question and a search query; retrieval, grading, generation, and verification use the self-contained question. Keep the raw user wording for the API and feedback record. History can clarify references such as “What about its default?”, but previous answers are never treated as source evidence or copied into the generation context. Each API turn still creates fresh graph state.

**Tests and gate:** verify first and follow-up turns, resolved pronouns, unknown sessions, session isolation, the history cap, an unsuccessful turn not being saved, and persistence across an application restart. A follow-up must retrieve current documentation and cite it; it must not answer from chat history alone. Document the single-instance SQLite scope and avoid promising simultaneous same-session ordering.

**Tradeoff:** bounded history is easy to explain but may miss a reference from an older turn. The accepted behavior is an `insufficient_context` response asking the user to restate the missing reference.

### Phase B3 — Search official documentation with Tavily when the corpus fails (implemented)

**Activation:** make Tavily optional through `WEB_FALLBACK_ENABLED` and `TAVILY_API_KEY`. With the flag off, the baseline graph and 15-case evaluation behave as before. Start with an official documentation domain allowlist (the FastAPI domain for this corpus), configurable if the corpus changes. Search only after the corpus has used its bounded rewrite budget and still has no relevant chunks. Make at most one Tavily request per query.

**Graph change:** `Grade(corpus)` routes to `SearchWeb` after exhausted rewrites when enabled; `SearchWeb → Grade(web)` reuses the existing relevance grader. Relevant web excerpts go through the same generator, citation validation, and B1 verifier. No relevant web excerpts lead to abstention. A Tavily outage after bounded operational retries surfaces as a service error, not a fabricated answer. The state needs only a `retrieval_origin` (`corpus` or `web`) and a `web_attempted` flag in addition to the existing fields.

**Adapter and provenance:** define a small `WebSearcher` protocol and one `TavilySearcher` implementation using the existing HTTP client dependency. Bound timeout, response size, result count (initially three), and excerpt length. Reject missing content, duplicate URLs, non-HTTP(S) URLs, and returned hosts outside the configured domain allowlist; use Tavily result text as the actual evidence presented to the model. Do not copy web hits into Chroma. Add `origin: corpus | web` to the returned source records so users can distinguish indexed documents from live search. Keep the source URL and excerpt attached to every citation. Treat Tavily snippets as lower-trust data and never follow instructions inside them.

**Tests and gate:** mock Tavily success, empty and irrelevant results, duplicate URLs, invalid URLs, timeout, 429/5xx retry, disabled fallback, one-search limit, and mixed provenance in API responses. Evaluate web-eligible questions in a separate dataset; preserve the original offline dataset with fallback disabled. Check that an out-of-corpus FastAPI question can gain a cited official-doc answer and that a genuinely unsupported question still abstains.

**Tradeoff:** web search makes answers less reproducible and costs an extra external request. Keep it explicitly labeled and optional. Tavily's Search API supports bounded result counts and domain restrictions; use those controls rather than building a general crawler.

### Phase B4 — Add a minimal Streamlit client (implemented)

**Scope:** one small `ui.py` that calls the existing FastAPI routes over HTTP. Use Streamlit's chat input and chat message components to display a question, answer, status, and clickable citations with `corpus`/`web` labels. Show a session reset control and store the current `session_id` and displayed transcript in Streamlit session state. Add thumbs up/down wired to `POST /feedback`. Keep document ingestion in the API and CLI.

**Boundaries:** Streamlit must not import the graph, open Chroma/SQLite, or hold Gemini/Tavily credentials. It uses only the API base URL, with a request timeout and clear messages for 422, 503, and connection failures. Put Streamlit in an optional UI dependency group so the API's normal install remains small.

**Tests and gate:** the API client is tested with mocked HTTP responses. Submission, citations, feedback, session continuation, and reset are kept in one small Streamlit module for a manual smoke check once the API has live credentials. The README documents the separate FastAPI and Streamlit commands.

### Final bonus acceptance check

Ruff and all deterministic tests must pass before provider testing. With credentials, run the live smoke test, the original evaluation with web fallback off, and separate bonus checks for a supported corpus answer, a corrected or abstained unsupported draft, a cited follow-up, a Tavily answer marked `web`, a failed web search, and UI feedback. The deterministic implementation is complete; final live acceptance remains pending until credentials are configured.

## 11. Key tradeoffs and assumptions

| Decision | Benefit | Cost or limitation |
| --- | --- | --- |
| Gemini for chat and embeddings | One credential and one provider setup | Provider latency, quotas, and costs affect both ingestion and queries |
| One stable text embedding model per collection | Consistent vector space and straightforward index validation | Changing models requires a full re-embedding |
| Persistent local Chroma | Easy setup and inspectable corpus | Intended for a single local instance |
| LLM grading per chunk | Explicit self-correction and clear evidence trail | More model calls; keep top-k small |
| SQLite for answer/feedback records | Small, reliable local persistence | Intended for a single local instance |
| Two query rewrites | Bounded recovery from poor retrieval | Some answerable questions may still abstain |
| Five-page FastAPI corpus | Small, coherent, easy to evaluate | Baseline abstains outside those pages; optional Tavily results are labeled `web` |

## 12. Reference documentation

- [LangGraph Graph API: state, conditional edges, retries, and timeouts](https://docs.langchain.com/oss/python/langgraph/use-graph-api)
- [LangGraph persistence: checkpointers and stores](https://docs.langchain.com/oss/python/langgraph/persistence) — reference for the memory tradeoff; B2 uses bounded SQLite answer history instead of graph checkpoints.
- [LangChain Gemini chat integration and structured output](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai)
- [LangChain Gemini embedding integration](https://docs.langchain.com/oss/python/integrations/embeddings/google_generative_ai)
- [Google Gemini model identifiers](https://ai.google.dev/gemini-api/docs/models)
- [Google Gemini embedding models and dimensions](https://ai.google.dev/gemini-api/docs/embeddings)
- [Chroma Python client and persistence](https://docs.trychroma.com/reference/python/client)
- [FastAPI file uploads](https://fastapi.tiangolo.com/tutorial/request-files/)
- [Tavily Search API: result limits and domain filters](https://docs.tavily.com/documentation/api-reference/endpoint/search)
- [Streamlit chat elements](https://docs.streamlit.io/develop/api-reference/chat) and [feedback widget](https://docs.streamlit.io/develop/api-reference/widgets/st.feedback)
