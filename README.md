# Enterprise RAG Knowledge Assistant

A production-shaped, domain-specific **Retrieval-Augmented Generation** platform:
upload your own documents, then ask questions in natural language and get answers
grounded in that corpus — with citations, streamed token by token, from a model
that runs entirely inside your own infrastructure.

---

## Architecture

```
┌───────────────────────────────────────────┐
│  FRONTEND LAYER  (frontend/Dockerfile)    │
│  Next.js 16 (App Router) + Tailwind CSS v4│
│  • upload UI          • SSE chat UI       │
└──────────────────┬────────────────────────┘
                   │  HTTPS / REST / SSE  (Bearer JWT)
                   ▼
┌───────────────────────────────────────────┐
│  API GATEWAY & SECURITY                   │
│  • JWT validation (PyJWT, HS256)          │
│  • Manual sliding-window rate limiting    │
│    (custom Redis sorted-set algorithm)    │
│  • CORS, request-id, security headers     │
└──────────────────┬────────────────────────┘
                   ▼
┌───────────────────────────────────────────┐
│  BACKEND CORE API  (backend/Dockerfile)   │
│  FastAPI + Pydantic v2                    │
│  routers → services → repositories        │
└───────┬───────────────────────────┬───────┘
        │                           │
        ▼                           ▼
┌──────────────────────┐   ┌────────────────────────┐
│ ASYNC WORKERS        │   │ LLM INFERENCE LAYER    │
│ Celery + Redis queue │   │ transformers pipeline  │
│ + LangChain pipeline │   │ Qwen2.5-0.5B-Instruct  │
└──────────┬───────────┘   └────────────────────────┘
           ▼
┌───────────────────────────────────────────┐
│  DATA & VECTOR STORAGE                    │
│  ├── PostgreSQL  (users, docs, chat)      │
│  ├── ChromaDB    (embedded + persistent)  │
│  └── Redis       (cache, rate limits, MQ) │
└───────────────────────────────────────────┘
                   ▲
┌──────────────────┴────────────────────────┐
│  TESTING & QUALITY CONTROL (Pytest)       │
└───────────────────────────────────────────┘
```

### Stack

| Layer | Technology |
| --- | --- |
| Frontend | Next.js 16 (App Router) + Tailwind CSS v4 + Dockerfile |
| API gateway / security | JWT (PyJWT) + **manual sliding-window rate limiting** |
| Backend core API | FastAPI + Pydantic v2 + SQLAlchemy 2.0 + Dockerfile |
| Async pipeline | Celery + Redis queue + **LangChain** (loading / chunking / retrieval) |
| Relational DB | PostgreSQL |
| Vector DB | **ChromaDB (embedded + persistent)** — no separate server |
| Cache / broker | Redis (cache, rate-limit counters, Celery broker/result backend) |
| LLM inference | Hugging Face `transformers` + **Qwen2.5-0.5B-Instruct** |
| Testing | Pytest (unit + integration) |

### Layer responsibilities

1. **Frontend layer** — document upload interface, chat interface, streaming
   rendering of LLM output via Server-Sent Events. `EventSource` cannot send an
   `Authorization` header and the endpoint must be a `POST`, so the client
   consumes `fetch` + `ReadableStream` and parses SSE framing itself
   (`src/lib/sse.js`).
2. **API gateway & security** — HS256 access tokens (30 min) and refresh tokens
   (7 days) issued by `app/core/security.py`; per-request validation in
   `app/api/dependencies.py`. Rate limiting is a **sliding-window log** over
   Redis sorted sets executed atomically in a `MULTI/EXEC` pipeline, with scopes
   `auth`, `chat`, `upload`, `default` (`app/core/rate_limit.py`).
3. **Backend core API** — `endpoints → services → repositories → models`.
   Endpoints never touch the ORM directly; services never touch HTTP types.
4. **Async pipeline & workers** — LangChain owns loading, recursive text
   splitting and the retriever chain (`app/rag/`). Celery + Redis run parsing,
   chunking and embedding off the request path. Every dispatch is mirrored into a
   durable `background_tasks` table because Redis is a cache, not a system of
   record.
5. **Data & vector storage** — PostgreSQL holds users, documents, the chunk
   ledger, conversations and tasks. ChromaDB runs as an embedded
   `PersistentClient` writing to `CHROMA_PERSIST_DIR`; tenant isolation is
   enforced with `user_id` metadata filters. Redis provides cache, rate-limit
   counters and the Celery broker/result backend.
6. **LLM inference** — Qwen2.5-0.5B-Instruct loads lazily on first use and is
   cached per process; streaming uses `TextIteratorStreamer` on a background
   thread.
7. **Testing & quality control** — Pytest against a real PostgreSQL test
   database, with eager Celery and an in-memory cache so no Redis is needed, a
   stub LLM, hash embeddings and a real embedded Chroma collection in a temp
   directory.

---

## Quick start

**Prerequisites:** a running **PostgreSQL** server and a running **Redis** server.

```bash
# 0a. PostgreSQL role + database (once).
#     NOTE: the password is required - `createuser` alone creates a PASSWORDLESS
#     role, which cannot authenticate against the password in backend/.env.
sudo -u postgres psql -c "CREATE ROLE rag WITH LOGIN CREATEDB PASSWORD 'rag';"
sudo -u postgres psql -c "CREATE DATABASE rag OWNER rag;"

# 0b. Redis (defaults to localhost:6379; use your package manager if installed)
docker run -d --name rag-redis -p 6379:6379 redis:7-alpine

# 1. Backend
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m app.cli init-db
uvicorn app.main:app --reload --port 8000        # http://localhost:8000/docs

# 2. Celery worker - OPTIONAL, and off by default.
#
#    Ingestion runs inline in the API process (CELERY_TASK_ALWAYS_EAGER=true), so
#    a document is searchable as soon as it finishes. Only start this worker if
#    you set CELERY_TASK_ALWAYS_EAGER=false, and read the warning below first.
cd backend && source .venv/bin/activate
celery -A app.workers.worker:celery_app worker \
  --loglevel=info --concurrency=2 --queues=default,ingestion,maintenance

# 3. Frontend (second terminal)
cd frontend
npm install
npm run dev                                      # http://localhost:3000
```

Register at `/register` in the UI, or seed an account from the CLI:

```bash
cd backend
python -m app.cli create-user --email admin@example.com --password 'ChangeMe123' --admin
```

### Models: real vs. offline

`backend/.env` ships configured with the **real** models, so answers are
generated by an actual LLM:

```ini
EMBEDDING_PROVIDER=huggingface   # sentence-transformers/all-MiniLM-L6-v2
LLM_PROVIDER=transformers        # Qwen/Qwen2.5-0.5B-Instruct
LLM_STRUCTURED_OUTPUT=true       # outlines-constrained JSON (see below)
```

The embedding model and Qwen2.5-0.5B weights download on first use into
`HF_HOME` (`./data/hf` by default).

For a fully offline environment (CI, no model download), swap in the
deterministic providers:

```ini
EMBEDDING_PROVIDER=hash
LLM_PROVIDER=stub
```

> **`LLM_PROVIDER=stub` is not a language model.** It is a deterministic
> extractor that quotes retrieved chunks back verbatim, so replies read
> mechanically and cannot be made more detailed. It exists so the whole pipeline
> can be exercised without weights — use it for tests, not for real answers.

### Structured output

When `outlines` is installed and `LLM_STRUCTURED_OUTPUT=true`, generation is
constrained to a JSON schema (`app/llm/structured.py::AnswerWithCitations`:
`answer` prose + `key_points` bullets + `citations`). Because the decoder can
only emit tokens valid for that schema, even a 0.5B model returns reliable,
parseable output — no post-hoc parsing of free-form text. The chat UI renders
the structured fields (prose, bullet list, citation chips).

If `outlines` is unavailable, the system automatically falls back to plain-text
generation.

### Answer length (token budget)

Qwen2.5-0.5B-Instruct has **two** limits, and they are not the same number:

| Limit | Value | Applies to |
| --- | --- | --- |
| Context window | **32 768** tokens | `prompt + completion`, combined |
| Max generation length | **8 192** tokens | the completion alone |

`LLM_MAX_NEW_TOKENS` defaults to the full **8 192** — the longest answer the
model was trained to produce — so replies are as detailed as it can manage.
Both ceilings are declared once in `app/core/config.py`
(`LLM_MAX_GENERATION_TOKENS`, `LLM_CONTEXT_WINDOW_TOKENS`) and enforced in three
places:

* the settings validator rejects a value above either ceiling at startup;
* `POST /chat/completions` rejects `max_new_tokens > 8192` with a `422`;
* `TransformersLLMProvider._resolve_generation_budget` trims the budget when the
  rendered prompt is long enough that `prompt + budget` would overflow the
  context window (it logs a warning naming both numbers when it does).

Generation past 8 192 tokens is outside the training distribution and degrades
into repetition rather than a longer coherent answer, which is why the ceiling
exists rather than being left open to the context window.

> **Cost of the full budget.** On CPU a 0.5B model produces roughly 20–40
> tokens/sec, so a genuinely 8 192-token answer takes several minutes. Because
> the SSE endpoint holds the connection open for that long, the container's
> gunicorn `--timeout` is raised to 900 s. For snappier replies set
> `LLM_MAX_NEW_TOKENS=2048` in `backend/.env` — the model usually stops well
> before the ceiling on its own when the answer is complete.


### Running without a Celery worker

Ingestion is dispatched to the worker over Redis. To make the API execute tasks
inline instead (useful for a single-process demo), set in `backend/.env`:

```ini
CELERY_TASK_ALWAYS_EAGER=true
```

Uploads then block until parsing/embedding finishes, and the `task_id` returned
by `/documents/upload` completes immediately.

### Pointing at Dockerised PostgreSQL / Redis

Both services default to `localhost`. If you run them in containers, change the
hosts in `backend/.env`:

```ini
DATABASE_URL=postgresql+psycopg2://rag:rag@postgres:5432/rag
REDIS_URL=redis://redis:6379/0
```

---

## Running the tests

The suite needs **no external service at all** — no PostgreSQL, no Redis, no
model downloads. The database is an in-memory SQLite engine installed for the
test session, so `pytest` works on a clean machine:

```bash
cd backend
pytest                     # everything
pytest -m unit             # fast, isolated
pytest -m integration      # through the FastAPI app
pytest --cov=app --cov-report=term-missing
```

Everything else stays offline too: eager Celery, an in-memory cache, a stub LLM,
hash embeddings and a real embedded Chroma collection in a temporary directory.
Test configuration (including the `unit` / `integration` markers) lives in
`tests/conftest.py`.

> **Note on the database.** The application is PostgreSQL-only, and the test
> suite uses in-memory SQLite purely as a fixture. Both are SQLAlchemy over the
> same models, but SQLite is not the production engine — features it cannot
> reproduce (server-side defaults, native `JSONB`/`UUID` types, `TRUNCATE`
> semantics) are therefore not exercised by the suite.

---

## API surface

All routes are prefixed with `/api/v1`.

### Health
| Method | Path | Notes |
| --- | --- | --- |
| GET | `/health` | Liveness |
| GET | `/health/ready` | Readiness — database, cache, broker, embeddings, LLM |

### Authentication
| Method | Path | Notes |
| --- | --- | --- |
| POST | `/auth/register` | Rate-limited (`auth`) |
| POST | `/auth/login` | Returns access + refresh tokens |
| POST | `/auth/refresh` | Rotates the refresh token |
| GET | `/auth/me` | Current profile |
| POST | `/auth/logout` | Stateless sign-out |

### Documents
| Method | Path | Notes |
| --- | --- | --- |
| POST | `/documents/upload` | `multipart/form-data`; returns `202` + `task_id` |
| GET | `/documents` | Paginated; `status`, `search`, `collection`, `extension` filters |
| GET | `/documents/stats` | Counts per ingestion status |
| GET | `/documents/{id}` | Metadata |
| GET | `/documents/{id}/status` | Ingestion progress (poll this) |
| GET | `/documents/{id}/chunks` | Chunk ledger |
| PATCH | `/documents/{id}` | Rename / retag |
| POST | `/documents/{id}/reindex` | Re-dispatch ingestion |
| DELETE | `/documents/{id}` | Row + chunks + file + vectors |

### Chat
| Method | Path | Notes |
| --- | --- | --- |
| POST | `/chat/completions` | Non-streaming grounded answer |
| POST | `/chat/stream` | **SSE**: `start` → `sources` → `token`* → `done` |
| POST | `/chat/stream/{conversation_id}` | Same, with the thread id in the path |
| POST/GET | `/chat/conversations` | Create / list |
| GET/PATCH/DELETE | `/chat/conversations/{id}` | Fetch (with messages) / rename / delete |
| GET | `/chat/conversations/{id}/messages` | Message history |

### Tasks
| Method | Path | Notes |
| --- | --- | --- |
| GET | `/tasks` | Durable background-task audit trail |
| GET | `/tasks/{id}` | By record id or Celery id |
| POST | `/tasks/maintenance/health` | Dispatch a worker probe |

### SSE event contract

```
event: start      data: {"conversation_id": "...", "model_name": "Qwen/Qwen2.5-0.5B-Instruct"}
event: sources    data: {"sources": [{"chunk_id","document_id","document_name","score","snippet","chunk_index"}]}
event: token      data: {"text": " Refund", "index": 7}
event: done       data: {"message_id","conversation_id","answer","latency_ms","token_count"}
event: error      data: {"message": "...", "code": "generation_error"}
```

---

## Repository layout

```
.
├── README.md                  # this document (architecture + operations)
├── backend/
│   ├── .env                   # backend configuration (see below)
│   ├── .gitignore
│   ├── .dockerignore
│   ├── Dockerfile             # multi-stage, CPU torch, model prefetch
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py            # FastAPI factory + lifespan
│   │   ├── cli.py             # init-db / create-user / reindex / health
│   │   ├── core/              # config, logging, security, rate limiting, errors
│   │   ├── db/                # engine/session, cache, Chroma wrapper, bootstrap
│   │   ├── models/            # users, documents, chunks, conversations, tasks
│   │   ├── schemas/           # Pydantic request/response contracts
│   │   ├── repositories/      # the only place that talks SQL
│   │   ├── services/          # business logic
│   │   ├── rag/               # loaders, chunking, embeddings, retriever, pipeline
│   │   ├── llm/               # provider interface, Qwen2.5, stub, prompt builder
│   │   ├── workers/           # Celery app + ingestion/maintenance tasks
│   │   └── api/               # dependencies, middleware, errors, v1 routers
│   └── tests/                 # unit/ + integration/ (config in conftest.py)
└── frontend/
    ├── .env                   # NEXT_PUBLIC_API_BASE_URL
    ├── .dockerignore
    ├── Dockerfile             # multi-stage standalone Next.js build
    ├── public/                # static assets
    └── src/
        ├── app/               # landing, login, register, /app console
        ├── components/ui.js   # shared Tailwind primitives
        └── lib/               # api client, auth context, SSE parser
```

### Configuration

Configuration is environment-based and read from `backend/.env` and
`frontend/.env` (see `app/core/config.py`, which loads both `backend/.env` and
the repository root `.env` if present).

`backend/.env` is the single source of truth for the backend: application
settings, JWT secrets, rate limits, database/Redis URLs, Chroma persistence,
upload policy, embeddings, chunking and the LLM provider. Every value is
documented inline, and its defaults run the stack with no external services.

---

## Design decisions worth knowing

* **Manual rate limiting, not a library.** `app/core/rate_limit.py` implements a
  sliding-window log over Redis sorted sets inside a single `MULTI/EXEC`
  pipeline, so concurrent workers cannot over-admit a caller. An in-memory store
  is retained as a safety net for when Redis is unreachable, but Redis is the
  configured backend.
* **Embedded Chroma.** `PersistentClient` writes straight to disk inside the
  Python process — one less service to operate.
* **One process owns the collection.** Chroma's durable rows and its **in-memory
  ANN index** are separate: `count()` reads the rows, but `query()` reads the
  index of the process that asks. Two processes over one `CHROMA_PERSIST_DIR`
  (API + worker) therefore desynchronise — the worker writes the rows, the API's
  index stays empty, and *every* query returns zero hits while `count()` reports
  vectors. `CELERY_TASK_ALWAYS_EAGER=true` (the default) avoids this by running
  ingestion in the API process. If you must split them, point
  `CHROMA_PERSIST_DIR` at a shared volume **and** expect to re-ingest documents
  before they become searchable from the API; retrieval logs a loud
  "ANN index is out of sync" error when it detects the mismatch.
* **Durable task ledger.** Celery results live in Redis, which is a cache. Every
  dispatch is also written to `background_tasks`, so `GET /tasks/{id}` and the
  dashboard keep working after a Redis flush.
* **Tenant isolation in the query, not the prompt.** Every retrieval is filtered
  by `user_id` metadata, so cross-tenant chunks can never enter a context window.
* **Pluggable providers.** `LLM_PROVIDER` and `EMBEDDING_PROVIDER` swap real
  models for deterministic stubs, which keeps the test suite offline and fast
  without weakening what it verifies.
* **SSE over `fetch`, not `EventSource`.** `EventSource` cannot send an
  `Authorization` header, and the endpoint must be a `POST` to carry a body.
* **No root tooling files.** `docker-compose.yml`, `Makefile` and `.gitignore`
  at the repository root were removed by request; the Dockerfiles in `backend/`
  and `frontend/` remain the containerisation path, and all commands are
  documented above.
* **Chroma telemetry is replaced, not merely muted.** `chromadb` calls
  `posthog.capture(distinct_id, event, properties)` — the posthog v2/v3
  positional API — while declaring only `posthog>=2.4.0` with no upper bound.
  A current posthog therefore installs and every Chroma event logs
  `capture() takes 1 positional argument but 3 were given`.
  `anonymized_telemetry=False` does **not** fix this (it only sets
  `posthog.disabled = True`, which skips the network send but still runs the
  mismatched call). `app/db/vector_store.py` instead supplies a no-op
  telemetry client through Chroma's own `chroma_product_telemetry_impl`
  extension point, so the broken call is never reached regardless of the
  installed posthog version. `requirements.txt` also caps posthog below 4 for
  reproducibility. The errors were cosmetic — ingestion still completed — but
  they polluted every worker log with stack traces.

## Security notes

* Passwords: bcrypt (cost 12); inputs over 72 bytes are pre-hashed to respect
  bcrypt's limit.
* Login returns an identical error for unknown email and wrong password (no
  account enumeration).
* Ownership is enforced in the repository layer, so another tenant's resource
  yields `404`, not `403` — no existence leakage.
* Upload policy: extension allowlist, size ceiling, SHA-256 deduplication.
* Rate limiting is per identity (user id when authenticated, else client IP) and
  emits `X-RateLimit-*` / `Retry-After` headers.
* Set a real `SECRET_KEY` in `backend/.env` before any non-local deployment.

## Known limitations

* Refresh tokens are stateless; revocation would need a denylist (not
  implemented).
* The in-memory rate-limit fallback is per-process — use Redis for multi-worker
  deployments.
* Chroma is embedded and effectively single-writer per process; heavy concurrent
  ingestion should be serialised through the Celery `ingestion` queue.
* `Base.metadata.create_all()` bootstraps the schema (via `python -m app.cli
  init-db`). Use Alembic for production schema evolution.

## License

Provided as-is for internal/enterprise use.
