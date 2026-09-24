# Research Paper Assistant

A local, single-user research assistant for text-based scientific PDFs. The
application will use retrieval-augmented generation to answer questions only
from selected papers and provide verified page-level citations.

The authoritative scope and implementation decisions are recorded in
`PROJECT_DECISIONS.txt`. Completed work is tracked in `PROJECT_PROGRESS.txt`.

## Current foundation

The backend provides:

- FastAPI application setup
- environment-based configuration
- SQLite schema initialization and connectivity checks
- Ollama readiness checks, including required model availability
- Qdrant readiness checks
- liveness and readiness endpoints
- validated PDF upload with configurable size limits
- generated-name local PDF storage and SHA-256 duplicate detection
- persistent paper metadata and ingestion-job state
- paper listing, detail, and deletion endpoints
- automatic page-preserving text extraction, chunking, and batched local embeddings
- Qdrant indexing with paper/page citation payloads and scoped vector deletion
- persisted ingestion states, interrupted-job recovery, and failed-job retries
- a development retrieval endpoint with selected-paper filtering and diverse source excerpts

The frontend provides:

- React, Vite, and strict TypeScript setup
- a responsive local workspace interface
- live SQLite, Qdrant, and Ollama status reporting
- empty-library and backend-offline states
- PDF upload with automatic ingestion-status updates and failure explanations
- explicit selection of up to 20 ready papers and a question form
- retrieved source excerpts with paper titles and physical PDF page numbers
- a development proxy to the FastAPI backend
- formatting, linting, tests, type checking, and production builds

## Prerequisites

- Python 3.12 through 3.14
- `uv`
- Node.js 24 or newer and npm
- Docker with Docker Compose (for the packaged application)
- Ollama available at `http://127.0.0.1:11434`
- `gemma4:12b` and `qwen3-embedding:0.6b` installed in Ollama

## Repository contents

The repository includes application source, dependency lockfiles, tests,
Docker configuration, evaluation fixtures, and project documentation.
Local `.env` settings, uploaded PDFs, SQLite data, Qdrant storage, downloaded
models, and installed dependencies are not included. An existing local library
stays on its original machine; a new checkout starts with an empty library.
Use `.env.example` as the template when configuring a new checkout.

## Packaged application

Ollama runs directly on the Linux host so it can use the AMD GPU. The backend,
frontend, and Qdrant run in Docker with host networking, allowing them to reach
Ollama while keeping all application ports bound to localhost.

Ensure Ollama and the Docker daemon are running, then start the complete
application with one command:

```bash
docker compose up --build
```

If the earlier standalone `research-paper-qdrant` container is still present,
migrate it once before the first Compose startup:

```bash
docker stop research-paper-qdrant
docker rm research-paper-qdrant
```

These commands remove only the old container, not its
`research-paper-qdrant-storage` volume. Compose attaches the replacement
Qdrant container to the same volume.

Open `http://127.0.0.1:5173`. The API and its interactive documentation are at
`http://127.0.0.1:8000` and `http://127.0.0.1:8000/docs`.

SQLite data and future uploaded papers persist in the local `data/` directory.
Qdrant data persists in the named `research-paper-qdrant-storage` Docker
volume. The Compose project reuses that volume if it already exists.

Stop the application with:

```bash
docker compose down
```

This does not remove persistent data. Avoid `docker compose down --volumes`
unless the Qdrant index should be permanently deleted.

The Compose topology uses Linux host networking because Ollama remains bound
to host localhost for privacy and direct GPU access.

## Query papers in the web app

Open `http://127.0.0.1:5173` and use **Upload a PDF** in the paper library.
Processing status updates automatically. Once a paper is `ready`, select its
checkbox, enter a question under **Search your papers**, and click **Search
papers**. Select multiple papers to search them together. Results show matching
passages, paper titles, and physical PDF page numbers.

The current interface retrieves evidence; generated answers and streaming chat
remain the next milestone. Failed uploads and searches show errors in the page.
For failed ingestion, use **Why did processing fail?** to see the explanation.
Deletion and ingestion retries remain available through the API documentation.

Browser search uses the existing retrieval endpoint. For Compose, set
`RETRIEVAL_DEBUG_ENABLED=true` in `.env` (already enabled in this local workspace)
before startup. This route remains disabled by default for production deployments.

## Manual development setup

```bash
uv sync
cp .env.example .env
cd frontend && npm install
```

Start the backend:

```bash
uv run uvicorn research_paper_assistant.main:app --reload
```

The API is available at `http://127.0.0.1:8000`. Interactive documentation is
available at `http://127.0.0.1:8000/docs`.

In a second terminal, start the frontend:

```bash
cd frontend
npm run dev
```

The workspace is available at `http://127.0.0.1:5173`. Vite proxies `/api`
requests to the backend at `http://127.0.0.1:8000`.

## Endpoints

```text
GET /                         Application metadata
GET /api/v1/health/live       Process liveness
GET /api/v1/health/ready      SQLite, Qdrant, and Ollama readiness
GET /api/v1/papers            Current paper library
POST /api/v1/papers           Upload one PDF as multipart field `file`
GET /api/v1/papers/{id}       Paper metadata and ingestion state
DELETE /api/v1/papers/{id}    Delete paper vectors, stored PDF, and metadata
POST /api/v1/papers/{id}/retry Requeue a failed ingestion attempt
POST /api/v1/debug/retrieval  Inspect retrieval (development or explicit local opt-in)
```

The readiness endpoint returns HTTP 503 when any required dependency or Ollama
model is unavailable.

Uploads must have a `.pdf` filename, a PDF content type, a valid PDF signature,
and a readable PDF structure. Password-protected, corrupt, duplicate, and
oversized files are rejected. The default upload limit is 50 MiB and can be
changed with `MAX_PDF_SIZE_BYTES`. User filenames are retained only as metadata;
stored files always use generated UUID names.

## Paper ingestion

Upload through the API documentation at `http://127.0.0.1:8000/docs`, or:

```bash
curl --fail -F 'file=@/path/to/paper.pdf;type=application/pdf' \
  http://127.0.0.1:5173/api/v1/papers
```

The upload returns HTTP 201 with `status: pending`. A background worker polls
the persisted SQLite queue and progresses each paper through `processing` to
`ready` or `failed`. Poll `GET /api/v1/papers/{id}` to check status and read
`error_message`. The frontend supports uploads and updates statuses automatically
while papers are pending or processing.

Text is extracted with PyMuPDF in page order, normalized, and split within each
physical PDF page. Citation page numbers are 1-based PDF page positions, not
printed page labels. Completely scanned or text-empty documents fail with an
OCR explanation. Blank and image-only pages in mixed PDFs are skipped; only
extractable text is indexed. There is no OCR or figure interpretation.

Initial chunk sizes use a **character-based token estimate** (four characters
per token for English prose), not the Qwen tokenizer. The defaults target 700
estimated tokens with 100 estimated tokens of overlap: at most 2,800 characters,
with roughly 400 characters of overlap. Cuts prefer whitespace and preserve
exact normalized source excerpts. Token counts for non-English text, formulas,
and code can differ substantially. Ollama truncation is disabled so oversized
model inputs fail explicitly instead of silently losing evidence.

The default collection is `paper_chunks_qwen3_0_6b_v1`, using 1,024-dimensional
cosine vectors. Each point includes `paper_id`, `paper_title`, `page_number`,
`chunk_id`, per-page `chunk_index`, `text`, and `embedding_model`. A keyword
index on `paper_id` supports later paper-filtered retrieval. Embeddings are
validated for count, dimension, and finite nonzero values. Qdrant writes must
finish before a paper becomes ready.

`CHUNK_TOKENS`, `CHUNK_OVERLAP_TOKENS`, `EMBEDDING_BATCH_SIZE` (default 16),
`INDEXING_TIMEOUT_SECONDS` (default 120), `INGESTION_POLL_SECONDS` (default 1),
`EMBEDDING_DIMENSIONS`, and `QDRANT_COLLECTION` are configurable in `.env` for
both development and Compose. If changing embedding models, use a new collection
and rebuild the library; never mix embeddings from different models even if
their dimensions match. The app does not migrate existing indexes automatically.

Failed jobs retain their original PDF and error. After resolving the problem:

```bash
curl --fail -X POST http://127.0.0.1:5173/api/v1/papers/PAPER_ID/retry
```

Retry returns HTTP 202 and clears the previous error; retrying a paper that is
not failed returns HTTP 409. At startup, interrupted `processing` jobs are
requeued automatically. Each attempt removes stale vectors before writing
deterministic chunk IDs, preventing duplicates after restart. Failed attempts
try to remove partial vectors; if Qdrant is unavailable, the error reports
unconfirmed cleanup. Such papers must not be used for retrieval.

Deletion waits for active ingestion of that paper, then removes its vectors,
PDF, and SQLite records. If vector cleanup fails, records and the PDF remain
so deletion can be retried. If file cleanup fails after vectors were removed,
the paper is marked failed. A long ingestion may outlast an HTTP client's
timeout when deletion is requested; check the library before retrying.

Run **one backend process** against this SQLite database and storage directory.
The worker and per-paper locks run in that process; multiple Uvicorn workers
or backend replicas are not supported. PDF processing runs off the event loop,
and one job at a time bounds simultaneous GPU work. No new database columns,
Redis, or separate task service are required.

The adapters follow the [Ollama embedding API](https://docs.ollama.com/api/embed)
and [Qdrant point API](https://api.qdrant.tech/api-reference/points/upsert-points).
Retrieval is available through the debug API below. Answer generation is the
next milestone.

For a repeatable live test after starting Compose, run:

```bash
uv run python scripts/smoke_ingestion.py
```

This uploads an 18-page PDF larger than 1 MiB through Nginx, verifies its
vectors and page payloads, tests blank-PDF failure and retry, then removes
only the test's own PDFs, vectors, metadata, and ingestion jobs.

## Retrieval development API

`POST /api/v1/debug/retrieval` accepts a question and an explicit list of ready
paper IDs. No chat or generation model is called. In development/test the route
is enabled; in production it is absent from routing and OpenAPI unless
`RETRIEVAL_DEBUG_ENABLED=true`. The local Compose workspace can opt in by adding
that setting to `.env` and running `docker compose up --build --detach`.

```bash
curl --fail -H 'Content-Type: application/json' \
  -d '{"question":"How was the experiment evaluated?","paper_ids":["PAPER_UUID"],"limit":8}' \
  http://127.0.0.1:5173/api/v1/debug/retrieval
```

After trimming leading/trailing whitespace, questions must contain 1-2,000
characters. Selections
must contain 1-20 distinct UUIDs. `limit` defaults to 8 (range 1-8). A missing
paper returns 404; any selected paper that is pending, processing, or failed
returns 409. Requests never silently fall back to searching the whole library
or just the ready subset. Invalid requests return 422; dependency/index errors
and the 30-second retrieval deadline return 503.

The question uses Qwen's instruction-prefixed query format; document embeddings
remain unchanged. Qdrant returns at most 20 candidates filtered by selected
`paper_id` values and the configured embedding model. Maximal marginal
relevance (MMR, relevance weight 0.7) retains the best hit and balances the
remaining passages' relevance against similarity to already selected passages.
Whitespace-equivalent repeated text within a paper is suppressed. Fewer than
eight passages can be returned when there are fewer eligible or distinct hits;
the selection does not force equal representation of each selected paper.

The response includes `question`, `paper_ids`, `candidate_count` (before
deduplication/diversity selection), and `sources`. Each source contains:

- `source_id`: 1-based position in this response, suitable for later citation markers
- `paper_id`, `paper_title`: paper identity and authoritative SQLite title
- `page_number`: physical 1-based PDF page position
- `chunk_id`: stable indexed chunk UUID
- `excerpt`: the full normalized indexed chunk text
- `score`: original dense cosine similarity, not an answer confidence score

Sources appear in MMR selection order, so scores need not be descending.
Vectors are used internally and are not included in the public response.
The service checks payload ownership, page bounds, chunk UUID mapping, embedding
model, and vector validity. It holds ordered per-paper locks while retrieving
to prevent concurrent deletion/re-ingestion from invalidating source metadata.
This retains the existing one-backend-process constraint.

An optional `score_threshold` from -1 to 1 can be supplied for experiments.
There is no default cutoff: a nearest passage is not necessarily sufficient
evidence. Empty results return HTTP 200 with `sources: []`; answerability and
insufficient-evidence behavior remain work for the generation milestone.

Run the labeled synthetic baseline against the local stack:

```bash
uv run python scripts/evaluate_retrieval.py
```

The script uses `evals/retrieval_baseline.json` to generate three temporary
three-page PDFs and query nine known questions. It reports top-1 accuracy,
hit-at-3, source-page checks, and filtering checks, then removes its own uploads
and vectors. Its exit gate is at least 80% top-1 accuracy and 100% hit-at-3.
This small synthetic corpus verifies the pipeline; representative real-paper
evaluation is still needed before claiming general retrieval quality or tuning
an answerability threshold.

Implementation references: [Qwen query instructions](https://github.com/QwenLM/Qwen3-Embedding#usage)
and the [Qdrant query API](https://api.qdrant.tech/api-reference/search/query-points).

## Quality checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest

cd frontend
npm run format:check
npm run lint
npm run typecheck
npm test
npm run build
```
