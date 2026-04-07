# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

x64rag is a dual-SDK Python package providing two AI pipelines:
- **Retrieval SDK** (`src/x64rag/retrieval/`) — Document ingestion, multi-path semantic search, LLM-grounded generation
- **Reasoning SDK** (`src/x64rag/reasoning/`) — Text analysis, classification, clustering, compliance checking, evaluation, pipeline composition

Both SDKs share common infrastructure in `src/x64rag/common/` (errors, language model config, logging, concurrency, CLI utilities). Each SDK has its own `common/` that re-exports from the shared common — never duplicate code between them.

## Commands

All tasks run via [poethepoet](https://github.com/nat-n/poethepoet). Prefix with `uv run` if not in the venv:

```bash
poe format                    # ruff format
poe check                     # ruff lint
poe check:fix                 # ruff lint with auto-fix
poe typecheck                 # mypy src/
poe test                      # pytest (asyncio_mode=auto, pythonpath=src)
poe test:cov                  # pytest with coverage
poe baml:generate:retrieval   # regenerate retrieval BAML clients
poe baml:generate:reasoning   # regenerate reasoning BAML clients
```

Run a single test: `pytest src/x64rag/retrieval/tests/test_search.py::test_name -v`

## Architecture

### Package Structure

```
src/x64rag/
├── __init__.py          # Re-exports everything from both SDKs
├── cli.py               # Unified CLI: x64rag retrieval ... / x64rag reasoning ...
├── common/              # Shared across both SDKs
│   ├── errors.py        # X64RagError, ConfigurationError (base classes)
│   ├── language_model.py # LanguageModelConfig, build_registry (BAML ClientRegistry)
│   ├── logging.py       # get_logger (env: X64RAG_LOG_ENABLED, X64RAG_LOG_LEVEL)
│   ├── startup.py       # BAML version check (parameterized per SDK)
│   ├── concurrency.py   # run_concurrent helper
│   └── cli.py           # ConfigError, CONFIG_DIR, load_dotenv
├── retrieval/
│   ├── common/           # Re-exports from x64rag.common + retrieval-specific (models, formatting, hashing, page_range)
│   ├── server.py         # RagServer — main entry point, wires all modules
│   ├── modules/
│   │   ├── ingestion/    # chunk/ (chunker, parsers, batch), analyze/ (structured 3-phase), embeddings/, vision/, tree/ (TOC detection, structure building)
│   │   ├── retrieval/    # search/ (vector, keyword/BM25, reranking/, rewriting/), refinement/, enrich/, judging, tree/ (BAML tool-use loop)
│   │   ├── generation/   # service, step, grounding, confidence
│   │   ├── knowledge/    # manager (CRUD), migration
│   │   └── evaluation/   # metrics (ExactMatch, F1, LLMJudge), retrieval_metrics
│   ├── stores/           # vector/ (Qdrant), metadata/ (SQLAlchemy), document/ (Postgres, filesystem), graph/ (Neo4j)
│   ├── cli/              # Click commands, config loader, output formatters
│   └── baml/             # baml_src/ (edit) + baml_client/ (generated, do not edit)
└── reasoning/
    ├── common/           # Re-exports from x64rag.common
    ├── modules/
    │   ├── analysis/     # AnalysisService — intent, dimensions, entities, context tracking
    │   ├── classification/ # ClassificationService — LLM or hybrid kNN→LLM
    │   ├── clustering/   # ClusteringService — K-Means, HDBSCAN, LLM labeling
    │   ├── compliance/   # ComplianceService — policy violation checking
    │   ├── evaluation/   # EvaluationService — similarity + LLM judge scoring
    │   └── pipeline/     # Pipeline — sequential step composition
    ├── protocols.py      # BaseEmbeddings, BaseVectorStore (structural typing)
    ├── cli/              # Click commands, config loader, output formatters
    └── baml/             # baml_src/ (edit) + baml_client/ (generated, do not edit)
```

### Entry Points

- **Retrieval:** `RagServer` in `server.py` — async context manager. `async with RagServer(config) as rag:`
- **Reasoning:** Services are standalone (`AnalysisService`, `ClassificationService`, etc.). `Pipeline` composes them sequentially.
- **CLI:** `x64rag retrieval <cmd>` / `x64rag reasoning <cmd>` (also standalone: `x64rag-retrieval`, `x64rag-reasoning`)
- **SDK import:** `from x64rag import RagServer, Pipeline, AnalysisService` — top-level re-exports everything from both SDKs

### Retrieval Pipeline Flow

The retrieval pipeline in `RagServer` runs in this order:

1. **Query rewriting** (pre-retrieval, optional) — HyDE, multi-query, or step-back. Expands 1 query into multiple variants via an LLM call. Configured via `RetrievalConfig.query_rewriter`.
2. **Multi-path search** (per query) — up to 6 concurrent paths, results merged via reciprocal rank fusion:
   - **Vector** — Dense similarity (always on) + SPLADE sparse vectors (hybrid search in Qdrant when `sparse_embeddings` configured)
   - **Keyword/BM25** — In-memory BM25 via `rank-bm25` (`bm25_enabled=True`). Auto-disabled when sparse embeddings are configured since SPLADE supersedes it.
   - **Document** — Full-text + substring search (requires document store)
   - **Graph** — Entity lookup + N-hop traversal (requires graph store)
   - **Enrich** — Structured retrieval with field filtering (requires metadata store)
   - **Tree** — LLM reasoning over hierarchical document structure (requires metadata store + `TreeSearchConfig.enabled`)
3. **Reranking** (optional) — Cross-encoder reranking against original query (Cohere, Voyage)
4. **Chunk refinement** (optional) — Extractive (context window) or abstractive (LLM summarization) refinement
5. **Generation** (for `query()` only) — Grounding gate → LLM relevance gate → optional clarification → LLM generation

### Error Hierarchy

```
X64RagError (common base)
├── ConfigurationError (shared)
├── RagError (retrieval)
│   ├── IngestionError, ParseError, EmptyDocumentError, EmbeddingError, IngestionInterruptedError, TreeIndexingError
│   ├── RetrievalError, GenerationError, TreeSearchError
│   └── StoreError, DuplicateSourceError, SourceNotFoundError
└── AceError (reasoning)
    ├── AnalysisError, ClassificationError, ClusteringError
    ├── ComplianceError, EvaluationError
```

### LLM Integration

All LLM calls go through BAML for structured output parsing, retry/fallback policies, and observability. Each SDK has its own `baml_src/` (source definitions) and `baml_client/` (auto-generated — never edit). After modifying `.baml` files, regenerate with `poe baml:generate:retrieval` or `poe baml:generate:reasoning`.

`LanguageModelConfig` in `common/language_model.py` builds a BAML `ClientRegistry` with primary + optional fallback provider routing.

## Key Patterns

- **Protocol-based abstraction** — No inheritance; `Protocol` classes define interfaces (`BaseEmbeddings`, `BaseVectorStore`, `BaseReranking`, etc.). Any conforming object works.
- **Async-first** — All I/O is async. Services use `async def`, stores use asyncpg/aiosqlite.
- **Service pattern** — Each module has a `Service` class with dependencies injected via `__init__`.
- **Shared common, SDK-specific re-exports** — SDK `common/` modules are thin re-exports from `x64rag.common`. Retrieval-specific utilities (models, formatting, hashing, page_range) stay in retrieval's own `common/`.
- **Config dataclasses** — Pydantic V2 or plain dataclasses with `__post_init__` validation.

## Linting & Style

- Ruff: line-length 120, target py312, rules: E, F, I, UP, B, SIM
- MyPy: python 3.12, ignores missing imports
- Both tools exclude `baml_client/` directories

## Testing

- pytest with `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed
- Tests use `AsyncMock` and `SimpleNamespace` for lightweight mocking
- Tests in `tests/` subdirectories within each SDK + inline `test_*.py` in some modules
- 487 tests total across both SDKs

## Environment Variables

- `X64RAG_LOG_ENABLED=true` / `X64RAG_LOG_LEVEL=DEBUG` — SDK logging
- `X64RAG_PROVIDER`, `X64RAG_MODEL`, `X64RAG_API_KEY` — Override reasoning CLI provider/model/key
- `BAML_LOG=info|warn|debug` — BAML runtime logging
- Config lives at `~/.config/x64rag/config.toml` + `.env`
