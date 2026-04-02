<img width="1536" height="480" alt="x64rag-ember" src="https://github.com/user-attachments/assets/c9395228-56eb-4c16-a0dc-076e1c8b34ab" />

## Retrieval

Composable retrieval-augmented generation. Ingest documents, search across vector, document, and graph stores in parallel, generate grounded answers with quality gates.

[Documentation](src/x64rag/retrieval/README.md) · [Examples](examples/retrieval)

```python
from x64rag.retrieval import RagServer, RagServerConfig, PersistenceConfig, IngestionConfig
from x64rag.retrieval import QdrantVectorStore, OpenAIEmbeddings

config = RagServerConfig(
    persistence=PersistenceConfig(
        vector_store=QdrantVectorStore(url="http://localhost:6333", collection="docs"),
    ),
    ingestion=IngestionConfig(
        embeddings=OpenAIEmbeddings(api_key="...", model="text-embedding-3-small"),
    ),
)

async with RagServer(config) as rag:
    await rag.ingest("manual.pdf", knowledge_id="equipment")
    result = await rag.query("How do I replace the filter?", knowledge_id="equipment")
    print(result.answer)
```

## Reasoning

Analysis, classification, compliance, evaluation, clustering, and pipeline composition. Each service is standalone — use one or compose them through pipelines.

[Documentation](src/x64rag/reasoning/README.md) · [Get Started](examples/reasoning)

```python
from x64rag.reasoning import AnalysisService, AnalysisConfig, DimensionDefinition
from x64rag.common.language_model import LanguageModelConfig, LanguageModelClientConfig

lm = LanguageModelConfig(
    client=LanguageModelClientConfig(provider="anthropic", model="claude-sonnet-4-20250514", api_key="..."),
)

analyzer = AnalysisService(lm_config=lm)
result = await analyzer.analyze(
    "My order FB-12345 hasn't arrived and I need it by Friday.",
    config=AnalysisConfig(
        dimensions=[DimensionDefinition("urgency", "How time-sensitive", "0.0-1.0")],
        summarize=True,
    ),
)
print(f"{result.primary_intent} — urgency: {result.dimensions['urgency'].value}")
```

## Installation

```bash
uv add x64rag

uv add "x64rag[graph]"                  # graph + Neo4j support
uv add "x64rag[cli]"                    # CLI support

uv sync --all-extras
uv run poe format                       # ruff format
uv run poe check                        # ruff lint
uv run poe check:fix                    # ruff lint + auto-fix
uv run poe typecheck                    # mypy type checking
uv run poe test                         # pytest
uv run poe baml:generate:retrieval      # regenerate retrieval BAML client
uv run poe baml:generate:reasoning      # regenerate reasoning BAML client
```

## Observability

All LLM calls go through [BAML](https://docs.boundaryml.com/) for structured output parsing, retry/fallback policies, and observability.

**Boundary Studio** — Set `boundary_api_key` in any `LanguageModelConfig` to enable automatic cloud tracing with token counts, latency, and function-level tracking.

**Programmatic** — Use `baml_py.Collector` for in-process token usage tracking.

## Env Variables

```bash
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
COHERE_API_KEY=
VOYAGE_API_KEY=

X64RAG_PROVIDER=
X64RAG_MODEL=
X64RAG_API_KEY=

X64RAG_LOG_ENABLED=false    # true / false
X64RAG_LOG_LEVEL=INFO       # DEBUG, INFO, WARNING, ERROR
BAML_LOG=warn               # info, warn, debug
```
