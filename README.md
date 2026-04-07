<img width="1536" height="480" alt="x64rag-ember(2)" src="https://github.com/user-attachments/assets/ce3262aa-7453-48d4-992d-94a6a5dedd2a" />

## Retrieval-Augmented Generation

Composable retrieval-augmented generation. Ingest documents, search across vector, document, graph, and tree stores in parallel, generate grounded answers with quality gates.

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
    await rag.ingest("annual_report.pdf", knowledge_id="reports", tree_index=True)  # tree indexing for structured docs
    result = await rag.query("How do I replace the filter?", knowledge_id="equipment")
    print(result.answer)
```

```bash
x64rag retrieval init
x64rag retrieval ingest manual.pdf -k equipment
x64rag retrieval query "how to replace the filter?" -k equipment
x64rag retrieval retrieve "part number 8842-A" -k equipment
```

## Reasoning-Augmented Generation

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

```bash
x64rag reasoning init
x64rag reasoning analyze "My order FB-12345 hasn't arrived and I need it by Friday"
x64rag reasoning classify "I want my money back" --categories categories.json
x64rag reasoning compliance "We'll give you 150% refund" --references policy.md
```

## Installation

```bash
uv add x64rag                           # add x64rag to your project

uv add "x64rag[graph]"                  # x64rag + graph + Neo4j support
uv add "x64rag[cli]"                    # x64rag + CLI support

uv sync --extra dev                     # setup with dev optional
uv sync --all-extras                    # setup with all optional

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
