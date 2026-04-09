<img width="200" alt="x64rag-ember(2)" src="https://github.com/user-attachments/assets/ce3262aa-7453-48d4-992d-94a6a5dedd2a" />

Retrieval / Reasoning Augmented Generation (RAG) Python SDK

## Retrieval-Augmented Generation

Modular retrieval-augmented generation with pluggable pipeline methods. No mandatory vector database — configure the retrieval paths you need: dense and sparse vector search, BM25 keyword matching, full-text document search, entity-relationship graph traversal, and LLM-powered tree navigation over hierarchical document structure. Methods run concurrently with per-method weights, results fuse via reciprocal rank fusion, and each method handles its own errors independently. Ingest through chunking, vision-analyzed drawings, or LLM entity extraction. Generate grounded answers with score gates, relevance judgment, and clarification flows.

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

```python
# Access individual retrieval methods for fine-grained control
async with RagServer(config) as rag:
    await rag.ingest("manual.pdf", knowledge_id="equipment")

    # Use individual methods directly
    vector_chunks = await rag.retrieval.vector.search("pressure specs", top_k=20)
    doc_chunks = await rag.retrieval.document.search("pressure specs", top_k=10)

    # Or let the pipeline handle everything
    result = await rag.query("What are the pressure specifications?", knowledge_id="equipment")
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

## Why x64rag

### vs. Long Context LLM

Long context windows and lexical search work for small document sets where you know what terms to look for. They break when knowledge bases grow beyond context limits, when users ask semantic questions ("how to change oil" vs. the manual's "lubricant replacement procedure"), and when you need entity relationships that no amount of text matching can surface. x64rag makes vector search optional — you can run document-store-only or graph-only configs — but when you need semantic understanding across thousands of pages, concurrent multi-path retrieval with fusion outperforms any single method.

### vs. Vectorless Tree Search (PageIndex)

Tree-based retrieval proves that LLM reasoning over document structure beats vector similarity for navigating long structured documents. x64rag's tree search follows the same principle — build a hierarchical index, let the LLM navigate it. The difference is that tree search is one method in the pipeline, not the only one. When the tree can't find the answer (wrong section, entity not in the TOC), vector search, BM25, document search, and graph traversal are all running in parallel. Tree search adds structural precision; the other methods add breadth and resilience.

### vs. LangChain / LlamaIndex

Framework orchestrators provide abstractions over LLM calls, prompt chains, and retrieval steps. x64rag is not an orchestrator — it's a retrieval engine. It owns the full pipeline from document parsing through chunk embedding, multi-path search, reranking, and grounded generation. No chain composition, no prompt templates, no agent loops. One `async with RagServer(config) as rag:` gives you ingestion, retrieval, and generation with quality gates. Use it inside LangChain if you want, or use it standalone — the SDK handles the retrieval problem end-to-end without external framework dependencies.

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
