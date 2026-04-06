from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from x64rag.retrieval.common.errors import ConfigurationError
from x64rag.retrieval.common.language_model import LanguageModelConfig
from x64rag.retrieval.common.logging import get_logger
from x64rag.retrieval.common.models import RetrievedChunk, Source
from x64rag.retrieval.modules.generation.models import QueryResult, StepResult, StreamEvent
from x64rag.retrieval.modules.generation.service import GenerationService
from x64rag.retrieval.modules.generation.step import StepGenerationService
from x64rag.retrieval.modules.ingestion.analyze.service import StructuredIngestionService
from x64rag.retrieval.modules.ingestion.chunk.chunker import SemanticChunker
from x64rag.retrieval.modules.ingestion.chunk.service import IngestionService
from x64rag.retrieval.modules.ingestion.embeddings.base import BaseEmbeddings
from x64rag.retrieval.modules.ingestion.embeddings.sparse.base import BaseSparseEmbeddings
from x64rag.retrieval.modules.ingestion.vision.base import BaseVision
from x64rag.retrieval.modules.knowledge.manager import KnowledgeManager
from x64rag.retrieval.modules.knowledge.migration import check_embedding_migration
from x64rag.retrieval.modules.retrieval.enrich.service import StructuredRetrievalService
from x64rag.retrieval.modules.retrieval.refinement.base import BaseChunkRefiner
from x64rag.retrieval.modules.retrieval.search.keyword import KeywordSearch
from x64rag.retrieval.modules.retrieval.search.reranking.base import BaseReranking
from x64rag.retrieval.modules.retrieval.search.rewriting.base import BaseQueryRewriter
from x64rag.retrieval.modules.retrieval.search.service import RetrievalService
from x64rag.retrieval.modules.retrieval.search.vector import VectorSearch
from x64rag.retrieval.stores.document.base import BaseDocumentStore
from x64rag.retrieval.stores.graph.base import BaseGraphStore
from x64rag.retrieval.stores.metadata.base import BaseMetadataStore
from x64rag.retrieval.stores.vector.base import BaseVectorStore

logger = get_logger("server")

SUPPORTED_STRUCTURED_EXTENSIONS = {".xml", ".l5x"}

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use only the provided context to answer questions. "
    "Cite sources with page numbers when available. If the context does not contain "
    "enough information to answer, say so."
)


@dataclass
class PersistenceConfig:
    vector_store: BaseVectorStore
    metadata_store: BaseMetadataStore | None = None
    document_store: BaseDocumentStore | None = None
    graph_store: BaseGraphStore | None = None


@dataclass
class IngestionConfig:
    embeddings: BaseEmbeddings
    vision: BaseVision | None = None
    chunk_size: int = 500
    chunk_overlap: int = 50
    dpi: int = 300
    lm_config: LanguageModelConfig | None = None
    sparse_embeddings: BaseSparseEmbeddings | None = None
    parent_chunk_size: int = 0
    parent_chunk_overlap: int = 200
    contextual_chunking: bool = True

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ConfigurationError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ConfigurationError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError("chunk_overlap must be less than chunk_size")
        if self.parent_chunk_size < 0:
            raise ConfigurationError("parent_chunk_size must be non-negative")
        if self.parent_chunk_size > 0 and self.parent_chunk_size <= self.chunk_size:
            raise ConfigurationError("parent_chunk_size must be greater than chunk_size")


@dataclass
class RetrievalConfig:
    top_k: int = 5
    reranker: BaseReranking | None = None
    query_rewriter: BaseQueryRewriter | None = None
    bm25_enabled: bool = False
    bm25_max_indexes: int = 16
    source_type_weights: dict[str, float] | None = None
    cross_reference_enrichment: bool = True
    enrich_lm_config: LanguageModelConfig | None = None
    parent_expansion: bool = True
    chunk_refiner: BaseChunkRefiner | None = None

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ConfigurationError("top_k must be positive")


@dataclass
class GenerationConfig:
    lm_config: LanguageModelConfig | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    grounding_enabled: bool = False
    grounding_threshold: float = 0.5
    relevance_gate_enabled: bool = False
    relevance_gate_model: LanguageModelConfig | None = None
    guiding_enabled: bool = False
    step_lm_config: LanguageModelConfig | None = None

    def __post_init__(self) -> None:
        if self.grounding_threshold < 0 or self.grounding_threshold > 1:
            raise ConfigurationError("grounding_threshold must be between 0 and 1")
        if self.relevance_gate_enabled and not self.grounding_enabled:
            raise ConfigurationError("relevance_gate_enabled requires grounding_enabled")
        if self.relevance_gate_enabled and not self.relevance_gate_model:
            raise ConfigurationError("relevance_gate_enabled requires relevance_gate_model")
        if self.guiding_enabled and not self.relevance_gate_enabled:
            raise ConfigurationError("guiding_enabled requires relevance_gate_enabled")


@dataclass
class TreeIndexingConfig:
    """Configuration for tree-based document indexing."""

    enabled: bool = False
    model: LanguageModelConfig | None = None
    toc_scan_pages: int = 20
    max_pages_per_node: int = 10
    max_tokens_per_node: int = 20_000
    generate_summaries: bool = True
    generate_description: bool = True

    def __post_init__(self) -> None:
        if self.toc_scan_pages < 1:
            raise ConfigurationError("toc_scan_pages must be positive")
        if self.max_pages_per_node < 1:
            raise ConfigurationError("max_pages_per_node must be positive")
        if self.max_tokens_per_node < 1:
            raise ConfigurationError("max_tokens_per_node must be positive")


@dataclass
class TreeSearchConfig:
    """Configuration for tree-based search."""

    enabled: bool = False
    model: LanguageModelConfig | None = None
    max_steps: int = 5
    max_context_tokens: int = 50_000

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ConfigurationError("max_steps must be positive")
        if self.max_context_tokens < 1:
            raise ConfigurationError("max_context_tokens must be positive")


@dataclass
class RagServerConfig:
    persistence: PersistenceConfig
    ingestion: IngestionConfig
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    tree_indexing: TreeIndexingConfig = field(default_factory=TreeIndexingConfig)
    tree_search: TreeSearchConfig = field(default_factory=TreeSearchConfig)


def _derive_embedding_model_name(embeddings: BaseEmbeddings) -> str:
    """Build fingerprint string from provider class and model name."""
    cls_name = type(embeddings).__name__.lower().replace("embeddings", "")
    return f"{cls_name}:{embeddings.model}"


class RagServer:
    def __init__(self, config: RagServerConfig) -> None:
        self._config = config
        self._initialized = False

        self._unstructured_ingestion: IngestionService | None = None
        self._structured_ingestion: StructuredIngestionService | None = None
        self._unstructured_retrieval: RetrievalService | None = None
        self._structured_retrieval: StructuredRetrievalService | None = None
        self._generation_service: GenerationService | None = None
        self._knowledge_manager: KnowledgeManager | None = None
        self._keyword_search: KeywordSearch | None = None
        self._step_service: StepGenerationService | None = None

        self._retrieval_by_collection: dict[str, tuple[RetrievalService, StructuredRetrievalService | None]] = {}
        self._ingestion_by_collection: dict[str, IngestionService] = {}

        self._chunker: SemanticChunker | None = None
        self._embedding_model_name: str = ""

    @property
    def knowledge(self) -> KnowledgeManager:
        self._check_initialized()
        assert self._knowledge_manager is not None
        return self._knowledge_manager

    @property
    def collections(self) -> list[str]:
        """Available Qdrant collections."""
        store = self._config.persistence.vector_store
        if hasattr(store, "collections"):
            return store.collections  # type: ignore[no-any-return]
        return []

    async def initialize(self) -> None:
        """Wire all modules and check embedding model consistency."""
        cfg = self._config
        persistence = cfg.persistence
        ingestion = cfg.ingestion
        retrieval = cfg.retrieval
        gen = cfg.generation

        logger.info("ragserver initializing")

        if persistence.metadata_store:
            await persistence.metadata_store.initialize()
            logger.info("persistence: vector + metadata stores")
        else:
            logger.info("persistence: vector store only")

        if persistence.document_store:
            await persistence.document_store.initialize()
            logger.info("persistence: document store enabled")

        if persistence.graph_store:
            await persistence.graph_store.initialize()
            logger.info("persistence: graph store enabled")

        vector_size = await ingestion.embeddings.embedding_dimension()
        await persistence.vector_store.initialize(vector_size)

        self._embedding_model_name = _derive_embedding_model_name(ingestion.embeddings)

        self._chunker = SemanticChunker(
            chunk_size=ingestion.chunk_size,
            chunk_overlap=ingestion.chunk_overlap,
            parent_chunk_size=ingestion.parent_chunk_size,
            parent_chunk_overlap=ingestion.parent_chunk_overlap,
        )
        chunker = self._chunker

        if retrieval.bm25_enabled and ingestion.sparse_embeddings:
            logger.warning("sparse_embeddings configured — bm25_enabled ignored (sparse vectors replace BM25)")
        elif retrieval.bm25_enabled:
            self._keyword_search = KeywordSearch(persistence.vector_store, max_indexes=retrieval.bm25_max_indexes)

        self._unstructured_ingestion = IngestionService(
            embeddings=ingestion.embeddings,
            chunker=chunker,
            vector_store=persistence.vector_store,
            embedding_model_name=self._embedding_model_name,
            source_type_weights=retrieval.source_type_weights,
            metadata_store=persistence.metadata_store,
            on_ingestion_complete=self._on_ingestion_complete,
            vision_parser=ingestion.vision,
            document_store=persistence.document_store,
            sparse_embeddings=ingestion.sparse_embeddings,
            contextual_chunking=ingestion.contextual_chunking,
        )

        if persistence.metadata_store:
            self._structured_ingestion = StructuredIngestionService(
                embeddings=ingestion.embeddings,
                vector_store=persistence.vector_store,
                metadata_store=persistence.metadata_store,
                embedding_model_name=self._embedding_model_name,
                vision=ingestion.vision,
                dpi=ingestion.dpi,
                source_type_weights=retrieval.source_type_weights,
                on_ingestion_complete=self._on_ingestion_complete,
                lm_config=ingestion.lm_config,
                document_store=persistence.document_store,
                graph_store=persistence.graph_store,
            )
            if not ingestion.vision:
                logger.warning("no vision provider — structured PDF analysis disabled")

        vector_search = VectorSearch(
            vector_store=persistence.vector_store,
            embeddings=ingestion.embeddings,
            sparse_embeddings=ingestion.sparse_embeddings,
            parent_expansion=retrieval.parent_expansion,
        )
        self._unstructured_retrieval = RetrievalService(
            vector_search=vector_search,
            keyword_search=self._keyword_search,
            reranking=retrieval.reranker,
            top_k=retrieval.top_k,
            source_type_weights=retrieval.source_type_weights,
            document_store=persistence.document_store,
            query_rewriter=retrieval.query_rewriter,
            graph_store=persistence.graph_store,
            chunk_refiner=retrieval.chunk_refiner,
        )

        self._structured_retrieval = StructuredRetrievalService(
            vector_store=persistence.vector_store,
            embeddings=ingestion.embeddings,
            lm_config=retrieval.enrich_lm_config,
            top_k=retrieval.top_k,
            enrich_cross_references=retrieval.cross_reference_enrichment,
        )

        self._retrieval_by_collection.clear()
        store_collections: list[str] = (
            persistence.vector_store.collections if hasattr(persistence.vector_store, "collections") else []
        )
        for coll_name in store_collections:
            if coll_name == store_collections[0]:
                self._retrieval_by_collection[coll_name] = (
                    self._unstructured_retrieval,
                    self._structured_retrieval,
                )
                continue

            scoped_store = persistence.vector_store.scoped(coll_name)  # type: ignore[attr-defined]
            scoped_retrieval = self._build_retrieval_pipeline(scoped_store, ingestion, retrieval, persistence)
            self._retrieval_by_collection[coll_name] = scoped_retrieval
            logger.info("retrieval pipeline built for collection '%s'", coll_name)

        if gen.lm_config:
            relevance_gate_lm_config = gen.relevance_gate_model if gen.relevance_gate_enabled else None
            self._generation_service = GenerationService(
                lm_config=gen.lm_config,
                system_prompt=gen.system_prompt,
                grounding_enabled=gen.grounding_enabled,
                grounding_threshold=gen.grounding_threshold,
                relevance_gate_enabled=gen.relevance_gate_enabled,
                guiding_enabled=gen.guiding_enabled,
                relevance_gate_lm_config=relevance_gate_lm_config,
            )
            logger.info("generation: enabled")
        else:
            logger.info("generation: disabled (retrieval-only mode)")

        if gen.step_lm_config:
            self._step_service = StepGenerationService(lm_config=gen.step_lm_config)
            logger.info("step generation: enabled")

        if gen.grounding_threshold > 0 and not gen.lm_config:
            raise ConfigurationError("generation provider required for grounding gate")

        self._knowledge_manager = KnowledgeManager(
            vector_store=persistence.vector_store,
            metadata_store=persistence.metadata_store,
            on_source_removed=self._on_source_removed,
            document_store=persistence.document_store,
            graph_store=persistence.graph_store,
        )

        stale = await check_embedding_migration(
            metadata_store=persistence.metadata_store,
            embedding_model_name=self._embedding_model_name,
        )
        if stale:
            logger.warning("%d sources are stale and need re-ingestion", stale)

        self._initialized = True

        flows = self._enabled_flows()
        logger.info("ragserver ready — %s flows enabled", ", ".join(flows) if flows else "none")

    async def shutdown(self) -> None:
        """Cleanup all store connections."""
        persistence = self._config.persistence
        try:
            await persistence.vector_store.shutdown()
        except Exception:
            logger.exception("error shutting down vector store")
        if persistence.metadata_store:
            try:
                await persistence.metadata_store.shutdown()
            except Exception:
                logger.exception("error shutting down metadata store")
        if persistence.document_store:
            try:
                await persistence.document_store.shutdown()
            except Exception:
                logger.exception("error shutting down document store")
        if persistence.graph_store:
            try:
                await persistence.graph_store.shutdown()
            except Exception:
                logger.exception("error shutting down graph store")
        self._initialized = False
        logger.info("ragserver shut down")

    async def __aenter__(self) -> RagServer:
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        await self.shutdown()

    async def ingest(
        self,
        file_path: str | Path,
        knowledge_id: str | None = None,
        source_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        page_range: str | None = None,
        resume_from_chunk: int = 0,
        on_progress: Callable[[int, int], Awaitable[None]] | None = None,
        collection: str | None = None,
    ) -> Source:
        """Ingest a file. Routes to unstructured or structured based on extension."""
        self._check_initialized()
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext in SUPPORTED_STRUCTURED_EXTENSIONS and self._structured_ingestion:
            source = await self._structured_ingestion.analyze(
                file_path=file_path,
                knowledge_id=knowledge_id,
                source_type=source_type,
                metadata=metadata,
                page_range=page_range,
            )
            source = await self._structured_ingestion.synthesize(source.source_id)
            source = await self._structured_ingestion.ingest(source.source_id)
            return source

        ingestion_svc = self._get_ingestion(collection)
        return await ingestion_svc.ingest(
            file_path=file_path,
            knowledge_id=knowledge_id,
            source_type=source_type,
            metadata=metadata,
            page_range=page_range,
            resume_from_chunk=resume_from_chunk,
            on_progress=on_progress,
        )

    async def ingest_text(
        self,
        content: str,
        knowledge_id: str | None = None,
        source_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        collection: str | None = None,
    ) -> Source:
        """Ingest raw text content into the RAG pipeline."""
        self._check_initialized()
        ingestion_svc = self._get_ingestion(collection)
        return await ingestion_svc.ingest_text(
            content=content, knowledge_id=knowledge_id, source_type=source_type, metadata=metadata
        )

    async def analyze(
        self,
        file_path: str | Path,
        knowledge_id: str | None = None,
        source_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        page_range: str | None = None,
        collection: str | None = None,
    ) -> Source:
        """Structured phase 1: per-page analysis."""
        self._check_initialized()
        if not self._structured_ingestion:
            raise ConfigurationError("metadata store required for structured ingestion")
        return await self._structured_ingestion.analyze(
            file_path=file_path,
            knowledge_id=knowledge_id,
            source_type=source_type,
            metadata=metadata,
            page_range=page_range,
        )

    async def synthesize(self, source_id: str) -> Source:
        """Structured phase 2: cross-page synthesis."""
        self._check_initialized()
        if not self._structured_ingestion:
            raise ConfigurationError("metadata store required for structured ingestion")
        return await self._structured_ingestion.synthesize(source_id)

    async def complete_ingestion(self, source_id: str) -> Source:
        """Structured phase 3: embed + store."""
        self._check_initialized()
        if not self._structured_ingestion:
            raise ConfigurationError("metadata store required for structured ingestion")
        return await self._structured_ingestion.ingest(source_id)

    async def query(
        self,
        text: str,
        knowledge_id: str | None = None,
        history: list[tuple[str, str]] | None = None,
        min_score: float | None = None,
        collection: str | None = None,
    ) -> QueryResult:
        """Full pipeline: retrieval + grounding + LLM generation."""
        self._check_initialized()
        if not self._generation_service:
            raise RuntimeError("query() requires generation to be configured")

        chunks = await self._retrieve_chunks(text, knowledge_id, history, min_score, collection)
        return await self._generation_service.generate(query=text, chunks=chunks, history=history)

    async def query_stream(
        self,
        text: str,
        knowledge_id: str | None = None,
        history: list[tuple[str, str]] | None = None,
        min_score: float | None = None,
        collection: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Full pipeline with streaming: retrieval + grounding + streamed LLM generation."""
        self._check_initialized()
        if not self._generation_service:
            raise RuntimeError("query_stream() requires generation to be configured")

        chunks = await self._retrieve_chunks(text, knowledge_id, history, min_score, collection)
        async for event in self._generation_service.generate_stream(query=text, chunks=chunks, history=history):
            yield event

    async def retrieve(
        self,
        text: str,
        knowledge_id: str | None = None,
        min_score: float | None = None,
        collection: str | None = None,
    ) -> list[RetrievedChunk]:
        """Low-level retrieval only, no LLM generation."""
        self._check_initialized()
        return await self._retrieve_chunks(text, knowledge_id, None, min_score, collection)

    async def generate_step(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        context: str | None = None,
    ) -> StepResult:
        """Generate a single reasoning step from retrieved chunks.

        Use with retrieve() to build iterative retrieval loops. The consumer
        owns the loop, stopping conditions, and query enrichment between iterations.
        """
        self._check_initialized()
        if not self._step_service:
            raise RuntimeError("generate_step() requires step_lm_config in GenerationConfig")

        return await self._step_service.generate_step(
            query=query,
            chunks=chunks,
            context=context,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts using the configured provider."""
        self._check_initialized()
        return await self._config.ingestion.embeddings.embed(texts)

    async def embed_single(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""
        self._check_initialized()
        vectors = await self._config.ingestion.embeddings.embed([text])
        return vectors[0]

    @staticmethod
    def _build_retrieval_query(text: str, history: list[tuple[str, str]] | None) -> str:
        """Enrich the retrieval query with recent conversation context.

        When history is available, appends key terms from recent exchanges so
        retrieval can find results relevant to the ongoing conversation — not
        just the latest message in isolation.
        """
        if not history:
            return text

        recent = history[-3:]
        context_parts = []
        for human_msg, _assistant_msg in recent:
            context_parts.append(human_msg)

        context = " ".join(context_parts)
        return f"{text}\n\nConversation context: {context}"

    async def _on_ingestion_complete(self, knowledge_id: str | None) -> None:
        """Callback after ingestion — invalidates BM25 cache for the knowledge_id."""
        if self._keyword_search:
            await self._keyword_search.invalidate(knowledge_id)

    async def _on_source_removed(self, knowledge_id: str | None) -> None:
        """Callback after source removal — invalidates BM25 cache for the knowledge_id."""
        if self._keyword_search:
            await self._keyword_search.invalidate(knowledge_id)

    def _build_retrieval_pipeline(
        self,
        vector_store: BaseVectorStore,
        ingestion: IngestionConfig,
        retrieval: RetrievalConfig,
        persistence: PersistenceConfig,
    ) -> tuple[RetrievalService, StructuredRetrievalService | None]:
        vs = VectorSearch(
            vector_store=vector_store,
            embeddings=ingestion.embeddings,
            sparse_embeddings=ingestion.sparse_embeddings,
            parent_expansion=retrieval.parent_expansion,
        )
        kw = KeywordSearch(vector_store, max_indexes=retrieval.bm25_max_indexes) if self._keyword_search else None
        unstructured = RetrievalService(
            vector_search=vs,
            keyword_search=kw,
            reranking=retrieval.reranker,
            top_k=retrieval.top_k,
            source_type_weights=retrieval.source_type_weights,
            document_store=persistence.document_store,
            query_rewriter=retrieval.query_rewriter,
            graph_store=persistence.graph_store,
            chunk_refiner=retrieval.chunk_refiner,
        )
        structured = StructuredRetrievalService(
            vector_store=vector_store,
            embeddings=ingestion.embeddings,
            lm_config=retrieval.enrich_lm_config,
            top_k=retrieval.top_k,
            enrich_cross_references=retrieval.cross_reference_enrichment,
        )
        return unstructured, structured

    def _build_ingestion_service(self, vector_store: BaseVectorStore) -> IngestionService:
        assert self._chunker is not None
        cfg = self._config
        return IngestionService(
            embeddings=cfg.ingestion.embeddings,
            chunker=self._chunker,
            vector_store=vector_store,
            embedding_model_name=self._embedding_model_name,
            source_type_weights=cfg.retrieval.source_type_weights,
            metadata_store=cfg.persistence.metadata_store,
            on_ingestion_complete=self._on_ingestion_complete,
            vision_parser=cfg.ingestion.vision,
            document_store=cfg.persistence.document_store,
            sparse_embeddings=cfg.ingestion.sparse_embeddings,
            contextual_chunking=cfg.ingestion.contextual_chunking,
        )

    async def _retrieve_chunks(
        self,
        text: str,
        knowledge_id: str | None,
        history: list[tuple[str, str]] | None,
        min_score: float | None,
        collection: str | None,
    ) -> list[RetrievedChunk]:
        """Shared retrieval: unstructured + structured merge + score filter."""
        unstructured, structured = self._get_retrieval(collection)
        retrieval_query = self._build_retrieval_query(text, history)

        if structured:
            unstructured_chunks, structured_chunks = await asyncio.gather(
                unstructured.retrieve(query=retrieval_query, knowledge_id=knowledge_id),
                structured.retrieve(query=retrieval_query, knowledge_id=knowledge_id),
            )
            chunks = self._merge_retrieval_results(unstructured_chunks, structured_chunks)
        else:
            chunks = await unstructured.retrieve(query=retrieval_query, knowledge_id=knowledge_id)

        if min_score is not None:
            chunks = [c for c in chunks if c.score >= min_score]
        return chunks

    def _get_retrieval(self, collection: str | None) -> tuple[RetrievalService, StructuredRetrievalService | None]:
        """Return retrieval pipeline for *collection* (default if None)."""
        if collection and collection in self._retrieval_by_collection:
            return self._retrieval_by_collection[collection]
        assert self._unstructured_retrieval is not None
        return self._unstructured_retrieval, self._structured_retrieval

    def _get_ingestion(self, collection: str | None) -> IngestionService:
        """Return ingestion service for *collection*, lazily building if needed."""
        if not collection:
            assert self._unstructured_ingestion is not None
            return self._unstructured_ingestion

        if collection in self._ingestion_by_collection:
            return self._ingestion_by_collection[collection]

        store = self._config.persistence.vector_store
        if not hasattr(store, "scoped"):
            raise ConfigurationError(f"Vector store does not support multi-collection (requested: {collection!r})")

        scoped_store = store.scoped(collection)  # type: ignore[union-attr]
        svc = self._build_ingestion_service(scoped_store)
        self._ingestion_by_collection[collection] = svc
        return svc

    def _check_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "RagServer not initialized. Use 'async with RagServer(config) as rag:' "
                "or call 'await rag.initialize()' first."
            )

    def _enabled_flows(self) -> list[str]:
        flows = ["unstructured"]
        if self._structured_ingestion:
            flows.append("structured")
        if self._generation_service:
            flows.append("generation")
        return flows

    @staticmethod
    def _merge_retrieval_results(
        unstructured: list[RetrievedChunk],
        structured: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Merge unstructured and structured results, dedup by chunk_id, sort by score."""
        seen: set[str] = set()
        merged: list[RetrievedChunk] = []
        for c in unstructured:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                merged.append(c)
        for c in structured:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                merged.append(c)
        merged.sort(key=lambda x: x.score, reverse=True)
        return merged
