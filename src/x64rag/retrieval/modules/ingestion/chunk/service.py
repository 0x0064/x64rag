import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from x64rag.retrieval.common.errors import (
    ConfigurationError,
    DuplicateSourceError,
    EmptyDocumentError,
    IngestionInterruptedError,
)
from x64rag.retrieval.common.hashing import file_hash
from x64rag.retrieval.common.logging import get_logger
from x64rag.retrieval.common.models import Source, SparseVector, VectorPoint
from x64rag.retrieval.common.page_range import parse_page_range
from x64rag.retrieval.modules.ingestion.chunk.chunker import SemanticChunker
from x64rag.retrieval.modules.ingestion.chunk.context import contextualize_chunks
from x64rag.retrieval.modules.ingestion.chunk.parsers.pdf import PDFParser
from x64rag.retrieval.modules.ingestion.chunk.parsers.text import TextParser
from x64rag.retrieval.modules.ingestion.embeddings.base import BaseEmbeddings
from x64rag.retrieval.modules.ingestion.embeddings.sparse.base import BaseSparseEmbeddings
from x64rag.retrieval.modules.ingestion.embeddings.utils import embed_batched
from x64rag.retrieval.modules.ingestion.models import ChunkedContent, ParsedPage
from x64rag.retrieval.modules.ingestion.vision.base import BaseVision
from x64rag.retrieval.modules.ingestion.vision.constants import IMAGE_EXTENSIONS
from x64rag.retrieval.stores.document.base import BaseDocumentStore
from x64rag.retrieval.stores.metadata.base import BaseMetadataStore
from x64rag.retrieval.stores.vector.base import BaseVectorStore

logger = get_logger("chunk/ingestion")

INGESTION_BATCH_SIZE = 20

FILE_PARSERS_BY_EXTENSION: dict[str, type] = {
    ".pdf": PDFParser,
    ".txt": TextParser,
    ".md": TextParser,
    ".text": TextParser,
}

IngestionProgress = Callable[[int, int], Awaitable[None]]


class IngestionService:
    def __init__(
        self,
        embeddings: BaseEmbeddings,
        chunker: SemanticChunker,
        vector_store: BaseVectorStore,
        embedding_model_name: str,
        source_type_weights: dict[str, float] | None = None,
        metadata_store: BaseMetadataStore | None = None,
        on_ingestion_complete: Callable[[str | None], Awaitable[None]] | None = None,
        vision_parser: BaseVision | None = None,
        document_store: BaseDocumentStore | None = None,
        sparse_embeddings: BaseSparseEmbeddings | None = None,
        contextual_chunking: bool = True,
    ) -> None:
        self._embeddings = embeddings
        self._chunker = chunker
        self._vector_store = vector_store
        self._metadata_store = metadata_store
        self._embedding_model_name = embedding_model_name
        self._source_type_weights = source_type_weights
        self._on_ingestion_complete = on_ingestion_complete
        self._vision_parser = vision_parser
        self._document_store = document_store
        self._sparse_embeddings = sparse_embeddings
        self._contextual_chunking = contextual_chunking

    def _resolve_weight(self, source_type: str | None) -> float:
        if self._source_type_weights is None:
            return 1.0
        if source_type is None:
            return 1.0
        if source_type not in self._source_type_weights:
            raise ConfigurationError(
                f"source_type '{source_type}' is not defined in source_type_weights. "
                f"Valid types: {sorted(self._source_type_weights.keys())}"
            )
        return self._source_type_weights[source_type]

    async def _check_duplicate(self, hash_value: str, knowledge_id: str | None) -> None:
        if not self._metadata_store:
            return
        existing = await self._metadata_store.list_sources(knowledge_id=knowledge_id)
        for source in existing:
            if source.file_hash == hash_value:
                raise DuplicateSourceError(
                    f"File already ingested as source {source.source_id} (hash={hash_value[:12]}...)"
                )

    async def _embed_sparse_safe(self, texts: list[str]) -> list[SparseVector] | None:
        """Embed sparse vectors with graceful fallback on failure."""
        if not self._sparse_embeddings:
            return None
        try:
            return await self._sparse_embeddings.embed_sparse(texts)
        except Exception as exc:
            logger.warning("sparse embedding failed, continuing without: %s", exc)
            return None

    async def _embed_and_store_incremental(
        self,
        source_id: str,
        chunks: list[ChunkedContent],
        knowledge_id: str | None,
        source_type: str | None,
        source_weight: float,
        tags: list[str],
        metadata: dict[str, Any],
        hash_value: str | None,
        resume_from_chunk: int = 0,
        on_progress: IngestionProgress | None = None,
    ) -> Source:
        total = len(chunks)
        chunks_to_process = chunks[resume_from_chunk:]

        if not chunks_to_process and resume_from_chunk == 0:
            raise EmptyDocumentError("Document produced no content to ingest")

        completed = resume_from_chunk

        for batch_start in range(0, len(chunks_to_process), INGESTION_BATCH_SIZE):
            batch = chunks_to_process[batch_start : batch_start + INGESTION_BATCH_SIZE]
            texts = [c.embedding_text for c in batch]

            try:
                vectors = await embed_batched(self._embeddings, texts)
            except Exception as exc:
                logger.exception("embedding failed at chunk %d/%d for source=%s", completed, total, source_id)
                raise IngestionInterruptedError(
                    f"Embedding failed at chunk {completed}/{total}: {exc}",
                    completed_chunk_index=completed,
                    source_id=source_id,
                ) from exc

            sparse_vectors = await self._embed_sparse_safe(texts)

            if vectors:
                await self._vector_store.initialize(len(vectors[0]))

            points = self._build_points(
                source_id,
                batch,
                vectors,
                sparse_vectors,
                tags,
                metadata,
                knowledge_id,
                source_type,
                source_weight,
                chunk_index_offset=resume_from_chunk + batch_start,
            )

            try:
                await self._vector_store.upsert(points)
            except Exception as exc:
                logger.exception("vector upsert failed at chunk %d/%d for source=%s", completed, total, source_id)
                raise IngestionInterruptedError(
                    f"Vector storage failed at chunk {completed}/{total}: {exc}",
                    completed_chunk_index=completed,
                    source_id=source_id,
                ) from exc

            completed = resume_from_chunk + batch_start + len(batch)
            logger.info("embed batch complete (%d/%d chunks)", completed, total)

            if on_progress:
                await on_progress(completed, total)

        source = Source(
            source_id=source_id,
            metadata=metadata,
            tags=tags,
            chunk_count=total,
            embedding_model=self._embedding_model_name,
            file_hash=hash_value,
            created_at=datetime.now(UTC),
            knowledge_id=knowledge_id,
            source_type=source_type,
            source_weight=source_weight,
        )

        if self._metadata_store:
            await self._metadata_store.create_source(source)

        if self._on_ingestion_complete:
            await self._on_ingestion_complete(knowledge_id)

        return source

    async def ingest(
        self,
        file_path: str | Path,
        knowledge_id: str | None = None,
        source_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        page_range: str | None = None,
        resume_from_chunk: int = 0,
        on_progress: IngestionProgress | None = None,
    ) -> Source:
        """Ingest a file with auto-detection: PDF, text, markdown, or image."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        ext = file_path.suffix.lower()
        source_weight = self._resolve_weight(source_type)
        metadata = metadata or {}

        hash_value = await asyncio.to_thread(file_hash, file_path)
        if resume_from_chunk == 0:
            await self._check_duplicate(hash_value, knowledge_id)

        pages_filter = parse_page_range(page_range) if page_range else None

        if ext in IMAGE_EXTENSIONS:
            if not self._vision_parser:
                raise ConfigurationError("vision provider required for image ingestion")
            logger.info("processing file: %s (%s, image)", file_path.name, ext)
            pages = await self._vision_parser.parse(str(file_path), pages=pages_filter)
        elif ext in FILE_PARSERS_BY_EXTENSION:
            logger.info("processing file: %s (%s)", file_path.name, ext)
            parser_cls = FILE_PARSERS_BY_EXTENSION[ext]
            parser = parser_cls()
            pages = parser.parse(str(file_path), pages=pages_filter)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Supported: {sorted(FILE_PARSERS_BY_EXTENSION.keys())}")

        if not pages:
            raise EmptyDocumentError(
                f"Document produced no content to ingest: {file_path.name}",
                reason="no_text_content",
            )

        source_id = str(uuid4())

        if self._document_store:
            full_text = "\n\n".join(f"[Page {p.page_number}]\n{p.content}" for p in pages)
            title = (metadata or {}).get("name", file_path.name)
            await self._document_store.store_content(
                source_id=source_id,
                knowledge_id=knowledge_id,
                source_type=source_type,
                title=title,
                content=full_text,
            )

        chunks = self._chunker.chunk(pages)
        logger.info("%d chunks from %d pages", len(chunks), len(pages))

        if not chunks:
            raise EmptyDocumentError(f"Document produced no content to ingest: {file_path.name}")

        if self._contextual_chunking:
            source_name = (metadata or {}).get("name", file_path.name)
            chunks = contextualize_chunks(chunks, source_name=source_name, source_type=source_type)

        source = await self._embed_and_store_incremental(
            source_id=source_id,
            chunks=chunks,
            knowledge_id=knowledge_id,
            source_type=source_type,
            source_weight=source_weight,
            tags=[],
            metadata=metadata,
            hash_value=hash_value,
            resume_from_chunk=resume_from_chunk,
            on_progress=on_progress,
        )

        logger.info("complete: %s --> %d chunks, source_id=%s", file_path.name, len(chunks), source_id)
        return source

    async def ingest_text(
        self,
        content: str,
        knowledge_id: str | None = None,
        source_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Source:
        source_weight = self._resolve_weight(source_type)
        metadata = metadata or {}

        logger.info("text ingestion started: %d chars", len(content))

        pages = [ParsedPage(page_number=1, content=content)]
        chunks = self._chunker.chunk(pages)

        if not chunks:
            raise EmptyDocumentError("Text content produced no chunks to ingest")

        if self._contextual_chunking:
            source_name = (metadata or {}).get("name", "text-input")
            chunks = contextualize_chunks(chunks, source_name=source_name, source_type=source_type)

        texts = [c.embedding_text for c in chunks]
        vectors = await embed_batched(self._embeddings, texts)

        sparse_vectors = await self._embed_sparse_safe(texts)

        source_id = str(uuid4())

        if self._document_store:
            title = (metadata or {}).get("name", "text-input")
            await self._document_store.store_content(
                source_id=source_id,
                knowledge_id=knowledge_id,
                source_type=source_type,
                title=title,
                content=content,
            )

        if vectors:
            await self._vector_store.initialize(len(vectors[0]))

        source = Source(
            source_id=source_id,
            metadata=metadata,
            tags=[],
            chunk_count=len(chunks),
            embedding_model=self._embedding_model_name,
            file_hash=None,
            created_at=datetime.now(UTC),
            knowledge_id=knowledge_id,
            source_type=source_type,
            source_weight=source_weight,
        )

        points = self._build_points(
            source_id, chunks, vectors, sparse_vectors, [], metadata, knowledge_id, source_type, source_weight
        )
        try:
            await self._vector_store.upsert(points)
            if self._metadata_store:
                await self._metadata_store.create_source(source)
        except Exception:
            logger.exception("ingestion failed, attempting cleanup for source=%s", source_id)
            try:
                await self._vector_store.delete({"source_id": source_id})
            except Exception:
                logger.exception("cleanup failed for source=%s", source_id)
            raise

        if self._on_ingestion_complete:
            await self._on_ingestion_complete(knowledge_id)

        logger.info("text ingestion complete: source=%s, chunks=%d", source_id, len(chunks))
        return source

    @staticmethod
    def _build_points(
        source_id: str,
        chunks: list[ChunkedContent],
        vectors: list[list[float]],
        sparse_vectors: list[SparseVector] | None,
        tags: list[str],
        metadata: dict[str, Any],
        knowledge_id: str | None,
        source_type: str | None,
        source_weight: float,
        chunk_index_offset: int = 0,
    ) -> list[VectorPoint]:
        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            sparse = sparse_vectors[idx] if sparse_vectors else None
            point_id = chunk.parent_id if chunk.chunk_type == "parent" and chunk.parent_id else str(uuid4())
            points.append(
                VectorPoint(
                    point_id=point_id,
                    vector=vector,
                    sparse_vector=sparse,
                    payload={
                        "content": chunk.content,
                        "context": chunk.context,
                        "contextualized": chunk.contextualized,
                        "page_number": chunk.page_number,
                        "section": chunk.section,
                        "chunk_index": chunk_index_offset + idx,
                        "source_id": source_id,
                        "knowledge_id": knowledge_id,
                        "source_type": source_type,
                        "source_weight": source_weight,
                        "chunk_type": chunk.chunk_type,
                        "parent_id": chunk.parent_id,
                        "tags": tags,
                        "source_name": metadata.get("name", ""),
                        "file_url": metadata.get("file_url", ""),
                    },
                )
            )
        return points
