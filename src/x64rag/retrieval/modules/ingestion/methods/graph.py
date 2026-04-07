from __future__ import annotations

from typing import Any

from x64rag.retrieval.common.logging import get_logger
from x64rag.retrieval.modules.ingestion.models import ChunkedContent, ParsedPage
from x64rag.retrieval.stores.graph.base import BaseGraphStore

logger = get_logger("ingestion.methods.graph")


class GraphIngestion:
    """Graph ingestion stub.

    Entity extraction from unstructured text requires LLM calls that are only
    available through ``StructuredIngestionService``.  This stub satisfies the
    ``BaseIngestionMethod`` protocol so it can be listed for future use but
    currently skips with a warning.  ``GraphRetrieval`` (querying) still works
    for entities created via the structured pipeline.
    """

    def __init__(self, graph_store: BaseGraphStore) -> None:
        self._store = graph_store

    @property
    def name(self) -> str:
        return "graph"

    async def ingest(
        self,
        source_id: str,
        knowledge_id: str | None,
        source_type: str | None,
        source_weight: float,
        title: str,
        full_text: str,
        chunks: list[ChunkedContent],
        tags: list[str],
        metadata: dict[str, Any],
        hash_value: str | None = None,
        pages: list[ParsedPage] | None = None,
    ) -> None:
        logger.warning(
            "graph ingestion skipped — entity extraction requires StructuredIngestionService"
        )

    async def delete(self, source_id: str) -> None:
        await self._store.delete_by_source(source_id)
