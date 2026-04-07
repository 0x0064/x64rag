from __future__ import annotations

import time
from typing import Any

from x64rag.retrieval.common.logging import get_logger
from x64rag.retrieval.modules.ingestion.models import ChunkedContent, ParsedPage
from x64rag.retrieval.stores.graph.base import BaseGraphStore

logger = get_logger("ingestion.methods.graph")


class GraphIngestion:
    """Extract entities and store in graph store."""

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
        start = time.perf_counter()
        try:
            await self._store.store_entities(source_id=source_id, content=full_text, knowledge_id=knowledge_id)
            elapsed = (time.perf_counter() - start) * 1000
            logger.info("entities stored in %.1fms", elapsed)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("failed in %.1fms — %s", elapsed, exc)

    async def delete(self, source_id: str) -> None:
        await self._store.delete_entities(source_id)
