import asyncio
from dataclasses import replace
from functools import partial

import voyageai

from x64rag.retrieval.common.logging import get_logger
from x64rag.retrieval.common.models import RetrievedChunk

logger = get_logger(__name__)


class VoyageReranking:
    def __init__(self, api_key: str, model: str = "rerank-2.5") -> None:
        self._client = voyageai.Client(api_key=api_key)
        self._model = model

    async def rerank(self, query: str, results: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
        if not results:
            return []

        documents = [r.content for r in results]

        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                partial(
                    self._client.rerank,
                    query=query,
                    documents=documents,
                    model=self._model,
                    top_k=top_k,
                ),
            )
        except Exception:
            logger.exception("voyage rerank failed, returning unranked")
            return results[:top_k]

        reranked = []
        for item in response.results:
            reranked.append(replace(results[item.index], score=item.relevance_score))

        return reranked
