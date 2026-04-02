from dataclasses import replace

import cohere

from x64rag.retrieval.common.logging import get_logger
from x64rag.retrieval.common.models import RetrievedChunk

logger = get_logger(__name__)


class CohereReranking:
    def __init__(self, api_key: str, model: str = "rerank-english-v3.0") -> None:
        self._client = cohere.AsyncClientV2(api_key=api_key)
        self._model = model

    async def rerank(self, query: str, results: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
        if not results:
            return []

        documents = [r.content for r in results]

        try:
            response = await self._client.rerank(
                query=query,
                documents=documents,
                model=self._model,
                top_n=top_k,
            )
        except Exception:
            logger.exception("cohere rerank failed, returning unranked")
            return results[:top_k]

        reranked = []
        for item in response.results:
            reranked.append(replace(results[item.index], score=item.relevance_score))

        return reranked
