import asyncio
from typing import Any

from x64rag.retrieval.common.errors import RetrievalError
from x64rag.retrieval.common.logging import get_logger
from x64rag.retrieval.common.models import ContentMatch, RetrievedChunk
from x64rag.retrieval.modules.retrieval.refinement.base import BaseChunkRefiner
from x64rag.retrieval.modules.retrieval.search.fusion import reciprocal_rank_fusion
from x64rag.retrieval.modules.retrieval.search.keyword import KeywordSearch
from x64rag.retrieval.modules.retrieval.search.reranking.base import BaseReranking
from x64rag.retrieval.modules.retrieval.search.rewriting.base import BaseQueryRewriter
from x64rag.retrieval.modules.retrieval.search.vector import VectorSearch
from x64rag.retrieval.stores.document.base import BaseDocumentStore
from x64rag.retrieval.stores.graph.base import BaseGraphStore
from x64rag.retrieval.stores.graph.models import GraphResult

logger = get_logger("search/retrieval")


class RetrievalService:
    def __init__(
        self,
        vector_search: VectorSearch,
        keyword_search: KeywordSearch | None = None,
        reranking: BaseReranking | None = None,
        top_k: int = 5,
        source_type_weights: dict[str, float] | None = None,
        document_store: BaseDocumentStore | None = None,
        query_rewriter: BaseQueryRewriter | None = None,
        graph_store: BaseGraphStore | None = None,
        chunk_refiner: BaseChunkRefiner | None = None,
    ) -> None:
        self._vector_search = vector_search
        self._keyword_search = keyword_search
        self._reranking = reranking
        self._top_k = top_k
        self._source_type_weights = source_type_weights
        self._document_store = document_store
        self._query_rewriter = query_rewriter
        self._graph_store = graph_store
        self._chunk_refiner = chunk_refiner

    async def retrieve(
        self,
        query: str,
        knowledge_id: str | None = None,
        top_k: int | None = None,
        tree_chunks: list[RetrievedChunk] | None = None,
    ) -> list[RetrievedChunk]:
        if not query or not query.strip():
            return []

        top_k = top_k if top_k is not None else self._top_k
        fetch_k = top_k * 4
        filters = self._build_filters(knowledge_id)

        logger.info('query: "%s" (knowledge_id=%s)', query[:80], knowledge_id)

        queries = [query]
        if self._query_rewriter:
            try:
                rewritten = await self._query_rewriter.rewrite(query)
                queries.extend(rewritten)
                if rewritten:
                    logger.info(
                        "query rewriting: %d total queries (1 original + %d rewritten)",
                        len(queries),
                        len(rewritten),
                    )
            except Exception as exc:
                logger.exception("query rewriter failed: %s — proceeding with original query", exc)

        search_tasks = [self._search_single_query(q, fetch_k, filters, knowledge_id) for q in queries]
        query_results = await asyncio.gather(*search_tasks)

        all_result_lists: list[list[RetrievedChunk]] = []
        for result_lists in query_results:
            all_result_lists.extend(result_lists)

        if tree_chunks:
            all_result_lists.append(tree_chunks)
            logger.info("%d tree search candidates added to fusion", len(tree_chunks))

        if len(all_result_lists) > 1:
            fused = reciprocal_rank_fusion(all_result_lists, source_type_weights=self._source_type_weights)
            logger.info("%d unique after reciprocal rank fusion", len(fused))
        elif all_result_lists:
            fused = self._apply_source_weights(all_result_lists[0])
        else:
            fused = []

        if self._reranking and fused:
            fused = await self._reranking.rerank(query, fused, top_k=top_k)
            logger.info("top %d selected after reranking", len(fused))
        else:
            fused = fused[:top_k]

        if self._chunk_refiner and fused:
            fused = await self._chunk_refiner.refine(query, fused)
            logger.info("chunk refinement: %d chunks after refinement", len(fused))

        return fused

    async def _search_single_query(
        self,
        query: str,
        fetch_k: int,
        filters: dict[str, Any] | None,
        knowledge_id: str | None,
    ) -> list[list[RetrievedChunk]]:
        """Run all search paths (vector, keyword, document, graph) for a single query.

        Each path is dispatched as a named coroutine and results are collected by key,
        avoiding fragile positional indexing.
        """
        try:
            named_tasks: dict[str, Any] = {
                "vector": self._vector_search.search(query=query, top_k=fetch_k, filters=filters),
            }

            if self._keyword_search:
                named_tasks["keyword"] = self._keyword_search.search(
                    query=query, top_k=fetch_k, knowledge_id=knowledge_id
                )

            if self._document_store:
                named_tasks["fulltext"] = self._document_store.search_content(
                    query=query, knowledge_id=knowledge_id, top_k=fetch_k
                )

            if self._graph_store:
                named_tasks["graph"] = self._graph_store.query_graph(
                    query=query, knowledge_id=knowledge_id, max_hops=2, top_k=fetch_k
                )

            gathered = await asyncio.gather(*named_tasks.values())
            results_by_key = dict(zip(named_tasks.keys(), gathered, strict=True))

            vector_results = results_by_key["vector"]
            result_lists: list[list[RetrievedChunk]] = [vector_results]

            if "keyword" in results_by_key:
                keyword_results = results_by_key["keyword"]
                logger.info("%d vector + %d bm25 candidates", len(vector_results), len(keyword_results))
                result_lists.append(keyword_results)

            if "fulltext" in results_by_key:
                fulltext_matches = results_by_key["fulltext"]
                if fulltext_matches:
                    fulltext_chunks = self._content_matches_to_chunks(fulltext_matches)
                    logger.info("%d fulltext candidates", len(fulltext_chunks))
                    result_lists.append(fulltext_chunks)

            if "graph" in results_by_key:
                graph_results = results_by_key["graph"]
                if graph_results:
                    graph_chunks = self._graph_results_to_chunks(graph_results)
                    logger.info("%d graph candidates", len(graph_chunks))
                    result_lists.append(graph_chunks)

        except Exception as exc:
            raise RetrievalError(f"Search failed: {exc}") from exc

        return result_lists

    @staticmethod
    def _graph_results_to_chunks(results: list[GraphResult]) -> list[RetrievedChunk]:
        """Convert graph traversal results to RetrievedChunk for fusion."""
        chunks: list[RetrievedChunk] = []
        for result in results:
            lines = [f"{result.entity.name} ({result.entity.entity_type})"]
            if result.entity.value:
                lines.append(f"  Specifications: {result.entity.value}")

            for path in result.paths:
                parts: list[str] = []
                for i, entity_name in enumerate(path.entities):
                    if i > 0 and i - 1 < len(path.relationships):
                        parts.append(f"-[{path.relationships[i - 1]}]->")
                    parts.append(entity_name)
                lines.append(f"  Path: {' '.join(parts)}")

            for connected in result.connected_entities[:5]:
                lines.append(f"  Connected: {connected.name} ({connected.entity_type})")

            chunks.append(
                RetrievedChunk(
                    chunk_id=f"graph:{result.entity.name}:{result.entity.entity_type}",
                    source_id=result.entity.properties.get("source_id", ""),
                    content="\n".join(lines),
                    score=result.relevance_score,
                    source_metadata={
                        "retrieval_type": "graph",
                        "entity_name": result.entity.name,
                        "entity_type": result.entity.entity_type,
                        "connected_count": len(result.connected_entities),
                    },
                )
            )
        return chunks

    @staticmethod
    def _content_matches_to_chunks(matches: list[ContentMatch]) -> list[RetrievedChunk]:
        chunks = []
        for match in matches:
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"fulltext:{match.source_id}",
                    source_id=match.source_id,
                    content=match.excerpt,
                    score=match.score,
                    source_type=match.source_type,
                    source_metadata={
                        "title": match.title,
                        "match_type": match.match_type,
                    },
                )
            )
        return chunks

    def _apply_source_weights(self, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not self._source_type_weights:
            return results
        from dataclasses import replace

        weighted = []
        for r in results:
            weighted.append(replace(r, score=r.score * r.source_weight))
        weighted.sort(key=lambda x: x.score, reverse=True)
        return weighted

    @staticmethod
    def _build_filters(knowledge_id: str | None) -> dict[str, Any] | None:
        if knowledge_id is None:
            return None
        return {"knowledge_id": knowledge_id}
