# src/x64rag/retrieval/tests/test_graph_retrieval_method.py
from unittest.mock import AsyncMock

from x64rag.retrieval.modules.retrieval.methods.graph import GraphRetrieval
from x64rag.retrieval.stores.graph.models import GraphEntity, GraphPath, GraphResult


def _make_graph_results():
    return [
        GraphResult(
            entity=GraphEntity(
                name="Motor M1",
                entity_type="motor",
                category="electrical",
                value="480V 3-phase",
                properties={"source_id": "src-2"},
            ),
            connected_entities=[
                GraphEntity(name="Breaker CB-3", entity_type="breaker", category="electrical"),
                GraphEntity(name="VFD-3", entity_type="vfd", category="electrical"),
            ],
            paths=[
                GraphPath(
                    entities=["Motor M1", "Breaker CB-3", "Panel MCC-1"],
                    relationships=["POWERED_BY", "FEEDS"],
                ),
            ],
            relevance_score=0.95,
        ),
    ]


async def test_search_converts_graph_results():
    store = AsyncMock()
    store.query_graph = AsyncMock(return_value=_make_graph_results())

    method = GraphRetrieval(graph_store=store, weight=0.7)
    assert method.name == "graph"
    assert method.weight == 0.7

    results = await method.search(query="Motor M1", top_k=5, knowledge_id="kb-1")
    assert len(results) == 1
    assert results[0].chunk_id == "graph:Motor M1:motor"
    assert results[0].score == 0.95
    assert "480V 3-phase" in results[0].content
    assert "POWERED_BY" in results[0].content
    assert results[0].source_metadata["retrieval_type"] == "graph"
    store.query_graph.assert_called_once_with(query="Motor M1", knowledge_id="kb-1", max_hops=2, top_k=5)


async def test_search_empty():
    store = AsyncMock()
    store.query_graph = AsyncMock(return_value=[])

    method = GraphRetrieval(graph_store=store)
    results = await method.search(query="nothing", top_k=5)
    assert results == []


async def test_error_returns_empty():
    store = AsyncMock()
    store.query_graph = AsyncMock(side_effect=RuntimeError("neo4j down"))

    method = GraphRetrieval(graph_store=store)
    results = await method.search(query="test", top_k=5)
    assert results == []
