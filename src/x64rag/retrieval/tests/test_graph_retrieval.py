from types import SimpleNamespace
from unittest.mock import AsyncMock

from x64rag.retrieval.common.models import RetrievedChunk
from x64rag.retrieval.modules.retrieval.methods.graph import GraphRetrieval
from x64rag.retrieval.modules.retrieval.search.service import RetrievalService
from x64rag.retrieval.stores.graph.models import GraphEntity, GraphPath, GraphResult


def _make_service(graph_method=None, document_method=None):
    mock_vector = SimpleNamespace(
        name="vector",
        weight=1.0,
        search=AsyncMock(
            return_value=[
                RetrievedChunk(chunk_id="chunk-1", source_id="src-1", content="Vector result", score=0.8),
            ]
        ),
    )
    methods = [mock_vector]
    if document_method is not None:
        methods.append(document_method)
    if graph_method is not None:
        methods.append(graph_method)
    return RetrievalService(
        retrieval_methods=methods,
        reranking=None,
        top_k=5,
    )


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


def _make_graph_method(graph_results):
    """Build a mock graph retrieval method that returns pre-converted chunks."""
    chunks = GraphRetrieval._convert(graph_results)
    return SimpleNamespace(
        name="graph",
        weight=1.0,
        search=AsyncMock(return_value=chunks),
    )


async def test_retrieve_with_graph_store():
    graph_results = _make_graph_results()
    mock_graph = _make_graph_method(graph_results)

    service = _make_service(graph_method=mock_graph)
    results = await service.retrieve(query="what connects to Motor M1?", knowledge_id="kb-1")

    mock_graph.search.assert_called_once()
    assert len(results) >= 2
    graph_chunks = [r for r in results if r.chunk_id.startswith("graph:")]
    assert len(graph_chunks) == 1


async def test_retrieve_without_graph_store():
    service = _make_service(graph_method=None)
    results = await service.retrieve(query="test query", knowledge_id="kb-1")

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"


async def test_graph_results_to_chunks_basic():
    results = _make_graph_results()
    chunks = GraphRetrieval._convert(results)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_id == "graph:Motor M1:motor"
    assert chunk.source_id == "src-2"
    assert chunk.score == 0.95
    assert "Motor M1 (motor)" in chunk.content
    assert "480V 3-phase" in chunk.content
    assert "POWERED_BY" in chunk.content
    assert "Breaker CB-3" in chunk.content
    assert chunk.source_metadata["retrieval_type"] == "graph"
    assert chunk.source_metadata["entity_name"] == "Motor M1"
    assert chunk.source_metadata["connected_count"] == 2


async def test_graph_results_to_chunks_no_value():
    results = [
        GraphResult(
            entity=GraphEntity(name="Pump P-1", entity_type="pump", category="mechanical", properties={}),
            connected_entities=[],
            paths=[],
            relevance_score=0.5,
        ),
    ]
    chunks = GraphRetrieval._convert(results)

    assert len(chunks) == 1
    assert "Specifications:" not in chunks[0].content
    assert chunks[0].chunk_id == "graph:Pump P-1:pump"


async def test_graph_results_to_chunks_empty():
    chunks = GraphRetrieval._convert([])
    assert chunks == []


async def test_graph_store_empty_result_no_fusion():
    mock_graph = SimpleNamespace(
        name="graph",
        weight=1.0,
        search=AsyncMock(return_value=[]),
    )

    service = _make_service(graph_method=mock_graph)
    results = await service.retrieve(query="test query", knowledge_id="kb-1")

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"


async def test_graph_and_document_store_together():
    graph_results = _make_graph_results()
    mock_graph = _make_graph_method(graph_results)

    mock_document = SimpleNamespace(
        name="document",
        weight=1.0,
        search=AsyncMock(
            return_value=[
                RetrievedChunk(
                    chunk_id="fulltext:src-3",
                    source_id="src-3",
                    content="excerpt",
                    score=0.7,
                    source_metadata={"title": "Doc", "match_type": "fulltext"},
                ),
            ]
        ),
    )

    service = _make_service(graph_method=mock_graph, document_method=mock_document)
    results = await service.retrieve(query="Motor M1", knowledge_id="kb-1")

    assert len(results) == 3
