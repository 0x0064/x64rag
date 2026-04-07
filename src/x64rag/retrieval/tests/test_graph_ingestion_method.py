from unittest.mock import AsyncMock

from x64rag.retrieval.modules.ingestion.methods.graph import GraphIngestion


async def test_ingest_is_stub_that_skips():
    store = AsyncMock()
    method = GraphIngestion(graph_store=store)
    assert method.name == "graph"
    await method.ingest(
        source_id="src-1",
        knowledge_id="kb-1",
        source_type=None,
        source_weight=1.0,
        title="Test",
        full_text="Entity A connects to Entity B.",
        chunks=[],
        tags=[],
        metadata={},
    )
    # Stub should NOT call any store methods during ingest
    store.assert_not_awaited()


async def test_delete_calls_delete_by_source():
    store = AsyncMock()
    store.delete_by_source = AsyncMock()
    method = GraphIngestion(graph_store=store)
    await method.delete("src-1")
    store.delete_by_source.assert_called_once_with("src-1")
