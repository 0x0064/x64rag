from unittest.mock import AsyncMock

from x64rag.retrieval.modules.ingestion.methods.graph import GraphIngestion


async def test_ingest_delegates_to_store():
    store = AsyncMock()
    store.store_entities = AsyncMock()
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
    store.store_entities.assert_called_once()


async def test_delete():
    store = AsyncMock()
    store.delete_entities = AsyncMock()
    method = GraphIngestion(graph_store=store)
    await method.delete("src-1")
    store.delete_entities.assert_called_once_with("src-1")
