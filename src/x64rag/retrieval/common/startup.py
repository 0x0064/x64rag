"""Startup checks for retrieval SDK."""

from x64rag.common.startup import check_baml as _check_baml


def check_baml() -> None:
    _check_baml("retrieval", "x64rag.retrieval.baml.baml_client")
