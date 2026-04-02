"""Startup checks for reasoning SDK."""

from x64rag.common.startup import check_baml as _check_baml


def check_baml() -> None:
    _check_baml("reasoning", "x64rag.reasoning.baml.baml_client")
