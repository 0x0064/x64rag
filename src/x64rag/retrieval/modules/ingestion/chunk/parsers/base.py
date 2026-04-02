from typing import Protocol

from x64rag.retrieval.modules.ingestion.models import ParsedPage


class BaseParser(Protocol):
    def parse(self, file_path: str, pages: set[int] | None = None) -> list[ParsedPage]: ...
