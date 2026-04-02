from __future__ import annotations

from typing import Any, Protocol


class BaseEmbeddings(Protocol):
    @property
    def model(self) -> str: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def embedding_dimension(self) -> int: ...


class BaseVectorStore(Protocol):
    async def scroll(
        self,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        offset: str | None = None,
    ) -> tuple[list[Any], str | None]: ...

    async def search(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[Any]: ...
