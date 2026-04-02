from openai import AsyncOpenAI


class OpenAIEmbeddings:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", **kwargs) -> None:
        self._client = AsyncOpenAI(api_key=api_key, max_retries=kwargs.pop("max_retries", 3))
        self._model = model
        self._dimension: int | None = None

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]

    async def embedding_dimension(self) -> int:
        if self._dimension is None:
            vectors = await self.embed(["dimension probe"])
            self._dimension = len(vectors[0])
        return self._dimension
