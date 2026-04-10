from baml_py import errors as baml_errors

from x64rag.retrieval.baml.baml_client.async_client import b
from x64rag.retrieval.common.errors import GenerationError
from x64rag.retrieval.common.formatting import chunks_to_context
from x64rag.retrieval.common.language_model import LanguageModelClient, build_registry
from x64rag.retrieval.common.logging import get_logger
from x64rag.retrieval.common.models import RetrievedChunk
from x64rag.retrieval.modules.generation.models import StepResult

logger = get_logger("generation/step")


class StepGenerationService:
    """Single reasoning step generation for iterative retrieval loops."""

    def __init__(self, lm_config: LanguageModelClient) -> None:
        self._lm_config = lm_config

    async def generate_step(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        context: str | None = None,
    ) -> StepResult:
        """Generate a single reasoning step from retrieved chunks."""
        if not query or not query.strip():
            raise GenerationError("Query must not be empty")

        chunk_context = chunks_to_context(chunks) if chunks else "(No context retrieved)"
        prior_reasoning = context or ""

        registry = build_registry(self._lm_config)

        try:
            result = await b.GenerateReasoningStep(
                query=query,
                context=chunk_context,
                prior_reasoning=prior_reasoning,
                baml_options={"client_registry": registry},
            )

            text = result.text.strip()

            logger.info(
                "reasoning step: %d chars, is_final=%s",
                len(text),
                result.is_final,
            )

            return StepResult(text=text, done=bool(result.is_final))

        except baml_errors.BamlValidationError as exc:
            raise GenerationError(f"GenerateReasoningStep returned unparseable response: {exc}") from exc
        except Exception as exc:
            raise GenerationError(f"GenerateReasoningStep failed: {exc}") from exc
