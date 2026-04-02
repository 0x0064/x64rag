from dataclasses import dataclass

from baml_py import errors as baml_errors

from x64rag.retrieval.baml.baml_client.async_client import b
from x64rag.retrieval.common.language_model import LanguageModelConfig, build_registry
from x64rag.retrieval.common.logging import get_logger

logger = get_logger("retrieval/rewriting/hyde")


@dataclass
class HyDeRewriter:
    """Generate a hypothetical document passage to use as the search query.

    Uses HyDE (Hypothetical Document Embeddings) — the LLM writes a short
    passage that would appear in a document answering the query.  The passage
    is embedded instead of the question, closing the vocabulary gap between
    question-style and document-style text.
    """

    lm_config: LanguageModelConfig

    async def rewrite(self, query: str, conversation_context: str | None = None) -> list[str]:
        registry = build_registry(self.lm_config)
        context_hint = f"\n\nConversation context: {conversation_context}" if conversation_context else ""

        try:
            result = await b.GenerateHypotheticalDocument(
                query + context_hint,
                baml_options={"client_registry": registry},
            )
            logger.info("hyde: generated %d-char hypothetical passage", len(result.passage))
            return [result.passage]

        except baml_errors.BamlValidationError as exc:
            logger.exception(
                "GenerateHypotheticalDocument failed: LLM returned unparseable response — "
                "falling back to original query. Detail: %s",
                exc,
            )
            return []
        except Exception as exc:
            logger.exception("GenerateHypotheticalDocument failed: %s — falling back to original query", exc)
            return []
