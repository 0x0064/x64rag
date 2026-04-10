import logging
import os

_BAML_LOG_ENV = "BAML_LOG"
_X64RAG_BAML_LOG_ENV = "X64RAG_BAML_LOG"


def _propagate_baml_log_env() -> None:
    """Wire X64RAG_BAML_LOG to BAML_LOG (what BAML actually reads) so users have
    one namespaced env var instead of touching BAML's internal name directly.

    An explicitly-set BAML_LOG wins — we only propagate when BAML_LOG is unset.
    """
    user_value = os.getenv(_X64RAG_BAML_LOG_ENV)
    if user_value and not os.getenv(_BAML_LOG_ENV):
        os.environ[_BAML_LOG_ENV] = user_value


def get_logger(module: str) -> logging.Logger:
    _propagate_baml_log_env()
    logger = logging.getLogger(f"x64rag.{module}")
    if os.getenv("X64RAG_LOG_ENABLED", "false").lower() == "true":
        level = os.getenv("X64RAG_LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, level, logging.INFO))
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            logger.addHandler(handler)
            logger.propagate = False
    else:
        logger.setLevel(logging.CRITICAL)
    return logger
