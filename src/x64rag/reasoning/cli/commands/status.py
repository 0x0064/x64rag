from __future__ import annotations

import os
from typing import Any

import click

from x64rag.reasoning.cli import cli, run_async
from x64rag.reasoning.cli.config import build_lm_config, load_config
from x64rag.reasoning.cli.constants import CONFIG_FILE, ENV_FILE, ConfigError
from x64rag.reasoning.cli.output import OutputMode, print_error, print_json
from x64rag.reasoning.common.language_model import LanguageModelClient


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Validate config and test LLM connection."""
    mode: OutputMode = ctx.obj["output_mode"]
    config_path = ctx.obj["config_path"]

    try:
        toml = load_config(config_path)
    except ConfigError as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None

    if mode == OutputMode.PRETTY:
        path = config_path or str(CONFIG_FILE)
        click.echo(f"Config: {path}")
        if os.path.exists(ENV_FILE):
            click.echo(f".env: {ENV_FILE}")
        else:
            click.echo(".env: not found (API keys must be in environment)")

    try:
        lm_config = build_lm_config(toml)
    except ConfigError as e:
        print_error(str(e), mode)
        raise SystemExit(1) from None

    if mode == OutputMode.PRETTY:
        click.echo(f"Provider: {lm_config.provider.provider}")
        click.echo(f"Model: {lm_config.provider.model}")
        if lm_config.fallback:
            click.echo(f"Fallback: {lm_config.fallback.provider}/{lm_config.fallback.model}")
        if toml.get("embeddings"):
            emb = toml["embeddings"]
            click.echo(f"Embeddings: {emb.get('provider', 'openai')}/{emb.get('model', 'default')}")
        if toml.get("vector_store"):
            vs = toml["vector_store"]
            click.echo(f"Vector store: {vs.get('provider', 'qdrant')} @ {vs.get('url', 'localhost')}")

    run_async(_test_connection(lm_config, toml, mode))


async def _test_connection(lm_config: LanguageModelClient, toml: dict[str, Any], mode: OutputMode) -> None:
    from x64rag.reasoning.modules.analysis.service import AnalysisService

    try:
        service = AnalysisService(lm_config=lm_config)
        await service.analyze("test", config=None)
        if mode == OutputMode.PRETTY:
            click.echo("LLM: connected")
            click.echo("\nReady.")
        else:
            status_data: dict[str, Any] = {
                "status": "ok",
                "provider": lm_config.provider.provider,
                "model": lm_config.provider.model,
            }
            if toml.get("embeddings"):
                status_data["embeddings"] = toml["embeddings"].get("provider", "openai")
            if toml.get("vector_store"):
                status_data["vector_store"] = toml["vector_store"].get("provider", "qdrant")
            print_json(status_data)
    except Exception as e:
        print_error(f"LLM connection failed: {e}", mode)
        raise SystemExit(1) from None
