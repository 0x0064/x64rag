"""Unified CLI entry point for x64rag."""

from __future__ import annotations

import click

import x64rag.reasoning.cli.commands.analyze
import x64rag.reasoning.cli.commands.analyze_context
import x64rag.reasoning.cli.commands.classify
import x64rag.reasoning.cli.commands.compliance
import x64rag.reasoning.cli.commands.evaluate
import x64rag.reasoning.cli.commands.init
import x64rag.reasoning.cli.commands.status
import x64rag.retrieval.cli.commands.ingest
import x64rag.retrieval.cli.commands.init
import x64rag.retrieval.cli.commands.knowledge
import x64rag.retrieval.cli.commands.query
import x64rag.retrieval.cli.commands.retrieve
import x64rag.retrieval.cli.commands.session
import x64rag.retrieval.cli.commands.status
from x64rag.reasoning.cli import cli as reasoning_cli
from x64rag.retrieval.cli import cli as retrieval_cli


@click.group()
@click.version_option(package_name="x64rag")
def main() -> None:
    """x64rag — Retrieval-Augmented Generation + Reasoning-Augmented Classification."""


main.add_command(retrieval_cli, name="retrieval")
main.add_command(reasoning_cli, name="reasoning")


if __name__ == "__main__":
    main()
