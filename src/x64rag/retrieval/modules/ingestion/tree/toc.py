"""TOC detection, parsing, and section position verification."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any

from x64rag.retrieval.common.logging import get_logger

logger = get_logger(__name__)


class TocPath(Enum):
    WITH_PAGE_NUMBERS = "with_page_numbers"
    WITHOUT_PAGE_NUMBERS = "without_page_numbers"
    NO_TOC = "no_toc"


@dataclass
class PageContent:
    index: int
    text: str
    token_count: int


@dataclass
class TocInfo:
    path: TocPath
    toc_pages: list[PageContent]


async def detect_toc(
    pages: list[PageContent],
    toc_scan_pages: int,
    registry: Any,
) -> TocInfo:
    """Scan the first N pages for a table of contents.

    Calls BAML DetectTableOfContents on each page within the scan range and
    returns a TocInfo describing whether a TOC was found and which path to
    follow (with page numbers, without, or no TOC).
    """
    from x64rag.retrieval.baml.baml_client.async_client import b

    scan_limit = min(toc_scan_pages, len(pages))
    candidates = pages[:scan_limit]

    toc_pages: list[PageContent] = []
    has_page_numbers = False

    for page in candidates:
        result = await b.DetectTableOfContents(
            page_text=page.text,
            baml_options={"client_registry": registry},
        )

        if result.has_toc:
            toc_pages.append(page)
            if result.has_page_numbers:
                has_page_numbers = True

    if not toc_pages:
        logger.debug("no TOC detected in first %d pages", scan_limit)
        return TocInfo(path=TocPath.NO_TOC, toc_pages=[])

    path = TocPath.WITH_PAGE_NUMBERS if has_page_numbers else TocPath.WITHOUT_PAGE_NUMBERS
    logger.debug("TOC detected: path=%s, toc_pages=%d", path.value, len(toc_pages))
    return TocInfo(path=path, toc_pages=toc_pages)


async def parse_toc(
    toc_pages: list[PageContent],
    registry: Any,
) -> list[dict[str, Any]]:
    """Parse TOC pages into structured section entries.

    Concatenates the text of all TOC pages and calls BAML ParseTableOfContents.
    Returns a list of dicts with keys: structure, title, page (optional).
    """
    from x64rag.retrieval.baml.baml_client.async_client import b

    combined_text = "\n".join(page.text for page in toc_pages)

    result = await b.ParseTableOfContents(
        toc_text=combined_text,
        baml_options={"client_registry": registry},
    )

    entries = []
    for entry in result.entries:
        entries.append({
            "structure": entry.structure,
            "title": entry.title,
            "page": entry.page,
        })

    logger.debug("parsed %d TOC entries", len(entries))
    return entries


async def find_section_starts(
    entries: list[dict[str, Any]],
    pages: list[PageContent],
    registry: Any,
) -> list[dict[str, Any]]:
    """For TOC entries without page numbers, find where each section starts.

    Calls BAML FindSectionStart for each entry to locate the page where
    that section heading appears. Returns updated entries with page numbers.
    """
    from x64rag.retrieval.baml.baml_client.async_client import b

    pages_text = "\n".join(
        f"--- Page {page.index} ---\n{page.text}" for page in pages
    )

    updated: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("page") is not None:
            updated.append(entry)
            continue

        page_num = await b.FindSectionStart(
            section_title=entry["title"],
            pages_text=pages_text,
            baml_options={"client_registry": registry},
        )

        updated.append({
            **entry,
            "page": page_num,
        })
        logger.debug("found start for '%s' at page %d", entry["title"], page_num)

    return updated


async def verify_section_positions(
    sections: list[dict[str, Any]],
    pages: list[PageContent],
    registry: Any,
) -> list[dict[str, Any]]:
    """Verify that each section heading actually appears on its claimed page.

    Runs BAML VerifySectionPosition concurrently for all sections. Sections
    that fail verification are marked with verified=False.
    """
    from x64rag.retrieval.baml.baml_client.async_client import b

    page_by_index = {page.index: page for page in pages}

    async def _verify(section: dict[str, Any]) -> dict[str, Any]:
        page_index = section.get("page")
        if page_index is None or page_index not in page_by_index:
            return {**section, "verified": False}

        page = page_by_index[page_index]
        verified = await b.VerifySectionPosition(
            title=section["title"],
            page_text=page.text,
            baml_options={"client_registry": registry},
        )

        return {**section, "verified": verified}

    tasks = [asyncio.create_task(_verify(section)) for section in sections]
    results = list(await asyncio.gather(*tasks))

    verified_count = sum(1 for r in results if r.get("verified"))
    logger.debug("verified %d/%d section positions", verified_count, len(results))

    return results
