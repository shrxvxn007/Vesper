"""MD&A section extractor.

Implements a BeautifulSoup-based extractor for the "Management's Discussion and
Analysis" section of US 10-Q (Item 2) and 10-K (Item 7) filings. The extractor
locates the section header using regex heuristics robust to whitespace and
inline tags, then walks forward until the next Item-level heading.

The module is exposed as both a function :func:`extract_mda_section` and a
stateful class :class:`MDAParser` for callers that need to amortise parser setup.
"""

from __future__ import annotations

import re
from typing import Final

from bs4 import BeautifulSoup, Tag

# Strict regex: starts with "Item", digits + optional dot, then "Management's".
# We match the heading even if it spans inline tags by relying on ``bs4``'s
# ``get_text`` after locating the parent header tag.
_MDA_HEADING_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*item\s*(\d+)\s*\.?\s*[\.:]?\s*management['\u2019]?s\s+discussion\s+and\s+analysis",
    re.IGNORECASE,
)
# Negative lookahead pattern: subsequent Item header stops the section.
_NEXT_ITEM_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*item\s*\d+\s*[\.:]?\s",
    re.IGNORECASE,
)


# Item numbers that correspond to MD&A in each form type.
MDA_ITEM_NUMBER: Final[dict[str, int]] = {
    "10-Q": 2,
    "10-K": 7,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_mda_section(html: str, form_type: str) -> str:
    """Extract the MD&A section text from a 10-Q/10-K HTML blob.

    Args:
        html: The full filing HTML as a string.
        form_type: Filing form code; only ``"10-Q"`` and ``"10-K"`` are
            supported. MD&A is Item 2 in 10-Q and Item 7 in 10-K.

    Returns:
        Cleaned plain-text MD&A. Returns an empty string if the section cannot
        be located, never raises for missing-section (callers can treat that
        as "no signal for this period").

    Raises:
        ValueError: If ``form_type`` is not a supported filing form.
    """
    target_item = MDA_ITEM_NUMBER.get(form_type.upper())
    if target_item is None:
        raise ValueError(
            f"Unsupported form_type={form_type!r}; expected one of {list(MDA_ITEM_NUMBER)}"
        )
    parser = MDAParser()
    return parser.extract(html, item_number=target_item, form_type=form_type.upper())


def clean_text(text: str) -> str:
    """Strip boilerplate, collapse whitespace, lowercase the result.

    Args:
        text: Raw text from a section.

    Returns:
        Lowercased single-line string with non-alphanumeric punctuation
        removed and tokens separated by a single space.
    """
    text = re.sub(r"<[^>]+>", " ", text, flags=re.IGNORECASE)
    text = text.lower()
    # Replace any character that is NOT a-z, 0-9, or whitespace with a space.
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Stateful parser
# ---------------------------------------------------------------------------


class MDAParser:
    """Robust, stateful MD&A section locator.

    Amortises the construction of a BeautifulSoup parser. For small batches
    of filings this is barely faster than calling :func:`extract_mda_section`
    directly, but helps when iterating over thousands of filings.

    Args:
        parser: ``bs4`` parser backend. Defaults to ``"lxml"`` which handles
            broken HTML gracefully; fall back to ``"html.parser"`` if lxml
            is unavailable in the deployment environment.
    """

    def __init__(self, parser: str = "lxml") -> None:
        self._parser = parser

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def extract(self, html: str, *, item_number: int, form_type: str) -> str:
        """Extract the body text of the MD&A item.

        Args:
            html: HTML text of the entire filing.
            item_number: Expected item number for the MD&A section.
            form_type: ``"10-Q"`` or ``"10-K"``; recorded for telemetry.

        Returns:
            Cleaned MDA text. Empty if not found.
        """
        soup = BeautifulSoup(html, self._parser)
        header_tag = self._locate_header(soup, item_number)
        if header_tag is None:
            return ""
        body_tags = self._collect_body(header_tag)
        if not body_tags:
            return ""
        text = " ".join(self._safe_get_text(t) for t in body_tags)
        return clean_text(text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _heading_text(tag: Tag) -> str:
        return re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()

    def _locate_header(self, soup: BeautifulSoup, item_number: int) -> Tag | None:
        """Find the header tag for ``Item <item_number>. Management's..."``.

        We do this by scanning every tag's text and looking for an exact match
        on the regex. The first match that also contains the right item number
        is returned. Returns ``None`` if not found.
        """
        target_prefix = f"item {item_number}"
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "p", "div", "b", "strong"]):
            text = self._heading_text(tag)
            if not text:
                continue
            m = _MDA_HEADING_RE.match(text)
            if m is None:
                continue
            # The first integer inside the pattern is the item number.
            try:
                found_item = int(m.group(1))
            except (TypeError, ValueError):
                continue
            if found_item != item_number:
                continue
            if target_prefix not in text.lower():
                continue
            return tag
        return None

    def _collect_body(self, header_tag: Tag) -> list[Tag]:
        """Walk forward from ``header_tag`` until the next Item-level header.

        Returns the list of sibling tags that constitute the body.
        """
        body: list[Tag] = []
        # Walk both same-level (sibling) elements, then up if needed.
        current: Tag | None = header_tag
        # First, collect same-level siblings after the heading itself.
        for sib in header_tag.find_next_siblings():
            text = self._heading_text(sib)
            if not text:
                # Likely a wrapper; recurse into children.
                for child in sib.find_all(["p", "div"], recursive=True):
                    body.append(child)
                continue
            if _NEXT_ITEM_RE.match(text) and "management" not in text.lower():
                # Stop at next "Item X." heading.
                break
            body.append(sib)
        # Fallback: if no body was captured at sibling level, take descendants.
        if not body:
            body.extend(header_tag.find_all_next(["p", "div"]))
            # Trim leading siblings until just after the heading.
            current = header_tag
            while body and body[0] is current:
                body.pop(0)
        return body

    @staticmethod
    def _safe_get_text(tag: Tag) -> str:
        try:
            return tag.get_text(" ", strip=True)
        except Exception:  # pragma: no cover - defensive guard
            return ""
