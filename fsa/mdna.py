"""MD&A extraction from SEC filing HTML.

MD&A ("Management's Discussion and Analysis") is narrative HTML — there is
no structured marker for it, so extraction is heuristic: strip the HTML to
text, then locate the section boundaries by their item headings. The first
"Item 7" hit is usually the table of contents; we therefore take the
longest candidate segment, which is reliably the real section.

Pure stdlib (regex + html), no BeautifulSoup dependency, so it runs
anywhere the fsa package does.
"""

from __future__ import annotations

import html as _html
import re

# Section boundaries. 10-K: Item 7 -> Item 7A/8. 10-Q: Item 2 -> Item 3/4.
_BOUNDS = {
    "10-K": (
        re.compile(r"item\s*7\s*[.:\-–—]?\s*management", re.I),
        re.compile(r"item\s*7a\s*[.:\-–—]?\s*quantitative"
                   r"|item\s*8\s*[.:\-–—]?\s*financial\s+statements", re.I),
    ),
    "10-Q": (
        re.compile(r"item\s*2\s*[.:\-–—]?\s*management", re.I),
        re.compile(r"item\s*3\s*[.:\-–—]?\s*quantitative"
                   r"|item\s*4\s*[.:\-–—]?\s*controls", re.I),
    ),
}

_MIN_SECTION_CHARS = 2000  # anything shorter is a TOC hit, not the section
_MAX_SECTION_CHARS = 400_000


def html_to_text(html: str) -> str:
    """Filing HTML -> readable plain text with paragraph breaks."""
    html = re.sub(r"(?is)<(script|style|head).*?</\1>", " ", html)
    # Block-level tags become newlines so paragraphs survive.
    html = re.sub(r"(?i)</?(p|div|br|tr|li|h[1-6]|table|td)[^>]*>", "\n", html)
    text = re.sub(r"<[^>]+>", " ", html)
    text = _html.unescape(text).replace("\xa0", " ").replace("’", "'")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_mdna(text: str, form: str = "10-K") -> str | None:
    """Return the MD&A section from filing *text*, or None if not found.

    Heuristic: collect every candidate (start heading -> next end heading)
    and keep the LATEST one that is long enough. The real section always
    follows the table-of-contents hit, so latest-start beats longest —
    a TOC-anchored span can swallow the TOC plus the body and win on
    length while starting in the wrong place.
    """
    start_re, end_re = _BOUNDS.get(form, _BOUNDS["10-K"])
    best: str | None = None
    for m in start_re.finditer(text):
        s = m.start()
        e_match = end_re.search(text, s + 100)
        e = e_match.start() if e_match else min(len(text),
                                                s + _MAX_SECTION_CHARS)
        segment = text[s:e].strip()
        if len(segment) >= _MIN_SECTION_CHARS:
            best = segment  # later starts overwrite earlier ones
    return best[:_MAX_SECTION_CHARS] if best else None


def chunk_text(text: str, target_chars: int = 1400) -> list[str]:
    """Split MD&A into paragraph-aligned chunks of roughly target size,
    for keyword search with readable excerpts."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        if buf and len(buf) + len(p) + 2 > target_chars:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
        # A single monster paragraph still gets emitted (split hard).
        while len(buf) > 2 * target_chars:
            chunks.append(buf[:target_chars])
            buf = buf[target_chars:]
    if buf:
        chunks.append(buf)
    return chunks
