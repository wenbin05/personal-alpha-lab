from __future__ import annotations

import hashlib
import html
import re
from typing import Iterable


MIN_USEFUL_TEXT_CHARS = 40
SUSPICIOUS_LARGE_TEXT_CHARS = 1_000_000

_SCRIPT_STYLE_RE = re.compile(r"<(script|style).*?>.*?</\1>", flags=re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_SPACES_RE = re.compile(r"[ \t\r\f\v]+")


def strip_html_tags(text: str) -> str:
    """Remove common HTML markup while preserving readable line breaks."""
    if not text:
        return ""
    without_scripts = _SCRIPT_STYLE_RE.sub(" ", text)
    with_breaks = re.sub(r"</?(p|br|div|li|tr|h[1-6])[^>]*>", "\n", without_scripts, flags=re.IGNORECASE)
    return _TAG_RE.sub(" ", with_breaks)


def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    lines = [_SPACES_RE.sub(" ", line).strip() for line in text.splitlines()]
    compact = "\n".join(line for line in lines)
    compact = _BLANK_LINES_RE.sub("\n\n", compact)
    return compact.strip()


def clean_text(raw_text: str, strip_html: bool = True) -> str:
    text = html.unescape(raw_text or "")
    if strip_html:
        text = strip_html_tags(text)
    return normalize_whitespace(text)


def compute_text_hash(text: str, fallback_parts: Iterable[str | None] | None = None) -> str:
    cleaned = normalize_whitespace(text or "")
    if cleaned:
        payload = cleaned
    else:
        payload = "|".join((part or "").strip() for part in (fallback_parts or []) if (part or "").strip())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def preview_text(text: str, limit: int = 2_000) -> str:
    cleaned = normalize_whitespace(text or "")
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "\n\n[preview truncated]"


def text_quality_warnings(raw_text: str, cleaned_text: str | None = None) -> list[str]:
    cleaned = cleaned_text if cleaned_text is not None else clean_text(raw_text)
    warnings: list[str] = []
    if not normalize_whitespace(raw_text or ""):
        warnings.append("Raw text is empty.")
    if len(cleaned or "") < MIN_USEFUL_TEXT_CHARS:
        warnings.append("Cleaned text is very short; future extraction may have little context.")
    if len(raw_text or "") > SUSPICIOUS_LARGE_TEXT_CHARS:
        warnings.append("Raw text is unusually large; preview and parsing may be slow.")
    return warnings


def join_warnings(*groups: Iterable[str] | str | None) -> str:
    warnings: list[str] = []
    for group in groups:
        if group is None:
            continue
        if isinstance(group, str):
            parts = [part.strip() for part in group.split(";") if part.strip()]
        else:
            parts = [str(part).strip() for part in group if str(part).strip()]
        warnings.extend(parts)
    deduped = list(dict.fromkeys(warnings))
    return "; ".join(deduped)
