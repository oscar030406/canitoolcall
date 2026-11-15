"""``canitoolcall lookup``: answer "does family X parse on engine Y?" from a ``matrix.json``.

The source is the published matrix by default, or a local file or URL given
with ``--matrix``. Only the standard library is used for HTTP, with a timeout.
The schema is ``spec/matrix.schema.json``; this reader accepts schema major
version 1 and rejects anything else.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

DEFAULT_SITE = "https://redd34.github.io/canitoolcall/"
DEFAULT_MATRIX = DEFAULT_SITE + "matrix.json"
SUPPORTED_MAJOR = 1
TIMEOUT = 15.0
MAX_BYTES = 32 * 1024 * 1024


class LookupError_(Exception):
    """A user-facing failure (bad source, unknown name, unsupported schema)."""


def _is_url(source: str) -> bool:
    return urlsplit(source).scheme in ("http", "https")


def load_matrix(source: str, timeout: float = TIMEOUT) -> dict[str, Any]:
    """Read and version-check a matrix.json from a path or an http(s) URL."""
    try:
        if _is_url(source):
            req = urllib.request.Request(source, headers={"User-Agent": "canitoolcall-lookup"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read(MAX_BYTES + 1)
        else:
            raw = Path(source).read_bytes()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        reason = getattr(e, "reason", e)
        raise LookupError_(f"cannot read {source}: {reason}") from e
    if len(raw) > MAX_BYTES:
        raise LookupError_(f"{source} is larger than {MAX_BYTES} bytes")
    try:
        data = json.loads(raw)
    except ValueError as e:
        raise LookupError_(f"{source} is not valid JSON: {e}") from e
    check_version(data, source)
    return data  # type: ignore[no-any-return]


def check_version(data: Any, source: str = "matrix.json") -> None:
    if not isinstance(data, dict):
        raise LookupError_(f"{source} is not a matrix.json object")
    version = data.get("schema_version")
    if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+", version):
        raise LookupError_(
            f"{source} has no usable schema_version (got {version!r}); "
            "it predates the versioned schema or is not a matrix.json"
        )
    major = int(version.split(".")[0])
    if major != SUPPORTED_MAJOR:
        raise LookupError_(
            f"{source} has schema_version {version}, but this canitoolcall understands "
            f"major version {SUPPORTED_MAJOR}; "
            "upgrade canitoolcall"
        )
    for key in ("families", "engines", "cells"):
        if not isinstance(data.get(key), list):
            raise LookupError_(f"{source} is malformed: {key!r} is missing or not a list")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def resolve_family(data: Mapping[str, Any], query: str) -> tuple[str, str]:
    """Map a slug, family name or model name to a family slug.

    Returns ``(slug, how)``. Matching order: exact slug; family display name;
    a model repo id (full or the part after ``org/``) from ``family_info``;
    then a unique slug/name/model substring. Case-insensitive throughout.
    """
    families: list[str] = list(data["families"])
    info: Mapping[str, Any] = data.get("family_info") or {}
    q = _norm(query)
    if not q:
        raise LookupError_("empty family or model name")

    for slug in families:
        if _norm(slug) == q:
            return slug, "family slug"
    for slug in families:
        if _norm(str(info.get(slug, {}).get("name", ""))) == q:
            return slug, "family name"
    exact = _hits(families, info, lambda m: q in {_norm(m), _norm(m.split("/", 1)[-1])}, models_only=True)
    if len(exact) == 1:
        return exact[0], "model name"
    if len(exact) > 1:
        raise _ambiguous(query, exact)

    loose = _hits(families, info, lambda s: q in _norm(s))
    if len(loose) == 1:
        return loose[0], "partial match"
    if len(loose) > 1:
        raise _ambiguous(query, loose)
    note = "" if info else " (this matrix.json has no model list, so only family slugs and names were matched)"
    raise LookupError_(f"unknown family or model {query!r}; families: {', '.join(families)}{note}")


def _hits(families: list[str], info: Mapping[str, Any], match: Any, models_only: bool = False) -> list[str]:
    out: list[str] = []
    for slug in families:
        meta = info.get(slug, {})
        names = [str(m) for m in meta.get("models", [])]
        if not models_only:
            names += [slug, str(meta.get("name", ""))]
        if any(match(n) for n in names):
            out.append(slug)
    return out


def _ambiguous(query: str, slugs: list[str]) -> LookupError_:
    return LookupError_(f"{query!r} matches several families ({', '.join(slugs)}); use a family slug")


def lookup(
    data: Mapping[str, Any], query: str, engine: str | None = None, base_url: str = DEFAULT_SITE
) -> dict[str, Any]:
    """Answer the question; returns a JSON-able result or raises :class:`LookupError_`."""
    family, how = resolve_family(data, query)
    engines = sorted({str(e["name"]) for e in data["engines"]})
    if engine is not None and engine.lower() not in {e.lower() for e in engines}:
        raise LookupError_(f"unknown engine {engine!r}; engines in this matrix: {', '.join(engines)}")
    cells = [
        c
        for c in data["cells"]
        if c["family"] == family and (engine is None or str(c["engine"]).lower() == engine.lower())
    ]
    if not cells:
        raise LookupError_(f"the matrix has no cell for {family} on {engine or 'any engine'}")
    cells.sort(key=lambda c: (str(c["engine"]), str(c["engine_version"])))
    return {
        "query": query,
        "family": family,
        "matched_by": how,
        "schema_version": data["schema_version"],
        "cells": [
            {
                "engine": c["engine"],
                "engine_version": c["engine_version"],
                "status": c["status"],
                "pass_rate": c.get("pass_rate"),
                "counts": c.get("counts", {}),
                "run_at": c.get("run_at"),
                "url": urljoin(base_url, str(c["page"])),
            }
            for c in cells
        ],
    }


def format_table(result: Mapping[str, Any]) -> str:
    """A short human table for :func:`lookup` output."""
    lines = [f"{result['family']} (matched {result['query']!r} by {result['matched_by']})"]
    rows = []
    for c in result["cells"]:
        rate = "n/a" if c["pass_rate"] is None else f"{c['pass_rate'] * 100:.0f}%"
        counts = " ".join(f"{k}={v}" for k, v in c["counts"].items() if v)
        rows.append((f"{c['engine']} {c['engine_version']}", c["status"], rate, counts or "-", c["url"]))
    head = ("ENGINE", "STATUS", "STRICT", "COUNTS", "PAGE")
    widths = [max(len(r[i]) for r in [head, *rows]) for i in range(4)]
    for r in [head, *rows]:
        lines.append("  ".join(r[i].ljust(widths[i]) for i in range(4)) + "  " + r[4])
    return "\n".join(lines)
