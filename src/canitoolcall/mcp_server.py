"""``canitoolcall mcp``: a stdio MCP server for the matrix lookup and the live probe.

Needs the optional extra (``pip install canitoolcall[mcp]``, the official ``mcp``
SDK). The tool bodies below are plain functions that import nothing outside the
core install; only :func:`build_server` and :func:`serve` import ``mcp``.

Tools: ``lookup``, ``list_families``, ``list_engines`` (all read ``matrix.json``,
chosen once at startup with ``--matrix``, never by the caller) and ``probe``
(sends requests to a caller-supplied endpoint).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Collection
from typing import Any
from urllib.parse import urljoin

from canitoolcall.lookup import DEFAULT_MATRIX, DEFAULT_SITE, LookupError_, load_matrix
from canitoolcall.lookup import lookup as lookup_matrix

DEFAULT_API_KEY_ENV = "CANITOOLCALL_API_KEY"
MAX_TIMEOUT_S = 120.0
MAX_CONCURRENCY = 8

MCP_MISSING_HINT = "the MCP server needs the optional extra: pip install 'canitoolcall[mcp]'"


def _site_base(source: str) -> str:
    return urljoin(source, ".") if source.startswith(("http://", "https://")) else DEFAULT_SITE


def _load(source: str) -> dict[str, Any]:
    try:
        return load_matrix(source)
    except LookupError_ as e:
        raise ValueError(str(e)) from None


def tool_lookup(source: str, name: str, engine: str | None = None) -> dict[str, Any]:
    """Same answer as ``canitoolcall lookup NAME --engine E --json``; raises ``ValueError`` on a bad question."""
    try:
        return lookup_matrix(load_matrix(source), name, engine or None, _site_base(source))
    except LookupError_ as e:
        raise ValueError(str(e)) from None


def tool_list_families(source: str) -> dict[str, Any]:
    data = _load(source)
    info = data.get("family_info") or {}
    return {
        "schema_version": data["schema_version"],
        "families": [
            {"slug": slug, "name": info.get(slug, {}).get("name"), "models": info.get(slug, {}).get("models", [])}
            for slug in data["families"]
        ],
    }


def tool_list_engines(source: str) -> dict[str, Any]:
    data = _load(source)
    versions: dict[str, set[str]] = {}
    for e in data["engines"]:
        versions.setdefault(str(e["name"]), set()).add(str(e["version"]))
    return {
        "schema_version": data["schema_version"],
        "engines": [{"name": n, "versions": sorted(v)} for n, v in sorted(versions.items())],
    }


def tool_probe(
    base_url: str,
    model: str,
    api_key_env: str | None = None,
    stream: bool = True,
    timeout_s: float = 60.0,
    concurrency: int = 4,
    *,
    allowed_key_envs: Collection[str] = (DEFAULT_API_KEY_ENV,),
) -> dict[str, Any]:
    """Run ``canitoolcall probe`` against ``base_url``; returns the JSON report (never the key)."""
    from canitoolcall.probe import ProbeRequestError, check_base_url, check_reachable, insecure_key_transport, probe

    check_base_url(base_url)  # ValueError with a readable message
    if not model.strip():
        raise ValueError("model must not be empty")
    if not 0 < timeout_s <= MAX_TIMEOUT_S:
        raise ValueError(f"timeout_s must be in (0, {MAX_TIMEOUT_S:g}]")
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise ValueError(f"concurrency must be in [1, {MAX_CONCURRENCY}]")
    api_key: str | None = None
    if api_key_env:
        # A tool caller must not be able to name any variable (say AWS_SECRET_ACCESS_KEY) and have it sent to
        # an endpoint of its choosing, so only variables the server's operator allowed at startup are read.
        if api_key_env not in allowed_key_envs:
            raise ValueError(
                f"api_key_env {api_key_env!r} is not allowed; the server was started allowing: "
                f"{', '.join(sorted(allowed_key_envs)) or '(none)'} (see --allow-api-key-env)"
            )
        api_key = os.environ.get(api_key_env) or None
    if api_key and insecure_key_transport(base_url):
        raise ValueError("refusing to send an API key over plain http to a non-loopback host; use https")
    try:
        check_reachable(base_url, timeout_s=min(timeout_s, 10.0))
    except ProbeRequestError as e:
        raise ValueError(str(e)) from None
    report = probe(
        base_url,
        model,
        api_key=api_key,
        stream_modes=(False, True) if stream else (False,),
        timeout_s=timeout_s,
        concurrency=concurrency,
    )
    result = report.to_dict()
    result["ok"] = not any(o.status in ("fail", "error") for o in report.outcomes)
    return result


def build_server(matrix: str | None = None, allowed_key_envs: Collection[str] = (DEFAULT_API_KEY_ENV,)) -> Any:
    """Build the ``MCPServer``; raises ``ImportError`` (with a hint) if the extra is missing."""
    try:
        from mcp.server.mcpserver import MCPServer
        from mcp.server.mcpserver.exceptions import ToolError
        from mcp_types import ToolAnnotations
    except ImportError as e:
        raise ImportError(MCP_MISSING_HINT) from e

    from canitoolcall import __version__

    source = matrix or DEFAULT_MATRIX
    allowed = tuple(allowed_key_envs)
    server = MCPServer(
        "canitoolcall",
        version=__version__,
        instructions=(
            "CanIToolCall: does a model family's tool-call/reasoning output parse on an inference engine "
            "(vLLM, SGLang, llama.cpp, Ollama, transformers)? Use lookup for the published matrix and probe "
            "to test a live OpenAI-compatible server."
        ),
    )

    async def run(fn: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any) -> dict[str, Any]:
        # Our own ValueErrors carry user-safe messages; anything else stays server-side (the SDK hides it).
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        except ValueError as e:
            raise ToolError(str(e)) from None

    read_only = ToolAnnotations(read_only_hint=True, open_world_hint=True)

    @server.tool(annotations=read_only)
    async def lookup(name: str, engine: str | None = None) -> dict[str, Any]:
        """Does a model family parse on an engine? `name` is a family slug, family name or model name
        (e.g. "qwen3-hermes" or "Qwen/Qwen3.5-9B"); `engine` optionally limits to one engine.
        Reads the matrix.json the server was started with (the published one by default)."""
        return await run(tool_lookup, source, name, engine)

    @server.tool(annotations=read_only)
    async def list_families() -> dict[str, Any]:
        """List the model families in the matrix with their display names and model repos."""
        return await run(tool_list_families, source)

    @server.tool(annotations=read_only)
    async def list_engines() -> dict[str, Any]:
        """List the engines in the matrix with the versions that were tested."""
        return await run(tool_list_engines, source)

    @server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=True))
    async def probe(
        base_url: str,
        model: str,
        api_key_env: str | None = None,
        stream: bool = True,
        timeout_s: float = 60.0,
        concurrency: int = 4,
    ) -> dict[str, Any]:
        """Run the canitoolcall live probe. THIS SENDS REQUESTS to `base_url`: about 10 scripted
        chat/completions calls with tool definitions (twice each when `stream` is true) to the
        OpenAI-compatible server at `base_url` (e.g. http://localhost:8000/v1) using `model`.
        Only call it for an endpoint the user named. `api_key_env` is the NAME of an environment
        variable on the machine running this server that holds the key (never the key itself); it
        must be one the server's operator allowed, and it is sent only as the Authorization header
        to `base_url`, never logged or returned. Omit it for no key. Returns per-scenario outcomes."""
        return await run(
            tool_probe,
            base_url,
            model,
            api_key_env,
            stream,
            timeout_s,
            concurrency,
            allowed_key_envs=allowed,
        )

    return server


def serve(matrix: str | None = None, allowed_key_envs: Collection[str] = (DEFAULT_API_KEY_ENV,)) -> None:
    """Run the server on stdio until the client disconnects."""
    build_server(matrix, allowed_key_envs).run("stdio")
