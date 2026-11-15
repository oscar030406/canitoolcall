"""``canitoolcall mcp``: a stdio MCP server for the matrix lookup and the live probe.

Needs the optional extra (``pip install canitoolcall[mcp]``, the official ``mcp``
SDK). The tool bodies below are plain functions that import nothing outside the
core install; only :func:`build_server` and :func:`serve` import ``mcp``.

Tools: ``lookup``, ``list_families``, ``list_engines`` (all read ``matrix.json``,
chosen once at startup with ``--matrix``, never by the caller) and ``probe``
(sends requests only to endpoints the OPERATOR allowed at startup; see :class:`ProbePolicy`).
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlsplit

from canitoolcall.lookup import DEFAULT_MATRIX, DEFAULT_SITE, LookupFailure, load_matrix
from canitoolcall.lookup import lookup as lookup_matrix

MAX_TIMEOUT_S = 120.0
MAX_CONCURRENCY = 8

MCP_MISSING_HINT = "the MCP server needs the optional extra: pip install 'canitoolcall[mcp]'"


def _site_base(source: str) -> str:
    return urljoin(source, ".") if source.startswith(("http://", "https://")) else DEFAULT_SITE


def _load(source: str) -> dict[str, Any]:
    try:
        return load_matrix(source)
    except LookupFailure as e:
        raise ValueError(str(e)) from None


def tool_lookup(source: str, name: str, engine: str | None = None) -> dict[str, Any]:
    """Same answer as ``canitoolcall lookup NAME --engine E --json``; raises ``ValueError`` on a bad question."""
    try:
        return lookup_matrix(load_matrix(source), name, engine or None, _site_base(source))
    except LookupFailure as e:
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


DENIED = "target not allowed by server configuration"
"""The only thing a tool caller learns about a refused target or key: no resolved IPs, no hints."""

_CGNAT = ipaddress.ip_network("100.64.0.0/10")
_METADATA = frozenset(ipaddress.ip_address(a) for a in ("169.254.169.254", "fd00:ec2::254"))


def _normalize(url: str) -> tuple[str, str, int, str]:
    """(scheme, host, port, path) with the default port made explicit and no trailing slash."""
    p = urlsplit(url)
    port = p.port or (443 if p.scheme == "https" else 80)
    return p.scheme, (p.hostname or "").rstrip(".").lower(), port, p.path.rstrip("/")


def _covers(allowed: str, requested: str, *, exact: bool = False) -> bool:
    """Same scheme, host and port, and ``requested``'s path equals or sits under ``allowed``'s."""
    a, r = _normalize(allowed), _normalize(requested)
    if a[:3] != r[:3]:
        return False
    return a[3] == r[3] if exact else r[3] == a[3] or r[3].startswith(a[3] + "/")


def _is_loopback_host(host: str) -> bool:
    host = host.rstrip(".").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _unsafe_address(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Not public unicast: private, loopback, link-local, CGNAT, multicast, unspecified, reserved or metadata."""
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return (
        not ip.is_global
        or ip.is_multicast
        or ip.is_unspecified
        or ip.is_link_local
        or ip.is_loopback
        or ip in _METADATA
        or (isinstance(ip, ipaddress.IPv4Address) and ip in _CGNAT)
    )


@dataclass(frozen=True)
class ProbePolicy:
    """What the operator allowed ``probe`` to reach, fixed when the server starts.

    * By default only loopback (127.0.0.0/8, ::1, ``localhost``) and no API key is ever sent.
    * ``allow_base_urls``: base URLs (scheme, host, port and a path prefix) allowed beyond loopback.
    * ``key_bindings``: ``{ENV_VAR: base_url}``; the variable's value is sent only to that base URL, and the
      binding also allows that URL as a target.
    """

    allow_base_urls: tuple[str, ...] = ()
    key_bindings: dict[str, str] = field(default_factory=dict)

    @classmethod
    def parse(cls, allow_base_urls: Iterable[str] = (), allow_api_key_envs: Iterable[str] = ()) -> ProbePolicy:
        """Build from ``--allow-base-url URL`` and ``--allow-api-key-env VAR=URL`` values; ``ValueError`` if bad."""
        from canitoolcall.probe import check_base_url

        urls = tuple(check_base_url(u) for u in allow_base_urls)
        bindings: dict[str, str] = {}
        for spec in allow_api_key_envs:
            var, sep, url = spec.partition("=")
            var, url = var.strip(), url.strip()
            if not sep or not var or not url:
                raise ValueError(f"--allow-api-key-env must be VAR=BASE_URL (got {var or spec!r})")
            if var in bindings:
                raise ValueError(f"--allow-api-key-env {var} given twice")
            bindings[var] = check_base_url(url)
        return cls(urls, bindings)

    def target_allowed(self, base_url: str) -> bool:
        if _is_loopback_host(urlsplit(base_url).hostname or ""):
            return True
        return any(_covers(u, base_url) for u in (*self.allow_base_urls, *self.key_bindings.values()))

    def key_for(self, api_key_env: str, base_url: str) -> bool:
        """True if ``api_key_env`` is bound to exactly this base URL."""
        bound = self.key_bindings.get(api_key_env)
        return bound is not None and _covers(bound, base_url, exact=True)

    def connector(self, base_url: str) -> Callable[..., socket.socket]:
        """A ``socket.create_connection`` replacement that vets the resolved addresses, then connects to one.

        Loopback targets must resolve only to loopback. Any other target must resolve only to public unicast
        addresses, unless the operator allowlisted that base URL with an IP literal (they named the address).
        Resolving and connecting happen together, so DNS cannot change the answer in between.
        """
        host = urlsplit(base_url).hostname or ""
        loopback = _is_loopback_host(host)
        trusted_literal = _is_ip_literal(host) and any(_covers(u, base_url) for u in self.allow_base_urls)

        def connect(
            address: tuple[str, int], timeout: float | None = socket.getdefaulttimeout(), source_address: Any = None
        ) -> socket.socket:
            name, port = address
            infos = socket.getaddrinfo(name, port, type=socket.SOCK_STREAM)
            ips = [ipaddress.ip_address(str(i[4][0]).split("%")[0]) for i in infos]
            if not ips:
                raise OSError(DENIED)
            if loopback:
                ok = all(ip.is_loopback for ip in ips)
            else:
                ok = trusted_literal or not any(_unsafe_address(ip) for ip in ips)
            if not ok:
                raise OSError(DENIED)
            last: OSError | None = None
            for info in infos:
                try:
                    return socket.create_connection((str(info[4][0]), port), timeout, source_address)
                except OSError as e:
                    last = e
            raise last or OSError(DENIED)

        return connect


def _hostport(base_url: str) -> tuple[str, int]:
    p = urlsplit(base_url)
    return p.hostname or "", p.port or (443 if p.scheme == "https" else 80)


def tool_probe(
    base_url: str,
    model: str,
    api_key_env: str | None = None,
    stream: bool = True,
    timeout_s: float = 60.0,
    concurrency: int = 4,
    *,
    policy: ProbePolicy | None = None,
) -> dict[str, Any]:
    """Run ``canitoolcall probe`` against ``base_url`` if the operator's ``policy`` allows it; never returns the key."""
    from canitoolcall.probe import ProbeRequestError, check_base_url, guarded_opener, insecure_key_transport, probe

    policy = policy or ProbePolicy()
    check_base_url(base_url)  # ValueError with a readable message
    if not model.strip():
        raise ValueError("model must not be empty")
    if not 0 < timeout_s <= MAX_TIMEOUT_S:
        raise ValueError(f"timeout_s must be in (0, {MAX_TIMEOUT_S:g}]")
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise ValueError(f"concurrency must be in [1, {MAX_CONCURRENCY}]")
    if not policy.target_allowed(base_url):
        raise ValueError(DENIED)
    api_key: str | None = None
    if api_key_env:
        # The caller may only name a variable the operator bound to exactly this base URL (a model cannot ask
        # for AWS_SECRET_ACCESS_KEY, nor send a bound key to another host).
        if not policy.key_for(api_key_env, base_url):
            raise ValueError(DENIED)
        api_key = os.environ.get(api_key_env) or None
    if api_key and insecure_key_transport(base_url):
        raise ValueError("refusing to send an API key over plain http to a non-loopback host; use https")
    connect = policy.connector(base_url)
    # One vetted connection up front: a refused or unreachable target fails fast with a generic message.
    try:
        connect(_hostport(base_url), min(timeout_s, 10.0)).close()
    except OSError:
        raise ValueError(f"cannot reach the target, or {DENIED}") from None
    try:
        report = probe(
            base_url,
            model,
            api_key=api_key,
            stream_modes=(False, True) if stream else (False,),
            timeout_s=timeout_s,
            concurrency=concurrency,
            opener=guarded_opener(connect),
        )
    except ProbeRequestError:
        raise ValueError("probe failed") from None
    result = report.to_dict()
    result["ok"] = not any(o.status in ("fail", "error") for o in report.outcomes)
    return result


def build_server(matrix: str | None = None, policy: ProbePolicy | None = None) -> Any:
    """Build the ``MCPServer``; raises ``ImportError`` (with a hint) if the extra is missing."""
    try:
        from mcp.server.mcpserver import MCPServer
        from mcp.server.mcpserver.exceptions import ToolError
        from mcp_types import ToolAnnotations
    except ImportError as e:
        raise ImportError(MCP_MISSING_HINT) from e

    from canitoolcall import __version__

    source = matrix or DEFAULT_MATRIX
    probe_policy = policy or ProbePolicy()
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
        """Run the canitoolcall live probe. THIS SENDS REQUESTS to `base_url`: the 9 built-in scripted
        chat/completions calls with tool definitions (twice each when `stream` is true) to the
        OpenAI-compatible server at `base_url` (e.g. http://localhost:8000/v1) using `model`.
        Only call it for an endpoint the user named. The server's operator decides which targets are
        reachable (loopback by default); anything else is refused. `api_key_env` is the NAME of an
        environment variable the operator bound to exactly this `base_url`; it is sent only as the
        Authorization header to it, never logged or returned. Omit it for no key. Returns
        per-scenario outcomes."""
        return await run(
            tool_probe,
            base_url,
            model,
            api_key_env,
            stream,
            timeout_s,
            concurrency,
            policy=probe_policy,
        )

    return server


def serve(matrix: str | None = None, policy: ProbePolicy | None = None) -> None:
    """Run the server on stdio until the client disconnects."""
    build_server(matrix, policy).run("stdio")
