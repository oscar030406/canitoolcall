"""In-process MCP client <-> `canitoolcall mcp` server tests.

The matrix is a tiny hand-written file in a tmp dir (mock data, not results) and the probe talks to the mock
OpenAI-compatible server from ``conftest.py``. No real network and no real key.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anyio
import pytest

pytest.importorskip("mcp")

from mcp import Client

from canitoolcall.cli import EXIT_ERROR, main
from canitoolcall.mcp_server import build_server

from .conftest import MockOpenAI

MATRIX = {
    "schema_version": "1.0",
    "canitoolcall_version": "0.0.0",
    "families": ["alpha", "beta"],
    "family_info": {
        "alpha": {"name": "Alpha Family", "models": ["acme/Alpha-7B"]},
        "beta": {"name": "Beta Family", "models": ["acme/Beta-1B"]},
    },
    "engines": [
        {"name": "vllm", "version": "1.0", "commit": None, "run_started_at": "t", "run_finished_at": "t"},
        {"name": "sglang", "version": "2.0", "commit": None, "run_started_at": "t", "run_finished_at": "t"},
    ],
    "cells": [
        {
            "family": "alpha",
            "engine": "vllm",
            "engine_version": "1.0",
            "status": "pass",
            "pass_rate": 1.0,
            "counts": {"pass": 3},
            "run_at": "t",
            "page": "cells/vllm/1.0/alpha.html",
        },
        {
            "family": "alpha",
            "engine": "sglang",
            "engine_version": "2.0",
            "status": "fail",
            "pass_rate": 0.5,
            "counts": {"pass": 1, "fail": 1},
            "run_at": "t",
            "page": "cells/sglang/2.0/alpha.html",
        },
    ],
}

DEFAULT_ALLOWED = ("CANITOOLCALL_API_KEY",)


@pytest.fixture
def matrix_path(tmp_path: Path) -> Path:
    p = tmp_path / "matrix.json"
    p.write_text(json.dumps(MATRIX), encoding="utf-8")
    return p


def call(
    tool: str, args: dict[str, Any] | None = None, *, matrix: Path, allowed: tuple[str, ...] = DEFAULT_ALLOWED
) -> tuple[bool, Any, str]:
    """Run one tool call through an in-process client; returns (is_error, structured content, text)."""
    out: list[tuple[bool, Any, str]] = []

    async def go() -> None:
        async with Client(build_server(str(matrix), allowed)) as client:
            r = await client.call_tool(tool, args or {})
            text = " ".join(getattr(c, "text", "") for c in r.content)
            out.append((bool(r.is_error), r.structured_content, text))

    anyio.run(go)
    return out[0]


def test_lists_the_four_tools(matrix_path: Path) -> None:
    names: list[str] = []

    async def go() -> None:
        async with Client(build_server(str(matrix_path))) as client:
            tools = (await client.list_tools()).tools
            names.extend(t.name for t in tools)
            probe_tool = next(t for t in tools if t.name == "probe")
            assert "SENDS REQUESTS" in (probe_tool.description or "")

    anyio.run(go)
    assert sorted(names) == ["list_engines", "list_families", "lookup", "probe"]


def test_lookup_matches_the_cli_answer(matrix_path: Path) -> None:
    from canitoolcall.lookup import lookup as lookup_matrix

    err, data, _ = call("lookup", {"name": "acme/alpha-7b", "engine": "VLLM"}, matrix=matrix_path)
    assert not err
    assert data["family"] == "alpha" and data["matched_by"] == "model name"
    assert [c["engine"] for c in data["cells"]] == ["vllm"]
    assert data == lookup_matrix(MATRIX, "acme/alpha-7b", "VLLM")

    err, data, _ = call("lookup", {"name": "alpha"}, matrix=matrix_path)
    assert not err and [c["status"] for c in data["cells"]] == ["fail", "pass"]  # sglang, vllm


def test_lookup_errors_are_tool_errors(matrix_path: Path) -> None:
    err, _, text = call("lookup", {"name": "nope"}, matrix=matrix_path)
    assert err and "unknown family or model" in text
    err, _, text = call("lookup", {"name": "beta"}, matrix=matrix_path)
    assert err and "no cell" in text


def test_bad_matrix_is_a_tool_error(tmp_path: Path) -> None:
    err, _, text = call("list_engines", matrix=tmp_path / "missing.json")
    assert err and "cannot read" in text


def test_list_families_and_engines(matrix_path: Path) -> None:
    err, data, _ = call("list_families", matrix=matrix_path)
    assert not err
    assert data["families"][0] == {"slug": "alpha", "name": "Alpha Family", "models": ["acme/Alpha-7B"]}
    err, data, _ = call("list_engines", matrix=matrix_path)
    assert not err
    assert data["engines"] == [{"name": "sglang", "versions": ["2.0"]}, {"name": "vllm", "versions": ["1.0"]}]


def test_probe_against_mock_endpoint(matrix_path: Path, mock_openai: MockOpenAI) -> None:
    args = {"base_url": mock_openai.base_url, "model": "mock-model", "stream": False}
    err, data, _ = call("probe", args, matrix=matrix_path)
    assert not err
    assert data["ok"] is True and data["model"] == "mock-model"
    assert data["summary"]["fail"] == 0 and data["summary"]["pass"] > 0
    assert mock_openai.requests and all(not body.get("stream") for _, body in mock_openai.requests)


def test_probe_reports_failures(matrix_path: Path, mock_openai: MockOpenAI) -> None:
    mock_openai.override("single-call", stream=False, tool_calls=())
    args = {"base_url": mock_openai.base_url, "model": "m", "stream": False}
    err, data, _ = call("probe", args, matrix=matrix_path)
    assert not err and data["ok"] is False and data["summary"]["fail"] >= 1


def test_probe_sends_the_named_key_but_never_returns_it(
    matrix_path: Path, mock_openai: MockOpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-test-not-a-real-key-123"
    monkeypatch.setenv("MCP_TEST_KEY", secret)
    args = {"base_url": mock_openai.base_url, "model": "m", "api_key_env": "MCP_TEST_KEY", "stream": False}
    err, data, text = call("probe", args, matrix=matrix_path, allowed=(*DEFAULT_ALLOWED, "MCP_TEST_KEY"))
    assert not err
    assert any(h.get("Authorization") == f"Bearer {secret}" for h, _ in mock_openai.requests)
    assert secret not in json.dumps(data) and secret not in text


def test_probe_refuses_env_vars_the_operator_did_not_allow(
    matrix_path: Path, mock_openai: MockOpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOME_OTHER_SECRET", "hunter2")
    args = {"base_url": mock_openai.base_url, "model": "m", "api_key_env": "SOME_OTHER_SECRET"}
    err, _, text = call("probe", args, matrix=matrix_path)
    assert err and "not allowed" in text and "hunter2" not in text
    assert not mock_openai.requests


def test_probe_refuses_key_over_plain_http_to_remote_host(matrix_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CANITOOLCALL_API_KEY", "sk-test")
    args = {"base_url": "http://example.invalid/v1", "model": "m", "api_key_env": "CANITOOLCALL_API_KEY"}
    err, _, text = call("probe", args, matrix=matrix_path)
    assert err and "plain http" in text


@pytest.mark.parametrize(
    "args",
    [
        {"base_url": "ftp://x/v1", "model": "m"},
        {"base_url": "http://user:pw@127.0.0.1:1/v1", "model": "m"},
        {"base_url": "http://127.0.0.1:1/v1", "model": " "},
        {"base_url": "http://127.0.0.1:1/v1", "model": "m", "timeout_s": 10_000},
        {"base_url": "http://127.0.0.1:1/v1", "model": "m", "concurrency": 100},
    ],
)
def test_probe_validates_arguments(matrix_path: Path, args: dict[str, Any]) -> None:
    err, _, text = call("probe", args, matrix=matrix_path)
    assert err and "pw" not in text


def test_cli_reports_missing_extra(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import builtins

    real_import: Callable[..., Any] = builtins.__import__

    def fake(name: str, *a: Any, **k: Any) -> Any:
        if name == "mcp" or name.startswith(("mcp.", "mcp_types")):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    assert main(["mcp"]) == EXIT_ERROR
    assert "canitoolcall[mcp]" in capsys.readouterr().err
