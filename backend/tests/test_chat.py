"""
问答接口测试 — 非流式 / 流式 SSE / 错误处理 / 多轮对话持久化
"""
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

import app.main as main_module


def _make_docs():
    return [Document(page_content="RAG 是检索增强生成，结合检索与生成模型。")]


def _build_index(client):
    """上传一份文档以建立索引"""
    with patch.object(main_module, "parse_file", return_value=_make_docs()):
        client.post(
            "/api/upload",
            files={"file": ("a.pdf", b"%PDF-1.4 a", "application/pdf")},
        )


def test_chat_without_index(client):
    """索引为空时非流式问答应返回 400"""
    res = client.post("/api/chat", json={"query": "什么是 RAG？"})
    assert res.status_code == 400
    assert "为空" in res.json()["detail"]


def test_chat_stream_without_index(client):
    """索引为空时流式问答应返回 400"""
    res = client.post("/api/chat/stream", json={"query": "什么是 RAG？"})
    assert res.status_code == 400


def test_chat_stream_without_api_key(client, monkeypatch):
    """无 API Key 时流式问答应返回 500"""
    _build_index(client)
    monkeypatch.setattr(main_module, "DEEPSEEK_API_KEY", "")
    res = client.post("/api/chat/stream", json={"query": "什么是 RAG？"})
    assert res.status_code == 500
    assert "DEEPSEEK_API_KEY" in res.json()["detail"]


def test_chat_non_streaming_returns_answer_and_persists_history(client, monkeypatch):
    """非流式问答返回回答与 conversation_id，并持久化对话历史"""
    _build_index(client)
    monkeypatch.setattr(main_module, "DEEPSEEK_API_KEY", "test-key")

    invoked = []

    class _SharedLLM:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            invoked.append(messages)
            return AIMessage(content="RAG 是检索增强生成。")

    shared = _SharedLLM()
    monkeypatch.setattr("app.agent.graph.ChatOpenAI", lambda *a, **kw: shared)

    r1 = client.post("/api/chat", json={"query": "什么是 RAG？"})
    assert r1.status_code == 200
    body1 = r1.json()
    assert "RAG" in body1["answer"]
    conv_id = body1["conversation_id"]
    assert conv_id

    # 第二轮：携带同一 conversation_id，应将历史传入 LLM
    r2 = client.post("/api/chat", json={"query": "它有什么优点？", "conversation_id": conv_id})
    assert r2.status_code == 200
    assert r2.json()["conversation_id"] == conv_id

    # 第二次 invoke 的消息中应包含第一轮的用户问题（历史被回传）
    assert any(
        any(getattr(m, "content", "") == "什么是 RAG？" for m in msgs)
        for msgs in invoked
    ), "第二轮应将历史用户消息传入 LLM"


def test_chat_stream_sse_format(client, monkeypatch):
    """流式问答应产出 conversation_id / status / token / [DONE] 事件"""
    _build_index(client)
    monkeypatch.setattr(main_module, "DEEPSEEK_API_KEY", "test-key")

    class _Chunk:
        def __init__(self, content):
            self.content = content

    class _StreamLLM:
        def __init__(self, *a, **kw):
            pass

        async def astream(self, messages):
            for tok in ["RAG", "是", "检索增强生成"]:
                yield _Chunk(tok)

    monkeypatch.setattr("langchain_openai.ChatOpenAI", _StreamLLM)

    res = client.post("/api/chat/stream", json={"query": "什么是 RAG？"})
    assert res.status_code == 200
    text = res.text

    # 解析 SSE 事件
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            events.append(line[6:].strip())

    assert "[DONE]" in events
    # 应包含 conversation_id、status、token 三类事件
    import json as _json
    types = []
    for ev in events:
        if ev == "[DONE]":
            continue
        types.append(_json.loads(ev)["type"])
    assert "conversation_id" in types
    assert "status" in types
    assert "token" in types
