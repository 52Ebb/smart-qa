"""
索引状态与清空接口测试
"""
from unittest.mock import patch

from langchain_core.documents import Document

import app.main as main_module


def _make_docs():
    return [Document(page_content="测试文档内容，用于索引状态检查。")]


def test_index_status_empty(client):
    """初始状态应为未索引"""
    res = client.get("/api/index/status")
    assert res.status_code == 200
    data = res.json()
    assert data["indexed"] is False
    assert data["document_count"] == 0
    assert data["bm25_indexed"] is False


def test_index_status_after_upload(client):
    """上传后状态应反映已索引"""
    with patch.object(main_module, "parse_file", return_value=_make_docs()):
        client.post(
            "/api/upload",
            files={"file": ("a.pdf", b"%PDF-1.4 a", "application/pdf")},
        )
    res = client.get("/api/index/status").json()
    assert res["indexed"] is True
    assert res["document_count"] >= 1
    assert res["bm25_indexed"] is True


def test_clear_index(client):
    """清空索引后状态应归零，且对话历史被清空"""
    with patch.object(main_module, "parse_file", return_value=_make_docs()):
        client.post(
            "/api/upload",
            files={"file": ("a.pdf", b"%PDF-1.4 a", "application/pdf")},
        )
    main_module._conversations["dummy"] = [{"role": "user", "content": "hi"}]

    res = client.delete("/api/index")
    assert res.status_code == 200

    status = client.get("/api/index/status").json()
    assert status["indexed"] is False
    assert status["document_count"] == 0
    assert status["bm25_indexed"] is False
    # 对话历史应被一并清空
    assert len(main_module._conversations) == 0
