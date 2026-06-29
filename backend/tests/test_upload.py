"""
上传接口测试 — 路径穿越防护、chunk_id 唯一性、文件类型/大小校验
"""
from unittest.mock import patch

from langchain_core.documents import Document

import app.main as main_module


def _make_docs():
    return [
        Document(page_content="智能文档问答系统基于 RAG 架构，支持 PDF 和 Word 文档的上传与检索。"),
        Document(page_content="系统使用 BGE 中文嵌入模型与 BM25 关键词检索进行混合检索。"),
    ]


def test_unsupported_file_type(client):
    """上传不支持的格式应返回 400"""
    res = client.post(
        "/api/upload",
        files={"file": ("test.txt", b"hello", "text/plain")},
    )
    assert res.status_code == 400
    assert "不支持" in res.json()["detail"]


def test_path_traversal_neutralized(client):
    """
    路径穿越文件名应被清洗：原始代码 DATA_DIR/{file_id}_{filename} 会把 ../../ 解析出 DATA_DIR，
    修复后取 Path(filename).name 剥离路径成分，文件安全落盘到 DATA_DIR 内。
    """
    with patch.object(main_module, "parse_file", return_value=_make_docs()):
        res = client.post(
            "/api/upload",
            files={"file": ("../../evil.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
    assert res.status_code == 200
    data_dir = main_module.DATA_DIR
    tmp_root = data_dir.parent
    # 文件应落在 DATA_DIR 内，且不存在于父目录（穿越被阻断）
    assert not (tmp_root / "evil.pdf").exists()
    saved = list(data_dir.glob("*_evil.pdf"))
    assert len(saved) == 1, f"应存在一个清洗后的文件，实际: {list(data_dir.glob('*'))}"


def test_upload_unique_chunk_ids_no_overwrite(client):
    """
    C1 回归: 多次上传的 chunk_id 必须全局唯一，第二次上传不得覆盖第一次的向量。
    断言: 两次上传后索引总数 == 两次 chunk 数之和（旧代码会停留在一份数量）。
    """
    with patch.object(main_module, "parse_file", return_value=_make_docs()):
        r1 = client.post(
            "/api/upload",
            files={"file": ("a.pdf", b"%PDF-1.4 a", "application/pdf")},
        )
        r2 = client.post(
            "/api/upload",
            files={"file": ("b.pdf", b"%PDF-1.4 b", "application/pdf")},
        )
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["chunk_count"] == 2
    assert r2.json()["chunk_count"] == 2

    status = client.get("/api/index/status").json()
    # 两次上传共 4 个 chunk，证明没有因 ID 冲突而覆盖
    assert status["document_count"] == 4, f"期望 4，实际 {status['document_count']}"
    assert status["indexed"] is True
    assert status["bm25_indexed"] is True


def test_oversize_file_rejected(client, monkeypatch):
    """超过大小限制应返回 413"""
    monkeypatch.setattr(main_module, "MAX_UPLOAD_SIZE", 100)
    payload = b"x" * 200
    res = client.post(
        "/api/upload",
        files={"file": ("big.pdf", payload, "application/pdf")},
    )
    assert res.status_code == 413
    assert "过大" in res.json()["detail"]
    # 文件应被清理
    assert not list(main_module.DATA_DIR.glob("big.pdf"))


def test_doc_extension_rejected(client):
    """旧版 .doc 二进制格式不再支持，应返回 400"""
    res = client.post(
        "/api/upload",
        files={"file": ("legacy.doc", b"\xd0\xcf\x11\xe0", "application/msword")},
    )
    assert res.status_code == 400
