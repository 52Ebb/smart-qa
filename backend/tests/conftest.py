"""
pytest 共享夹具 — 隔离的临时目录、模拟的本地模型、状态重置
"""
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# ============================================================
# 1. 在导入 app 之前，把 DATA_DIR / CHROMA_DIR 重定向到临时目录
#    注意: 必须在 app.retrieval.vector_store 被 import 之前完成，
#    否则 `from app.config import CHROMA_DIR` 会绑定到原始路径
# ============================================================
_TMP_ROOT = Path(tempfile.mkdtemp(prefix="sdocqa_test_"))
TMP_DATA_DIR = _TMP_ROOT / "data"
TMP_CHROMA_DIR = _TMP_ROOT / "chroma"
TMP_DATA_DIR.mkdir(parents=True, exist_ok=True)
TMP_CHROMA_DIR.mkdir(parents=True, exist_ok=True)

import app.config as _config  # noqa: E402
_config.DATA_DIR = TMP_DATA_DIR
_config.CHROMA_DIR = TMP_CHROMA_DIR


# ============================================================
# 2. 用轻量假模型替换 SentenceTransformer / CrossEncoder，
#    避免测试下载/加载真实 BGE 模型
# ============================================================
class FakeSentenceTransformer:
    """返回固定维度的伪嵌入向量，使 Chroma 索引/查询流程可跑通"""

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.zeros(16, dtype=np.float32)
        return np.zeros((len(texts), 16), dtype=np.float32)


class FakeCrossEncoder:
    """返回常数相关性分数，使 rerank 流程可跑通"""

    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs):
        return np.array([0.5] * len(pairs), dtype=np.float32)


import app.retrieval.embeddings as _emb_mod  # noqa: E402
import app.retrieval.reranker as _rer_mod  # noqa: E402
_emb_mod.SentenceTransformer = FakeSentenceTransformer
_rer_mod.CrossEncoder = FakeCrossEncoder


# ============================================================
# 3. 现在安全地导入 app
# ============================================================
import app.main as main_module  # noqa: E402
from app.main import app  # noqa: E402
from app.retrieval.vector_store import get_vector_store  # noqa: E402
from app.retrieval.bm25 import BM25Retriever  # noqa: E402


@pytest.fixture()
def client():
    """同步 TestClient，可处理 async 与 def 端点"""
    from starlette.testclient import TestClient
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_index_state():
    """每个测试前重置索引/对话/数据目录，保证用例隔离"""
    main_module._conversations.clear()
    try:
        main_module.vector_store.delete_collection()
    except Exception:
        pass
    main_module.vector_store = get_vector_store()
    main_module.bm25_retriever = BM25Retriever()
    main_module.init_tools(main_module.vector_store, main_module.bm25_retriever)
    for f in TMP_DATA_DIR.glob("*"):
        if f.is_file():
            f.unlink(missing_ok=True)
    # 重置编译后的 agent 图单例，确保每个用例的 LLM mock 重新生效
    import app.agent.graph as _graph_mod
    _graph_mod._compiled_graph = None
    yield


@pytest.fixture()
def fake_doc_texts():
    """返回用于模拟解析的文档文本"""
    return [
        "智能文档问答系统基于 RAG 架构，支持 PDF 和 Word 文档的上传与检索。",
        "系统使用 BGE 中文嵌入模型与 BM25 关键词检索进行混合检索，再经 CrossEncoder 重排序。",
    ]
