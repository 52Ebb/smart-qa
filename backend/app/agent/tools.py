"""
Agent 工具定义 — 封装文档查询、关键词搜索、追问澄清 3 个 Tool
"""
from typing import List

from langchain_core.tools import tool

from app.retrieval.hybrid import hybrid_search
from app.retrieval.vector_store import Chroma
from app.retrieval.bm25 import BM25Retriever


# 全局检索器实例，由 Agent 启动时注入
_vector_store: Chroma | None = None
_bm25_retriever: BM25Retriever | None = None


def init_tools(vector_store: Chroma, bm25_retriever: BM25Retriever) -> None:
    """注入检索器实例到工具模块"""
    global _vector_store, _bm25_retriever
    _vector_store = vector_store
    _bm25_retriever = bm25_retriever


@tool
def search_documents(query: str) -> str:
    """
    在已索引的文档库中执行语义+关键词混合检索。
    当用户询问关于文档的具体内容、需要查找某个知识点或概念时使用此工具。
    参数:
        query: 搜索查询，建议使用具体的关键词或问题描述
    返回:
        检索到的相关文档片段（最多 5 条），每条包含内容摘要和来源页码
    """
    if _vector_store is None or _bm25_retriever is None:
        return "错误: 文档检索系统尚未初始化，请先上传文档。"

    results = hybrid_search(
        query=query,
        vector_store=_vector_store,
        bm25_retriever=_bm25_retriever,
    )

    if not results:
        return "未找到与查询相关的文档内容。请尝试更换关键词或扩大搜索范围。"

    # 格式化输出
    formatted = []
    for i, (text, score) in enumerate(results, 1):
        # 截断过长文本，保留前 300 字
        preview = text[:300] + ("..." if len(text) > 300 else "")
        formatted.append(f"[文档{i}] (相关性: {score:.3f})\n{preview}")

    return "\n\n".join(formatted)


@tool
def keyword_search(keywords: str) -> str:
    """
    通过关键词在文档中进行精确匹配搜索。
    适用于用户明确提到某个术语、名称或代码片段时使用。
    参数:
        keywords: 搜索关键词，多个关键词用空格分隔
    返回:
        包含关键词的文档片段
    """
    if _bm25_retriever is None:
        return "错误: 文档检索系统尚未初始化，请先上传文档。"

    results = _bm25_retriever.search(keywords, top_k=5)

    if not results:
        return f"未找到包含 '{keywords}' 的文档内容。"

    formatted = []
    for i, (text, score) in enumerate(results, 1):
        preview = text[:300] + ("..." if len(text) > 300 else "")
        formatted.append(f"[关键词匹配{i}] (BM25: {score:.3f})\n{preview}")

    return "\n\n".join(formatted)


@tool
def ask_clarify(question: str) -> str:
    """
    当用户问题模糊不清或歧义时，向用户发起追问以澄清意图。
    使用时机：用户问题过于宽泛、缺少上下文、或存在多种解释可能时。
    参数:
        question: 向用户提出的澄清问题
    返回:
        用户的澄清意图（模拟，实际由 Agent 处理）
    """
    return f"需要澄清: {question}"
