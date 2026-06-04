"""
Reranker — 使用本地 Cross-Encoder 模型对检索结果进行重排序
替代简历中的 Cohere Rerank（免费、无需 API Key）
"""
from typing import List, Tuple

from sentence_transformers import CrossEncoder

from app.config import RERANKER_MODEL_NAME, RERANKER_DEVICE


# 全局单例
_reranker_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    """懒加载 CrossEncoder 模型"""
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(
            RERANKER_MODEL_NAME,
            device=RERANKER_DEVICE,
        )
    return _reranker_model


def rerank(
    query: str,
    candidates: List[str],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    对候选文档进行重排序
    参数:
        query: 用户查询
        candidates: 候选文档文本列表
        top_k: 返回数
    返回:
        [(文本, 相关性分数), ...] 按分数降序排列
    """
    if not candidates:
        return []

    model = _get_model()
    # 构造 (query, doc) 对
    pairs = [(query, doc) for doc in candidates]
    scores = model.predict(pairs)

    # 按分数降序排列，取 top_k
    # scores 可能是单个 float 或 list
    if not isinstance(scores, list):
        scores = scores.tolist()

    scored_docs = list(zip(candidates, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return scored_docs[:top_k]
