"""
混合检索 — 融合向量检索和 BM25 关键词检索，再经过 Reranker 重排序
"""
from typing import List, Tuple

from chromadb import PersistentClient
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma

from app.config import (
    CHROMA_DIR,
    VECTOR_SEARCH_TOP_K,
    BM25_SEARCH_TOP_K,
    RERANK_TOP_K,
)
from app.retrieval.vector_store import get_vector_store
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.reranker import rerank


def _reciprocal_rank_fusion(
    vector_results: List[Tuple[str, float]],
    bm25_results: List[Tuple[str, float]],
    k: int = 60,
) -> List[str]:
    """
    RRF (Reciprocal Rank Fusion) 融合多种检索结果
    参数:
        k: RRF 平滑参数，默认 60（推荐值）
    返回:
        融合后的文本列表（去重）
    """
    rrf_scores = {}

    # 向量检索结果 — 相似度分数越高越好，排名越靠前
    for rank, (text, _) in enumerate(vector_results, start=1):
        rrf_scores[text] = rrf_scores.get(text, 0) + 1.0 / (k + rank)

    # BM25 检索结果 — BM25 分数越高越好
    for rank, (text, _) in enumerate(bm25_results, start=1):
        rrf_scores[text] = rrf_scores.get(text, 0) + 1.0 / (k + rank)

    # 按 RRF 分数降序排列
    sorted_texts = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [text for text, _ in sorted_texts]


def hybrid_search(
    query: str,
    vector_store: Chroma,
    bm25_retriever: BM25Retriever,
    vector_top_k: int = VECTOR_SEARCH_TOP_K,
    bm25_top_k: int = BM25_SEARCH_TOP_K,
    rerank_top_k: int = RERANK_TOP_K,
) -> List[Tuple[str, float]]:
    """
    混合检索流程:
    1. 向量检索 (语义相似度)
    2. BM25 关键词检索 (精确匹配)
    3. RRF 融合
    4. Cross-Encoder Rerank 重排序
    返回:
        [(文本内容, 相关性分数), ...]
    """
    # 步骤 1: 向量检索
    from app.retrieval.vector_store import search_by_vector
    vector_results = search_by_vector(vector_store, query, top_k=vector_top_k)

    # 步骤 2: BM25 检索
    bm25_results = bm25_retriever.search(query, top_k=bm25_top_k)

    # 步骤 3: RRF 融合
    fused_texts = _reciprocal_rank_fusion(vector_results, bm25_results)

    # 步骤 4: Rerank 重排序
    if fused_texts:
        reranked = rerank(query, fused_texts, top_k=rerank_top_k)
        return reranked

    # 如果融合结果为空，直接用向量检索结果
    return vector_results[:rerank_top_k]
