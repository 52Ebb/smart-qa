"""
BM25 关键词检索 — 基于词频的稀疏检索，与向量检索互补
"""
from typing import List, Tuple

import jieba
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """BM25 检索器，使用 jieba 中文分词"""

    def __init__(self):
        self._corpus: List[str] = []  # 原始文本语料
        self._tokenized_corpus: List[List[str]] = []  # 分词后的语料
        self._bm25: BM25Okapi | None = None

    def index(self, texts: List[str]) -> None:
        """构建 BM25 索引"""
        self._corpus = texts
        self._tokenized_corpus = [list(jieba.cut(text)) for text in texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """BM25 检索，返回 (文本内容, BM25 分数) 列表"""
        if self._bm25 is None:
            return []

        tokenized_query = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokenized_query)

        # 按分数降序排列
        scored_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        return [(self._corpus[i], scores[i]) for i in scored_indices if scores[i] > 0]

    @property
    def is_indexed(self) -> bool:
        return self._bm25 is not None
