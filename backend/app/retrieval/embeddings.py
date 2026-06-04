"""
嵌入模型封装 — 使用本地 BGE 中文模型，无需 API Key
"""
from typing import List

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE


# 全局单例，避免重复加载模型
_embedding_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """懒加载 SentenceTransformer 模型"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device=EMBEDDING_DEVICE,
        )
    return _embedding_model


class BGEZhEmbeddings(Embeddings):
    """BGE 中文 Embedding，兼容 LangChain Embeddings 接口"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        model = _get_model()
        # BGE 模型建议对文档添加 "为这个句子生成表示以用于检索相关文章：" 前缀
        # 参考: https://huggingface.co/BAAI/bge-small-zh-v1.5
        texts_with_prefix = [
            f"为这个句子生成表示以用于检索相关文章：{t}" for t in texts
        ]
        embeddings = model.encode(
            texts_with_prefix,
            normalize_embeddings=True,  # 归一化，便于余弦相似度计算
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        model = _get_model()
        # 查询文本使用不同的前缀
        query_with_prefix = f"为这个句子生成表示以用于检索相关文章：{text}"
        embedding = model.encode(
            query_with_prefix,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()
