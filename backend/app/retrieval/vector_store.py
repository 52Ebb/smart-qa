"""
向量存储 — 基于 ChromaDB 实现文档索引和相似度检索
"""
from typing import List, Tuple

from chromadb import PersistentClient
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import CHROMA_DIR, VECTOR_SEARCH_TOP_K
from app.retrieval.embeddings import BGEZhEmbeddings


def get_vector_store(collection_name: str = "documents") -> Chroma:
    """获取 ChromaDB 向量存储实例"""
    embeddings = BGEZhEmbeddings()
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        client=PersistentClient(
            path=str(CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        ),
    )
    return vector_store


def add_documents_to_store(
    vector_store: Chroma,
    chunks: List[Document],
    batch_size: int = 32,
) -> None:
    """批量添加文档块到向量存储"""
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    # chunk_id 由 chunker 保证全局唯一（含 file_id 前缀），直接用作 ChromaDB 主键
    ids = [f"chunk_{chunk.metadata['chunk_id']}" for chunk in chunks]

    # 分批添加，避免一次性添加过多导致内存问题
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        vector_store.add_texts(
            texts=batch_texts,
            metadatas=batch_metadatas,
            ids=batch_ids,
        )


def search_by_vector(
    vector_store: Chroma,
    query: str,
    top_k: int = VECTOR_SEARCH_TOP_K,
) -> List[Tuple[str, float]]:
    """向量相似度检索，返回 (文本内容, 相似度分数) 列表"""
    results = vector_store.similarity_search_with_score(query, k=top_k)
    return [(doc.page_content, score) for doc, score in results]
