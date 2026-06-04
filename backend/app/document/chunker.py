"""
文档分块器 — 将文档拆分为适合检索的文本块
"""
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP


# 中文文本分割器，使用常见的中文分隔符
_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n\n",  # 段落
        "\n",    # 换行
        "。",    # 中文句号
        "！",    # 感叹号
        "？",    # 问号
        "；",    # 分号
        "，",    # 逗号
        ".",     # 英文句号
        " ",     # 空格
        "",      # 逐字分割（最后的回退方案）
    ],
    length_function=len,  # 使用字符长度
    is_separator_regex=False,
)


def chunk_documents(documents: List[Document]) -> List[Document]:
    """将文档列表切分为更小的文本块"""
    chunks = _text_splitter.split_documents(documents)
    # 为每个 chunk 的 metadata 添加 chunk_id，便于后续引用追踪
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = str(i)
    return chunks
