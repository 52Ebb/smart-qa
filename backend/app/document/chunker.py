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


def chunk_documents(documents: List[Document], file_id: str = "") -> List[Document]:
    """
    将文档列表切分为更小的文本块。
    参数:
        documents: 待分块的文档列表
        file_id: 上传文件标识，用于生成全局唯一的 chunk_id，避免不同文档间的 ID 冲突
    """
    chunks = _text_splitter.split_documents(documents)
    # 为每个 chunk 的 metadata 添加全局唯一的 chunk_id
    # 格式: {file_id}_{序号}，确保多次上传不会覆盖既有向量
    prefix = f"{file_id}_" if file_id else ""
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{prefix}{i}"
    return chunks
