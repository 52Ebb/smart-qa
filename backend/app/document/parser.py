"""
文档解析器 — 支持 PDF 和 Word (.docx) 格式
"""
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document


def parse_pdf(file_path: str) -> List[Document]:
    """解析 PDF 文件，返回 LangChain Document 列表"""
    loader = PyPDFLoader(file_path)
    return loader.load()


def parse_docx(file_path: str) -> List[Document]:
    """解析 Word 文件，返回 LangChain Document 列表"""
    loader = Docx2txtLoader(file_path)
    return loader.load()


def parse_file(file_path: str) -> List[Document]:
    """根据文件扩展名自动选择合适的解析器"""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        # 注意: 仅支持 OOXML(.docx)；旧版二进制 .doc 格式无法由 docx2txt 解析
        return parse_docx(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}，仅支持 PDF 和 Word(.docx) 文档")
