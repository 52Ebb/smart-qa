"""
全局配置管理
"""
import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 文档上传目录
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# ChromaDB 持久化目录
CHROMA_DIR = BASE_DIR / "chroma_db"

# DeepSeek API 配置（兼容 OpenAI SDK）
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"  # DeepSeek 对话模型

# 本地 Embedding 模型名称（使用 BGE 中文小模型，无需 API Key）
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DEVICE = "cpu"  # 可改为 "cuda" 如果有 GPU

# 本地 Reranker 模型
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
RERANKER_DEVICE = "cpu"

# 文档分块参数
CHUNK_SIZE = 500  # 每个文本块的最大字符数
CHUNK_OVERLAP = 50  # 相邻块的重叠字符数

# 检索参数
VECTOR_SEARCH_TOP_K = 10  # 向量检索返回数
BM25_SEARCH_TOP_K = 10  # BM25 检索返回数
RERANK_TOP_K = 5  # 重排序后返回数

# Agent 参数
MAX_AGENT_ITERATIONS = 5  # Agent 最大迭代次数
