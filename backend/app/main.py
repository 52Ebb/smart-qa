"""
FastAPI 主入口 — 文档上传、问答 SSE 流式输出
"""
import os
import uuid
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import DATA_DIR
from app.document.parser import parse_file
from app.document.chunker import chunk_documents
from app.retrieval.vector_store import get_vector_store, add_documents_to_store
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import hybrid_search
from app.agent.graph import build_agent_graph
from app.agent.tools import init_tools

app = FastAPI(
    title="智能文档问答系统",
    description="基于 LangGraph 的 RAG 文档问答系统，支持 PDF/Word 文档",
    version="1.0.0",
)

# 允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局检索器实例
vector_store = get_vector_store()
bm25_retriever = BM25Retriever()

# 注入到 Agent 工具模块
init_tools(vector_store, bm25_retriever)


# ==================== 数据模型 ====================

class ChatRequest(BaseModel):
    """问答请求"""
    query: str
    conversation_id: Optional[str] = None  # 会话 ID，用于多轮对话


class ChatResponse(BaseModel):
    """问答响应"""
    answer: str
    sources: List[str] = []
    conversation_id: str


class IndexStatus(BaseModel):
    """索引状态"""
    indexed: bool
    document_count: int
    bm25_indexed: bool


# ==================== 会话存储（简易内存版） ====================
# 生产环境应使用 Redis 等存储
_conversations: dict = {}


# ==================== API 路由 ====================

@app.get("/")
async def root():
    """健康检查"""
    return {"status": "ok", "service": "智能文档问答系统"}


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档接口 — 支持 PDF 和 Word (.docx)
    1. 保存文件
    2. 解析文档内容
    3. 分块
    4. 建立向量索引和 BM25 索引
    """
    # 验证文件类型
    ext = Path(file.filename or "").suffix.lower()
    if ext not in (".pdf", ".docx", ".doc"):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext}，仅支持 PDF 和 Word 文档",
        )

    # 保存上传文件
    file_id = uuid.uuid4().hex[:8]
    saved_path = DATA_DIR / f"{file_id}_{file.filename}"
    with open(saved_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        # 解析文档
        documents = parse_file(str(saved_path))

        # 分块
        chunks = chunk_documents(documents)

        # 写入向量存储
        add_documents_to_store(vector_store, chunks)

        # 更新 BM25 索引 — 收集所有已索引文本重建索引
        # 注意: ChromaDB 不直接提供全量文本获取，这里用简化方案
        # 将当前 chunks 的文本追加到 BM25 索引
        chunk_texts = [chunk.page_content for chunk in chunks]
        if bm25_retriever.is_indexed:
            # 如果已索引，需要重建（BM25 不支持增量更新）
            # 从 ChromaDB 获取所有文本重新索引
            all_texts = _get_all_indexed_texts()
            all_texts.extend(chunk_texts)
            bm25_retriever.index(all_texts)
        else:
            bm25_retriever.index(chunk_texts)

        return {
            "message": f"文档上传成功，共处理 {len(chunks)} 个文本块",
            "file_id": file_id,
            "chunk_count": len(chunks),
            "file_name": file.filename,
        }

    except Exception as e:
        # 清理临时文件
        if saved_path.exists():
            saved_path.unlink()
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")


@app.get("/api/index/status")
async def index_status() -> IndexStatus:
    """查询索引状态"""
    try:
        collection = vector_store._collection
        doc_count = collection.count()
    except Exception:
        doc_count = 0

    return IndexStatus(
        indexed=doc_count > 0,
        document_count=doc_count,
        bm25_indexed=bm25_retriever.is_indexed,
    )


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    问答接口 — 非流式，完整回答后返回
    """
    if not bm25_retriever.is_indexed:
        raise HTTPException(
            status_code=400,
            detail="文档索引为空，请先上传文档",
        )

    # 构建 Agent 并执行
    agent = build_agent_graph()

    conversation_id = request.conversation_id or uuid.uuid4().hex[:8]
    history = _conversations.get(conversation_id, [])

    result = agent.invoke({
        "messages": history,
        "query": request.query,
        "documents": [],
        "need_clarify": False,
        "final_answer": "",
        "iteration_count": 0,
    })

    answer = result.get("final_answer", "")

    # 保存对话历史
    history.append({"role": "user", "content": request.query})
    history.append({"role": "assistant", "content": answer})
    _conversations[conversation_id] = history

    return ChatResponse(
        answer=answer,
        sources=result.get("documents", []),
        conversation_id=conversation_id,
    )


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    问答接口 — SSE 流式输出
    逐 token 推送回答内容
    """
    if not bm25_retriever.is_indexed:
        raise HTTPException(
            status_code=400,
            detail="文档索引为空，请先上传文档",
        )

    async def generate():
        """生成 SSE 事件流"""
        import asyncio

        # 第一步: 执行检索
        results = hybrid_search(
            query=request.query,
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
        )

        # 构建文档上下文
        doc_context = "\n\n---\n\n".join(
            [text[:500] for text, _ in results]
        ) if results else ""

        yield f"data: {_sse_event('status', '检索完成，正在生成回答...')}\n\n"
        await asyncio.sleep(0.1)

        # 第二步: 调用 DeepSeek 流式生成
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage

        from app.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
        from app.agent.graph import SYSTEM_PROMPT

        if not DEEPSEEK_API_KEY:
            yield f"data: {_sse_event('error', '未配置 DEEPSEEK_API_KEY 环境变量')}\n\n"
            yield "data: [DONE]\n\n"
            return

        llm = ChatOpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            model=DEEPSEEK_MODEL,
            temperature=0.3,
            max_tokens=2048,
            streaming=True,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"""基于以下文档内容回答用户问题。

## 检索到的文档内容
{doc_context if doc_context else '（未检索到相关文档）'}

## 用户问题
{request.query}

## 要求
- 如果文档中有相关信息，请准确回答并注明引用来源
- 如果文档中没有相关信息，请明确告知用户
- 使用中文回答"""),
        ]

        try:
            for chunk in llm.stream(messages):
                if chunk.content:
                    yield f"data: {_sse_event('token', chunk.content)}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {_sse_event('error', str(e))}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        },
    )


@app.delete("/api/index")
async def clear_index():
    """清空所有文档索引"""
    global vector_store, bm25_retriever
    try:
        # 清空 ChromaDB collection
        vector_store.delete_collection()
        # 重新创建 collection
        vector_store = get_vector_store()
        bm25_retriever = BM25Retriever()
        init_tools(vector_store, bm25_retriever)
        return {"message": "索引已清空"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清空索引失败: {str(e)}")


# ==================== 辅助函数 ====================

def _sse_event(event_type: str, data: str) -> str:
    """构造 SSE 事件消息"""
    import json
    return json.dumps({"type": event_type, "content": data}, ensure_ascii=False)


def _get_all_indexed_texts() -> List[str]:
    """从 ChromaDB 获取所有已索引的文本（用于重建 BM25）"""
    try:
        collection = vector_store._collection
        results = collection.get()
        if results and "documents" in results:
            return results["documents"] or []
    except Exception:
        pass
    return []


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
