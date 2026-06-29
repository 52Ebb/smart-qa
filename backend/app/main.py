"""
FastAPI 主入口 — 文档上传、问答 SSE 流式输出
"""
import json
import uuid
import threading
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import (
    DATA_DIR,
    MAX_UPLOAD_SIZE,
    SUPPORTED_FILE_TYPES,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
)
from app.document.parser import parse_file
from app.document.chunker import chunk_documents
from app.retrieval.vector_store import get_vector_store, add_documents_to_store
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import hybrid_search
from app.agent.graph import get_agent_graph, history_to_messages
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

# 索引变更互斥锁：保护上传/清空对共享索引的并发修改，避免在途请求读到半成品状态
_index_lock = threading.Lock()


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
def upload_document(file: UploadFile = File(...)):
    """
    上传文档接口 — 支持 PDF 和 Word (.docx)
    1. 保存文件（流式落盘 + 大小限制 + 路径穿越防护）
    2. 解析文档内容
    3. 分块（chunk_id 含 file_id 前缀，全局唯一）
    4. 建立向量索引和 BM25 索引（加锁，避免与清空操作竞态）
    """
    # 验证文件类型
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext or '未知'}，仅支持 PDF 和 Word(.docx) 文档",
        )

    # 清洗文件名，防止路径穿越（取纯文件名，剥离任何路径成分）
    safe_name = Path(file.filename or "").name
    if not safe_name:
        safe_name = f"upload{ext}"

    file_id = uuid.uuid4().hex[:8]
    saved_path = DATA_DIR / f"{file_id}_{safe_name}"

    # 二次校验：解析后路径必须仍在 DATA_DIR 内
    try:
        saved_path.resolve().relative_to(DATA_DIR.resolve())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="非法的文件路径",
        )

    # 流式落盘，并累计大小，超限立即中止
    written = 0
    try:
        with open(saved_path, "wb") as f:
            while True:
                chunk = file.file.read(1024 * 1024)  # 1MB 缓冲
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_SIZE:
                    f.close()
                    saved_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件过大，超过 {MAX_UPLOAD_SIZE // (1024 * 1024)}MB 限制",
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        if saved_path.exists():
            saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")

    try:
        # 解析文档（阻塞 IO/CPU，本端点为 def，FastAPI 自动在线程池运行）
        documents = parse_file(str(saved_path))

        # 分块 — 传入 file_id 生成全局唯一 chunk_id
        chunks = chunk_documents(documents, file_id=file_id)

        # 加锁：向量写入 + BM25 重建需作为一个原子操作，避免与 clear_index 交错
        with _index_lock:
            # 写入向量存储
            add_documents_to_store(vector_store, chunks)

            # 重建 BM25 索引 — 从 ChromaDB 读取全量文本（已含本次新增），不再额外 extend
            all_texts = _get_all_indexed_texts()
            if all_texts:
                bm25_retriever.index(all_texts)
            # 空语料保护：all_texts 为空时跳过，避免 BM25Okapi 在空集上崩溃

        return {
            "message": f"文档上传成功，共处理 {len(chunks)} 个文本块",
            "file_id": file_id,
            "chunk_count": len(chunks),
            "file_name": safe_name,
        }

    except Exception as e:
        # 清理临时文件
        if saved_path.exists():
            saved_path.unlink(missing_ok=True)
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
def chat(request: ChatRequest):
    """
    问答接口 — 非流式，完整回答后返回
    走 LangGraph ReAct Agent，正确派发工具调用并支持多轮迭代。
    本端点为 def：agent.invoke() 为阻塞调用，由 FastAPI 线程池承载，不阻塞事件循环。
    """
    if not bm25_retriever.is_indexed:
        raise HTTPException(
            status_code=400,
            detail="文档索引为空，请先上传文档",
        )

    agent = get_agent_graph()

    conversation_id = request.conversation_id or uuid.uuid4().hex[:8]
    history = _conversations.get(conversation_id, [])

    # 将历史转为 LangChain 消息对象并追加当前 query
    from langchain_core.messages import HumanMessage
    messages = history_to_messages(history)
    messages.append(HumanMessage(content=request.query))

    result = agent.invoke({
        "messages": messages,
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
    逐 token 推送回答内容，首个事件回传 conversation_id 以支持多轮对话。
    """
    if not bm25_retriever.is_indexed:
        raise HTTPException(
            status_code=400,
            detail="文档索引为空，请先上传文档",
        )

    if not DEEPSEEK_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="未配置 DEEPSEEK_API_KEY 环境变量",
        )

    conversation_id = request.conversation_id or uuid.uuid4().hex[:8]

    async def generate():
        """生成 SSE 事件流"""
        import asyncio
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        from app.agent.graph import SYSTEM_PROMPT

        # 第一个事件：回传 conversation_id，供前端绑定后续多轮对话
        yield f"data: {_sse_event('conversation_id', conversation_id)}\n\n"

        # 第一步: 执行检索（阻塞，丢到线程池避免阻塞事件循环）
        results = await asyncio.to_thread(
            hybrid_search,
            query=request.query,
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
        )

        # 构建文档上下文
        doc_context = "\n\n---\n\n".join(
            [text[:500] for text, _ in results]
        ) if results else ""

        yield f"data: {_sse_event('status', '检索完成，正在生成回答...')}\n\n"

        # 第二步: 调用 DeepSeek 流式生成（异步迭代，不阻塞事件循环）
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

        full_answer_parts: List[str] = []
        try:
            async for chunk in llm.astream(messages):
                if chunk.content:
                    full_answer_parts.append(chunk.content)
                    yield f"data: {_sse_event('token', chunk.content)}\n\n"

            # 保存对话历史（与 /api/chat 保持一致，支持多轮）
            history = _conversations.get(conversation_id, [])
            history.append({"role": "user", "content": request.query})
            history.append({"role": "assistant", "content": "".join(full_answer_parts)})
            _conversations[conversation_id] = history

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
def clear_index():
    """清空所有文档索引（加锁，避免与上传/问答竞态）"""
    global vector_store, bm25_retriever
    with _index_lock:
        try:
            # 清空 ChromaDB collection
            vector_store.delete_collection()
            # 重新创建 collection
            vector_store = get_vector_store()
            bm25_retriever = BM25Retriever()
            init_tools(vector_store, bm25_retriever)
            # 清空对话历史（引用的文档已不存在）
            _conversations.clear()
            # 清理上传的原始文档文件
            for f in DATA_DIR.glob("*"):
                if f.is_file():
                    f.unlink(missing_ok=True)
            return {"message": "索引已清空"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"清空索引失败: {str(e)}")


# ==================== 辅助函数 ====================

def _sse_event(event_type: str, data: str) -> str:
    """构造 SSE 事件消息"""
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
