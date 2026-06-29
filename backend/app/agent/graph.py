"""
LangGraph ReAct Agent — 条件边路由，支持多轮对话与真正的工具派发
使用 langgraph.prebuilt.ToolNode 正确解析 LLM 的 tool_calls 并执行回边循环。
"""
from typing import List, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    MAX_AGENT_ITERATIONS,
)
from app.agent.state import AgentState
from app.agent.tools import search_documents, keyword_search, ask_clarify


# Agent 可用的工具列表
AGENT_TOOLS = [search_documents, keyword_search, ask_clarify]

# 系统提示词 — 定义 Agent 的行为规则
SYSTEM_PROMPT = """你是一个智能文档问答助手，帮助用户从文档库中查找和解答问题。

你可以使用以下工具:
- search_documents: 对文档库执行语义+关键词混合检索，适合查找概念、知识点的详细解释
- keyword_search: 精确关键词匹配，适合查找特定术语、名称、代码
- ask_clarify: 当用户问题模糊不清时，向用户追问以明确需求

工作规则:
1. 收到用户问题后，首先判断问题是否清晰明确。如果模糊不清，使用 ask_clarify 工具追问
2. 对明确的问题，使用 search_documents 进行文档检索
3. 如果首次检索结果不理想，尝试用 keyword_search 补充检索
4. 基于检索到的文档内容生成准确回答，必须注明信息来源（文档编号）
5. 回答使用中文，简洁准确，不要编造文档中没有的信息
6. 如果多次检索都未找到相关信息，诚实地告知用户文档库中没有相关内容
"""


def _build_llm() -> ChatOpenAI:
    """构建 DeepSeek LLM 实例（兼容 OpenAI SDK）"""
    return ChatOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        model=DEEPSEEK_MODEL,
        temperature=0.3,  # 文档问答保持低温度，减少幻觉
        max_tokens=2048,
    )


def history_to_messages(history: List[dict]) -> List[BaseMessage]:
    """将外部存储的对话历史（dict 形式）转换为 LangChain 消息对象"""
    msgs: List[BaseMessage] = []
    for m in history:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    return msgs


def agent_node(state: AgentState) -> dict:
    """
    Agent 决策节点 — 调用 LLM 决定是调用工具还是直接回答。
    保留原生 AIMessage（含 tool_calls 元信息），供 ToolNode 正确派发。
    """
    llm = _build_llm()
    llm_with_tools = llm.bind_tools(AGENT_TOOLS)

    messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(state.get("messages", []))

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response],  # 原生 AIMessage，保留 tool_calls
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def should_continue(state: AgentState) -> Literal["tools", "finalize"]:
    """
    条件边路由:
    - 超过最大迭代次数 → 强制结束
    - 最后一条 AIMessage 含 tool_calls → 执行工具
    - 否则 → 结束并生成最终回答
    """
    if state.get("iteration_count", 0) >= MAX_AGENT_ITERATIONS:
        return "finalize"

    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
    return "finalize"


def finalize_node(state: AgentState) -> dict:
    """
    生成最终回答:
    - 正常情况: ReAct 循环结束时最后一条 AIMessage 的内容即 LLM 基于工具结果给出的回答，直接提取
    - 兜底情况: 超过迭代上限仍存在未处理工具调用时，基于已收集的工具结果再调一次 LLM 生成回答
    同时收集所有 ToolMessage 内容作为 sources 返回。
    """
    messages = state.get("messages", [])

    # 收集工具返回的文档内容作为来源
    documents: List[str] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            documents.append(m.content if isinstance(m.content, str) else str(m.content))

    # 取最后一条 AIMessage 的文本作为最终回答
    last_ai_content = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            last_ai_content = (m.content or "")
            break

    if last_ai_content:
        final_answer = last_ai_content
    else:
        # 兜底：迭代上限耗尽且无直接回答，基于已有工具上下文强制生成
        llm = _build_llm()
        doc_context = "\n\n---\n\n".join(documents) if documents else "（未检索到相关文档）"
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"""基于以下文档内容回答用户问题。

## 检索到的文档内容
{doc_context}

## 用户问题
{state.get('query', '')}

## 要求
- 如果文档中有相关信息，请准确回答并注明引用来源
- 如果文档中没有相关信息，请明确告知用户
- 使用中文回答"""),
        ])
        final_answer = response.content if hasattr(response, "content") else str(response)

    return {"final_answer": final_answer, "documents": documents}


def build_agent_graph():
    """
    构建 LangGraph ReAct Agent 图
    流程: agent → (tools | finalize) → END，tools 执行后回到 agent 形成多轮推理循环
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(AGENT_TOOLS))
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "finalize": "finalize"},
    )

    # 工具执行后回到 agent 节点继续推理（真正的 ReAct 循环）
    workflow.add_edge("tools", "agent")
    workflow.add_edge("finalize", END)

    return workflow.compile()


# 编译后的图单例，避免每请求重复编译
_compiled_graph = None


def get_agent_graph():
    """获取编译好的 Agent 图单例"""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_agent_graph()
    return _compiled_graph
