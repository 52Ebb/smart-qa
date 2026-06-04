"""
LangGraph ReAct Agent — 条件边路由，支持多轮对话
"""
import json
from typing import Literal

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
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


def agent_node(state: AgentState) -> dict:
    """
    Agent 决策节点 — 调用 LLM 决定是调用工具还是直接回答
    """
    llm = _build_llm()
    llm_with_tools = llm.bind_tools(AGENT_TOOLS)

    # 构建消息列表
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # 添加历史消息
    for msg in state.get("messages", []):
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    # 添加当前查询
    messages.append(HumanMessage(content=state["query"]))

    # 调用 LLM
    response = llm_with_tools.invoke(messages)

    # 检查是否有工具调用
    has_tool_calls = hasattr(response, "tool_calls") and response.tool_calls

    return {
        "messages": [
            {"role": "assistant", "content": response.content or ""},
        ],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def should_continue(state: AgentState) -> Literal["tools", "finalize"]:
    """
    条件边路由:
    - 如果最后一条消息包含 tool_calls → 执行工具
    - 如果超过最大迭代次数 → 强制结束
    - 否则 → 直接生成最终回答
    """
    if state.get("iteration_count", 0) >= MAX_AGENT_ITERATIONS:
        return "finalize"

    # 检查最后一条 assistant 消息是否有 tool_calls
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        # 如果 LLM 请求了工具调用，且有工具调用的内容
        # LangGraph 的 tool_calls 信息在 AIMessage 的 tool_calls 属性
        # 这里检查 response 中是否有 tool_calls
        # 简化处理: 如果消息内容为空，可能有 tool_calls
        if not last_msg.get("content"):
            return "tools"

    return "finalize"


def tools_node(state: AgentState) -> dict:
    """
    工具执行节点 — 执行 Agent 请求的工具调用
    由于 LangGraph ToolNode 需要原生消息格式，这里使用简化版工具路由
    """
    messages = state.get("messages", [])
    if not messages:
        return {"messages": []}

    last_msg = messages[-1]
    query = state.get("query", "")

    # 根据消息内容判断需要调用哪个工具
    # 实际项目中应解析 tool_calls，这里做简化处理
    # 尝试用 search_documents 检索
    result = search_documents.invoke({"query": query})

    return {
        "messages": [
            {"role": "tool", "content": str(result)},
        ],
        "documents": [str(result)],
        "query": query,
    }


def finalize_node(state: AgentState) -> dict:
    """
    生成最终回答 — 基于检索到的文档上下文
    """
    llm = _build_llm()

    # 构建最终回答的上下文
    documents = state.get("documents", [])
    doc_context = "\n\n---\n\n".join(documents) if documents else "（未检索到相关文档）"

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""基于以下文档内容回答用户问题。

## 检索到的文档内容
{doc_context}

## 用户问题
{state['query']}

## 要求
- 如果文档中有相关信息，请准确回答并注明引用来源
- 如果文档中没有相关信息，请明确告知用户
- 如果文档信息不完整，可以说明局限性
- 使用中文回答"""),
    ]

    response = llm.invoke(messages)
    content = response.content if hasattr(response, "content") else str(response)

    return {
        "final_answer": content,
        "messages": [
            {"role": "assistant", "content": content},
        ],
    }


def build_agent_graph() -> StateGraph:
    """
    构建 LangGraph ReAct Agent 图
    流程: agent → (tools | finalize) → END
    """
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("finalize", finalize_node)

    # 设置入口点
    workflow.set_entry_point("agent")

    # 添加条件边: agent → tools 或 finalize
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "finalize": "finalize",
        },
    )

    # tools 节点执行后 → finalize（也可以回到 agent 继续思考）
    workflow.add_edge("tools", "finalize")

    # finalize → END
    workflow.add_edge("finalize", END)

    return workflow.compile()
