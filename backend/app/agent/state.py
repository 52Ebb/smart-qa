"""
Agent 状态定义 — LangGraph 状态管理
"""
import operator
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """ReAct Agent 的状态数据结构"""
    # 对话历史 + 当前消息，使用 LangChain 原生消息对象（含 tool_calls 等元信息）
    # operator.add 实现追加而非覆盖，工具返回的 ToolMessage 也会被追加进来
    messages: Annotated[List[BaseMessage], operator.add]
    # 当前用户问题（供 finalize 节点构造提示词）
    query: str
    # 检索到的相关文档（作为 sources 返回）
    documents: List[str]
    # 是否需要进一步澄清问题
    need_clarify: bool
    # 最终回答
    final_answer: str
    # Agent 当前迭代次数
    iteration_count: int
