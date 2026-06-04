"""
Agent 状态定义 — LangGraph 状态管理
"""
from typing import Annotated, List, TypedDict
import operator


class AgentState(TypedDict):
    """ReAct Agent 的状态数据结构"""
    # 对话历史消息，operator.add 实现追加而非覆盖
    messages: Annotated[List[dict], operator.add]
    # 当前用户问题
    query: str
    # 检索到的相关文档
    documents: List[str]
    # 是否需要进一步澄清问题
    need_clarify: bool
    # 最终回答（SSE 流式输出时使用）
    final_answer: str
    # Agent 当前迭代次数
    iteration_count: int
