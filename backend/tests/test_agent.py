"""
Agent 图逻辑测试 — 工具调用派发、ReAct 回边循环、迭代上限、历史转换
"""
from collections import deque
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import app.agent.graph as graph_mod
from app.agent.graph import get_agent_graph, history_to_messages
from app.config import MAX_AGENT_ITERATIONS


def test_history_to_messages():
    """dict 历史应正确转为 LangChain 消息对象"""
    history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，有什么可以帮你？"},
    ]
    msgs = history_to_messages(history)
    assert len(msgs) == 2
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "你好"
    assert isinstance(msgs[1], AIMessage)
    assert msgs[1].content == "你好，有什么可以帮你？"


def _scripted_llm(responses):
    """返回一个共享的脚本化 LLM，按顺序弹出响应"""
    queue = deque(responses)

    class _Shared:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            return queue.popleft()

    return _Shared(), queue


def test_agent_dispatches_tool_calls_and_loops():
    """
    H4/H6 回归: Agent 应正确解析 tool_calls，执行工具，回边再推理，最终给出回答。
    """
    responses = [
        AIMessage(
            content="",
            tool_calls=[{"name": "search_documents", "args": {"query": "RAG"}, "id": "c1"}],
        ),
        AIMessage(content="根据文档，RAG 是检索增强生成。"),
    ]
    shared, queue = _scripted_llm(responses)

    with patch.object(graph_mod, "ChatOpenAI", lambda *a, **kw: shared), \
         patch("app.agent.tools.hybrid_search", return_value=[("RAG 是检索增强生成。", 0.9)]):
        agent = get_agent_graph()
        result = agent.invoke({
            "messages": [HumanMessage(content="什么是 RAG？")],
            "query": "什么是 RAG？",
            "documents": [],
            "need_clarify": False,
            "final_answer": "",
            "iteration_count": 0,
        })

    assert "RAG" in result["final_answer"]
    # documents 应收集到工具返回内容（作为 sources）
    assert len(result["documents"]) >= 1
    # 队列应被消费完（两次 LLM 调用：一次工具决策、一次最终回答）
    assert len(queue) == 0


def test_agent_iteration_limit_terminates():
    """
    H5 回归: 当 LLM 持续请求工具调用时，迭代上限应强制终止并给出兜底回答。
    旧代码的 tools→finalize 直边使循环无法多轮，迭代上限形同虚设；修复后应能正确触发。
    """
    # 前 MAX_AGENT_ITERATIONS 次：始终返回工具调用（无 content）
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "search_documents", "args": {"query": "x"}, "id": "c1"}],
    )
    responses = [tool_call_msg] * MAX_AGENT_ITERATIONS
    # 兜底回答（finalize 在超过上限时调用一次 LLM 生成回答）
    responses.append(AIMessage(content="已达迭代上限，基于现有信息作答。"))
    shared, queue = _scripted_llm(responses)

    with patch.object(graph_mod, "ChatOpenAI", lambda *a, **kw: shared), \
         patch("app.agent.tools.hybrid_search", return_value=[("文档片段", 0.5)]):
        agent = get_agent_graph()
        result = agent.invoke({
            "messages": [HumanMessage(content="问题")],
            "query": "问题",
            "documents": [],
            "need_clarify": False,
            "final_answer": "",
            "iteration_count": 0,
        })

    # 应正常终止并产出非空回答
    assert result["final_answer"]
    assert "迭代上限" in result["final_answer"]
    # 队列应被消费完
    assert len(queue) == 0


def test_agent_direct_answer_without_tools():
    """当 LLM 直接给出回答（无 tool_calls）时，应跳过工具直接结束"""
    shared, queue = _scripted_llm([AIMessage(content="我直接回答。")])
    with patch.object(graph_mod, "ChatOpenAI", lambda *a, **kw: shared), \
         patch("app.agent.tools.hybrid_search", return_value=[]):
        agent = get_agent_graph()
        result = agent.invoke({
            "messages": [HumanMessage(content="你好")],
            "query": "你好",
            "documents": [],
            "need_clarify": False,
            "final_answer": "",
            "iteration_count": 0,
        })
    assert result["final_answer"] == "我直接回答。"
    assert len(queue) == 0
