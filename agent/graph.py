from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from agent.nodes import rewrite_node, retrieve_node, generate_node, grade_docs_node


def _decide_to_generate(state: AgentState) -> str:
    """grade_docs_node 이후 분기: 재작성 필요 여부에 따라 라우팅"""
    if state.get("should_rewrite"):
        return "rewrite"
    return "generate"


def build_graph():
    """LangGraph StateGraph를 조립하여 컴파일된 그래프를 반환"""
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_docs", grade_docs_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_docs")
    graph.add_conditional_edges(
        "grade_docs",
        _decide_to_generate,
        {
            "rewrite": "rewrite",
            "generate": "generate",
        },
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()
