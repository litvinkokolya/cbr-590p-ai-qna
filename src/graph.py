from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.nodes import (
    ask_human,
    check_completeness,
    generate_answer,
    parse_request,
    retrieve,
    route_after_check,
)
from src.state import State


def build_graph():
    graph = StateGraph(State)

    graph.add_node("parse_request", parse_request)
    graph.add_node("check_completeness", check_completeness)
    graph.add_node("ask_human", ask_human)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate_answer", generate_answer)

    graph.add_edge(START, "parse_request")
    graph.add_edge("parse_request", "check_completeness")
    graph.add_conditional_edges(
        "check_completeness",
        route_after_check,
        {"retrieve": "retrieve", "ask_human": "ask_human"},
    )
    graph.add_edge("ask_human", "check_completeness")  # петля до полноты данных
    graph.add_edge("retrieve", "generate_answer")
    graph.add_edge("generate_answer", END)

    # MemorySaver сохраняет state между interrupt и resume
    return graph.compile(checkpointer=MemorySaver())
