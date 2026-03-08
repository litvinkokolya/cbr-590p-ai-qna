from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.config import Settings
from src.nodes import AgentNodes
from src.rag import get_retriever
from src.state import State


def build_graph(settings: Settings):
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )
    retriever = get_retriever(settings)
    nodes = AgentNodes(llm=llm, retriever=retriever)

    graph = StateGraph(State)

    graph.add_node("check_completeness", nodes.check_completeness)
    graph.add_node("ask_human", nodes.ask_human)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("generate_answer", nodes.generate_answer)

    graph.add_edge(START, "check_completeness")
    graph.add_conditional_edges(
        "check_completeness",
        nodes.route_after_check,
        {"retrieve": "retrieve", "ask_human": "ask_human"},
    )
    graph.add_edge("ask_human", "check_completeness")  # петля до полноты данных
    graph.add_edge("retrieve", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile(checkpointer=MemorySaver())
