from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from src.graph import build_graph
from src.state import CompletenessCheck


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.llm_model = "gpt-4o-mini"
    settings.openai_api_key = "sk-test"
    settings.retriever_k = 8
    return settings


@patch("src.graph.get_retriever")
@patch("src.graph.ChatOpenAI")
def test_graph_complete_question(mock_chat_openai, mock_get_retriever, mock_settings):
    """Полный вопрос проходит граф без HITL и возвращает ответ."""
    mock_llm = MagicMock()
    mock_retriever = MagicMock()
    mock_chat_openai.return_value = mock_llm
    mock_get_retriever.return_value = mock_retriever

    mock_structured = MagicMock()
    mock_structured.invoke.return_value = CompletenessCheck(
        is_complete=True,
        missing_info=[],
        clarification_question="",
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mock_retriever.invoke.return_value = [MagicMock(page_content="3.11. I категория — резерв 0%")]
    mock_llm.invoke.return_value = AIMessage(content="Ставка резерва 0% для I категории.")

    graph = build_graph(mock_settings)
    config = {"configurable": {"thread_id": "test-complete"}}

    result = graph.invoke(
        {
            "messages": [HumanMessage(content="Ставка резерва для I категории?")],
            "missing_info": [],
            "clarification_question": "",
            "context": "",
            "is_complete": False,
        },
        config=config,
    )

    snapshot = graph.get_state(config)
    assert not snapshot.next

    last_message = result["messages"][-1]
    assert isinstance(last_message, AIMessage)
    assert last_message.content == "Ставка резерва 0% для I категории."


@patch("src.graph.get_retriever")
@patch("src.graph.ChatOpenAI")
def test_graph_hitl_flow(mock_chat_openai, mock_get_retriever, mock_settings):
    """Неполный вопрос — граф прерывается на HITL, после ответа завершается."""
    mock_llm = MagicMock()
    mock_retriever = MagicMock()
    mock_chat_openai.return_value = mock_llm
    mock_get_retriever.return_value = mock_retriever

    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = [
        CompletenessCheck(
            is_complete=False,
            missing_info=["категория качества"],
            clarification_question="Укажите категорию качества ссуды.",
        ),
        CompletenessCheck(
            is_complete=True,
            missing_info=[],
            clarification_question="",
        ),
    ]
    mock_llm.with_structured_output.return_value = mock_structured
    mock_retriever.invoke.return_value = [
        MagicMock(page_content="3.11. III категория — резерв 21-50%")
    ]
    mock_llm.invoke.return_value = AIMessage(content="Резерв 21-50% согласно п. 3.11.")

    graph = build_graph(mock_settings)
    config = {"configurable": {"thread_id": "test-hitl"}}

    graph.invoke(
        {
            "messages": [HumanMessage(content="Какая ставка резерва?")],
            "missing_info": [],
            "clarification_question": "",
            "context": "",
            "is_complete": False,
        },
        config=config,
    )

    snapshot = graph.get_state(config)
    assert snapshot.next
    assert snapshot.tasks[0].interrupts[0].value == "Укажите категорию качества ссуды."

    result = graph.invoke(Command(resume="III категория"), config=config)

    snapshot = graph.get_state(config)
    assert not snapshot.next

    last_message = result["messages"][-1]
    assert isinstance(last_message, AIMessage)
    assert last_message.content == "Резерв 21-50% согласно п. 3.11."
