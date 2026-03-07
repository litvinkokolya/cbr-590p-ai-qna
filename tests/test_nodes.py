from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.nodes import (
    CompletenessCheck,
    ask_human,
    check_completeness,
    generate_answer,
    parse_request,
    retrieve,
    route_after_check,
)


# --- Фикстуры ---


@pytest.fixture
def initial_state():
    """Минимальный state после parse_request."""
    return {
        "messages": [HumanMessage(content="Какая ставка резерва?")],
        "missing_info": [],
        "clarification_question": "",
        "context": "",
        "is_complete": False,
    }


@pytest.fixture
def complete_state():
    """State с достаточными данными для ответа."""
    return {
        "messages": [
            HumanMessage(content="Какая ставка резерва для III категории качества без обеспечения?")
        ],
        "missing_info": [],
        "clarification_question": "",
        "context": "3.11. III категория качества — резерв 21-50%",
        "is_complete": True,
    }


# --- Тесты parse_request ---


def test_parse_request_initializes_state(initial_state):
    result = parse_request(initial_state)
    assert result["missing_info"] == []
    assert result["clarification_question"] == ""
    assert result["context"] == ""
    assert result["is_complete"] is False


# --- Тесты check_completeness ---


@patch("src.nodes.llm")
def test_check_completeness_incomplete(mock_llm, initial_state):
    """Неполный запрос — LLM возвращает is_complete=False."""
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = CompletenessCheck(
        is_complete=False,
        missing_info=["категория качества ссуды"],
        clarification_question="Укажите категорию качества ссуды (I-V).",
    )
    mock_llm.with_structured_output.return_value = mock_structured

    result = check_completeness(initial_state)

    assert result["is_complete"] is False
    assert "категория качества ссуды" in result["missing_info"]
    assert result["clarification_question"] == "Укажите категорию качества ссуды (I-V)."


@patch("src.nodes.llm")
def test_check_completeness_complete(mock_llm, complete_state):
    """Полный запрос — LLM возвращает is_complete=True."""
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = CompletenessCheck(
        is_complete=True,
        missing_info=[],
        clarification_question="",
    )
    mock_llm.with_structured_output.return_value = mock_structured

    result = check_completeness(complete_state)

    assert result["is_complete"] is True
    assert result["missing_info"] == []


# --- Тесты route_after_check ---


def test_route_incomplete(initial_state):
    assert route_after_check(initial_state) == "ask_human"


def test_route_complete(complete_state):
    assert route_after_check(complete_state) == "retrieve"


# --- Тесты ask_human ---


@patch("src.nodes.interrupt")
def test_ask_human_adds_messages(mock_interrupt, initial_state):
    """ask_human добавляет вопрос агента и ответ пользователя в messages."""
    initial_state["clarification_question"] = "Укажите категорию качества ссуды."
    mock_interrupt.return_value = "III категория"

    result = ask_human(initial_state)

    messages = result["messages"]
    assert len(messages) == 2
    assert isinstance(messages[0], AIMessage)
    assert messages[0].content == "Укажите категорию качества ссуды."
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "III категория"


@patch("src.nodes.interrupt")
def test_ask_human_uses_state_question(mock_interrupt, initial_state):
    """ask_human берёт вопрос из state, а не вызывает LLM повторно."""
    initial_state["clarification_question"] = "Тип заёмщика?"
    mock_interrupt.return_value = "Юридическое лицо"

    ask_human(initial_state)

    # interrupt должен быть вызван с вопросом из state
    mock_interrupt.assert_called_once_with("Тип заёмщика?")


# --- Тесты retrieve ---


@patch("src.nodes.retriever")
def test_retrieve_returns_context(mock_retriever, initial_state):
    """retrieve склеивает найденные чанки в строку context."""
    mock_retriever.invoke.return_value = [
        MagicMock(page_content="Чанк 1 из 590-П"),
        MagicMock(page_content="Чанк 2 из 590-П"),
    ]

    result = retrieve(initial_state)

    assert "Чанк 1 из 590-П" in result["context"]
    assert "Чанк 2 из 590-П" in result["context"]


@patch("src.nodes.retriever")
def test_retrieve_uses_last_human_message(mock_retriever):
    """retrieve берёт последнее сообщение пользователя, а не первое."""
    state = {
        "messages": [
            HumanMessage(content="Первый вопрос"),
            AIMessage(content="Уточните категорию"),
            HumanMessage(content="III категория без обеспечения"),
        ],
        "missing_info": [],
        "clarification_question": "",
        "context": "",
        "is_complete": True,
    }
    mock_retriever.invoke.return_value = []

    retrieve(state)

    mock_retriever.invoke.assert_called_once_with("III категория без обеспечения")


# --- Тесты generate_answer ---


@patch("src.nodes.llm")
def test_generate_answer_returns_ai_message(mock_llm, complete_state):
    """generate_answer возвращает AIMessage с ответом."""
    mock_llm.invoke.return_value = AIMessage(content="Ставка резерва 21-50% согласно п. 3.11.")

    result = generate_answer(complete_state)

    messages = result["messages"]
    assert len(messages) == 1
    assert isinstance(messages[0], AIMessage)
    assert "3.11" in messages[0].content
