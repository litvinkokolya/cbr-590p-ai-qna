from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.nodes import AgentNodes
from src.state import CompletenessCheck

# --- Фикстуры ---


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def mock_retriever():
    return MagicMock()


@pytest.fixture
def nodes(mock_llm, mock_retriever):
    return AgentNodes(llm=mock_llm, retriever=mock_retriever)


@pytest.fixture
def initial_state():
    """Минимальный state для начала диалога."""
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


# --- Тесты check_completeness ---


def test_check_completeness_incomplete(nodes, mock_llm, initial_state):
    """Неполный запрос — LLM возвращает is_complete=False."""
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = CompletenessCheck(
        is_complete=False,
        missing_info=["категория качества ссуды"],
        clarification_question="Укажите категорию качества ссуды (I-V).",
    )
    mock_llm.with_structured_output.return_value = mock_structured

    result = nodes.check_completeness(initial_state)

    assert result["is_complete"] is False
    assert "категория качества ссуды" in result["missing_info"]
    assert result["clarification_question"] == "Укажите категорию качества ссуды (I-V)."


def test_check_completeness_complete(nodes, mock_llm, complete_state):
    """Полный запрос — LLM возвращает is_complete=True."""
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = CompletenessCheck(
        is_complete=True,
        missing_info=[],
        clarification_question="",
    )
    mock_llm.with_structured_output.return_value = mock_structured

    result = nodes.check_completeness(complete_state)

    assert result["is_complete"] is True
    assert result["missing_info"] == []


# --- Тесты route_after_check ---


def test_route_incomplete(initial_state):
    assert AgentNodes.route_after_check(initial_state) == "ask_human"


def test_route_complete(complete_state):
    assert AgentNodes.route_after_check(complete_state) == "retrieve"


# --- Тесты ask_human ---


@patch("src.nodes.interrupt")
def test_ask_human_adds_messages(mock_interrupt, nodes, initial_state):
    """ask_human добавляет вопрос агента и ответ пользователя в messages."""
    initial_state["clarification_question"] = "Укажите категорию качества ссуды."
    mock_interrupt.return_value = "III категория"

    result = nodes.ask_human(initial_state)

    messages = result["messages"]
    assert len(messages) == 2
    assert isinstance(messages[0], AIMessage)
    assert messages[0].content == "Укажите категорию качества ссуды."
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "III категория"


@patch("src.nodes.interrupt")
def test_ask_human_uses_state_question(mock_interrupt, nodes, initial_state):
    """ask_human берёт вопрос из state, а не вызывает LLM повторно."""
    initial_state["clarification_question"] = "Тип заёмщика?"
    mock_interrupt.return_value = "Юридическое лицо"

    nodes.ask_human(initial_state)

    mock_interrupt.assert_called_once_with("Тип заёмщика?")


# --- Тесты retrieve ---


def test_retrieve_returns_context(nodes, mock_retriever, initial_state):
    """retrieve склеивает найденные чанки в строку context."""
    mock_retriever.invoke.return_value = [
        MagicMock(page_content="Чанк 1 из 590-П"),
        MagicMock(page_content="Чанк 2 из 590-П"),
    ]

    result = nodes.retrieve(initial_state)

    assert "Чанк 1 из 590-П" in result["context"]
    assert "Чанк 2 из 590-П" in result["context"]


def test_retrieve_joins_human_messages(nodes, mock_retriever):
    """retrieve склеивает последние _RETRIEVE_WINDOW сообщений пользователя в один запрос."""
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

    nodes.retrieve(state)

    mock_retriever.invoke.assert_called_once_with("Первый вопрос III категория без обеспечения")


# --- Тесты generate_answer ---


def test_generate_answer_returns_ai_message(nodes, mock_llm, complete_state):
    """generate_answer возвращает AIMessage с ответом."""
    mock_llm.invoke.return_value = AIMessage(content="Ставка резерва 21-50% согласно п. 3.11.")

    result = nodes.generate_answer(complete_state)

    messages = result["messages"]
    assert len(messages) == 1
    assert isinstance(messages[0], AIMessage)
    assert "3.11" in messages[0].content
