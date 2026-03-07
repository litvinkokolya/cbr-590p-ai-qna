from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from src.graph import build_graph
from src.nodes import CompletenessCheck


@patch("src.nodes.retriever")
@patch("src.nodes.llm")
def test_graph_complete_question(mock_llm, mock_retriever):
    """Полный вопрос проходит граф без HITL и возвращает ответ."""
    # check_completeness возвращает is_complete=True
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = CompletenessCheck(
        is_complete=True,
        missing_info=[],
        clarification_question="",
    )
    mock_llm.with_structured_output.return_value = mock_structured

    # retrieve находит чанки
    mock_retriever.invoke.return_value = [MagicMock(page_content="3.11. I категория — резерв 0%")]

    # generate_answer возвращает ответ
    mock_llm.invoke.return_value = AIMessage(content="Ставка резерва 0% для I категории.")

    graph = build_graph()
    config = {"configurable": {"thread_id": "test-complete"}}

    result = graph.invoke(
        {"messages": [HumanMessage(content="Ставка резерва для I категории?")]},
        config=config,
    )

    # граф завершился без прерывания
    snapshot = graph.get_state(config)
    assert not snapshot.next

    last_message = result["messages"][-1]
    assert isinstance(last_message, AIMessage)
    assert last_message.content == "Ставка резерва 0% для I категории."


@patch("src.nodes.retriever")
@patch("src.nodes.llm")
def test_graph_hitl_flow(mock_llm, mock_retriever):
    """Неполный вопрос — граф прерывается на HITL, после ответа завершается."""
    mock_structured = MagicMock()

    # первый вызов check_completeness — данных не хватает
    # второй вызов check_completeness (после уточнения) — данных достаточно
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

    graph = build_graph()
    config = {"configurable": {"thread_id": "test-hitl"}}

    # первый invoke — граф должен прерваться
    graph.invoke(
        {"messages": [HumanMessage(content="Какая ставка резерва?")]},
        config=config,
    )

    # проверяем что граф прерван
    snapshot = graph.get_state(config)
    assert snapshot.next  # граф ждёт ответа пользователя
    assert snapshot.tasks[0].interrupts[0].value == "Укажите категорию качества ссуды."

    # возобновляем с ответом пользователя
    result = graph.invoke(Command(resume="III категория"), config=config)

    # граф завершился
    snapshot = graph.get_state(config)
    assert not snapshot.next

    last_message = result["messages"][-1]
    assert isinstance(last_message, AIMessage)
    assert last_message.content == "Резерв 21-50% согласно п. 3.11."
