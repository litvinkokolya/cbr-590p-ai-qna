from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[
        list[BaseMessage], add_messages
    ]  # вся история диалога, авто добавление сообщений
    missing_info: list[str]  # чего не хватает для ответа
    clarification_question: str  # уточняющий вопрос сформулированный LLM
    context: str  # найденные чанки из 590-П
    is_complete: bool  # достаточно ли данных для ответа
