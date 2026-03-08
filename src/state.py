from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# --- Состояние диалога ---


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # история диалога, авто добавление
    missing_info: list[str]  # чего не хватает для ответа
    clarification_question: str  # уточняющий вопрос сформулированный LLM
    context: str  # найденные чанки из 590-П
    is_complete: bool  # достаточно ли данных для ответа


# --- Схема структурированного ответа LLM ---


class CompletenessCheck(BaseModel):
    is_complete: bool = Field(description="Достаточно ли данных для ответа по 590-П")
    missing_info: list[str] = Field(description="Список того, чего не хватает для ответа")
    clarification_question: str = Field(
        description="Готовый вопрос пользователю если данных не хватает, иначе пустая строка"
    )


# --- Возвращаемые типы узлов графа ---


class CheckCompletenessResult(TypedDict):
    is_complete: bool
    missing_info: list[str]
    clarification_question: str


class MessagesResult(TypedDict):
    messages: list[BaseMessage]


class RetrieveResult(TypedDict):
    context: str
