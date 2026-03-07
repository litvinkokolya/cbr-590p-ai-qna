from typing import TypedDict

from langchain_core.messages import BaseMessage


class State(TypedDict):
    messages: list[BaseMessage]  # вся история диалога
    missing_info: list[str]  # чего не хватает для ответа
    context: str  # найденные чанки из 590-П
    is_complete: bool  # достаточно ли данных для ответа
