from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.prompts import ANSWER_PROMPT, COMPLETENESS_PROMPT
from src.rag import get_retriever
from src.state import State

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = get_retriever()


# --- Структура ответа LLM при проверке полноты ---


class CompletenessCheck(BaseModel):
    is_complete: bool = Field(description="Достаточно ли данных для ответа по 590-П")
    missing_info: list[str] = Field(description="Список того, чего не хватает для ответа")
    clarification_question: str = Field(
        description="Готовый вопрос пользователю если данных не хватает, иначе пустая строка"
    )


# --- Узлы графа ---


def parse_request(state: State) -> dict:
    """Точка входа — инициализирует поля state.
    Здесь можно добавить логирование или предобработку текста."""
    return {
        "missing_info": [],
        "clarification_question": "",
        "context": "",
        "is_complete": False,
    }


def check_completeness(state: State) -> dict:
    """LLM анализирует историю диалога и решает достаточно ли данных для ответа."""
    structured_llm = llm.with_structured_output(CompletenessCheck)
    result: CompletenessCheck = structured_llm.invoke(
        [SystemMessage(content=COMPLETENESS_PROMPT)] + state["messages"]
    )
    return {
        "is_complete": result.is_complete,
        "missing_info": result.missing_info,
        "clarification_question": result.clarification_question,
    }


def ask_human(state: State) -> dict:
    """Останавливает граф через interrupt() и ждёт ответа пользователя."""
    user_answer = interrupt(state["clarification_question"])
    return {
        "messages": [
            AIMessage(content=state["clarification_question"]),
            HumanMessage(content=user_answer),
        ]
    }


def retrieve(state: State) -> dict:
    """RAG поиск по FAISS индексу 590-П.
    Берёт последнее сообщение пользователя и возвращает релевантные чанки."""
    last_human = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
    docs = retriever.invoke(last_human.content)
    context = "\n\n".join(doc.page_content for doc in docs)
    return {"context": context}


def generate_answer(state: State) -> dict:
    """Генерирует финальный ответ на основе найденных чанков из 590-П."""
    system = SystemMessage(content=ANSWER_PROMPT.format(context=state["context"]))
    response = llm.invoke([system] + state["messages"])
    return {"messages": [AIMessage(content=response.content)]}


# --- Routing ---


def route_after_check(state: State) -> str:
    """Conditional edge: решает куда идти после проверки полноты."""
    return "retrieve" if state["is_complete"] else "ask_human"
