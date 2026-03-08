import openai
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt

from src.prompts import ANSWER_PROMPT, COMPLETENESS_PROMPT
from src.state import (
    CheckCompletenessResult,
    CompletenessCheck,
    MessagesResult,
    RetrieveResult,
    State,
)

_RETRIEVE_WINDOW = 3  # сколько последних сообщений пользователя склеивать в запрос


class AgentNodes:
    def __init__(self, llm: ChatOpenAI, retriever: VectorStoreRetriever) -> None:
        self.llm = llm
        self.retriever = retriever

    def check_completeness(self, state: State) -> CheckCompletenessResult:
        """LLM анализирует историю диалога и решает достаточно ли данных для ответа."""
        structured_llm = self.llm.with_structured_output(CompletenessCheck)
        try:
            result: CompletenessCheck = structured_llm.invoke(
                [SystemMessage(content=COMPLETENESS_PROMPT)] + state["messages"]
            )
        except openai.AuthenticationError:
            raise RuntimeError("Ошибка аутентификации OpenAI: проверьте API ключ.")
        except openai.RateLimitError:
            raise RuntimeError("Превышен лимит запросов OpenAI. Попробуйте позже.")
        except openai.APITimeoutError:
            raise RuntimeError("Запрос к OpenAI завершился по таймауту. Попробуйте ещё раз.")
        except openai.APIConnectionError:
            raise RuntimeError("Не удалось подключиться к OpenAI. Проверьте интернет-соединение.")
        except openai.BadRequestError as e:
            raise RuntimeError(f"Некорректный запрос к OpenAI: {e}")
        except openai.InternalServerError:
            raise RuntimeError("Внутренняя ошибка OpenAI. Попробуйте позже.")
        return CheckCompletenessResult(
            is_complete=result.is_complete,
            missing_info=result.missing_info,
            clarification_question=result.clarification_question,
        )

    def ask_human(self, state: State) -> MessagesResult:
        """Останавливает граф через interrupt() и ждёт ответа пользователя."""
        user_answer = interrupt(state["clarification_question"])
        return MessagesResult(
            messages=[
                AIMessage(content=state["clarification_question"]),
                HumanMessage(content=user_answer),
            ]
        )

    def retrieve(self, state: State) -> RetrieveResult:
        """RAG поиск по FAISS индексу 590-П.
        Склеивает последние _RETRIEVE_WINDOW сообщений пользователя в единый запрос,
        чтобы не терять контекст после HITL-уточнений."""
        human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        query = " ".join(m.content for m in human_messages[-_RETRIEVE_WINDOW:])
        docs = self.retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)
        return RetrieveResult(context=context)

    def generate_answer(self, state: State) -> MessagesResult:
        """Генерирует финальный ответ на основе найденных чанков из 590-П."""
        system = SystemMessage(content=ANSWER_PROMPT.format(context=state["context"]))
        try:
            response = self.llm.invoke([system] + state["messages"])
        except openai.AuthenticationError:
            raise RuntimeError("Ошибка аутентификации OpenAI: проверьте API ключ.")
        except openai.RateLimitError:
            raise RuntimeError("Превышен лимит запросов OpenAI. Попробуйте позже.")
        except openai.APITimeoutError:
            raise RuntimeError("Запрос к OpenAI завершился по таймауту. Попробуйте ещё раз.")
        except openai.APIConnectionError:
            raise RuntimeError("Не удалось подключиться к OpenAI. Проверьте интернет-соединение.")
        except openai.BadRequestError as e:
            raise RuntimeError(f"Некорректный запрос к OpenAI: {e}")
        except openai.InternalServerError:
            raise RuntimeError("Внутренняя ошибка OpenAI. Попробуйте позже.")
        return MessagesResult(messages=[AIMessage(content=response.content)])

    @staticmethod
    def route_after_check(state: State) -> str:
        """Conditional edge: решает куда идти после проверки полноты."""
        return "retrieve" if state["is_complete"] else "ask_human"
