import uuid
import warnings

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory

from src.config import Settings
from src.graph import build_graph

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


def print_agent(text: str) -> None:
    print(f"\n\033[32mАгент:\033[0m {text}\n")


def run() -> None:
    # валидация конфига при запуске — упадёт сразу если нет ключа или неверные параметры
    settings = Settings()
    graph = build_graph(settings)

    session = PromptSession(
        history=InMemoryHistory(),
        message=HTML("<ansiyellow>Вы: </ansiyellow>"),
    )

    print("Чат-бот по Положению ЦБ РФ 590-П.")
    print("Для выхода введите 'exit' или нажмите Ctrl+C.\n")

    try:
        while True:
            user_input = session.prompt().strip()
            if not user_input:
                continue
            if user_input.lower() == "exit":
                break

            # каждый новый вопрос — отдельная сессия с уникальным thread_id
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}

            try:
                state = graph.invoke(
                    {
                        "messages": [HumanMessage(content=user_input)],
                        "missing_info": [],
                        "clarification_question": "",
                        "context": "",
                        "is_complete": False,
                    },
                    config=config,
                )

                # внутренний цикл обрабатывает HITL уточнения
                while True:
                    snapshot = graph.get_state(config)

                    if snapshot.next:
                        clarification = snapshot.tasks[0].interrupts[0].value
                        print_agent(clarification)

                        user_answer = session.prompt().strip()
                        if user_answer.lower() == "exit":
                            return

                        state = graph.invoke(
                            Command(resume=user_answer),
                            config=config,
                        )
                    else:
                        print_agent(state["messages"][-1].content)
                        break

            except RuntimeError as e:
                print(f"\n\033[31mОшибка:\033[0m {e}\n")

    except (KeyboardInterrupt, EOFError):
        print("\nДо свидания!")


if __name__ == "__main__":
    run()
