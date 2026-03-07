import warnings

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from graph import build_graph

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

graph = build_graph()

# thread_id позволяет вести несколько независимых диалогов
config = {"configurable": {"thread_id": "1"}}


def run():
    print("Чат-бот по Положению ЦБ РФ 590-П. Для выхода введите 'exit'.\n")

    user_input = input("Вы: ").strip()
    if user_input.lower() == "exit":
        return

    # первый запуск графа с вопросом пользователя
    state = graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
    )

    while True:
        # проверяем прерван ли граф на HITL
        snapshot = graph.get_state(config)

        if snapshot.next:
            # граф ждёт ответа пользователя — показываем уточняющий вопрос
            clarification = snapshot.tasks[0].interrupts[0].value
            print(f"\nАгент: {clarification}")

            user_answer = input("Вы: ").strip()
            if user_answer.lower() == "exit":
                break

            # возобновляем граф с ответом пользователя
            state = graph.invoke(
                Command(resume=user_answer),
                config=config,
            )
        else:
            # граф завершил работу — показываем финальный ответ
            last_message = state["messages"][-1]
            print(f"\nАгент: {last_message.content}\n")
            break


if __name__ == "__main__":
    run()
