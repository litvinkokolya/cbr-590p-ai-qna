import os
from dotenv import load_dotenv

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DOCX_PATH = "590-п.docx"
INDEX_PATH = "faiss_590p_index"


def build_index():
    """Загружает 590-П, нарезает на чанки и сохраняет FAISS индекс."""
    print("Загружаю документ...")
    loader = Docx2txtLoader(DOCX_PATH)
    documents = loader.load()

    print("Нарезаю на чанки...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documents)
    print(f"Чанков: {len(chunks)}")

    print("Создаю эмбеддинги и индекс (запрос к OpenAI)...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)
    print(f"Индекс сохранён в '{INDEX_PATH}'")


def load_retriever():
    """Загружает сохранённый индекс и возвращает retriever."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore.as_retriever(search_kwargs={"k": 5})


def get_retriever():
    """Возвращает retriever — строит индекс если его нет."""
    if not os.path.exists(INDEX_PATH):
        build_index()
    return load_retriever()


if __name__ == "__main__":
    # Запуск напрямую: python rag.py — строит индекс и проверяет поиск
    retriever = get_retriever()
    question = "Какая минимальная ставка резерва при стандартном качестве обслуживания долга?"
    docs = retriever.invoke(question)
    print(f"\nВопрос: {question}")
    print(f"Найдено чанков: {len(docs)}\n")
    for i, doc in enumerate(docs):
        print(f"--- Чанк {i+1} ---")
        print(doc.page_content)
        print()
