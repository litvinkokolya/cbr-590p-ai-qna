from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str = Field(description="OpenAI API ключ")
    llm_model: str = Field(default="gpt-4o-mini", description="Модель для генерации ответов")
    embeddings_model: str = Field(
        default="text-embedding-3-small", description="Модель для эмбеддингов"
    )
    retriever_k: int = Field(default=8, description="Количество чанков для RAG поиска")
    docx_path: str = Field(default="data/590-п.docx", description="Путь к документу 590-П")
    index_path: str = Field(default="data/faiss_590p_index", description="Путь к FAISS индексу")

    @field_validator("openai_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v.startswith("sk-"):
            raise ValueError("OPENAI_API_KEY должен начинаться с 'sk-'")
        return v

    @field_validator("retriever_k")
    @classmethod
    def validate_retriever_k(cls, v: int) -> int:
        if v < 1 or v > 20:
            raise ValueError("RETRIEVER_K должен быть от 1 до 20")
        return v
