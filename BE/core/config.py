from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = ""

    # Semantic Scholar (선택) — 없으면 무료 tier 사용
    semantic_scholar_api_key: str = ""

    # OpenAlex (선택) — 이메일 등록 시 polite pool(100 req/sec) 적용
    openalex_email: str = ""

    # 데이터베이스 — postgresql+asyncpg:// 형식 사용
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ai_research"

    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8001

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
