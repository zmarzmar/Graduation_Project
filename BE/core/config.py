from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# 로컬 개발 기본값 — 이 조합이면 운영 검증을 스킵한다.
_LOCAL_DB_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/ai_research"
_DEFAULT_JWT_SECRET = "change-me-in-production"


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = ""

    # Semantic Scholar (선택) — 없으면 무료 tier 사용
    semantic_scholar_api_key: str = ""

    # OpenAlex (선택) — 이메일 등록 시 polite pool(100 req/sec) 적용
    openalex_email: str = ""

    # 데이터베이스 — postgresql+asyncpg:// 형식 사용
    database_url: str = _LOCAL_DB_URL

    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8001

    # CORS
    cors_origins: str = "http://localhost:3000,https://paperpilot.cloud,https://www.paperpilot.cloud"

    # JWT
    jwt_secret_key: str = _DEFAULT_JWT_SECRET

    # PDF 업로드 상한 (바이트) — 기본 20MB
    max_pdf_upload_bytes: int = 20 * 1024 * 1024

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @model_validator(mode="after")
    def _validate_production_secrets(self):
        # 로컬 개발 기본 DB URL이면 검증 스킵. 그 외에는 운영으로 간주해
        # JWT 시크릿 기본값/빈값을 거부한다 — 기본 시크릿으로 발급된 토큰은
        # 사실상 공개 키로 서명된 것과 같다.
        if self.database_url == _LOCAL_DB_URL:
            return self
        if self.jwt_secret_key in ("", _DEFAULT_JWT_SECRET):
            raise ValueError(
                "JWT_SECRET_KEY must be set to a non-default value in production."
            )
        return self


settings = Settings()
