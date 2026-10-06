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

    # ChromaDB — 논문 Q&A의 벡터 색인. Postgres(paper_documents)에서 다시 만들 수 있는 파생 데이터다.
    chroma_host: str = "localhost"
    chroma_port: int = 8001

    # 논문 Q&A (RAG)
    embedding_model: str = "text-embedding-3-small"
    qa_model: str = "gpt-4o-mini"
    qa_top_k: int = 6
    # 가장 가까운 청크의 코사인 거리가 이 값을 넘으면 LLM을 부르지 않고 '근거를 확인하지 못함'으로 답한다.
    # 평가(evals/qa_eval.py)의 보정용 질문으로 정하는 값이다 — 1.0은 사실상 게이트를 끈 상태.
    qa_max_distance: float = 1.0

    # CORS
    cors_origins: str = "http://localhost:3000,https://paperpilot.cloud,https://www.paperpilot.cloud"

    # JWT
    jwt_secret_key: str = _DEFAULT_JWT_SECRET

    # PDF 업로드 상한 (바이트) — 기본 20MB
    max_pdf_upload_bytes: int = 20 * 1024 * 1024

    # 논문 본문 상한 (추정 토큰). 초과하면 자르지 않고 분석을 거부한다.
    # 가장 좁은 Analyzer(gpt-4o-mini, 128k)에서 시스템 프롬프트·출력 여유를 뺀 값.
    # Coder·Reviewer(o4-mini, 200k)는 본문 + 이전 코드 + 피드백 + 추론/출력 여유를 더해도 이 안에 들어온다.
    max_paper_tokens: int = 100_000

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
