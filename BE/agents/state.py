from typing import TypedDict


class AgentState(TypedDict):
    """LangGraph 에이전트 파이프라인 공유 상태"""

    # ── 입력 ──────────────────────────────────────────────────────────────
    mode: str           # "pdf" | "search" | "trend" | "analyze"
    user_query: str     # 사용자 입력 (키워드 또는 설명)
    pdf_pages: list[str]  # PDF 페이지별 텍스트 — 빈 페이지 포함, 인덱스 + 1 = 원본 페이지 번호
    pdf_text: str       # pdf_pages를 합친 전문 (pdf·analyze 모드, 없으면 초록 기반 분석)

    # ── 노드별 산출물 ──────────────────────────────────────────────────────
    plan: str           # Planner 실행 계획 (JSON 문자열)
    papers: list[dict]  # Researcher 수집 논문 목록
    paper_summary: str  # Analyzer 논문 요약
    paper_review: dict  # Analyzer 논문 리뷰 {"strengths": [...], "limitations": [...], "significance": "..."}
    key_formulas: list[dict]  # Analyzer 핵심 수식 [{"name": "...", "latex": "...", "description": "..."}]
    generated_code: str # Coder 생성 코드
    code_skip_reason: str  # Coder가 구현 대상이 아니라고 판단한 이유 — 있으면 코드 없이 Reviewer를 건너뛴다
    smoke_test: str     # 처음 나온 회차의 smoke_test — 이후 회차에도 이것으로 실행한다
    exec_status: str    # 실행 검사 결과: "" (안 함) | passed | code_error | timeout | oom | test_error | unavailable
    exec_detail: str    # 실행 검사 결과의 짧은 설명 (오류 끝부분)
    exec_feedback: str  # 실행 실패를 Coder에게 줄 피드백 — 있으면 Reviewer 피드백 대신 쓴다
    exec_disabled: bool # 인증·한도 문제로 이 분석에서는 실행 검사를 더 시도하지 않는다
    review_feedback: str# Reviewer 피드백
    review_passed: bool # Reviewer 검증 통과 여부

    # ── 트렌드 분석 산출물 (trend 모드 전용) ────────────────────────────────
    trend_analysis: dict  # {"paper_summaries": [...], "trending_keywords": [...]}

    # ── Self-Correction 루프 제어 ──────────────────────────────────────────
    iteration_count: int

    # ── 최종 출력 ──────────────────────────────────────────────────────────
    final_result: dict

    # ── SSE 스트리밍 메타 ──────────────────────────────────────────────────
    current_node: str   # 현재 실행 중인 노드 이름
    error: str | None   # 노드 실패 시 오류 메시지
