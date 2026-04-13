from typing import TypedDict


class AgentState(TypedDict):
    """LangGraph 에이전트 파이프라인 공유 상태"""

    # ── 입력 ──────────────────────────────────────────────────────────────
    mode: str           # "pdf" | "search" | "trend" | "analyze"
    user_query: str     # 사용자 입력 (키워드 또는 설명)
    pdf_text: str       # PDF 추출 텍스트 (mode == "pdf" 전용)

    # ── 노드별 산출물 ──────────────────────────────────────────────────────
    plan: str           # Planner 실행 계획 (JSON 문자열)
    papers: list[dict]  # Researcher 수집 논문 목록
    paper_summary: str  # Analyzer 논문 요약
    paper_review: dict  # Analyzer 논문 리뷰 {"strengths": [...], "limitations": [...], "significance": "..."}
    key_formulas: list[dict]  # Analyzer 핵심 수식 [{"name": "...", "latex": "...", "description": "..."}]
    generated_code: str # Coder 생성 코드
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
