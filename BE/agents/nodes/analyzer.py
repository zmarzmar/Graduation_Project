import json
import logging
import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.log_stream import emit_log
from agents.perf import log_elapsed
from agents.qa import Citation, validate_citations
from agents.state import AgentState
from agents.token_budget import ANALYZER_OUTPUT_BUDGET, ensure_complete, ensure_request_fits
from core.config import settings

logger = logging.getLogger(__name__)

_MODEL = settings.analyzer_model
_llm = ChatOpenAI(
    model=_MODEL,
    api_key=settings.openai_api_key,
    timeout=settings.llm_timeout_analyzer_seconds,
    max_retries=settings.llm_max_retries,
    max_tokens=ANALYZER_OUTPUT_BUDGET,  # 출력 예산 — 추론 토큰도 여기서 쓴다
    reasoning_effort=settings.analyzer_reasoning_effort,  # 추론 모델은 temperature를 받지 않는다
    model_kwargs={"response_format": {"type": "json_object"}},
)

_SYSTEM_PROMPT = """당신은 AI 논문을 분석하는 전문가입니다.
주어진 논문을 분석하고 아래 JSON 형식으로만 응답하세요.

{
  "summary_claims": [
    {
      "text": "요약 한 문장 (한국어)",
      "citations": [{"page": 3, "quote": "그 페이지에서 한 글자도 바꾸지 않고 그대로 복사한 영어 원문 한두 문장"}]
    }
  ],
  "review": {
    "strengths": ["강점1", "강점2", "강점3"],
    "limitations": ["한계점1", "한계점2"],
    "significance": "이 논문의 학문적/실용적 의의 (한국어, 2-3문장)"
  },
  "key_formulas": [
    {
      "name": "수식 이름 (예: Attention Score)",
      "latex": "LaTeX 수식 (예: \\\\text{Attention}(Q,K,V) = \\\\text{softmax}(\\\\frac{QK^T}{\\\\sqrt{d_k}})V)",
      "description": "이 수식이 하는 역할 설명 (한국어, 1-2문장)"
    }
  ]
}

규칙:
- summary_claims는 논문 핵심 요약이다: 한국어 3-5문장, 연구 목적·방법·결과·기여를 포함한다. 문장 하나가 claim 하나다
- 논문이 <page number="N"> 블록으로 주어지면, claim마다 그 문장의 근거가 된 citation을 하나 이상 단다: page는 블록 번호,
  quote는 그 블록에서 한 글자도 바꾸지 않고 그대로 복사한 한두 문장(대소문자·기호도 그대로, 단어를 넣거나 빼지 않는다).
  수식·표의 숫자 줄이 아닌 서술 문장을 인용한다
- claim 하나에는 자기 quote가 말하는 내용만 담는다. 근거를 댈 수 없는 내용은 쓰지 않는다
- <page> 블록이 없으면(초록만 주어진 경우) citations는 빈 배열로 둔다
- key_formulas는 논문에서 핵심적인 수식/알고리즘을 최대 3개까지만 추출
- 수식이 없는 논문이면 key_formulas는 빈 배열로
- LaTeX 수식은 인라인 수식 형태로 작성"""


def _extract_json(text: str) -> dict:
    """LLM 응답에서 JSON을 추출한다."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


@dataclass
class _Page:
    """validate_citations가 받는 구절 형태 — 여기서는 페이지 하나가 구절 하나다."""
    chunk_index: int
    page: int
    text: str


def _check_summary_claims(raw: object, pages: list[str]) -> list[dict]:
    """요약 문장마다 인용이 그 페이지에 실제로 있는지 확인한다. 확인되지 않은 문장도 빼지 않고 verified=False로 남긴다
    (요약이 중간에 비지 않게 — Q&A와 다른 점). 확인하는 것은 인용의 존재뿐이고, 인용이 문장을 뒷받침하는지는 아니다."""
    if not isinstance(raw, list):
        return []
    passages = [_Page(chunk_index=index, page=index + 1, text=text) for index, text in enumerate(pages)]
    claims: list[dict] = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
            continue
        valid: list[dict] = []
        dropped = 0
        for cited in item.get("citations") or []:
            if not (isinstance(cited, dict) and isinstance(cited.get("page"), int) and isinstance(cited.get("quote"), str)):
                dropped += 1
                continue
            # 모델은 인용문을 그대로 옮기고도 페이지 번호를 하나 틀리곤 한다 (2026-11-18 측정) — 적은 페이지에 없으면
            # 다른 페이지도 찾고, 찾은 페이지를 보여준다. 인용문 자체의 비교 기준은 그대로다
            order = [cited["page"]] + [n for n in range(1, len(passages) + 1) if n != cited["page"]]
            found = next((hit for n in order for hit in validate_citations([Citation(passage=n, quote=cited["quote"])], passages)[0]), None)
            if found:
                valid.append(found)
            else:
                dropped += 1
        claims.append({
            "text": item["text"].strip(),
            # 화면에 보여줄 인용은 검증을 통과한 것만 — 페이지 번호도 모델 출력이 아니라 확인한 페이지에서 가져온다
            "citations": [{"page": c["page"], "quote": c["quote"]} for c in valid],
            "verified": bool(valid) and not dropped,
        })
    return claims


async def analyzer_node(state: AgentState) -> dict:
    """논문 요약, 리뷰, 핵심 수식을 추출한다. (pdf/search 모드 전용)"""
    logger.info("[Analyzer] 시작")

    papers = state.get("papers", [])
    pdf_text = state.get("pdf_text", "")
    pdf_pages = state.get("pdf_pages") or []

    # 분석 대상 텍스트 구성 — PDF 텍스트 우선, 없으면 수집된 논문 사용
    if pdf_text and pdf_pages:
        # 요약 문장의 인용을 페이지 단위로 확인할 수 있게 페이지 번호를 붙인다 (빈 페이지는 번호만 건너뛴다)
        paper_text = "\n\n".join(
            f'<page number="{number}">\n{page}\n</page>' for number, page in enumerate(pdf_pages, start=1) if page.strip()
        )
        emit_log("analyzer", "PDF 전체 내용 분석 중...")
    elif pdf_text:
        paper_text = pdf_text
        emit_log("analyzer", "PDF 전체 내용 분석 중...")
    elif papers:
        first = papers[0]
        paper_text = f"제목: {first.get('title', '')}\n\n초록:\n{first.get('abstract', '')}"
        if tldr := first.get("tldr"):
            paper_text += f"\n\nTL;DR: {tldr}"
        emit_log("analyzer", f"논문 분석 중: {first.get('title', '')[:50]}...")
    else:
        logger.warning("[Analyzer] 분석할 논문 없음")
        return {
            "paper_summary": "",
            "summary_claims": [],
            "paper_review": {},
            "key_formulas": [],
            "current_node": "analyzer",
            "error": "분석할 논문이 없습니다.",
        }

    # LLM 호출 실패와 JSON 파싱 실패를 분리해서 처리
    try:
        emit_log("analyzer", "요약 및 리뷰 생성 중...")
        async with log_elapsed(logger, "external_call", node="analyzer", external="openai"):
            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=f"다음 논문을 분석해주세요:\n\n{paper_text}"),
            ]
            # 이전 코드·피드백까지 합친 전체 입력 + 출력 예산이 한도 안인지 호출 직전에 확인한다
            ensure_request_fits(_MODEL, messages, ANALYZER_OUTPUT_BUDGET)
            response = await _llm.ainvoke(messages)
            ensure_complete(response, ANALYZER_OUTPUT_BUDGET)
    except Exception as e:
        logger.error(f"[Analyzer] LLM API 호출 실패: {e}")
        return {
            "paper_summary": "",
            "summary_claims": [],
            "paper_review": {},
            "key_formulas": [],
            "current_node": "analyzer",
            "error": "LLM API 호출에 실패했습니다.",
        }

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        logger.warning("[Analyzer] JSON 파싱 실패 — _extract_json 폴백 시도")
        result = _extract_json(response.content)

    summary_claims = _check_summary_claims(result.get("summary_claims") if isinstance(result, dict) else None, pdf_pages)
    if summary_claims:
        result["summary"] = " ".join(claim["text"] for claim in summary_claims)

    # 해석하지 못한 응답을 빈 요약으로 조용히 넘기지 않는다 — 호출 실패와 같은 오류 형태로 돌려준다 (배열 응답도 여기서 거른다)
    if not isinstance(result, dict) or not isinstance(result.get("summary"), str) or not result["summary"].strip():
        logger.warning(f"[Analyzer] 응답 해석 실패 — 원문 앞부분: {str(response.content)[:500]!r}")
        emit_log("analyzer", "분석 응답을 해석하지 못했습니다")
        return {
            "paper_summary": "",
            "summary_claims": [],
            "paper_review": {},
            "key_formulas": [],
            "current_node": "analyzer",
            "error": "분석 응답을 해석하지 못했습니다.",
        }

    summary = result["summary"]
    # 형식이 다른 부가 필드는 비운다 — 요약은 살리고 len(None) 같은 예외로 그래프가 죽지 않게
    review = result.get("review") if isinstance(result.get("review"), dict) else {}
    formulas = result.get("key_formulas")
    key_formulas = [f for f in formulas if isinstance(f, dict)] if isinstance(formulas, list) else []

    emit_log("analyzer", "요약 완료")
    if key_formulas:
        emit_log("analyzer", f"핵심 수식 {len(key_formulas)}개 추출 완료")
    logger.info(f"[Analyzer] 완료 — 수식 {len(key_formulas)}개 추출")

    return {
        "paper_summary": summary,
        # 원문(PDF)이 없으면 인용을 확인할 수 없다 — 초록 기반 분석은 빈 목록이고 화면은 요약 문자열만 보여준다
        "summary_claims": summary_claims if pdf_pages else [],
        "paper_review": review,
        "key_formulas": key_formulas,
        "current_node": "analyzer",
        "error": None,
    }
