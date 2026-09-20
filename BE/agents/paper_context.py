from agents.state import AgentState


def paper_source_context(state: AgentState) -> str:
    """Coder와 Reviewer가 공유하는 '논문 원문 부분' 컨텍스트를 만든다.

    생성기와 검토기가 같은 근거를 보도록 한 곳에서 만든다. 통일하는 것은 논문 원문 부분뿐이고,
    Coder에만 들어가는 요약·핵심 수식·이전 코드 등은 각 노드가 따로 붙인다.
    본문은 자르지 않는다 — 길이 한도는 추출 단계(extract_pdf_pages)에서 이미 검사했다.
    """
    if pdf_text := state.get("pdf_text", ""):
        return f"[논문 전문]\n{pdf_text}"

    # 전문이 없으면 초록 기반 (최대 3편)
    papers = state.get("papers", [])
    if not papers:
        return "수집된 논문 없음"

    lines = []
    for i, paper in enumerate(papers[:3], 1):
        lines.append(f"## 논문 {i}: {paper.get('title', 'N/A')}")
        lines.append(f"저자: {', '.join(paper.get('authors', [])[:3])}")
        if tldr := paper.get("tldr"):
            lines.append(f"요약: {tldr}")
        lines.append(f"초록:\n{paper.get('abstract', '')[:800]}")
        lines.append("")
    return "\n".join(lines)
