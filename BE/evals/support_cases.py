"""'인용 구절이 그 문장을 뒷받침하는가'를 원문과 대조해 판정해 둔 사례와, LLM 판정기가 그 판정과 얼마나 맞는지 보는 스크립트.

실행: cd BE && uv run python -m evals.support_cases        (실제 API 호출 — CI에서는 돌리지 않는다)

왜 있는가
- 서버가 런타임에 확인하는 것은 '인용문이 구절에 실재하는가'뿐이다. 실재하는 인용을 달고도 문장이 인용보다 넓게 쓰일 수 있다.
- qa_eval.py의 LLM 판정(judge_support)이 그런 문장을 잡는지 확인하려면 정답이 정해진 사례가 필요하다.
  판정기가 이 사례들과 맞지 않으면 "판정기가 뒷받침된다고 했다"는 수치는 정확성의 근거가 되지 못한다.

사례의 출처와 한계
- 모두 실제 모델 출력이다: 운영(2026-10-06)과 로컬 재현에서 나온 문장과 그 문장에 달린, 서버 검증을 통과한 인용문.
- label은 구현한 쪽(Claude)이 논문 원문을 읽고 붙였다. 사람이 다시 확인한 것이 아니다 — 판정이 갈릴 수 있는 사례는 note에 이유를 적었다.
- over_broad = 인용문에 없는 내용이 문장에 들어 있다. 그 내용이 논문의 다른 곳에 있더라도(인용 누락) over_broad다:
  화면에서 문장 옆에 보이는 구절로는 그 문장을 확인할 수 없기 때문이다.
- 논문 3편, 8개 사례. 빈도를 말해 주는 표본이 아니라 판정기를 시험하는 고정 입력이다.

검증에서 '거부된' 인용 사례(참고문헌 번호를 뺀 인용, 문장 앞머리를 떼며 쉼표를 뺀 인용 — 둘 다 거짓 거부)는 API 없이 재현되므로
tests/test_rag.py의 test_known_rejections_from_a_real_paper_stay_rejected에 있다.

인용 검증의 결과와 문장의 정확성은 따로 판단한다. 쉼표 때문에 거부된 인용이 달려 있던 문장은
"이 방법은 ODQA 벤치마크에서 다른 압축 방법들보다 우수한 성능을 보인다."였다. 그 인용
("Our method, ACoRN, has shown improved performance over other compression methods.")이 통과했더라도
'ODQA 벤치마크에서'는 그 구절에 없다 (2쪽의 다른 문장 "Validated on three ODQA benchmarks, …"에 있다) — over_broad에 해당한다.
즉 이 사례는 거짓 거부이면서, 통과했다면 인용 범위 초과였을 문장이다. 거부된 문장이라 판정기 사례(CASES)에는 넣지 않았다.
"""

import asyncio

from evals.qa_eval import judge_support

ACORN = "ACoRN (arXiv 2504.12673)"
_ACORN_CORE = "이 논문의 핵심이 뭐야?"
_RECONSTRUCTS = ("Our method, ACoRN, reconstructs the training dataset through offline data augmentation to ensure "
                 "robustness against two types of retrieval noise, as described in Section III-C.")

CASES: list[dict] = [
    {
        "id": "acorn-prod-retrieval-performance", "paper": ACORN, "label": "over_broad", "question": _ACORN_CORE,
        "claim": "이 논문은 ACoRN이라는 훈련 방법을 제안하여, 정보 검색에서 발생하는 노이즈의 영향을 줄이고, "
                 "증거 문서의 요약 및 검색 성능을 향상시키는 것을 목표로 한다.",
        "quotes": [_RECONSTRUCTS],
        "note": "인용문은 노이즈 강건성만 말한다. '요약 능력 향상'은 2쪽의 다른 문장에 있고, '검색 성능 향상'은 모호하다 — "
                "논문은 검색기가 아니라 압축기를 훈련한다 (3쪽 그림 설명에 'retrieve evidential documents'라는 표현은 있다).",
    },
    {
        "id": "acorn-reconstruct-to-keep-information", "paper": ACORN, "label": "over_broad", "question": _ACORN_CORE,
        "claim": "ACoRN은 두 가지 유형의 검색 노이즈를 완화하고, 올바른 답변을 지원하는 정보를 손실하지 않도록 훈련 데이터셋을 재구성한다.",
        "quotes": [_RECONSTRUCTS.removesuffix(", as described in Section III-C.") + "."],
        "note": "정보 손실 감소는 인용문에 없다. 원문에서 그것은 데이터셋 재구성이 아니라 증거 문서 중심의 미세 조정으로 다룬다 (혼동).",
    },
    {
        "id": "acorn-reduces-information-loss", "paper": ACORN, "label": "over_broad", "question": _ACORN_CORE,
        "claim": "이 논문의 핵심은 ACoRN이라는 훈련 방법을 제안하여, 정보 손실을 줄이고 다양한 유형의 검색 노이즈에 대한 강인성을 향상시키는 것이다.",
        "quotes": [_RECONSTRUCTS],
        "note": "'정보 손실을 줄이고'가 인용문에 없다. 내용 자체는 2쪽의 목표 (ii)에 있다 (인용 누락).",
    },
    {
        "id": "acorn-extracts-key-information", "paper": ACORN, "label": "over_broad", "question": _ACORN_CORE,
        "claim": "ACoRN은 검색된 문서에서 핵심 정보를 추출하고 요약하는 데 효과적이며, 특히 높은 노이즈 문서 비율을 가진 데이터셋에서 성능이 향상된다.",
        "quotes": ["Validated on three ODQA benchmarks, it outperforms other methods, especially on datasets with a high noise-document ratio."],
        "note": "뒤 절은 인용문 그대로다. 앞 절(핵심 정보 추출·요약에 효과적)은 인용문에 없다.",
    },
    {
        "id": "acorn-prod-two-noise-types", "paper": ACORN, "label": "supported",
        "question": "ACoRN은 어떤 두 종류의 검색 노이즈를 다루나요?",
        "claim": "ACoRN은 두 가지 유형의 검색 노이즈를 다룹니다: (i) 검색된 문서가 주제적으로 관련이 있지만 잘못된 정보를 포함하는 경우(사실 오류 문서)와 "
                 "(ii) 검색된 문서가 질문에 대한 충분한 정보를 결여하고 있는 경우(무관한 문서).",
        "quotes": ["(i) retrieved documents that are thematically related to the query but contain incorrect informa-tion (Factual error documents), "
                   "and (ii) retrieved documents lacking sufficient information to answer the query (Irrele-vant documents)."],
        "note": "인용문이 문장 전체를 덮는다.",
    },
    {
        "id": "acorn-noise-robust-compression", "paper": ACORN, "label": "supported", "question": _ACORN_CORE,
        "claim": "이 논문의 핵심은 ACoRN이라는 방법을 통해 노이즈에 강한 추상적 압축을 구현하는 것이다.",
        "quotes": ["We propose ACoRN, an effective and efficient noise-robust abstractive compression method."],
        "note": "인용문이 문장 전체를 덮는다.",
    },
    {
        "id": "attention-encoder-layers", "paper": "Attention Is All You Need (arXiv 1706.03762)", "label": "supported",
        "question": "How many identical layers does the encoder stack have?",
        "claim": "The encoder is composed of a stack of N = 6 identical layers.",
        "quotes": ["The encoder is composed of a stack of N = 6 identical layers."],
        "note": "문장과 인용문이 같다.",
    },
    {
        "id": "loraplus-learning-rates", "paper": "LoRA+ (arXiv 2402.12354)", "label": "supported",
        "question": "How does LoRA+ set the learning rates of the adapter matrices A and B?",
        "claim": "In LoRA+, the learning rate of matrix B is set to be λ times that of matrix A, where λ is a fixed value much greater than 1.",
        "quotes": ["In LoRA+, we set the learning rate of B to be λ× that of A, where λ ≫1 is fixed."],
        "note": "인용문이 문장 전체를 덮는다.",
    },
]


async def run() -> None:
    """판정기를 사례마다 한 번씩 부르고, 사람이 붙인 label과 비교한다."""
    verdicts = [await judge_support(case["question"], case["claim"], case["quotes"]) for case in CASES]
    print(f"{'case':42}{'label':12}{'judge':12}match")
    for case, verdict in zip(CASES, verdicts):
        judged = "supported" if verdict.supported else "over_broad"
        print(f"{case['id']:42}{case['label']:12}{judged:12}{'yes' if judged == case['label'] else 'NO'}")
    for label in ("over_broad", "supported"):
        chosen = [(c, v) for c, v in zip(CASES, verdicts) if c["label"] == label]
        hits = sum(("supported" if v.supported else "over_broad") == label for _, v in chosen)
        print(f"judge agrees on {label:10}: {hits}/{len(chosen)}")
    print("\n판정기가 over_broad를 잡지 못하면 qa_eval의 'LLM judge said supported' 수치는 정확성의 근거가 아니다.")


if __name__ == "__main__":
    asyncio.run(run())
