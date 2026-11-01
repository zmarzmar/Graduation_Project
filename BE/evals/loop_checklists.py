"""자기수정 루프 평가의 논문별 체크리스트 — 생성 결과를 보기 전에 확정했다 (2026-10-27, 사용자 확정).

loop_eval.py의 결과에서 최초본(1회차)과 최종본(마지막 회차) 코드를 사람이 이 기준으로 판정한다.

판정 규칙
- 충족 / 미충족 두 가지. 일부만 구현된 것은 미충족으로 하고 빠진 것을 메모한다
- TODO·pass·NotImplementedError로 남긴 부분, 주석으로 수식만 적은 부분은 미충족
- PyTorch 내장 모듈에 맡긴 경우(예: nn.MultiheadAttention)는 충족으로 치되 '위임'으로 따로 표시한다
- 최초본과 최종본에 같은 항목·같은 조건을 적용한다
- 최종본에서 새로 생긴 문제를 따로 적는다: 충족 → 미충족으로 바뀐 항목, 새 문법 오류, 새 '확인 필요' import
- 범위는 Coder가 지시받은 Methods 섹션의 구현이다. 학습 설정(옵티마이저·warmup·label smoothing)과
  하이퍼파라미터 기본값(N=6, d_model=512, r 값)은 보지 않는다

근거의 쪽수는 arXiv PDF에서 추출한 텍스트로 확인했다. 절 번호와 문장의 대응은 체크리스트를 쓴 쪽이 읽고 붙였다.
"""

# (id, 요구사항, 논문 근거, 충족 조건)
CHECKLISTS: dict[str, list[tuple[str, str, str, str]]] = {
    "attention": [
        ("A1", "Scaled dot-product attention", "§3.2.1 식 (1), 4쪽",
         "QKᵀ를 √d_k로 나눈 뒤 softmax를 취하고 V에 곱한다. 나눗셈이 없거나 d_k가 아닌 값으로 나누면 미충족"),
        ("A2", "Multi-head attention", "§3.2.2, 5쪽",
         "Q·K·V를 각각 선형 투영하고 h개 head로 나눠(head 차원 = d_model/h) attention 뒤 이어 붙여 출력 선형층을 통과시킨다"),
        ("A3", "Decoder의 미래 위치 가리기", "§3.1 3쪽, §3.2.3 5쪽",
         "미래 위치를 가리는 마스크를 만드는 코드가 있고, decoder self-attention에서 softmax 전에 −∞(또는 동등한 값)로 적용한다"),
        ("A4", "Residual + LayerNorm", "§3.1, 3쪽",
         "encoder·decoder의 모든 sublayer가 LayerNorm(x + Sublayer(x)) 순서다. pre-norm은 미충족"),
        ("A5", "Position-wise FFN", "§3.3 식 (2), 5쪽",
         "선형 → ReLU → 선형이고, 안쪽 차원(d_ff)을 d_model과 따로 둔다"),
        ("A6", "사인·코사인 positional encoding", "§3.5, 6쪽",
         "짝수 차원 sin, 홀수 차원 cos, 주파수 10000^(2i/d_model)이고 임베딩에 더한다"),
        ("A7", "Encoder-decoder 구조", "§3.1 3쪽, §3.2.3 5쪽",
         "encoder layer는 sublayer 2개, decoder layer는 3개. decoder의 가운데 sublayer는 Q를 decoder에서, K·V를 encoder 출력에서 받는다"),
        ("A8", "임베딩에 √d_model 곱하기", "§3.4, 5쪽",
         "토큰 임베딩 출력에 √d_model을 곱한다"),
    ],
    "lora": [
        ("L1", "저랭크 갱신 forward", "§4.1 식 (3), 4쪽",
         "원래 층의 출력에 BAx를 더한다. B는 d×r, A는 r×k 모양"),
        ("L2", "사전학습 가중치 동결", "§4.1, 4쪽",
         "W₀가 gradient를 받지 않게 하는 코드가 있고(requires_grad=False 등) A·B만 학습 대상이다"),
        ("L3", "초기화", "§4.1, 4쪽",
         "B는 0, A는 무작위(0이 아님)로 초기화해 시작 시 BA = 0. A가 가우시안이 아니면(예: Kaiming) 충족으로 치되 메모한다"),
        ("L4", "α/r 스케일", "§4.1, 4쪽",
         "ΔWx(= BAx)에 α/r을 곱한다"),
        ("L5", "추론용 병합", "§4.1 'No Additional Inference Latency', 4쪽",
         "W = W₀ + BA(스케일 포함)로 합치는 기능이 있다"),
        ("L6", "적용 대상", "§4.2 5쪽, §5 6쪽",
         "LoRA를 attention의 query·value 투영(W_q, W_v)에 적용하는 코드가 있다. 범용 LoRA 층만 있고 어디에 붙이는지 없으면 미충족"),
    ],
}
