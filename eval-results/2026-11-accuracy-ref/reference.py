"""핵심 계산의 기준 구현과 고의 오답 — 생성 코드와 무관하게 논문 식으로만 쓴다 (사전 등록, SPEC.txt).

실행: python3 -I reference.py   (numpy만 쓴다. 기준 구현은 통과하고 고의 오답은 실패해야 한다)
생성 코드에 적용하는 것은 run_targets.py가 한다. 이 파일의 기대값·입력·허용 오차는 생성 코드 결과를 본 뒤에 바꾸지 않는다.
"""

import numpy as np

RTOL, ATOL = 1e-5, 1e-5  # 생성 코드가 float32로 계산해도 통과할 수 있는 크기. 값은 O(1)~O(100)


# ── Rotated LoRA R2: 리만 기울기 성분 (식 11) ─────────────────────────────
def rlora_components(G: np.ndarray, U: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """S = Uᵀ G V,  Up = G V − U S,  Vp = Gᵀ U − V Sᵀ  (§3.1 식 11, p3)"""
    S = U.T @ G @ V
    return S, G @ V - U @ S, G.T @ U - V @ S.T


def rlora_mutants() -> dict:
    def s_not_transposed(G, U, V):  # Vp에 Sᵀ 대신 S
        S = U.T @ G @ V
        return S, G @ V - U @ S, G.T @ U - V @ S

    def missing_us(G, U, V):  # Up에서 US 항 누락 (유클리드 기울기 그대로)
        S = U.T @ G @ V
        return S, G @ V, G.T @ U - V @ S.T

    return {"Vp에 Sᵀ 대신 S": s_not_transposed, "Up의 US 누락": missing_us}


RLORA_HAND = {
    "G": np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=float),
    "U": np.eye(3)[:, :2], "V": np.eye(4)[:, :2],
    # 손계산 (SPEC.txt): S = G의 왼쪽 위 2×2, Up = 셋째 행만 남은 GV, Vp = 셋째·넷째 행만 남은 GᵀU
    "S": np.array([[1, 2], [5, 6]], dtype=float),
    "Up": np.array([[0, 0], [0, 0], [9, 10]], dtype=float),
    "Vp": np.array([[0, 0], [0, 0], [3, 7], [4, 8]], dtype=float),
}


def rlora_random(seed: int = 0) -> dict:
    """직교 U·V가 좌표축이 아닌 입력 (m=5, n=4, r=2)"""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((5, 2)))
    V, _ = np.linalg.qr(rng.standard_normal((4, 2)))
    return {"G": rng.standard_normal((5, 4)), "U": U, "V": V}


def rlora_projection_identity(G: np.ndarray, U: np.ndarray, V: np.ndarray, comps: tuple) -> bool:
    """식 (10)의 사영과 식 (8)의 재구성이 같은지 — 기준 구현 자체를 원문의 다른 식으로 확인한다"""
    S, Up, Vp = comps
    m, n = G.shape
    proj = G - (np.eye(m) - U @ U.T) @ G @ (np.eye(n) - V @ V.T)
    return np.allclose(U @ S @ V.T + Up @ V.T + U @ Vp.T, proj, rtol=1e-10, atol=1e-10)


# ── AVQ V3: 코드워드별 값 합·개수 (식 3, 알고리즘 1, 식 15) ─────────────────
def avq_aggregate(K: np.ndarray, V: np.ndarray, C: np.ndarray, M0: int, n_children: int) -> tuple[np.ndarray, np.ndarray]:
    """알고리즘 1 (p9): 각 키를 가장 가까운 부모에 더하고(8행), 그 부모의 자식 중 가장 가까운 자식이
    부모보다 가까우면 그 자식에도 더한다(16행). 부모 집계에는 자식으로 간 키도 남는다 — 식 (15)의 V̄_stay = V̄_p − Σ V̄_c.
    K, V: [N, D], C: [M0 + M0·n_children, D] (부모 m의 자식은 M0 + m·n_children부터)"""
    total = C.shape[0]
    Vbar, n = np.zeros((total, V.shape[1])), np.zeros(total)
    for k, v in zip(K, V):
        d_parent = ((C[:M0] - k) ** 2).sum(1)
        a = int(d_parent.argmin())
        Vbar[a] += v
        n[a] += 1
        c0 = M0 + a * n_children
        d_child = ((C[c0:c0 + n_children] - k) ** 2).sum(1)
        c = int(d_child.argmin())
        if d_child[c] < d_parent[a]:
            Vbar[c0 + c] += v
            n[c0 + c] += 1
    return Vbar, n


def avq_mutants() -> dict:
    def exclude_moved(K, V, C, M0, nc):  # 자식으로 간 키를 부모 집계에서 뺀다
        Vbar, n = avq_aggregate(K, V, C, M0, nc)
        for m in range(M0):
            ch = slice(M0 + m * nc, M0 + (m + 1) * nc)
            Vbar[m] -= Vbar[ch].sum(0)
            n[m] -= n[ch].sum()
        return Vbar, n

    def mean_not_sum(K, V, C, M0, nc):  # 합 대신 평균
        Vbar, n = avq_aggregate(K, V, C, M0, nc)
        return Vbar / np.maximum(n, 1)[:, None], n

    return {"부모에서 자식으로 간 키 제외": exclude_moved, "합 대신 평균": mean_not_sum}


AVQ_HAND = {
    # 부모 2개(M0=2), 부모마다 자식 1개. 부모0=(0,0), 부모1=(10,0), 부모0의 자식=(2,0), 부모1의 자식=(10,5)
    "C": np.array([[0, 0], [10, 0], [2, 0], [10, 5]], dtype=float), "M0": 2, "n_children": 1,
    # k1은 부모0에 남고, k2는 부모0의 자식으로, k3은 부모1에 남고, k4는 부모1의 자식으로 간다
    "K": np.array([[0, 0.1], [1.5, 0], [9, 0], [10, 4]], dtype=float),
    "V": np.array([[1, 2], [3, 5], [7, 11], [13, 17]], dtype=float),
    # 손계산: 부모0 = v1+v2 = (4,7) n=2, 부모1 = v3+v4 = (20,28) n=2, 자식0 = v2 = (3,5) n=1, 자식1 = v4 = (13,17) n=1
    "Vbar": np.array([[4, 7], [20, 28], [3, 5], [13, 17]], dtype=float),
    "n": np.array([2, 2, 1, 1], dtype=float),
}


# ── Explicit Dropout D6: FFN 선형층의 정규화 항 (식 19, 부록 A.6 식 93–100) ─────
def ffn_regularizer(Z: np.ndarray, W: np.ndarray, p: float) -> float:
    """J = ½ · Tr((Zᵀ Z ⊙ p²) Wᵀ W) = ½ p² ‖Z Wᵀ‖_F²   (Z: 그 선형층의 입력 [tokens, in], W: [out, in] — 부록 A.6 식 93의 H = X Wᵀ)
    토큰 수로 나누지 않는다 — 식 (4)·(13)의 B는 마스크(에폭) 평균이고 XᵀX는 토큰에 대한 합이다"""
    return 0.5 * p ** 2 * float(np.trace(Z.T @ Z @ W.T @ W))


def dropout_mutants() -> dict:
    return {
        "토큰 수로 나눔": lambda Z, W, p: ffn_regularizer(Z, W, p) / Z.shape[0],
        "p² 대신 p": lambda Z, W, p: ffn_regularizer(Z, W, p) / p,
        # 두 번째 층에 첫 번째 층 입력의 Gram — 차원이 다르면 계산할 수 없다 (실패로 센다)
        "2층에 1층 입력 Gram": lambda Z, W, p: ffn_regularizer(Z[:, : W.shape[1] - 1], W, p),
    }


DROPOUT_HAND = {
    # 2층 입력 Z (토큰 2개, hidden 3), 2층 가중치 W (out 2 × in 3), p = 0.5
    "Z": np.array([[1, 0, 2], [0, 1, 1]], dtype=float),
    "W": np.array([[1, 0, 1], [0, 2, 0]], dtype=float),
    "p": 0.5,
    # 손계산: Z Wᵀ = [[3,0],[1,2]] → ‖·‖² = 9+0+1+4 = 14 → J = ½ · 0.25 · 14 = 1.75
    "J": 1.75,
}


def _close(a, b) -> bool:
    try:
        return all(np.allclose(np.asarray(x), np.asarray(y), rtol=RTOL, atol=ATOL) for x, y in zip(a, b))
    except Exception:
        return False


def self_check() -> list[str]:
    """기준 구현은 손계산·원문의 다른 식과 맞고, 고의 오답은 모두 실패하는지. 문제 목록을 돌려준다 (비면 통과)."""
    problems: list[str] = []
    h = RLORA_HAND
    if not _close(rlora_components(h["G"], h["U"], h["V"]), (h["S"], h["Up"], h["Vp"])):
        problems.append("RLoRA 기준 구현이 손계산과 다름")
    r = rlora_random()
    if not rlora_projection_identity(r["G"], r["U"], r["V"], rlora_components(r["G"], r["U"], r["V"])):
        problems.append("RLoRA 기준 구현이 식 (10) 사영과 다름")
    for name, f in rlora_mutants().items():
        if _close(f(h["G"], h["U"], h["V"]), (h["S"], h["Up"], h["Vp"])) and \
                _close(f(r["G"], r["U"], r["V"]), rlora_components(r["G"], r["U"], r["V"])):
            problems.append(f"RLoRA 고의 오답이 통과: {name}")

    a = AVQ_HAND
    ref = avq_aggregate(a["K"], a["V"], a["C"], a["M0"], a["n_children"])
    if not _close(ref, (a["Vbar"], a["n"])):
        problems.append("AVQ 기준 구현이 손계산과 다름")
    for name, f in avq_mutants().items():
        if _close(f(a["K"], a["V"], a["C"], a["M0"], a["n_children"]), (a["Vbar"], a["n"])):
            problems.append(f"AVQ 고의 오답이 통과: {name}")

    d = DROPOUT_HAND
    if not np.isclose(ffn_regularizer(d["Z"], d["W"], d["p"]), d["J"], rtol=RTOL, atol=ATOL):
        problems.append("Dropout 기준 구현이 손계산과 다름")
    for name, f in dropout_mutants().items():
        try:
            passed = np.isclose(f(d["Z"], d["W"], d["p"]), d["J"], rtol=RTOL, atol=ATOL)
        except Exception:
            passed = False
        if passed:
            problems.append(f"Dropout 고의 오답이 통과: {name}")
    return problems


if __name__ == "__main__":
    found = self_check()
    print("기준 구현 자체 검사:", "통과" if not found else found)
    raise SystemExit(1 if found else 0)
