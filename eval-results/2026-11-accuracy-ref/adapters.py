"""대상 코드와 기준 입력을 잇는 어댑터 — SPEC.txt 4절 규칙: 이름·인자·축 대응, 상태 주입, 출력·중간값 추출만 한다.
검사 대상 값(집계·Gram·성분)은 여기서 계산하지 않는다. 연결할 이름이 없으면 AdapterError(판단 불가)를 낸다.

각 함수는 (대상 모듈의 전역 dict, torch, 입력) → 대상 코드가 낸 값(numpy)을 돌려준다.
"""

import numpy as np


class AdapterError(Exception):
    """연결할 수 없음 — '평가 측 문제로 판단 불가'로 분류한다."""


def _get(ns: dict, name: str):  # noqa: ANN202
    if name not in ns:
        raise AdapterError(f"대상 코드에 {name}이(가) 없습니다")
    return ns[name]


def rlora(ns: dict, torch, G: np.ndarray, U: np.ndarray, V: np.ndarray) -> tuple:  # noqa: ANN001
    """_euclidean_grad_to_tangent_components(x, grad_output, U, V)는 G = grad_outputᵀ x를 쓴다 (코드 주석).
    x = 단위행렬(n×n), grad_output = Gᵀ로 넣으면 G가 그대로 전달된다 — 입력 대응일 뿐 성분 계산은 대상 코드가 한다."""
    fn = _get(ns, "_euclidean_grad_to_tangent_components")
    t = lambda a: torch.tensor(a, dtype=torch.float64)  # noqa: E731
    out = fn(t(np.eye(G.shape[1])), t(G.T.copy()), t(U), t(V))
    return tuple(getattr(out, k).detach().numpy() for k in ("S", "Up", "Vp"))


def avq(ns: dict, torch, K: np.ndarray, V: np.ndarray, C: np.ndarray, M0: int, n_children: int) -> tuple:  # noqa: ANN001
    """HierarchicalCodebook에 코드북을 주입하고 quantize_and_aggregate의 집계(attention이 쓰는 aggregated_values/counts)를 꺼낸다.
    대상 코드의 배치: 부모 [H, M0, D], 자식 [H, M0, C, D], 집계 인덱스는 부모 다음에 num_parents + 부모·C + c (기준과 같다)."""
    cls = _get(ns, "HierarchicalCodebook")
    D = C.shape[1]
    cb = cls(num_heads=1, num_parents=M0, num_children=n_children, dim=D).double()
    with torch.no_grad():
        cb.parent_codes.copy_(torch.tensor(C[:M0]).view(1, M0, D))
        cb.child_codes.copy_(torch.tensor(C[M0:]).view(1, M0, n_children, D))
    stats = cb.quantize_and_aggregate(torch.tensor(K).view(1, 1, -1, D), torch.tensor(V).view(1, 1, -1, D))
    return stats.aggregated_values[0, 0].detach().numpy(), stats.aggregated_counts[0, 0].detach().numpy()


def dropout(ns: dict, torch, p: float, tokens: int, d: int, h: int, seed: int) -> tuple[float, list]:  # noqa: ANN001
    """FFN 모듈(대상 코드의 초기화, seed 고정)과 입력을 만들고, 대상 코드의 feedforward_regularizer 값과
    forward hook으로 꺼낸 각 선형층의 실제 입력·가중치를 돌려준다 (기준값은 run_targets.py가 이 입력으로 reference에서 계산)."""
    ffn_cls = _get(ns, "FeedForwardNetworkExplicit")
    reg_cls = _get(ns, "ExplicitDropoutRegularizer")
    torch.manual_seed(seed)
    ffn = ffn_cls(embed_dim=d, hidden_dim=h).double()
    x = torch.randn(1, tokens, d, dtype=torch.float64)
    captured: dict = {}
    hooks = [lin.register_forward_hook(lambda m, inp, out, key=key: captured.__setitem__(key, inp[0].detach()))
             for key, lin in (("ff1", ffn.ff1), ("ff2", ffn.ff2))]
    out = ffn(x)
    for hk in hooks:
        hk.remove()
    reg = reg_cls(drop_prob=p)
    fn = reg.feedforward_regularizer
    # 최종본은 forward가 (출력, hidden)을 돌려주고 정규화에 hidden_x를 받는다 — 대상 코드 자신의 경로를 그대로 쓴다
    kwargs = {"hidden_x": out[1]} if isinstance(out, tuple) and "hidden_x" in fn.__code__.co_varnames else {}
    value = float(fn(x, ffn.ff1.weight, ffn.ff2.weight, **kwargs))
    layers = [(captured[k][0].numpy(), lin.weight.detach().numpy()) for k, lin in (("ff1", ffn.ff1), ("ff2", ffn.ff2))]
    return value, layers
