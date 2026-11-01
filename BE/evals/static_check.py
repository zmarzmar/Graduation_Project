"""생성 코드의 정적 검사 — AST로 읽기만 하고 실행하지 않는다.

알 수 있는 것은 두 가지뿐이다: 문법이 맞는가, 어떤 모듈을 import하는가.
패키지가 실제로 설치돼 있는지, 코드가 실행되는지는 이 검사로 판단하지 못한다.
"""

import ast
import sys

# 구현 초안이 써도 따로 확인할 필요가 없는 모듈. 그 밖의 모듈은 '확인 필요'로 표시만 한다 (실패가 아니다)
_KNOWN_MODULES = sys.stdlib_module_names | {"torch", "numpy"}


def static_check(code: str) -> dict:
    """문법 통과 여부와 import 목록을 돌려준다. 코드를 import하거나 실행하지 않는다."""
    result: dict = {"syntax_ok": False, "syntax_error": None, "imports": [], "unverified_imports": []}
    if not code.strip():
        result["syntax_error"] = "코드 없음"
        return result
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError) as e:  # ValueError: 널 바이트가 든 소스
        line = getattr(e, "lineno", None)
        result["syntax_error"] = f"{line}행: {e.msg}" if line else str(e)
        return result

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # 상대 import는 파일 하나짜리 초안에서 풀리지 않는다 — 점을 붙여 그대로 남긴다
            modules.add("." * node.level + (node.module or "") if node.level else node.module.split(".")[0])

    result["syntax_ok"] = True
    result["imports"] = sorted(modules)
    result["unverified_imports"] = sorted(m for m in modules if m not in _KNOWN_MODULES)
    return result
