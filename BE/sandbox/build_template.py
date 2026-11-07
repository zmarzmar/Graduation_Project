"""생성 코드 실행 검사용 E2B 템플릿을 만든다 (한 번만, 패키지를 바꿀 때만 다시).

실행: cd BE && uv run python -m sandbox.build_template
E2B_API_KEY가 .env에 있어야 한다. 실행 중에는 네트워크를 끄므로 필요한 패키지는 여기서 모두 설치한다.
검사 runner는 템플릿에 굽지 않고 실행마다 올린다 — runner를 고쳐도 템플릿을 다시 만들 필요가 없다.
버전을 고정한다 (uv.lock은 템플릿 안을 고정하지 않는다). 운영이 쓰는 빌드: template_id eq84nicqa9hyjmenbmud,
build cdfc4b0b-2296-4aa6-90ab-ca8cdb87d07d (2026-11-06) — 그 안의 버전을 2026-11-07 실행으로 확인해 아래에 적었다.
다시 만들면 새 build ID를 여기와 eval-results/2026-11-exec-check/E2B_PLAN.txt에 적는다.
"""

from dotenv import load_dotenv
from e2b import Template, default_build_logger

TEMPLATE_NAME = "paperpilot-exec-check"
TORCH_VERSION = "2.15.0+cpu"
NUMPY_VERSION = "2.4.6"
CPU_COUNT = 2
MEMORY_MB = 2048

template = (
    Template()
    .from_python_image("3.11")
    # root로 시스템 전체에 설치한다 — 사용자 폴더(~/.local)에 깔리면 격리 모드(python -I)에서 보이지 않는다
    .run_cmd(f"pip install --no-cache-dir torch=={TORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu"
             f" && pip install --no-cache-dir numpy=={NUMPY_VERSION}",
             user="root")
)

if __name__ == "__main__":
    load_dotenv()
    info = Template.build(template, TEMPLATE_NAME, cpu_count=CPU_COUNT, memory_mb=MEMORY_MB,
                          on_build_logs=default_build_logger())
    print(info)
