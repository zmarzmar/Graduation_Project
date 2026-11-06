"""생성 코드 실행 검사용 E2B 템플릿을 만든다 (한 번만, 패키지를 바꿀 때만 다시).

실행: cd BE && uv run python -m sandbox.build_template
E2B_API_KEY가 .env에 있어야 한다. 실행 중에는 네트워크를 끄므로 필요한 패키지는 여기서 모두 설치한다.
검사 runner는 템플릿에 굽지 않고 실행마다 올린다 — runner를 고쳐도 템플릿을 다시 만들 필요가 없다.
"""

from dotenv import load_dotenv
from e2b import Template, default_build_logger

TEMPLATE_NAME = "paperpilot-exec-check"
CPU_COUNT = 2
MEMORY_MB = 2048

template = (
    Template()
    .from_python_image("3.11")
    # root로 시스템 전체에 설치한다 — 사용자 폴더(~/.local)에 깔리면 격리 모드(python -I)에서 보이지 않는다
    .run_cmd("pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir numpy",
             user="root")
)

if __name__ == "__main__":
    load_dotenv()
    info = Template.build(template, TEMPLATE_NAME, cpu_count=CPU_COUNT, memory_mb=MEMORY_MB,
                          on_build_logs=default_build_logger())
    print(info)
