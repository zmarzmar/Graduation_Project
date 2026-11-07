# 생성 코드 실행 검사용 이미지 (평가 전용, 운영은 E2B 템플릿 sandbox/build_template.py)
# 빌드 (BE에서): docker build -t paperpilot-exec-check:dev -f evals/exec_check/Dockerfile .
# 네트워크 없이 돌리므로 필요한 패키지는 여기서 모두 설치한다. 실제 버전은 runner가 결과에 남긴다
FROM python:3.11-slim
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir numpy
COPY sandbox/runner.py /runner/runner.py
USER 65534:65534
WORKDIR /tmp
ENTRYPOINT ["python", "-I", "/runner/runner.py"]
