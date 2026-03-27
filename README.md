# AI Research Analyst

> AI 논문 자동 분석 및 코드 재현 Deep Research Agent — 졸업 프로젝트

---

## Overview

최신 AI 논문을 검색·분석하고 Methods 섹션을 기반으로 PyTorch/TensorFlow 코드 스켈레톤을 자동 생성
LangGraph 멀티 에이전트 파이프라인이 코드를 생성한 뒤 논문 이론과 대조 검증하는 **자기수정 루프**를 실행

---

## Features

- **PDF Upload** — 논문 PDF 업로드 → 자동 분석 및 코드 재현
- **Keyword Search** — 키워드로 arXiv + Semantic Scholar 논문 검색 및 분석
- **Trend Briefing** — 최신 트렌드 논문 Top N 요약 리포트 생성

---

## Tech Stack

- **Frontend**: Next.js, TypeScript, Tailwind CSS, shadcn/ui
- **Backend**: FastAPI, Python 3.11
- **AI**: LangGraph, OpenAI API (`gpt-4o-mini`, `o4-mini`)
- **DB**: ChromaDB, PostgreSQL

---

## Getting Started

```bash
# Backend
cd BE && uv sync
uv run uvicorn main:app --reload  # localhost:8000

# Frontend
cd FE && npm install
npm run dev  # localhost:3000
```

---

## Git Convention

| Emoji | Type | Description |
|-------|------|-------------|
| 🎉 | Start | Start new project |
| ✨ | Feat | Add new feature |
| 🐛 | Fix | Fix a bug |
| 🎨 | Design | Change UI/CSS |
| ♻️ | Refactor | Refactor code |
| 🔧 | Settings | Change configuration files |
| 🔥 | Remove | Delete files |
| 📝 | Docs | Update documentation |

커밋 형식: `{emoji} {Type}: {Description}`
