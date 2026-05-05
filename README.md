# 🐣 국립어린이과학관 AI 가이드

Streamlit Cloud 배포용 버전입니다.

## 🚀 Features

- 🤖 AI 챗봇 가이드 (GPT-4o-mini 기반, ReAct Agent + RAG)
- 🎤 음성 입출력 지원 (Whisper STT / OpenAI TTS)
- 🌏 다국어 지원 (한국어, English, 日本語, 中文)
- 📚 사후 학습 시스템 "또만나 놀이터" (퀴즈, 궁금해요!, 과학동화 + 오디오북)
- 🔍 RAG 기반 전시물 검색 (ChromaDB + CSV fallback)
- 💬 Google Forms 연동 피드백 설문 (어린이/성인 모드 분기)

## 🛠️ Recent Fixes (2026.05)

- 사이드바 UI 단순화: 디버그 옵션 숨김, 음성 기능 항상 활성화
- 사이드바 레이아웃 정리: 앱 제목 메인 화면 이동, 설문조사 하단 고정
- 사이드바 구분선 정리: 불필요한 중복 구분선 제거
- 다국어 Google Form URL 지원 (언어별 폼 분리 가능)
- Whisper STT 언어 자동 감지 (한국어 고정 → auto-detect)
- 챗봇 대화 맥락 전달 버그 수정: `st.session_state.messages`에서 이전 대화 직접 포함
- 또만나 놀이터 "궁금해요!" 다국어 답변 지시 + 오류 처리 강화
- 학습(`learning.py`) 및 음성(`voice.py`) 모듈 문법 오류 수정

## 📦 Requirements

- Python 3.11+
- OpenAI API Key

## 🔧 Setup

Streamlit Cloud Secrets에 다음을 추가하세요:

```toml
OPENAI_API_KEY = "your-api-key-here"
```

## 📱 Main File

`app_with_voice.py`

## 📝 Feedback Form

피드백 설문 URL: https://forms.gle/UvRfnMEwjUEZgFJJ8
