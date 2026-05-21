# 국립어린이과학관 AI 가이드 — 아키텍처 문서

> **검토 목적**: 이 문서는 앱 설계 전반에 대한 전문가 피드백을 받기 위해 작성되었습니다.  
> **최종 업데이트**: 2026-05-21  
> **배포 환경**: Streamlit Cloud  

---

## 1. 프로젝트 개요

국립어린이과학관(서울 종로구) 방문객을 위한 **AI 챗봇 가이드** 앱입니다.  
어린이·학부모를 주 대상으로 하며, 방문 안내부터 전시 학습까지 원스톱으로 지원합니다.

### 핵심 기능

| 기능 | 설명 |
|---|---|
| **챗봇 안내** | 운영시간·요금·예약·동선 등 FAQ + RAG 기반 자유 질의 |
| **음성 입출력** | 마이크 녹음(STT) + 답변 음성 출력(TTS) |
| **또만나 놀이터** | 사후 학습 시스템 — 퀴즈, 궁금해요!, 과학동화+오디오북 |
| **다국어 지원** | 한국어 / English / 日本語 / 中文 |
| **어린이·성인 모드** | UI 언어 난이도·이모지·어조 분기 |

---

## 2. 기술 스택

| 레이어 | 기술 | 용도 |
|---|---|---|
| **프레임워크** | Streamlit ≥ 1.30 | UI 렌더링, 세션 관리, 배포 |
| **LLM** | OpenAI GPT-4o-mini | 채팅 에이전트, 퀴즈/동화 생성, 번역, 의도 분류 |
| **Agent** | LangGraph ReAct Agent + MemorySaver | 도구 호출·대화 메모리 관리 |
| **RAG** | LangChain + ChromaDB + text-embedding-3-small | 전시물 정보 벡터 검색 |
| **STT** | OpenAI Whisper (whisper-1) | 음성 → 텍스트 |
| **TTS** | ElevenLabs (1순위) → edge-tts (2순위) → OpenAI tts-1 (3순위) | 텍스트 → 음성 |
| **분석** | Google Analytics 4 (Measurement Protocol, 서버사이드) | 이벤트 트래킹 |
| **피드백** | Google Forms + Google Sheets (gspread) | 사용자 만족도 수집 |
| **호스팅** | Streamlit Cloud | 서버리스 배포 |

---

## 3. 파일 구조

```
cloud-deployment/
├── app_with_voice.py      # 메인 앱 (UI, 라우팅, 에이전트 실행)   ~1,380줄
├── core.py                # 규칙 기반 핸들러, RAG 초기화, 도구 정의 ~4,060줄
├── learning.py            # 또만나 놀이터 (퀴즈/동화/Q&A)          ~2,400줄
├── voice.py               # STT/TTS 처리                           ~273줄
├── static_translations.py # 다국어 UI 텍스트 상수                   ~57,000자
├── requirements.txt       # 의존성
├── data/
│   ├── 행동놀이터.csv       # 전시물 데이터 (존별 1개)
│   ├── AI놀이터.csv
│   ├── 탐구놀이터.csv
│   └── ... (총 ~15개 존)
└── .streamlit/
    └── secrets.toml        # API 키 (git 제외)
```

---

## 4. 아키텍처 — 질의 처리 흐름

```
사용자 입력 (텍스트 or 음성)
       │
       ▼ (음성일 경우 Whisper STT 먼저 수행)
┌─────────────────────────────────────────────────────┐
│              route_intent()                         │
│  ┌───────────────────────────────────────────────┐  │
│  │ 1. clear_faq_keywords 목록 매칭               │  │
│  │    → intent = "notice" (즉시 규칙 응답)       │  │
│  │ 2. 나이 패턴/날짜/위치 등 정규식              │  │
│  │    → intent = "basic" (카테고리별 규칙 응답)  │  │
│  │ 3. 애매한 질문 → LLM 의도 분류(캐시 1h)       │  │
│  │    → intent = "llm_agent"                    │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
       │
  ┌────┴─────────┐
  ▼              ▼
"notice"/"basic"  "llm_agent"
  │               │
  ▼               ▼
answer_rule_based()   RAG 검색 (ChromaDB k=3)
(~70% 질문,              + LangGraph ReAct Agent
 LLM 호출 없음)          + GPT-4o-mini 스트리밍
  │               │
  └────┬──────────┘
       ▼
   최종 답변 출력
   (TTS 옵션: ElevenLabs → edge-tts → OpenAI)
```

---

## 5. 규칙 기반 핸들러 분류 목록

`classify_basic_category()`에서 처리되는 카테고리 (~30개):

| 카테고리 | 예시 질문 |
|---|---|
| `operating_hours` | 몇 시에 열어요? |
| `admission_fee` | 입장료 얼마예요? |
| `reservation_guide` | 예약 어떻게 해요? |
| `route_by_age` | 7살 어디 보면 좋아요? / 추천 동선 |
| `today_programs` | 오늘 프로그램 뭐 있어요? |
| `floor_guide` | 층별 안내 |
| `directions` | 오시는 길 |
| `parking` | 주차 |
| `facility_amenities` | 유모차 대여, 수유실, 물품보관함 |
| `science_show` | 과학쇼, 로봇쇼, 사이언스랩 |
| `planetarium` | 천체투영관 예약 |
| `robot_show` | 로봇쇼 시간 |
| `education_guide` | 교육 프로그램 안내 |
| `food_drink` | 도시락 반입, 매점 |
| `umbrella_rental` | 우산 빌려줘 |
| `lost_found` | 분실물 |
| `wifi_info` | 와이파이 |
| `pet_policy` | 반려동물 |
| `wheelchair_stroller` | 휠체어, 유모차 |
| `phone_charging` | 핸드폰 충전 |
| `late_arrival` | 늦게 도착하면 |
| ... | 등 총 ~30개 |

---

## 6. LLM 사용 현황

| 위치 | 모델 | 용도 | 캐시 |
|---|---|---|---|
| 메인 채팅 에이전트 | gpt-4o-mini | RAG + 대화 답변 | ✗ (스트리밍) |
| 퀴즈 생성 | gpt-4o-mini | 전시물 기반 퀴즈 | 세션 내 캐시 |
| 과학동화 생성 | gpt-4o-mini | 어린이 과학 스토리 | ✗ |
| 의도 분류 | gpt-4o-mini | 애매한 질문 분류 | 1시간 |
| 규칙 답변 번역 | gpt-4o-mini | 한국어 → 다국어 | 24시간 |
| 키워드 추출 (폴백) | gpt-4o-mini | CSV 존 키워드 추출 | 24시간 |
| 키워드 번역 | gpt-4o-mini | 한국어 → 다국어 | 24시간 |
| 임베딩 | text-embedding-3-small | RAG 벡터화 | 세션 |
| STT | whisper-1 | 음성 → 텍스트 | ✗ |
| TTS (3순위 폴백) | tts-1 | 텍스트 → 음성 | 세션 |

---

## 7. TTS 우선순위 체계

```
text_to_speech() 호출
  │
  ▼
1) ElevenLabs     ← ELEVENLABS_API_KEY 있을 때 (유료, 가장 자연스러운 음성)
   └─ 실패/키 없음 ↓
2) edge-tts       ← Microsoft Neural TTS (무료, 빠름, 기본값)
   └─ 실패 ↓
3) OpenAI tts-1   ← 최종 폴백 (유료)
```

**언어별 음성 설정**:

| 언어 | ElevenLabs Voice ID | edge-tts |
|---|---|---|
| 한국어 | uyVNoMrnUku1dZyVEXwD | ko-KR-InJoonNeural |
| English | 8LVfoRdkh4zgjr8v5ObE | en-US-AriaNeural |
| 日本語 | 3JDquces8E8bkmvbh6Bc | ja-JP-NanamiNeural |
| 中文 | vZZLclMx4wouUtKBRfZn | zh-CN-XiaoxiaoNeural |

---

## 8. RAG 시스템

- **벡터 DB**: ChromaDB (Streamlit Cloud에서 in-memory, 영구 저장 없음)
- **임베딩**: text-embedding-3-small
- **검색**: `similarity_search(query, k=3)`
- **데이터 소스**: `data/` 폴더 내 CSV 파일들 (전시물 설명, FAQ, 운영 안내 등)
- **초기화**: `@st.cache_resource(ttl=3600)` — 1시간마다 재구축
- **폴백**: RAG 실패 시 ChromaDB 없이 직접 CSV에서 유사 행 검색

### RAG 컨텍스트 구성

```
System Prompt (동적 생성, 사용자 모드·언어 반영)
+ [RAG 배경지식] similarity_search 결과 3개
+ 대화 히스토리 최근 10턴
+ 사용자 입력 (+ 언어 오버라이드 지시)
```

---

## 9. 또만나 놀이터 (사후 학습 시스템)

### 9-1. 퀴즈 생성 흐름

```
사용자가 존 선택 → 존 키워드 버튼 표시 (CSV 제목 컬럼 기반, 최대 12개)
  → 키워드 선택
  → quiz_detail 구성:
     ① 선택 전시물 [전시물 설명] + [체험 방법]
     ② 데이터 부실(< 200자)이면 같은 분류(category)의 이웃 전시물 자동 보충 (최대 500자)
  → generate_quiz(principle, exhibit_detail, prev_questions, variation_seed)
  → GPT-4o-mini → JSON (question, options, answer, explanation)
```

**품질 규칙**:
- 이전 3개 문제 이력 전달 → 반복 방지
- "제공된 전시물 데이터 외 과학 개념 창작 금지" 규칙 프롬프트에 명시
- 난이도 분기: 유아(유치~초등 저학년) / 초등(3~6학년)

### 9-2. 과학동화 생성

- 선택된 존(들)의 전시물 데이터에서 원리 추출
- `exhibit_summary` 구성: content/detail 있는 전시물 우선, 없으면 제목 목록으로 폴백
- "목록에 없는 전시물·과학 개념 창작 금지" 규칙 포함
- 동화 + TTS 오디오북 자동 생성

### 9-3. 키워드 버튼 추출 우선순위

1. CSV `제목` 컬럼에서 한/영 분리 추출 (가장 빠름, 캐시 불필요)
2. GPT-4o-mini LLM 키워드 추출 (1번 폴백, 24h 캐시)
3. 단어 빈도 기반 추출 (최종 폴백)

---

## 10. 주중/주말 분기 로직

`route_by_age` 핸들러에서 현재 시각(KST)을 기준으로 분기:

| 요일 | 동작 |
|---|---|
| 월요일 | 휴관일 안내 배너 표시 |
| 화~금 (평일) | "로봇순회(14:30)" 강조, "관람객 여유" 안내 |
| 토~일 (주말) | "일찍 도착 권장", "예약 필수" 강조 |
| 방학 월 (1, 2, 8월) | 방학 특수 운영 안내 추가 |

---

## 11. 외부 서비스 및 Secrets 구성

| 서비스 | Secret 키 | 필수 여부 | 비고 |
|---|---|---|---|
| OpenAI | `OPENAI_API_KEY` | **필수** | LLM, Whisper, TTS 폴백 |
| ElevenLabs | `ELEVENLABS_API_KEY` | 선택 | TTS 1순위, 없으면 edge-tts 사용 |
| ElevenLabs | `ELEVENLABS_VOICE_ID` | 선택 | 없으면 언어별 기본 voice_id 사용 |
| ElevenLabs | `ELEVENLABS_MODEL_ID` | 선택 | 없으면 eleven_multilingual_v2 |
| Google Analytics 4 | `GA4_API_SECRET` | 선택 | 없으면 이벤트 전송 스킵 |
| Google Analytics 4 | `GA4_MEASUREMENT_ID` | 선택 | 없으면 하드코딩 ID 사용 |
| Google Sheets | `gcp_service_account` | 선택 | 피드백 스프레드시트 저장 |
| Google Sheets | `app.feedback_sheet_id` | 선택 | 없으면 Sheets 저장 스킵 |

---

## 12. 성능 최적화 현황

| 최적화 | 방법 |
|---|---|
| 규칙 기반 우선 처리 | ~70% 질문은 LLM 없이 즉시 응답 |
| 의도 분류 캐시 | 동일 문장 1시간 캐시 (`@st.cache_data(ttl=3600)`) |
| 번역 캐시 | 규칙 기반 답변 번역 24시간 캐시 |
| 키워드 캐시 | 존 키워드 추출·번역 24시간 캐시 |
| TTS 캐시 | 답변별 음성 세션 내 캐시 (음성 변경 시 자동 갱신) |
| RAG 캐시 | ChromaDB 1시간 캐시 (`@st.cache_resource(ttl=3600)`) |
| LLM 재시도 | `max_retries=3` 설정 (모든 OpenAI 클라이언트) |
| 스트리밍 | 메인 채팅은 토큰 스트리밍으로 체감 속도 개선 |

---

## 13. 한계 및 개선 검토 사항

### 현재 한계

1. **ChromaDB 비영구화**: Streamlit Cloud에서 앱 재시작 시마다 벡터 DB 재구축 (약 30초~1분 소요)
2. **단일 모델 사용**: 모든 LLM 호출이 gpt-4o-mini 단일 모델 — 용도별 티어 분리 미적용
3. **CSV 데이터 수동 관리**: 전시물 데이터 업데이트 시 CSV 직접 수정 필요, CMS 없음
4. **세션 의존적 상태**: 새로고침·재접속 시 대화 이력 초기화 (MemorySaver는 세션 내만 유효)
5. **동시 사용자 확장성**: Streamlit Community 플랜은 단일 인스턴스, 동시 접속자 많을 때 성능 미보장

### 개선 가능 방향

1. **모델 티어 분리**: 메인 채팅/퀴즈/동화 → gpt-4o 이상, 분류/번역 → gpt-4o-mini 유지
2. **벡터 DB 외부화**: Pinecone 또는 Supabase pgvector로 영구 저장
3. **전시물 데이터 파이프라인**: CSV → Google Sheets 기반 CMS로 비개발자 관리 가능하도록
4. **프롬프트 버전 관리**: 현재 코드 내 하드코딩 → 별도 YAML/JSON 파일 분리
5. **사용자 인증**: 현재 완전 익명 → 선택적 사용자 식별으로 맞춤 추천 강화

---

## 14. 최근 개선 이력 (2026-05)

| 항목 | 내용 |
|---|---|
| 퀴즈 프롬프트 개선 | `quiz_detail`에 content + detail 두 필드 모두 주입 |
| 퀴즈 데이터 보충 | 부실 키워드(< 200자) 시 같은 category 이웃 전시물 자동 결합 |
| 동화 폴백 강화 | content/detail 없을 때 전시물 제목 목록으로 대체 |
| 동화 가드레일 | "목록 외 전시물·과학 개념 창작 금지" 규칙 프롬프트 추가 |
| route_by_age 분기 | 요일별 운영 차이 안내 (로봇순회·예약·붐빔 등) |
| 오류 메시지 다국어화 | 스트리밍 실패·STT 처리 등 에러 메시지 4개 언어 지원 |
| URL 링크 정리 | 홈페이지 URL 텍스트 직접 노출 제거 (link_button으로 대체) |
| 재시도 로직 | OpenAI 클라이언트 전체 `max_retries=3` 추가 |
| LLM prep 오류 처리 | RAG 검색 실패 시 다국어 에러 메시지 출력 |

---

## 15. 배포 구성

```toml
# .streamlit/config.toml
[server]
maxUploadSize = 10

[theme]
# 커스텀 테마 설정
```

**배포 URL**: Streamlit Cloud 자동 할당  
**Python 버전**: 3.11+  
**브랜치**: main → 자동 배포 (GitHub Actions 연동)

---

*이 문서는 `ARCHITECTURE.md`로 저장되었습니다.*
