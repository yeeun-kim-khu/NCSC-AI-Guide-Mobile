# 국립어린이과학관 AI 가이드 — 상세 코드 레퍼런스 (cloud-deployment)

> 이 문서는 **Claude 등 외부 LLM이 이 앱의 전체 구조와 모든 함수의 역할을 한 번에 파악**하고, 어떤 부분을 어떻게 고치거나 확장해야 할지 근거 있는 제안을 할 수 있도록 작성된 **완전 레퍼런스**입니다. 함수 단위 역할, 입출력, 호출 관계, 부수 효과, 주의 사항까지 모두 포함합니다.
>
> **대상 폴더**: `cloud-deployment/` (Streamlit Cloud 배포 버전 — 사용자가 "모바일"이라 부름)
> **최종 업데이트 기준**: 2026-04-27 패치 (search_directions 복원, 주차 안내 가드레일, 디버그 캡션 확장)

---

## 0. 한 줄 요약

**국립어린이과학관을 방문하는 어린이·가족·외국인을 위한 4개 국어(한/영/일/중) Streamlit 챗봇 + 방문 후 학습(퀴즈·질문·과학동화 + 오디오북) 시스템.** 챗봇은 LangGraph ReAct agent가 FAISS/Chroma RAG + 4개 커스텀 툴로 답변하며, 규칙 기반(rule-based) FAQ는 LLM을 거치지 않고 바로 응답한다.

---

## 1. 파일 구성 (cloud-deployment 루트)

```
cloud-deployment/
├── app_with_voice.py       # Streamlit 엔트리포인트 (UI, 라우팅, 음성 I/O, 챗봇 루프)
├── core.py                 # 상수, 규칙 기반 답변, RAG 인덱싱, LangChain 툴, 프롬프트
├── learning.py             # 사후 학습 탭 (퀴즈/질문/과학동화/오디오북)
├── voice.py                # STT (Whisper) + TTS (OpenAI) + 언어 코드 매핑
├── __init__.py             # 빈 파일 (패키지 식별용)
├── requirements.txt        # streamlit, langchain, openai, chroma 등 의존성
├── .streamlit/             # Streamlit 설정 (secrets.toml 등)
├── data/                   # CSV 전시물 데이터 + pages/ 서브폴더
│   ├── AI놀이터.csv
│   ├── 행동놀이터.csv
│   ├── 탐구놀이터.csv
│   ├── 관찰놀이터.csv
│   ├── 천체투영관.csv
│   ├── 관람료.csv
│   ├── 운영안내.csv
│   └── pages/*.csv
└── multilingual/           # 다국어 브로셔 (영/일/중 PDF + TXT + CSV)
    ├── Science Center Information_ENG_250318.{pdf,txt,csv}
    ├── Science Center Information_JPN_250318.{pdf,txt,csv}
    └── Science Center Information_CHN_250318.{pdf,txt,csv}
```

### Python 파일별 대략 크기
| 파일 | 크기 | 라인 수(대략) | 역할 |
|---|---:|---:|---|
| `core.py` | 67KB | ~1543 | 비즈니스 로직 본체 |
| `learning.py` | 72KB | ~1507 | 사후 학습 UI + 동화 생성 |
| `app_with_voice.py` | 41KB | ~755 | Streamlit 진입점 |
| `voice.py` | 4KB | ~140 | 음성 I/O |

---

## 2. 전체 아키텍처

### 2.1 레이어 다이어그램

```
┌──────────────────────────────────────────────────────────┐
│  app_with_voice.py :: main()                             │
│  └─ Sidebar (mode/lang/voice/debug/FAQ buttons)          │
│  └─ Tab1: 🏙️ 과학관 안내 (챗봇)                          │
│  └─ Tab2: 🥰 또만나 놀이터 (learning.render_post_visit_learning)
└──────────────────────────────────────────────────────────┘
                │
     ┌──────────┼──────────┐
     ▼          ▼          ▼
  core.py   learning.py  voice.py
     │          │          │
     │          └──→ core.load_zone_rows_from_csv
     │
     ├─ route_intent → "notice" | "basic" | "llm_agent"
     ├─ answer_rule_based (규칙 기반 즉답)
     ├─ get_tools() → LangChain @tool 4개
     ├─ initialize_vector_db → Chroma(OpenAI embeddings)
     └─ get_dynamic_prompt → 시스템 프롬프트 (언어/모드별)
```

### 2.2 챗봇 질문 처리 흐름 (app_with_voice.py의 핵심 루프)

```
사용자 질문 (텍스트 or 음성)
       │
       ▼
 route_intent(text)  ← core.py
       │
 ┌─────┼─────┬──────────┐
 ▼     ▼     ▼          ▼
"notice"  "basic"   "llm_agent"
 │         │           │
 │ answer_rule_based   agent.invoke()
 │    └→ 한국어        ├─ RAG similarity_search(k=3)
 │      원문 생성      ├─ system_prompt + rag_context
 │    └→ 외국어 모드   ├─ language_override prefix 주입
 │      translate_    ├─ create_react_agent → tool calling
 │      answer_cached │   (check_museum_closed_date,
 │                    │    search_directions, search_csc_live_info,
 │                    │    fetch_latest_notices)
 │                    └─ answer = result["messages"][-1].content
 │
 ▼
 st.markdown(answer)
 + KO/BT 디버그 캡션 (외국어 모드 + 디버그 체크박스 시)
 + render_tts_for_answer(answer) → OpenAI TTS → st.audio
 + render_source_buttons(sources)
 │
 ▼
 session_state.messages.append(assistant_msg)
```

### 2.3 사후 학습 흐름 (learning.render_post_visit_learning)

```
Tab2 "또만나 놀이터"
 ├─ 3개 서브탭: 퀴즈타임 / 궁금해요! / 과학동화
 │
 ├─ [퀴즈타임]
 │    ├─ 놀이터 선택 → get_zone_exhibits_from_rag (RAG → fallback CSV)
 │    ├─ extract_principles_from_exhibits (LLM)
 │    ├─ _render_keyword_tags (키워드 버튼)
 │    └─ generate_quiz(zone, principle, llm, language) → 4지선다
 │
 ├─ [궁금해요!]
 │    └─ 자유 질문 → vector_db.similarity_search → LLM 답변
 │
 └─ [과학동화]
      ├─ 놀이터 다중 선택
      ├─ generate_science_story(zones, exhibits, principles, language)
      │    └─ 3막 구조, 주인공/동반자/세계관 랜덤, CSV 전시물 2개를 마법 아이템으로
      └─ text_to_audiobook (ElevenLabs → Naver CLOVA → OpenAI TTS fallback)
```

---

## 3. 파일별 상세 분석

---

## 3.1 `core.py` — 비즈니스 로직 본체 (1543 lines)

> **config.py + rag.py + tools.py + utils.py + multilingual_loader.py의 통합본**

### 3.1.1 임포트 (L1-L23)
- `os, glob, re, requests, urllib3, time`: 파일/HTTP/정규식 기본
- `pandas`: CSV 처리
- `streamlit as st`: `st.session_state`, `st.cache_data`
- `bs4.BeautifulSoup`: 공지사항/홈페이지 HTML 파싱
- `requests.adapters.HTTPAdapter, urllib3.util.retry.Retry`: 재시도 세션
- `datetime, timezone, timedelta`, `zoneinfo.ZoneInfo`: KST 시각 처리
- `langchain.tools.tool`: `@tool` 데코레이터
- `langchain_openai.OpenAIEmbeddings`: RAG 임베딩
- `langchain_community.vectorstores.Chroma`: 벡터 DB
- `langchain_core.documents.Document`: 문서 객체

### 3.1.2 상수 (L25-L87)

| 상수 | 타입 | 내용 |
|---|---|---|
| `MUSEUM_BASE_URL` | `str` | `https://www.csc.go.kr` |
| `CSC_URLS` | `dict[str,str]` | 34개 공식 홈페이지 URL (공지/예약/전시관/프로그램별). **모든 출처 버튼·크롤링의 단일 진실 원천(SSOT)**. |
| `STATIC_EXHIBIT_INFO` | `dict[str,str]` | 주소·운영시간·관람료 fallback 3개 — RAG 실패 시 최소 생존 정보 |

### 3.1.3 규칙(라우팅/분류) 함수들

#### `route_intent(text: str) -> str` (L93-L118)
**역할**: 사용자 질문을 3개 intent로 라우팅 → `"notice"` | `"basic"` | `"llm_agent"`.
**로직**:
1. `st.session_state["awaiting_directions_origin"]`이 True면 바로 `llm_agent` + `directions_origin` 저장 (대화형 길찾기 2단계 처리)
2. "공지/공지사항/알림" → `notice`
3. "오시는길·교통·길찾기·어디" + "에서/출발/역" → `llm_agent` (search_directions 도구 유도)
4. "운영/휴관/관람료/주차/예약" → `llm_agent`
5. 기본값 → `llm_agent`
**부수 효과**: `st.session_state["directions_origin"]`에 출발지 저장.
**주의**: 현재 "basic" 리턴 경로가 코드에는 있지만, 실질적으로는 `classify_basic_category`가 `answer_rule_based` 내에서 호출됨. 즉 `route_intent`가 직접 `"basic"`을 반환하는 패스는 거의 쓰이지 않고, 대부분 `llm_agent`로 빠져 LLM이 처리.

#### `classify_basic_category(message: str) -> str` (L120-L139)
**역할**: `intent == "basic"`일 때 세부 카테고리 6개로 분류 → `floor_guide` | `facility_amenities` | `exhibit_guide` | `route_by_age` | `today_programs` | `planetarium_timetable` | `reservation_guide` | `operating_hours` | `admission_fee` | `parking` | `directions`.
**로직**: 키워드 리스트 순회해서 매칭되는 첫 번째 카테고리 반환. 매칭 없으면 `operating_hours` 기본값.

#### `check_closed_date(target_date: datetime) -> tuple[bool, str]` (L141-L152)
**역할**: 날짜 받아서 `(휴관여부, 사유문자열)` 반환. `get_today_status`·`check_museum_closed_date` 툴의 내부 헬퍼.
**휴관 규칙**: 월요일(일부 예외일 제외) + 1/1 + 설날/추석 당일 + 대체 휴관일 4개.

#### `get_today_status() -> str` (L154-L168)
**역할**: KST 현재 시각 기준으로 "오늘 휴관" / "개관 시간 이전/이후" / "운영 중" 한국어 문자열 반환.
**주의**: KST는 `datetime.now(timezone.utc) + timedelta(hours=9)`로 계산 (서버가 UTC일 수 있으므로).

### 3.1.4 규칙 기반 답변 — `answer_rule_based(intent, message, mode) -> str` (L170-L564)

**역할**: LLM을 거치지 않고 미리 작성된 장문 답변을 즉시 반환하는 FAQ 엔진. 속도 최적화 + 환각 방지.

**intent가 "notice"일 때** (L172-L199):
- "공지 3번 자세히" 같은 번호 요청 → `st.session_state["latest_notices"]` 캐시에서 해당 pkid 뽑아 `get_notice_detail_text` 호출
- "pkid=12345" 명시 → 바로 본문 가져옴
- 그 외 → `get_latest_notices_text(limit=5)`로 최신 5개 목록 반환 + session_state에 캐시

**intent가 "basic"일 때** (L200-L563): `classify_basic_category` 결과별로 분기하여 하드코딩된 마크다운 블록 반환. 각 블록 요약:

| category | 반환 내용 요약 | 라인 |
|---|---|---|
| `floor_guide` | 1/2/3층별 전시관·편의시설 + 입구(2층)·출구(1층) 안내 | L202-L231 |
| `facility_amenities` | 의무실·수유실·물품보관함·유모차 대여 등 편의시설 정리 | L233-L255 |
| `exhibit_guide` | 6개 놀이터 한 줄 소개 | L257-L267 |
| `route_by_age` | 유아/저학년/고학년별 추천 동선 | L269-L285 |
| `today_programs` | KST 월/요일 기반 **오늘의 과학쇼/전시해설/천체투영관** 동적 시간표 생성 | L287-L436 |
| `reservation_guide` | 예약 규정 (어린이 미동반 성인은 3일 전 proxima11@korea.kr) | L438-L474 |
| `planetarium_timetable` | 천체투영관 6회차 시간표 + 주의사항 | L476-L500 |
| `operating_hours` | `get_today_status()` 결과 + 휴관일 설명 | L502-L506 |
| `admission_fee` | 상설전시관/천체투영관 요금표 (마크다운 테이블) | L508-L540 |
| `parking` | **🔴 "전용 주차장 없음" 명시** (어린이 모드 분기). 최근 패치로 빈 문자열에서 복원됨. | L541-L544 |
| `directions` | 출발지 추출 후 없으면 `awaiting_directions_origin` 플래그 세우고 "어디서 출발?" 질문. 있으면 STATIC_FAQ 기반 안내. | L546-L562 |

**주의 사항 for LLM refactor**:
- "오늘의 프로그램"은 홀수월(1/3/5/7/9/11)과 짝수월에서 다른 프로그램을 반환 → `science_show_type = "사이언스랩" if month in [1,3,5,7,9,11] else "로봇쇼"` 규칙.
- `operating_hours`의 날짜·요일 계산은 모두 `datetime.now(timezone.utc) + timedelta(hours=9)` 직접 계산 (pytz/zoneinfo 사용 안 함).

### 3.1.5 RAG 시스템 (L566-L785)

#### `load_csv_data() -> list[Document]` (L570-L676)
**역할**: `data/*.csv` 및 `data/pages/*.csv` 파일들을 순회하며 LangChain Document 리스트로 변환.
**입력**: 없음 (파일시스템 직접 탐색)
**출력**: `Document(page_content, metadata)` 리스트
**동작**:
1. 내부 헬퍼 `load_csv_safe(path)`: 인코딩 4종(utf-8-sig, utf-8, cp949, euc-kr) 순차 시도
2. 헤더 정규화: 한글 컬럼명을 `title/content/detail/category`로 rename (synonyms 맵 사용)
3. `Unnamed` 컬럼이 많으면 첫 행을 헤더로 승격
4. zone 이름은 파일명에서 추출 ("AI놀이터.csv" → zone_name="AI놀이터")
5. 각 row를 `[zone_name] title\nCategory: ...\nContent: ...\nDetails: ...` 포맷으로 Document화
6. metadata: `{source: "csv_{zone_name}", title, category: zone_name, subcategory: category}`

**호출처**: `initialize_vector_db` (L768)

#### `load_multilingual_brochures() -> list[Document]` (L678-L751)
**역할**: `multilingual/` 폴더의 영/일/중 브로셔 3세트(PDF/TXT/CSV 각각)를 Document로 변환.
**출력**: Document 리스트 (metadata에 `language`, `file_type` 포함)
**주의**: PDF는 placeholder 문자열만 넣음 (실제 추출 미구현 — 향후 `pypdf` 추가 여지).

#### `initialize_vector_db() -> Chroma` (L753-L785)
**역할**: STATIC_EXHIBIT_INFO + CSV docs + 다국어 브로셔 docs를 합쳐 **매 실행마다 새로운 Chroma 벡터스토어 생성** (Streamlit Cloud는 영속성 없음).
**임베딩**: `OpenAIEmbeddings(model="text-embedding-3-small")`
**호출처**: `app_with_voice.load_rag_db()` (Streamlit cache_resource)

### 3.1.6 LangChain 도구 (L787-L1202)

#### `parse_html_tables_to_markdown(soup) -> str` (L791-L803)
BS4 `<table>` → 마크다운 표 변환 헬퍼.

#### `@tool check_museum_closed_date(date_str) -> str` (L805-L849)
**역할**: 특정 날짜 휴관 여부 확인.
**입력**: `"2026-03-24"` 또는 `"내일"`, `"모레"`
**출력**: `"Observation: 2026년 3월 24일(화요일)은 정상 운영일입니다."` 같은 문자열
**규칙**:
- `holiday_closed = {01-01, 02-17, 09-25}` (명절)
- `substitute_closed = {02-19, 03-03, 05-26, 08-18, 10-06}` (대체 휴관)
- `monday_exceptions`: 월요일이지만 개관하는 공휴일
- 기본값: 일반 월요일 → 정기휴관

#### `@tool search_directions(origin, destination="국립어린이과학관") -> str` (L851-L884)
**역할**: 출발지→목적지 대중교통 경로 안내 (실제 API 호출 없이 정적 가이드 생성).
**반환 예시**:
```
Observation: 강남역에서 국립어린이과학관까지의 경로 안내
정확한 실시간 경로는:
1. 네이버 지도...
일반적인 안내:
- 주소: 서울특별시 종로구 창경궁로 215
- 가까운 지하철역: 4호선 혜화역 4번 출구
- 전용 주차장이 없으므로 대중교통 이용 권장
```
**주의**: 현재는 **실제 경로 검색 API를 호출하지 않음** — LLM이 이 반환값을 기반으로 자연스럽게 안내문을 조립하는 구조. 향후 Naver Directions API / Kakao Mobility API 연동 여지 있음.

#### `@tool search_csc_live_info(keyword) -> str` (L886-L924)
**역할**: `CSC_URLS[keyword]` URL 크롤링 → `<script>/<style>/<nav>/<footer>/<header>` 제거 → 본문 텍스트 + 표 마크다운 반환 (최대 3000자).
**주의**: `verify=False`로 SSL 검증 무시 (CSC 사이트 인증서 문제). `urllib3.disable_warnings`로 경고 숨김.

#### `get_latest_notices_text(limit=5) -> str` (L927-L1053) + 헬퍼들
**역할**: `https://www.csc.go.kr/boardList.do?bbspkid=22` 공지사항 목록 크롤링.
**로직**:
1. `_fetch_html_bytes`: Retry 세션 + must_contain 마커(`goView`, `rbbs`, `boardList`) 검증
2. `soup.select("a[onclick*='goView']")`로 공지 앵커 추출
3. `onclick="goView('12345','0','1')"` 파싱해 pkid 추출
4. `/boardView.do?bbspkid=22&pkid={pkid}&num={num}` 링크 조립
5. 노이즈 제목(에러/메인 페이지) 필터링

#### `_resolve_notice_title(pkid, num) -> str` (L1055-L1081)
pkid만 알 때 해당 공지의 실제 제목을 다시 가져오는 헬퍼.

#### `_build_retry_session() -> requests.Session` (L1084-L1098)
urllib3 Retry(total=6, backoff_factor=1.5) 설정된 requests 세션.

#### `_read_response_bytes(resp, max_bytes=2MB) -> bytes` (L1101-L1112)
청크 단위로 읽다가 max_bytes 초과하면 중단.

#### `_fetch_html_bytes(url, headers, max_attempts=3, must_contain=None) -> bytes` (L1115-L1143)
재시도 + must_contain 마커 검증 + CSC 서버 특이사항(connection close, identity encoding) 처리된 HTML 바이트 다운로더.

#### `get_notice_detail_text(pkid) -> str` (L1146-L1186)
공지 개별 페이지 본문 텍스트 추출.

#### `@tool fetch_latest_notices(limit=5) -> str` (L1189-L1192)
LangChain tool 래퍼 — 내부적으로 `get_latest_notices_text` 호출.

#### `get_tools() -> list` (L1195-L1202)
**LangChain ReAct agent에 주입되는 4개 툴 반환**:
```python
[check_museum_closed_date, search_directions, search_csc_live_info, fetch_latest_notices]
```
**⚠️ 주의**: `search_directions`는 2026-04-26 패치로 복원됨. 이전 버전에서 누락되어 LLM이 환각으로 "주차 가능" 같은 거짓말을 생성하는 버그 있었음.

### 3.1.7 `get_dynamic_prompt(mode, language="한국어") -> str` (L1208-L1344)
**역할**: 언어별·사용자 모드별 LLM 시스템 프롬프트를 동적 생성.
**구성 요소**:
1. `[오늘 날짜] {KST} ({요일}) KST` — 휴관일 판단·"오늘의 프로그램"에 사용
2. `language_instruction`: 한국어/English/日本語/中文별 **"반드시 해당 언어로만 답하라"** 강제 지시
3. `safety_instruction`: 어린이 대상 욕설·폭력 금지 가드레일
4. `핵심 임무` + `답변해야 할 주요 영역` 5가지
5. `환각 방지 가드레일`: RAG/도구 결과 없으면 홈페이지 안내로 폴백 + **🔴 주차 안내 규칙 ("주차 가능" 금지, 자가용 권장 금지)**
6. `국립어린이과학관 위치 정보 (고정값)`: 주소, 지하철역, 버스 정류장, 전화번호
7. `길찾기 응대 규칙` 4가지: (1) 과학관→집 (2) 집→과학관 (3) 개인정보 보호 (4) search_directions 도구 강제
8. `OFFICIAL PLACE NAMES (MANDATORY GLOSSARY)`: 놀이터/전시관 10개의 한·영·일·중 고정 번역 매핑. 외국어 모드에서 LLM이 "Thought Playground" 같은 자유 번역 못하게 금지.
9. `mode == "어린이"`면 마지막에 "쉽고 재미있게, 이모지 활용" 추가

**호출처**: `app_with_voice.py`에서 매 챗봇 답변 전에 호출.

### 3.1.8 천체투영관 전용 (L1346-L1414)

#### `PLANETARIUM_VIDEO_INFO` (L1346-L1367)
5편 영상 메타데이터 dict (themes, fulldomedb_url).

#### `_load_planetarium_videos() -> list[dict]` (L1370-L1414)
`data/천체투영관.csv`에서 `category`가 `프로그램_`로 시작하는 행만 추출해 `{title, content, detail, category}` row 형식으로 변환. `PLANETARIUM_VIDEO_INFO`의 5개 제목과 매칭되는 것만 반환 (중복 제거).

### 3.1.9 `load_zone_rows_from_csv(zone_name) -> list[dict]` (L1417-L1484)
**역할**: `learning.py`에서 놀이터별 전시물 목록 로드 시 사용.
**특수 케이스**: `zone_name == "천체투영관"` → `_load_planetarium_videos()` 호출.
**일반 케이스**: `data/{zone_name이 포함된 파일명}.csv` 읽고 헤더 정규화 후 `{title, content, detail, category}` 리스트 반환.

### 3.1.10 번역·렌더링 유틸

#### `@st.cache_data(ttl=24h) translate_answer_cached(text, target_language) -> str` (L1487-L1512)
**역할**: 규칙 기반 한국어 답변을 외국어로 LLM 번역 (gpt-4o-mini, temperature=0).
**중요**: 마크다운 구조(헤딩/불릿/테이블)·이모지·숫자·고유명사·시간 **그대로 유지** 지시. 24시간 캐시.

#### `render_source_buttons(sources, language_mode, key_suffix) -> None` (L1515-L1542)
**역할**: 답변 아래에 "📚 참고 홈페이지 자세히 보기" expander 렌더링. URL 리스트(최대 5개)를 클릭 가능한 링크로 표시.

---

## 3.2 `learning.py` — 또만나 놀이터 시스템 (1507 lines)

> **post_visit_learning.py + audiobook_generator.py + visualization.py 통합본**

### 3.2.1 임포트 + 클라이언트 초기화 (L1-L21)
- `langchain_openai.ChatOpenAI`, `openai.OpenAI` (직접 SDK도 병행 사용)
- `from core import initialize_vector_db, load_zone_rows_from_csv`
- `client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))` **모듈 로드 시점** 고정 — 환경변수 없으면 이후 호출 시 에러
- `_safe_secret_get(key, default)`: `st.secrets.get` try/except 래퍼 (ElevenLabs/Naver 키용)

### 3.2.2 상수 (L23-L97)

| 상수 | 내용 |
|---|---|
| `ZONE_INFO` | 7개 놀이터 메타(층, 설명, has_data). `has_data=False`면 CSV 데이터 없음 |
| `ZONE_GROUPS` | UI 선택용 그룹핑 (1층놀이터/2층/천체투영관/빛놀이터) |
| `ZONE_GROUP_LABELS` | 한/영/일/중 4개 언어별 그룹 레이블 |

### 3.2.3 놀이터 선택 헬퍼 (L100-L117)

#### `_select_zones_by_group(prefix_key, language_mode) -> list[str]`
여러 놀이터를 체크박스로 선택 받는 UI 헬퍼. `prefix_key`로 session_state 키 분리 (quiz/story 다른 선택 유지).

### 3.2.4 CSV 프리로딩 (L120-L128)

#### `@st.cache_data _preload_all_zone_csv_rows()`
모든 놀이터 CSV를 한 번에 로드해 `{zone_name: rows}` dict 반환. 캐시되므로 세션 당 한 번만 실행.

### 3.2.5 키워드 추출 (L131-L264)

#### `_extract_zone_keywords(zone_rows, top_n=12) -> list[str]` (L134-L144)
전시물 title+category+content 합쳐서 단어 빈도 Counter → 상위 N개 한글 키워드.

#### `_extract_zone_keywords_from_titles(zone_rows, top_n=12) -> list[str]` (L147-L159)
전시물 title만 사용 (더 깔끔한 버전).

#### `@st.cache_data _extract_zone_keywords_llm(zone_name, language_mode, csv_compact_text)` (L162-L201)
gpt-4o-mini로 놀이터 대표 키워드 12개 추출 (LLM 기반, 24h 캐시).

#### `@st.cache_data _translate_keywords_cached(keywords_tuple, target_language)` (L204-L235)
한국어 키워드를 외국어로 일괄 번역 (gpt-4o-mini, 24h 캐시).

#### `_get_zone_keywords(zone_name, zone_rows, language_mode) -> list[tuple]` (L238-L264)
위 3개를 조합해 `[(ko_keyword, display_keyword), ...]` 반환. UI 버튼 렌더링용.

#### `_render_keyword_tags(zone_name, keyword_pairs, zone_rows, language_mode, mode, llm)` (L267-L320)
키워드 버튼들을 4열 그리드로 렌더링. `mode` 파라미터:
- `"exhibits"`: 클릭 시 관련 전시물 목록 표시
- `"quiz"`: 클릭 시 즉시 퀴즈 생성
- `"question"`: 클릭 시 질문 입력창 활성화

### 3.2.6 RAG 검색 (L323-L407)

#### `_load_exhibits_from_csv_direct(zone_name) -> list[dict]` (L326-L350)
RAG 실패 시 CSV에서 바로 전시물 로드하는 fallback. `load_zone_rows_from_csv` 래핑.

#### `get_zone_exhibits_from_rag(zone_name, vector_db) -> list[dict]` (L353-L407)
**역할**: 놀이터 이름으로 RAG에서 해당 전시물 정보 수집.
**로직**:
1. `vector_db.similarity_search(zone_name, k=10)` → metadata에서 zone_name 매칭되는 것만 필터
2. 결과 없으면 `_load_exhibits_from_csv_direct` fallback
3. `{title, content, detail, category}` dict 리스트 반환

### 3.2.7 과학원리 추출 (L409-L444)

#### `extract_principles_from_exhibits(exhibits, llm) -> tuple[list[str], str]` (L409-L444)
**입력**: 전시물 리스트, ChatOpenAI LLM
**출력**: `(과학원리 5개 리스트, 요약 텍스트)`
전시물 제목·내용을 LLM에 던져 "이 놀이터에서 배울 수 있는 과학원리 5개"를 JSON 형식으로 추출.

### 3.2.8 퀴즈 생성 (L447-L560)

#### `generate_quiz(zone_name, principle, llm, language="한국어", variation_seed=0) -> dict` (L449-L560)
**역할**: 놀이터 + 과학원리 하나 → 4지선다 퀴즈 1문항 생성.
**출력 구조**:
```python
{
    "question": "...",
    "choices": ["A", "B", "C", "D"],
    "correct_index": 0,
    "explanation": "..."
}
```
**특이사항**:
- `variation_seed`로 같은 원리여도 매번 다른 문제 생성 (random.seed + random.shuffle로 정답 인덱스 섞기)
- 4개 언어별 프롬프트 분기 (한/영/일/중)

### 3.2.9 과학동화 생성 (L563-L930) — **가장 복잡한 함수**

#### `_get_ui_glossary_rules(language_mode) -> str` (L565-L593)
외국어 동화 생성 시 장소명 고정 번역 규칙 반환 ("놀이터" → "Zone" 등).

#### `generate_science_story(zone_name, exhibits, principles, language="한국어") -> str` (L596-L930)
**역할**: 체험한 놀이터 기반 어린이 과학동화 생성.
**입력**:
- `zone_name`: 단일 놀이터 이름 또는 쉼표로 묶인 여러 놀이터
- `exhibits`: 실제 CSV 전시물 리스트
- `principles`: 과학원리 리스트
- `language`: 한/영/일/중
**구조**:
1. **랜덤 요소**: 주인공 이름, 동반자(친구/동물/로봇), 세계관 분위기 (마법 숲/우주정거장/미래도시 등) 선택
2. **CSV 재료 추출**: zone 정체성 한 줄, 분위기 재료 5개(제목), 핵심 마법 아이템 2개(제목+설명)
3. **프롬프트 조립**: 4개 언어별로 다른 프롬프트, 아래 규칙 주입
4. **개연성 규칙 (6~8세 대상)**:
   - **3막 구조 (총 6~8문단)**:
     - 1막 (2문단): 주인공의 평범한 순간 → 과학 원리와 직접 관련된 이상한 사건 → 하나의 명확한 목표
     - 2막 (3~4문단): 마법 아이템 시도 → 현상 작게 일어남 (감각 묘사) → 실패 → 동반자와 패턴 발견
     - 3막 (1~2문단): **아하 순간** — 주인공이 "아, 이게 {원리}구나!" 깨달음 → 원리로 위기 해결 → 1막 수수께끼 설명 → 따뜻한 마무리
   - 설명충 금지, 감각 묘사 강제
5. **gpt-4o-mini, temperature=0.9**로 다양성 확보
6. **반환**: 마크다운 동화 텍스트

**⚠️ 주의**:
- 이 프롬프트는 **자주 수정되는 부분**. 최근 작업: 3막 구조 도입, 원리 가시성 강화, CSV 전시물 2개를 "핵심 마법 아이템"으로 변환 강제.
- 장소명 번역은 `_get_ui_glossary_rules`로 고정.

### 3.2.10 오디오북 생성 (L932-L1041)

#### `text_to_audiobook(story_text, language="한국어", voice_override=None, speed_override=None) -> bytes` (L932-L1041)
**역할**: 동화 텍스트를 MP3 바이트로 변환.
**TTS 제공자 폴백 순서**:
1. **ElevenLabs** (`ELEVENLABS_API_KEY` 있으면) — 가장 자연스러움
2. **Naver CLOVA** (`NAVER_CLIENT_ID`+`NAVER_CLIENT_SECRET`) — 한국어 최적화
3. **OpenAI TTS-1** (fallback) — 기본
**언어별 보이스 매핑**: 예) 한국어는 ElevenLabs 한국어 voice ID, 영어는 Rachel 등.

### 3.2.11 역번역 (L1046-L1061)

#### `@st.cache_data(ttl=24h) _backtranslate_to_korean_cached(text, source_language) -> str` (L1046-L1061)
**역할**: 외국어 답변 → 한국어로 역번역 (gpt-4o-mini, temperature=0). 디버그용.
**호출처**:
- `app_with_voice.py`의 챗봇 답변 (rule-based & LLM 경로 + replay 루프)
- `learning.py`의 동화 본문 하단
- `learning.py`의 사후학습 탭 3개 정적 UI 텍스트

### 3.2.12 메인 UI — `render_post_visit_learning(...)` (L1064-L1507)

**시그니처**:
```python
def render_post_visit_learning(
    vector_db,
    language_mode="한국어",
    debug_show_korean: bool = False,
    debug_backtranslate: bool = False,
):
```

**구조**:
1. 내부 헬퍼 `_display_zone_name(zone)`: 외국어 모드에서 놀이터 이름을 공식 영어명으로 변환
2. `texts` dict: 4개 언어별 UI 문구 일괄 정의 (title, subtitle, tab 이름, 버튼 라벨 등)
3. 3개 서브탭 생성:
   - **퀴즈타임** (`tab_quiz`): 놀이터 선택 → 원리 추출 → 키워드 버튼 → 퀴즈 생성
   - **궁금해요!** (`tab_question`): 자유 질문창 → RAG + LLM 답변
   - **과학동화** (`tab_story`): 놀이터 다중 선택 → `generate_science_story` → 결과 마크다운 + "오디오북 변환" 버튼 → `text_to_audiobook`
4. 디버그 캡션: 정적 UI 텍스트 3곳 + 생성된 동화 본문에 KO 원문/역번역 표시

**Session state 키**:
- `post_learning_story`: 생성된 동화 텍스트
- `post_learning_story_zones`: 선택된 놀이터 리스트
- `post_learning_story_audio`: MP3 바이트 캐시
- `post_learning_quiz_{zone}`: 퀴즈 생성 결과 캐시

---

## 3.3 `app_with_voice.py` — Streamlit 진입점 (755 lines)

### 3.3.1 임포트 (L1-L15)
- `streamlit, uuid, warnings, urllib.parse, base64`
- `langchain_openai.ChatOpenAI`, `langgraph.prebuilt.create_react_agent`, `langgraph.checkpoint.memory.MemorySaver`
- `audio_recorder_streamlit.audio_recorder`
- `from core import ...` 8개 심볼
- `from voice import ...` 5개 심볼
- `from learning import render_post_visit_learning, _backtranslate_to_korean_cached`

### 3.3.2 `@st.cache_resource load_rag_db()` (L17-L24)
매 세션마다 한 번만 `initialize_vector_db` 호출 (Chroma는 비싼 작업).

### 3.3.3 `main()` 함수 구조 (L26-L750+)

**섹션별 분해**:

#### (a) 초기 세팅 (L26-L34)
- `warnings.filterwarnings`: langgraph deprecation 경고 숨김
- `st.session_state["language_mode"]` 기본값 "한국어"
- `prev_language_for_page` 캐시

#### (b) `ui_text` dict 정의 (L33-L218)
4개 언어 × 약 50개 UI 키 = 200여 개 문자열. 사이드바/탭/버튼/플레이스홀더 모두 다국어화.

#### (c) 내부 헬퍼 `t(key)` (L220-L222)
현재 언어 모드로 ui_text 조회. 미등록 키는 한국어 fallback.

#### (d) 페이지 설정 + API 키 로드 (L224-L234)
- `st.set_page_config(page_title, page_icon="🐣", layout="centered")`
- `OPENAI_API_KEY`가 env에 없으면 `st.secrets`에서 로드
- `vector_db = load_rag_db()`

#### (e) 사이드바 (L236-L348)

| 섹션 | 위젯 | 동작 |
|---|---|---|
| 사용자 모드 | selectbox (어린이/청소년·성인) | `user_mode` 결정 |
| 언어 | selectbox (한/영/일/중) | 변경 시 `thread_id` 재생성 + messages 초기화 + TTS 캐시 삭제 + rerun |
| 음성 섹션 | 체크박스 2개 | `enable_voice_input`, `enable_voice_output` |
| **디버그 섹션** | 체크박스 2개 | `debug_show_ko`, `debug_backtranslate` — 외국어 모드에서만 캡션 활성화 |
| FAQ 버튼 4개 | 층별/프로그램/동선/전시관 | 클릭 시 `pending_user_input` 세팅 후 rerun |
| 음성 입력 | `audio_recorder` | WAV 바이트 → `speech_to_text` → pending_user_input |
| 새로고침 | 버튼 | messages/thread_id/TTS 캐시 전체 초기화 |

#### (f) 메인 화면 (L350-L464)
- 앱 제목 + 인사말 (4개 언어 `intro_enhanced`)
- 사용자 모드/언어 변경 후 첫 렌더에는 info 배너
- "빠른 메뉴(모바일 추천)" expander — 4개 버튼

#### (g) 탭 구조 (L466-L710)
```python
tab1, tab2 = st.tabs(["🏙️ 과학관 안내", "🥰 또만나 놀이터"])
```

**tab1 (챗봇)** (L472-L706):
1. session_state 초기화 (messages, thread_id, debug_logs, tts_cache)
2. `system_prompt = get_dynamic_prompt(user_mode, language_mode)`
3. `llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)`
4. `agent = create_react_agent(model=llm, tools=get_tools(), checkpointer=MemorySaver())`
5. 내부 함수 `render_tts_for_answer(answer_text)`:
   - voice_output 켜있으면 `text_to_speech` 호출 (최대 1200자 잘라서)
   - 캐시 키: `f"{language_mode}::{tts_ns}::{hash(text)}"`
   - "🔊 음성으로 듣기" 버튼 표시
6. **메시지 replay 루프** (L513-L576):
   - 저장된 `st.session_state.messages` 순회
   - 각 메시지 `st.chat_message` 블록에 `st.markdown`
   - **디버그 캡션**: assistant 메시지 + 외국어 모드 + 체크박스 켜짐일 때 KO 원문(`msg["ko_original"]`) 또는 역번역(`_backtranslate_to_korean_cached`) 표시
   - `msg.get("ui") == "program_buttons"` / `"reservation_links"` 렌더링
   - 음성 출력: 각 assistant 메시지 밑에 TTS 버튼
7. **사용자 입력 처리** (L585-L705):
   - `st.chat_input` 또는 `pending_user_input` 체크
   - `route_intent(user_input)`로 분기
   - `intent in ["notice", "basic"]` 경로:
     - `answer_rule_based(intent, user_input, user_mode)` (한국어)
     - `ko_original = answer` 보관
     - 외국어 모드면 `translate_answer_cached` 적용
     - 디버그 캡션 표시 (KO 원문 + 역번역)
     - `render_source_buttons`, `render_tts_for_answer`
   - `intent == "llm_agent"` 경로:
     - `directions_origin` 있으면 user_input 재조립 (출발지/목적지 명시)
     - `vector_db.similarity_search(user_input, k=3)` → rag_context 생성
     - 외국어 모드면 `llm_user_input`에 언어 강제 prefix 붙임
     - `agent.invoke({messages: [system, user]}, config={thread_id})` 
     - `answer = result["messages"][-1].content`
     - **🔴 외국어 모드 디버그 캡션 (BT만)**
     - debug_info 수집 (RAG 결과 + 툴 호출 내역) → expander
8. `assistant_msg` 조립 (role, content, tts_autoplayed, ko_original, ui) → messages 리스트 append
9. 자동 스크롤 트리거 (`scroll_to_bottom` → hidden iframe)

**tab2 (또만나 놀이터)** (L708-L715):
```python
render_post_visit_learning(
    vector_db,
    st.session_state.get("language_mode", "한국어"),
    debug_show_korean=debug_show_ko,
    debug_backtranslate=debug_backtranslate,
)
```

#### (h) 엔트리포인트
```python
if __name__ == "__main__":
    main()
```

### 3.3.4 Session state 키 전수

| 키 | 설정처 | 설명 |
|---|---|---|
| `language_mode` | 사이드바 언어 selectbox | "한국어"/"English"/"日本語"/"中文" |
| `messages` | 챗봇 루프 | `[{role, content, ko_original, ui, tts_autoplayed}, ...]` |
| `thread_id` | uuid4 | LangGraph MemorySaver 세션 식별자 |
| `tts_cache` | `render_tts_for_answer` | `{cache_key: audio_bytes}` |
| `debug_logs` | LLM 답변 시 | RAG 검색 결과 + 툴 호출 내역 |
| `pending_user_input` | FAQ 버튼 | 다음 rerun에서 챗봇이 이 문자열을 user_input으로 처리 |
| `pending_ui_program_buttons` | "오늘의 프로그램" 버튼 | 다음 assistant 메시지에 4개 서브 버튼 표시 |
| `pending_ui_reservation_links` | "예약" 질문 감지 시 | 다음 assistant 메시지에 3개 예약 링크 버튼 |
| `awaiting_directions_origin` | route_intent | 다음 사용자 입력을 출발지로 해석 |
| `directions_origin` | route_intent | LLM 경로에서 user_input 재조립에 사용 |
| `latest_notices` | notice intent | 공지 번호 참조용 캐시 |
| `_last_audio_sig` | 음성 입력 중복 방지 | 같은 오디오 중복 처리 차단 |
| `audio_recorder_key` | uuid4 | audio_recorder 위젯 리셋용 |
| `mode_language_changed` | 사이드바 | 변경 후 info 배너 1회 표시 |
| `scroll_to_bottom` | 챗봇 답변 후 | hidden iframe으로 자동 스크롤 |
| `post_learning_story` | 동화 생성 | 결과 마크다운 |
| `post_learning_story_zones` | 동화 생성 | 선택된 놀이터 |
| `post_learning_story_audio` | 오디오북 변환 | MP3 바이트 |

---

## 3.4 `voice.py` — 음성 I/O (140 lines)

### 3.4.1 임포트 + 초기화 (L1-L14)
```python
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
```
모듈 로드 시점 고정. OpenAI SDK 직접 사용.

### 3.4.2 `speech_to_text(audio_bytes) -> str | None` (L11-L55)
**역할**: WAV 오디오 바이트 → 텍스트 (Whisper).
**로직**:
1. 바이트를 `tempfile.NamedTemporaryFile(suffix=".wav")`에 저장
2. `client.audio.transcriptions.create(model="whisper-1", file=...)` 호출
3. text 반환, 에러 시 None + 디버그 print

### 3.4.3 `text_to_speech(text, language="ko") -> bytes | None` (L57-L83)
**역할**: 텍스트 → MP3 바이트 (OpenAI tts-1).
**로직**:
1. `preprocess_tts_text(text, language)` 호출 (이모지 제거 등)
2. 언어별 voice 매핑 (`get_tts_cache_namespace`의 voice_map 재사용)
3. `client.audio.speech.create(model="tts-1", voice=voice, input=text)` 호출
4. `response.content` 반환

**⚠️ 주의**: 로컬 `voice.py`는 ElevenLabs fallback 지원하나 cloud 버전은 OpenAI 단일 제공자. 향후 ElevenLabs 이식 여지.

### 3.4.4 `get_tts_cache_namespace(language="ko") -> str` (L86-L94)
언어별 voice 매핑 문자열 반환 (`openai::tts-1::nova` 등). 캐시 키 구성용.

### 3.4.5 `get_language_code(language_mode) -> str` (L96-L104)
`"한국어" -> "ko"`, `"English" -> "en"`, `"日本語" -> "ja"`, `"中文" -> "zh"`.

### 3.4.6 `autoplay_audio(audio_bytes) -> None` (L106-L117)
base64 인코딩 후 `<audio autoplay>` HTML을 `st.markdown(unsafe_allow_html=True)`로 주입. 브라우저 자동재생 정책 주의.

### 3.4.7 `preprocess_tts_text(text, language="ko") -> str` (L120-L140)
한국어가 아닐 때 이모지·특수문자 제거 등 전처리. 마크다운 기호도 정리.

---

## 4. 데이터 디렉토리 (`data/`)

### 4.1 필수 CSV 스키마

모든 전시물 CSV는 아래 4개 컬럼 기대 (한글·영문 동의어 자동 매핑):
| 컬럼 | 동의어 |
|---|---|
| `title` | 전시물명, 전시물, 전시명, 제목, 명칭, 이름 |
| `content` | 내용, 설명, 전시내용, 본문 |
| `detail` | 세부설명, 상세, 상세설명 |
| `category` | 분류, 카테고리, 구분 |

### 4.2 파일별 역할
| 파일 | 사용처 |
|---|---|
| `AI놀이터.csv` | learning.py 퀴즈/동화 재료 |
| `행동놀이터.csv` | 동상 |
| `탐구놀이터.csv` | 동상 |
| `관찰놀이터.csv` | 동상 |
| `천체투영관.csv` | `_load_planetarium_videos`에서 `category='프로그램_*'` 행만 사용 |
| `관람료.csv` | RAG 인덱싱 (현재 `answer_rule_based`가 하드코딩 답변 우선) |
| `운영안내.csv` | RAG 인덱싱 |
| `pages/*.csv` | 추가 페이지 상세 데이터 (RAG만) |

---

## 5. 다국어 처리 아키텍처

### 5.1 3단 방어선

1. **UI 레이어**: `ui_text` dict (app_with_voice.py) + `texts` dict (learning.py) — 정적 문자열 전부 언어별 분리
2. **규칙 기반 답변 레이어**: `answer_rule_based`는 항상 한국어 반환 → `translate_answer_cached`가 24h 캐시된 LLM 번역으로 변환
3. **LLM 에이전트 레이어**: `get_dynamic_prompt`에 강력한 "{언어} 전용 답변" 지시 + FAQ 버튼 트리거된 한국어 질문엔 `llm_user_input`에 `_lang_override` prefix 주입

### 5.2 장소명 고정 glossary
시스템 프롬프트 L1316-L1338의 **MANDATORY GLOSSARY**:
- AI놀이터 → AIゾーン | AI区 | AI Zone
- 생각놀이터 → 考えるゾーン | 思考区 | Thinking Zone
- 탐구놀이터 → 探究ゾーン | 探究区 | Discovery Zone
- ... (10개)

LLM이 "Thought Playground" 등 자유 번역 못하게 "FIXED — do NOT invent" 강조.

### 5.3 디버그 캡션

외국어 모드 + 사이드바 디버그 체크박스 켜짐일 때:
- **KO 원문** (`ko_original`): rule-based 답변에서만 수집 (LLM 답변은 원문이 외국어라 N/A)
- **BT (역번역)**: `_backtranslate_to_korean_cached` 호출 결과

표시 위치:
1. 챗봇 새 답변 (rule-based + LLM 경로 양쪽)
2. 챗봇 메시지 replay 루프 (rerun 후에도 유지)
3. 사후학습 탭 3개 정적 헤더
4. 사후학습 동화 본문 (expander)

---

## 6. 외부 의존성 & 환경변수

### 6.1 필수
| 변수 | 용도 | 출처 |
|---|---|---|
| `OPENAI_API_KEY` | LLM(gpt-4o-mini) + 임베딩(text-embedding-3-small) + Whisper + TTS | env 또는 `.streamlit/secrets.toml` |

### 6.2 선택
| 변수 | 용도 | Fallback |
|---|---|---|
| `ELEVENLABS_API_KEY` | 오디오북 고품질 TTS | OpenAI TTS |
| `NAVER_CLIENT_ID` + `NAVER_CLIENT_SECRET` | 한국어 CLOVA TTS | OpenAI TTS |

### 6.3 requirements.txt 주요 패키지
```
streamlit
langchain, langchain-openai, langchain-community, langchain-core
langgraph
openai
chromadb
pandas
beautifulsoup4
requests, urllib3
audio-recorder-streamlit
```

---

## 7. LangGraph Agent 구조

### 7.1 ReAct Agent 구성
```python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
memory = MemorySaver()  # 세션 내 대화 기억
agent = create_react_agent(
    model=llm,
    tools=get_tools(),  # 4개 @tool
    checkpointer=memory,
)
```

### 7.2 Invocation
```python
config = {"configurable": {"thread_id": st.session_state.thread_id}}
messages = [
    {"role": "system", "content": f"{system_prompt}\n\n[RAG 배경지식]\n{rag_context}"},
    {"role": "user", "content": llm_user_input},
]
result = agent.invoke({"messages": messages}, config=config)
answer = result["messages"][-1].content
```

### 7.3 툴 호출 흐름
LLM이 스스로 판단 → 필요한 툴 호출 → Observation 받아 추론 반복 → 최종 답변 생성.

예시:
- "내일 열어?" → `check_museum_closed_date("내일")` → "정상 운영일" → 자연어 조립
- "강남에서 어떻게 가?" → `search_directions("강남역", "국립어린이과학관")` → 경로 정보 → 자연어 조립
- "공지사항" → `fetch_latest_notices(5)` → 목록 → 자연어 조립
- "예약 방법" → `search_csc_live_info("예약안내")` → 홈페이지 크롤링 → 자연어 조립

---

## 8. 알려진 이슈 & 확장 여지

### 8.1 해결된 이슈 (최근 패치 기록)
| 이슈 | 파일 | 수정 내용 |
|---|---|---|
| 외국어 모드에서 규칙 기반 답변이 한국어로 섞여 나감 | core.py | `translate_answer_cached` + 시스템 프롬프트 CRITICAL LANGUAGE RULE |
| LLM이 외국어 모드에서 한국어 트리거 무시하고 한국어 답변 | app_with_voice.py | `llm_user_input`에 `_lang_override` prefix 주입 |
| 장소명 "Thought Playground" 등 자유 번역 | core.py | OFFICIAL PLACE NAMES glossary "FIXED — do NOT invent" |
| 동화 개연성 부족, 과학 원리 묻힘 | learning.py | 3막 구조 + 아하 순간 + CSV 전시물 2개 = 마법 아이템 |
| 디버그 캡션 안 보임 | app_with_voice.py, learning.py | 챗봇 답변 + 동화 본문에 KO/BT 캡션 + replay 루프 유지 |
| 클라우드에서 "주차 가능" 환각 | core.py | `parking` 답변 복원 + 시스템 프롬프트에 주차 금지 가드레일 |
| 클라우드에서 길찾기 대화형 안내 실패 | core.py | `search_directions` @tool 복원 + `get_tools` 등록 + 길찾기 응대 규칙 |

### 8.2 알려진 제약
1. `search_directions`는 실제 API를 호출하지 않음 — Naver Directions / Kakao Mobility 연동 여지
2. `load_multilingual_brochures`는 PDF 본문 추출 안 함 — `pypdf` 추가 여지
3. Streamlit Cloud는 Chroma 영속성 없어 매 세션마다 RAG 재구축 (비용·속도 증가)
4. ElevenLabs/Naver TTS는 로컬에서만 지원, 클라우드 voice.py는 OpenAI 단일
5. 음성 입력은 브라우저 MediaRecorder 기반 — 일부 모바일 브라우저에서 녹음 품질 저하
6. 공지사항 크롤러는 CSC 홈페이지 HTML 구조 변경 시 깨짐 (selector 고정)

### 8.3 제안 가능한 개선 방향 (Claude 브레인스토밍용 시드)
1. **길찾기 실시간화**: `search_directions`에 Naver Directions API 연동
2. **오프라인 동화 저장**: `post_learning_story`를 사용자 계정에 연결, 히스토리 기능
3. **전시물 이미지**: 현재 텍스트 기반 → CSV에 image_url 컬럼 추가, `st.image` 렌더링
4. **음성 대화 지속**: 현재는 STT → 텍스트 질문 → TTS 답변 (한 턴). LangGraph 스트리밍 + WebRTC로 실시간 대화
5. **방문 전/중 구분**: 방문 전(예약 안내 우선) / 방문 중(길찾기·동선 우선) / 방문 후(퀴즈/동화) 3단 모드
6. **RAG 영속성**: Pinecone/Qdrant 클라우드 벡터 DB로 이전, 초기 로드 시간 단축
7. **멀티모달 질문**: 사용자가 전시물 사진 찍어 올리면 인식해서 관련 정보 제공 (gpt-4o 비전)
8. **부모용 리포트**: 아이가 생성한 퀴즈/동화 요약을 이메일 또는 PDF로
9. **접근성**: 시각장애 대응(고대비 모드), 청각장애 대응(자막 항상 켜짐), 난독증 대응(폰트 옵션)
10. **분석 대시보드**: 어떤 놀이터·원리가 가장 많이 선택되는지 집계 → 전시 기획 피드백

---

## 9. 개발자 빠른 참조

### 9.1 진입점
```
streamlit run app_with_voice.py
```

### 9.2 "X를 어디서 고쳐야 하지?" 맵

| 고치고 싶은 것 | 파일 | 함수·섹션 |
|---|---|---|
| FAQ 즉답 추가/수정 | `core.py` | `answer_rule_based` + `classify_basic_category` 키워드 |
| 새 LangChain 도구 추가 | `core.py` | `@tool` 데코레이터 함수 정의 + `get_tools()` 리스트 |
| LLM 시스템 프롬프트 | `core.py` | `get_dynamic_prompt` |
| 공식 장소명 번역 | `core.py` | `get_dynamic_prompt`의 MANDATORY GLOSSARY |
| 언어별 UI 문구 | `app_with_voice.py` | `ui_text` dict, `learning.py` `texts` dict |
| 사이드바 추가 버튼 | `app_with_voice.py` | `with st.sidebar:` 블록 |
| 챗봇 답변 뒤 UI 렌더 | `app_with_voice.py` | replay 루프 + `assistant_msg["ui"]` 분기 |
| 동화 프롬프트 | `learning.py` | `generate_science_story` 내부 `language_prompts` |
| 퀴즈 난이도·개수 | `learning.py` | `generate_quiz` |
| 오디오북 음성 제공자 | `learning.py` | `text_to_audiobook` fallback 순서 |
| STT/TTS 모델 변경 | `voice.py` | `speech_to_text`, `text_to_speech` |
| CSV 파일 추가 | `data/*.csv` + `load_csv_data`에 zone_name 매핑 추가 |
| 새 언어 추가 | 4곳: `ui_text`, `texts`, `language_prompts`, `translate_answer_cached`, `language_instruction`, `safety_instruction` 모두 |

### 9.3 디버그 팁
- 사이드바 "🧪 디버그" 체크박스로 KO 원문·역번역 보기
- LLM 답변 하단 "🔍 디버깅 정보 (도구 호출 내역)" expander에서 RAG 결과 + 툴 invocation 확인
- `print(...)` 문은 Streamlit Cloud 로그에서 확인 가능
- 세션 초기화: 사이드바 "대화 새로고침 🔄" 버튼 또는 언어 모드 변경

### 9.4 배포 (Streamlit Cloud)
1. Github push → Streamlit Cloud에서 자동 배포
2. `.streamlit/secrets.toml`에 `OPENAI_API_KEY = "sk-..."` 등록
3. `requirements.txt` 자동 설치
4. 메모리 제약: Chroma + 임베딩 로드로 초기 메모리 ~300MB

---

## 10. 라이선스 & 연락처

- 국립어린이과학관 공식 데이터 기반 (출처: `https://www.csc.go.kr`)
- 대표 전화: 02-3668-3350
- 위치: 서울특별시 종로구 창경궁로 215 (와룡동 2-1)
- 가까운 지하철역: 4호선 혜화역 4번 출구 (도보 15분), 1호선 종로5가역 2번 출구 (도보 20분)
- **전용 주차장 없음** — 대중교통 이용 권장

---

**이 문서로 충분하지 않은 부분이 있다면**: 해당 파일의 실제 코드를 `@file_path:line-range` 형식으로 인용해서 질문하면 정확한 답변을 받을 수 있습니다.
