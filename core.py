# core.py - 핵심 시스템 통합
# config.py + rag.py + tools.py + utils.py + multilingual_loader.py 통합

import os
import glob
import re
import requests
import pandas as pd
import urllib3
import time
import streamlit as st
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# 정적 사전 번역 (4개 언어 핵심 FAQ) — LLM 번역 우회용
from static_translations import get_static_answer, get_operating_hours_text

# SSL 경고 무시
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================================================
# CONSTANTS - 상수 정의
# ============================================================================

MUSEUM_BASE_URL = "https://www.csc.go.kr"

# 국립어린이과학관 주요 URL 맵
CSC_URLS = {
    # 홈페이지
    "홈페이지": f"{MUSEUM_BASE_URL}/index.do",
    
    # 과학관 소개
    "인사말": f"{MUSEUM_BASE_URL}/new1/introduce/introduce.jsp",
    "연혁": f"{MUSEUM_BASE_URL}/new1/introduce/overview.jsp",
    "조직도": f"{MUSEUM_BASE_URL}/new1/introduce/organization.jsp",
    "시설안내": f"{MUSEUM_BASE_URL}/new1/information/facility.jsp",
    
    # 관람안내
    "이용안내": f"{MUSEUM_BASE_URL}/new1/information/tourinfo.jsp",
    "오시는길": f"{MUSEUM_BASE_URL}/new1/information/direction.jsp",
    "교통안내": f"{MUSEUM_BASE_URL}/new1/information/direction.jsp",
    "자주묻는질문": f"{MUSEUM_BASE_URL}/boardList.do?bbspkid=10&type=F&page=1",
    "공지사항": f"{MUSEUM_BASE_URL}/boardList.do?bbspkid=22",
    
    # 예약
    "예약안내": f"{MUSEUM_BASE_URL}/new1/reservation/guide.jsp",
    "개인예약": f"{MUSEUM_BASE_URL}/new1/reservation/reservation_person.jsp",
    "단체예약": f"{MUSEUM_BASE_URL}/new1/reservation/reservation_group.jsp",
    "교육예약": f"{MUSEUM_BASE_URL}/new1/reservation/education_creation.jsp",
    
    # 전시관 - 상설전시관 (5개 놀이터)
    "탐구놀이터": f"{MUSEUM_BASE_URL}/new1/centers/exhibitinfo_1.jsp",
    "관찰놀이터": f"{MUSEUM_BASE_URL}/new1/centers/exhibitinfo_2.jsp",
    "행동놀이터": f"{MUSEUM_BASE_URL}/new1/centers/exhibitinfo_3.jsp",
    "생각놀이터": f"{MUSEUM_BASE_URL}/new1/centers/exhibitinfo_4.jsp",
    "AI놀이터": f"{MUSEUM_BASE_URL}/new1/centers/exhibitinfo_5.jsp",
    
    # 전시관 - 천문우주
    "천체투영관": f"{MUSEUM_BASE_URL}/new1/centers/space.jsp",
    "천체관측소": f"{MUSEUM_BASE_URL}/new1/centers/space_2.jsp",
    "메타버스과학관": f"{MUSEUM_BASE_URL}/new1/centers/metaverse.jsp",
    
    # 과학교육
    "과학교육실": f"{MUSEUM_BASE_URL}/new1/centers/ckdwkr.jsp",
    "창작교실1": f"{MUSEUM_BASE_URL}/new1/centers/exhibitinfo_49.jsp",
    "창작교실2": f"{MUSEUM_BASE_URL}/new1/centers/exhibitinfo_50.jsp",
    "창작교실3": f"{MUSEUM_BASE_URL}/new1/centers/exhibitinfo_51.jsp",
    "창작교실4": f"{MUSEUM_BASE_URL}/new1/centers/exhibitinfo_58.jsp",
    "어린이교실": f"{MUSEUM_BASE_URL}/new1/centers/exhibitinfo_59.jsp",
    
    # 문화행사
    "과학쇼": f"{MUSEUM_BASE_URL}/new1/centers/exhibitinfo_57.jsp",
    "전시해설": f"{MUSEUM_BASE_URL}/new1/centers/explanation.jsp",
}

# RAG vector DB core data (LLM reference)
# NOTE: Primary source is CSV files in data/ and data/pages/
# STATIC_EXHIBIT_INFO provides minimal fallback for critical info
STATIC_EXHIBIT_INFO = {
    "국립어린이과학관": "서울특별시 종로구 창경궁로 215. 4호선 혜화역 4번 출구 도보 10분. 문의: 02-3668-3350",
    "운영시간": "관람시간 09:30~17:30, 입장마감 16:30. 휴관: 매주 월요일, 1월 1일, 설날·추석 당일",
    "관람료": "상설전시관 성인 2,000원/청소년 1,000원/초등 1,000원/유아 무료. 천체투영관 성인 1,500원/청소년·초등 1,000원/유아(4~6세) 1,000원",
}

# ============================================================================
# RULES - 규칙 및 로직 함수
# ============================================================================

def route_intent(text: str) -> str:
    """사용자 질문의 의도를 파악하여 라우팅.

    반환 값:
    - "notice": 공지사항 조회 (크롤링 + 텍스트 가공)
    - "basic": 정적 FAQ (answer_rule_based 의 정형 답변)
    - "llm_agent": LangGraph ReAct 에이전트 (RAG + 도구 호출 필요한 동적 질의)
    """
    lowered = text.lower().strip()

    # 길찾기 2단계: 이전 답변이 출발지를 물어본 상태
    if st.session_state.get("awaiting_directions_origin") and lowered:
        st.session_state["awaiting_directions_origin"] = False
        st.session_state["directions_origin"] = text.strip()
        return "llm_agent"  # search_directions 툴 호출 필요

    # 공지사항 → 전용 크롤러 경로
    if any(token in lowered for token in ["공지", "공지사항", "알림"]):
        return "notice"

    # ── 동적 질의 (LLM 에이전트로 보내야 하는 케이스) ────────────────────────
    # 1) 길찾기에 출발지가 명시 → search_directions 툴 필요
    has_origin = bool(re.search(r"(에서|출발|출발지|역에서|집에서)", text)) or bool(re.search(r"[가-힣A-Za-z0-9]{2,}역", text))
    if has_origin and any(token in lowered for token in ["가", "길", "오", "교통", "어떻게"]):
        return "llm_agent"

    # 2) 특정 날짜 휴관 여부 → check_museum_closed_date 툴 필요
    if re.search(r"(\d+월\s*\d+일|\d{4}-\d{2}-\d{2}|내일|모레|이번\s*주|다음\s*주)", text) and any(
        t in lowered for t in ["열", "휴관", "운영", "여나", "여는", "쉬"]
    ):
        return "llm_agent"

    # 3) 천체투영관 영상/줄거리 자세히 → 동적 RAG 답변
    if "천체투영관" in lowered and any(t in lowered for t in ["뭐", "내용", "줄거리", "어떤", "코코몽", "키츠", "다이노소어", "바니"]):
        return "llm_agent"

    # ── 정적 FAQ 키워드 (basic 라우팅) ───────────────────────────────────────
    basic_keywords = [
        # 층/시설
        "층별", "1층", "2층", "층 안내", "게이트", "입구", "출구",
        "시설", "편의시설", "의무실", "수유실", "유아휴게", "물품보관",
        "보관함", "락커", "유모차", "휠체어", "대여",
        "안내데스크", "매표소", "꿈트리", "휴게실", "영유아놀이터", "하늘마당", "옥상",
        # 전시관/동선
        "전시관", "놀이터 안내", "동선", "연령", "나이", "추천", "유아", "초등", "저학년", "고학년", "4~7",
        # 프로그램/시간표
        "오늘의 프로그램", "오늘 프로그램", "프로그램", "과학쇼", "전시해설",
        "천체투영관 시간", "투영관 시간", "시간표", "상영", "회차", "빛놀이터",
        # 예약
        "예약", "예매", "방문신청", "방문 신청", "단체예약", "개인예약", "교육예약",
        "모바일 qr", "입장권", "정원", "1600",
        # 운영시간
        "운영", "휴관", "몇 시", "마감", "여나", "닫나", "열어", "여는 시간", "운영시간",
        # 가격
        "관람료", "입장료", "요금", "가격", "얼마",
        # 주차
        "주차", "주차장",
        # 길찾기 (출발지 없는 일반 질문 → directions 카테고리가 출발지 묻기)
        "오시는길", "오는길", "교통", "길찾기", "주소", "위치", "어디",
    ]
    if any(token in lowered for token in basic_keywords):
        return "basic"

    # 자유 질문, 전시물 상세, 과학 원리 등 → LLM + RAG
    return "llm_agent"

def classify_basic_category(message: str) -> str:
    """기본 질문 카테고리 분류.

    우선순위: 구체적 키워드를 가진 카테고리가 앞에 와야 '얼마', '시간' 같은
    범용 키워드로 잘못 분류되는 것을 막을 수 있다.
    특히 parking 을 admission_fee 앞에 두어 "주차비 얼마?" → parking 으로 분류되게 한다.
    """
    lowered = message.lower()
    rules = [
        # 가장 구체적인 카테고리 먼저
        ("reservation_guide", ["예약", "예매", "방문신청", "방문 신청", "단체예약", "개인예약", "교육예약", "모바일 qr", "입장권", "정원", "1600"]),
        ("planetarium_timetable", ["천체투영관 시간표", "투영관 시간표", "천체투영관 시간", "투영관 시간", "상영", "회차", "프로그램(투영관)", "코코몽", "키츠", "바니", "다이노"]),
        ("today_programs",  ["오늘의 프로그램", "오늘 프로그램", "오늘 뭐", "과학쇼", "전시해설", "오늘 해", "오늘의 행사", "빛놀이터"]),
        # parking 은 admission_fee 앞에 위치 (주차비/주차료의 '비/료' 를 admission 이 먹지 않도록)
        ("parking",         ["주차", "주차장", "주차비", "주차료", "주차 요금", "주차 되", "주차되", "파킹", "parking", "car park"]),
        ("admission_fee",   ["관람료", "입장료", "요금", "가격", "얼마", "얼만", "비용", "유료", "무료", "할인", "티켓값", "표값"]),
        ("exhibit_guide",   ["전시관", "전시관 안내", "놀이터 안내", "ai놀이터", "행동놀이터", "관찰놀이터", "탐구놀이터", "생각놀이터"]),
        ("directions",      ["오시는길", "오는길", "오시는 길", "오는 길", "교통", "길찾기", "길 찾", "주소", "위치", "어떻게 가", "어떻게 오", "어디에 있", "어디야", "가는 방법", "가는길"]),
        ("facility_amenities", ["시설", "편의시설", "의무실", "수유실", "유아휴게", "물품보관", "보관함", "락커", "유모차", "휠체어", "대여", "안내데스크", "매표소", "꿈트리", "휴게실", "영유아놀이터", "하늘마당", "옥상", "화장실"]),
        ("route_by_age",    ["동선", "연령별", "연령", "나이", "추천 코스", "추천 동선", "추천해", "4~7", "7세", "유아", "초등", "저학년", "고학년", "몇 살", "몇살"]),
        ("floor_guide",     ["층별", "층 안내", "1층", "2층", "3층", "게이트", "입구", "출구", "어느 층", "무슨 층"]),
        ("operating_hours", ["운영시간", "운영 시간", "운영", "휴관", "휴무", "몇 시", "몇시", "마감", "언제 열", "언제 닫", "여는 시간", "닫는 시간", "개관", "폐관", "여나요", "여나"]),
    ]
    for category, keywords in rules:
        if any(keyword in lowered for keyword in keywords):
            return category
    return "operating_hours"

def check_closed_date(target_date: datetime) -> tuple[bool, str]:
    """특정 날짜의 휴관 여부 확인"""
    month_day = target_date.strftime("%m-%d")
    weekday = target_date.weekday()
    weekday_kr = ["월", "화", "수", "목", "금", "토", "일"][weekday]

    if month_day == "01-01":
        return (True, f"{target_date.strftime('%m월 %d일')}({weekday_kr}요일)은 휴관일(1월 1일)입니다.")
    if weekday == 0:
        return (True, f"{target_date.strftime('%m월 %d일')}({weekday_kr}요일)은 정기휴관일(월요일)입니다.")

    return (False, f"{target_date.strftime('%m월 %d일')}({weekday_kr}요일)은 정상 운영일입니다.")

def get_today_status() -> str:
    """오늘 과학관 운영 상태 확인"""
    now_utc = datetime.now(timezone.utc)
    now = now_utc + timedelta(hours=9)  # KST = UTC+9
    is_closed, status_msg = check_closed_date(now)
    
    if is_closed:
        return status_msg
    
    current = now.time()
    if (current.hour < 9) or (current.hour == 9 and current.minute < 30):
        return "아직 개관 전이에요. 관람시간은 09:30~17:30이고, 입장 마감은 16:30이에요."
    if current.hour > 17 or (current.hour == 17 and current.minute >= 30):
        return "오늘 관람 시간은 종료됐어요. 관람시간은 09:30~17:30이고, 입장 마감은 16:30이에요."
    return "현재 정상 운영 중입니다! 관람시간은 09:30~17:30이고, 입장 마감은 16:30이에요."

def answer_rule_based(intent: str, message: str, mode: str) -> str:
    """규칙 기반 답변 생성"""
    if intent == "notice":
        m_num = re.search(r"공지\s*(?P<num>\d+)\s*(번)?\s*(자세히|상세)", message)
        m_pkid = re.search(r"pkid=(?P<pkid>\d+)", message)

        if m_pkid:
            return get_notice_detail_text(m_pkid.group("pkid"))

        if m_num:
            num = int(m_num.group("num"))
            cached = st.session_state.get("latest_notices")
            if isinstance(cached, list) and 1 <= num <= len(cached):
                _, href = cached[num - 1]
                m2 = re.search(r"pkid=(?P<pkid>\d+)", href)
                if m2:
                    return get_notice_detail_text(m2.group("pkid"))
            return "최근 공지사항 번호를 찾지 못했어요. 먼저 \"공지사항 알려줘\"라고 물어봐 주세요."

        text = get_latest_notices_text(limit=5)
        latest = []
        for line in text.splitlines():
            m = re.match(r"\d+\.\s+(?P<title>.+)", line.strip())
            if m:
                latest.append((m.group("title"), ""))
            elif line.strip().startswith("-") and latest:
                latest[-1] = (latest[-1][0], line.strip().lstrip("- "))
        if latest:
            st.session_state["latest_notices"] = latest
        return text
    if intent == "basic":
        category = classify_basic_category(message)
        if category == "floor_guide":
            return """층별 안내를 한눈에 보기 쉽게 정리해드릴게요! 😊

## 1층
- 매표소·안내데스크
- AI놀이터 / 생각놀이터 / 행동놀이터
- 천체투영관, 과학극장
- 어린이교실
- 수유실·유아휴게실, 의무실
- 물품보관함(락커)
- 유모차·휠체어 대여(신분증 제시)
- 꿈트리 동산(창경궁 방향)

## 2층
- 빛놀이터 / 탐구놀이터 / 관찰놀이터
- 창작교실
- 휴게실(영유아놀이터 포함)
- 물품보관함(락커)

## 3층(옥상)
- 하늘마당(옥상)
  - 과학관 퇴장 후 오른쪽으로 돌아 언덕을 따라 올라가면 돼요.

## (중요) 입구/출구 안내
- **입구: 2층 게이트**
- **출구: 1층 게이트**

## 입장 팁
- 1층 매표소에서 매표(또는 예약 확인) 후, 2층 입구 게이트로 들어오세요.
- 과학관 입구(2층)에서 예약한 입장권(QR코드) 확인 후 관람해주시기 바랍니다."""

        if category == "facility_amenities":
            return """편의시설 안내를 한 번에 정리해서 알려드릴게요! 😊

## 1층
- **의무실(First Aid)**: 1층 / 일반의약품 구비
- **수유실·유아휴게실(Baby Care)**: 1층 / 싱크대, 전자레인지, 쇼파 등
- **물품보관함(Locker)**: 1층 매표소 인근
- **유모차·휠체어 대여**: 1층 매표소·안내데스크에서 신분증 제시 후 대여
  - 수량: 유모차 5대 / 휠체어 2대
  - 유모차 이용: 36개월 이하
- **매표소·안내데스크(Tickets & Information)**: 1층
- **꿈트리 동산(Little Library)**: 1층(창경궁 방향)

## 2층
- **휴게실(Lounge)**: 2층
- **영유아놀이터(Baby Lounge)**: 2층 휴게실 내
- **물품보관함(Locker)**: 2층 휴게실 내부

## 3층(옥상)
- **하늘마당(Courtyard)**: 3층 옥상
  - 안내: 과학관 퇴장 후 오른쪽으로 돌아 언덕을 따라 올라가면 돼요.

대표번호는 모두 동일해요: **02-3668-3350**"""

        if category == "exhibit_guide":
            return """🎨 전시관(놀이터) 소개

국립어린이과학관의 상설전시관은 **1층과 2층에 총 6개의 놀이터 + 천체투영관 + 과학극장**으로 구성되어 있어요. 각 공간은 주제와 추천 연령이 달라요.

## 1층 전시관

### 🤖 AI놀이터 (1층)
- **주제**: 인공지능의 원리와 미션형 체험
- **추천 연령**: 초등 중·고학년 이상
- **특징**: 얼굴 인식, 음성 AI, 기계학습 기반 인터랙티브 전시
- **추천 체류시간**: 30~40분
- **관람 팁**: 미션을 해결하는 형태라 시간 여유를 두고 방문하세요.

### 🏃 행동놀이터 (1층)
- **주제**: 몸을 움직이며 배우는 건강·운동 과학
- **추천 연령**: 유아~초등 저학년
- **특징**: 점프·균형·반응속도 등 신체 활동 기반 체험
- **추천 체류시간**: 20~30분

### 💡 생각놀이터 (1층)
- **주제**: 어린이의 사고력·창의력 자극
- **참고**: 2026년 5월 개관 예정 (정확한 개관일은 공식 공지사항 확인)

### 🌌 천체투영관 (1층)
- **주제**: 돔 스크린으로 관람하는 별자리·우주 영상
- **추천 연령**: 연나이 4세 이상 (미취학은 보호자 동반 필수)
- **예약**: **100% 사전예약 필수** (상설전시관 예약도 별도 필요)
- **회차**: 하루 6회차 (40분 단위)

### 🎭 과학극장 (1층)
- **주제**: 사이언스랩·로봇쇼 공연
- **이용**: 무료 / 선착순 입장 / 정원 95명
- **입장 규칙**: 공연 시작 25분 전부터 입장 가능, 35분 전 마감

## 2층 전시관

### ✨ 빛놀이터 (2층)
- **주제**: 빛·숲·생태의 몰입형 미디어 체험
- **추천 연령**: 전 연령 (특히 유아~초등 저학년에 인기)
- **운영**: 회차제로 입장 (현장 안내 표지 확인)
- **추천 체류시간**: 30~40분

### 🔬 탐구놀이터 (2층)
- **주제**: 생활 속 도구·에너지·기계의 원리
- **추천 연령**: 초등 저~고학년
- **특징**: 직접 만지고 실험하는 체험형 전시
- **추천 체류시간**: 30~45분

### 🦖 관찰놀이터 (2층)
- **주제**: 공룡·화석·생물 표본 관찰
- **추천 연령**: 유아~초등 저학년 (공룡 좋아하는 친구들에게 인기!)
- **특징**: 실제 화석 및 대형 공룡 모형
- **추천 체류시간**: 20~30분

## 관람 동선 팁
- **입구는 2층 게이트, 출구는 1층 게이트**예요. 자연스러운 동선은 **2층 → 1층** 순서입니다.
- 점심·간식은 2층 **휴게실(영유아놀이터 포함)**에서 드실 수 있어요.
- 천체투영관·과학쇼 시간을 먼저 정해두고 전시 관람 사이에 끼워넣는 게 효율적입니다.

💬 특정 놀이터가 궁금하면 **"AI놀이터 전시물 뭐 있어?"** 처럼 이름을 말해주세요. 더 자세한 전시물 목록을 찾아드릴게요.

⚠️ 전시관 구성·운영은 변경될 수 있으니 방문 전 공식 홈페이지(www.csc.go.kr)에서 확인해 주세요."""

        if category == "route_by_age":
            return """🧒 연령별 추천 관람 동선

자녀 연령에 맞춰 과학관을 효율적으로 돌아볼 수 있는 동선을 추천해드려요. **전체 관람 시간은 2~3시간** 정도가 적당합니다.

## 👶 4~7세 유아

### 추천 동선 (2시간~2시간 30분)
1. **2층 입구 입장** → 🦖 관찰놀이터 (2층) — 공룡·화석 관람 (약 25분)
2. ✨ 빛놀이터 (2층) — 미디어 체험 (약 30분, 회차 확인)
3. 🍪 2층 휴게실·영유아놀이터 — 간식·수유·쉼 (약 20분)
4. 🏃 행동놀이터 (1층) — 몸으로 노는 체험 (약 25분)
5. 🎭 과학극장 (1층) — 11:40 또는 13:40 공연 (15분)

### 팁
- 유아는 집중시간이 짧으므로 **중간 휴식**을 꼭 챙겨주세요.
- 2층 **영유아놀이터**는 미취학 전용 공간이라 보호자와 함께 쉬기 좋습니다.
- **유모차 대여(5대)**: 1층 안내데스크 / 신분증 지참 / 36개월 이하 이용.
- 천체투영관 관람 시 **보호자 동반 필수** (4세 이상 입장).

## 🎒 초등 저학년 (8~10세)

### 추천 동선 (2시간 30분~3시간)
1. **2층 입구 입장** → 🔬 탐구놀이터 (2층) — 에너지·도구 실험 (약 40분)
2. 🦖 관찰놀이터 (2층) — 공룡·화석 (약 20분)
3. 🎭 과학극장 (1층) — 사이언스랩 또는 로봇쇼 (15분)
4. 🏃 행동놀이터 (1층) — 신체 활동 체험 (약 20분)
5. 🌌 천체투영관 (1층) — 회차 선택 (40분, 예약 필수)

### 팁
- 탐구놀이터에서 **직접 만지는 체험**이 많아 호기심을 자극합니다.
- 과학쇼는 **공연 시작 25분 전부터 입장 가능**하고 **35분 전 마감**이니 시간 여유를 두세요.
- '전시해설(「헬로 다이노!」/「짹짹 새 탐험대」)' 프로그램도 15분 정도로 좋아요.

## 🧑‍🎓 초등 고학년 (11~13세)

### 추천 동선 (2시간 30분~3시간)
1. **2층 입구 입장** → 🔬 탐구놀이터 (2층) — 원리 탐구 중심 (약 40분)
2. 🤖 AI놀이터 (1층) — AI 미션 체험 (약 40분)
3. 🎙️ 전시해설 프로그램 — 「헬로 다이노!」 또는 「짹짹 새 탐험대」 (15분)
4. 🌌 천체투영관 (1층) — '다이노소어'(14:00) 등 회차 추천 (예약 필수)

### 팁
- AI놀이터는 **미션 해결형**이라 집중이 필요합니다.
- 천체투영관의 '다이노소어'(14:00 회차)는 **초등학생 이상** 권장 콘텐츠입니다.
- 남는 시간이 있다면 1층 **행동놀이터**로 에너지를 발산해도 좋아요.

## 👨‍👩‍👧 형제·자매 동반 방문
- 연령 차가 큰 형제(예: 유아+고학년)는 **2층 빛놀이터·관찰놀이터**에서 시작하세요. 모두가 즐길 수 있는 구역입니다.
- 부모가 번갈아 담당하며 AI놀이터(고학년)와 행동놀이터(유아)로 나눠 이동하는 것도 좋은 방법입니다.
- 휴게실이 **2층**에 있으니 쉬는 시간은 2층 위주로 계획하세요.

## 🎟️ 예약·입장 핵심
- 하루 **최대 1,600명**으로 제한 → 주말·공휴일은 예약 권장.
- **천체투영관은 100% 사전예약**.
- 인터넷 예매는 관람일 **14일 전 0시**부터 가능.

💬 "6살인데 어디부터 보면 좋을까?" 처럼 **구체적 나이와 현재 위치(1층/2층)**를 말해주시면 더 정확한 동선을 짜드릴게요!"""

        if category == "today_programs":
            now_utc = datetime.now(timezone.utc)
            now_kst = now_utc + timedelta(hours=9)
            month = now_kst.month
            month_label = f"{month}월"
            weekday_kr = ["월", "화", "수", "목", "금", "토", "일"][now_kst.weekday()]
            is_weekend = now_kst.weekday() >= 5

            science_show_type = "사이언스랩" if month in [1, 3, 5, 7, 9, 11] else "로봇쇼"
            explanation_type = "헬로 다이노!" if month in [1, 3, 5, 7, 9, 11] else "짹짹 새 탐험대"

            if ("전시해설" in message) and ("자세" in message or "상세" in message):
                return """전시해설 안내입니다. 😊

## 기본 정보
- 자유롭게 참여하며 즐기는 전시품 과학해설
- 참여대상: 어린이 및 보호자
- 운영시간: 약 15분
- 비용: 무료

## 프로그램(월별)
- **스폿해설 「헬로 다이노!」** (1, 3, 5, 7, 9, 11월)
  - 백악기 시대 공룡 이야기로 과학을 쉽고 재미있게 알아봐요.
  - 장소: 2층 공룡 전시물 앞
  - 참여방법: 자유 관람
- **전시톡톡해설 「짹짹 새 탐험대」** (2, 4, 6, 8, 10, 12월)
  - 다양한 새들의 특징을 퀴즈로 알아봐요.
  - 장소: 1층 과학극장
  - 참여방법: 선착순 입장 및 관람
  - 입장 안내: 35분부터 입장 가능 / 39분 59초에 입장 마감
  - 진행 중에는 입장/퇴장 불가
- **단체해설 「북적북적 과학관」** (개학기간 3~7월)
  - 단체 예약 프로그램(신청콕으로 예약)
  - 장소: 2층 공룡 전시물

## 시간표(요약)
- 방학기간(1~2월 / 8월)
  - 14:40, 15:40: 해설(공룡/새) (화~일)
- 개학기간(3~7월 / 9~12월)
  - 화~금 10:40: 단체해설
  - 토/일 10:40: 해설(공룡/새)
  - 토/일 14:40, 15:40: 해설(공룡/새)

* 프로그램 내용 및 시간표는 운영 상황에 따라 변동될 수 있어요. 공휴일은 주말 일정과 동일하게 운영합니다."""

            if ("과학쇼" in message) and ("자세" in message or "상세" in message):
                return """과학쇼 안내입니다. 😊

## 기본 정보
- 자유롭게 참여하며 즐기는 **과학 실험 & 로봇쇼**
- 참여대상: 어린이 및 보호자
- 소요시간: 약 15분
- 정원: 95명
- 비용: 무료
- 장소: **1층 과학극장**

## 이용방법(중요)
- 선착순 입장 및 관람
- **35분부터 입장 가능**, **39분 59초에 입장 마감**
- 원활한 진행을 위해 프로그램 진행 중에는 **입장/퇴장 불가**

## 프로그램 구성
- **사이언스랩**: 드라이아이스/비눗방울/공기/힘 등 과학 원리를 실험 시연으로 재미있게 전달
- **로봇쇼 「로봇 프렌즈!」**: 다양한 로봇을 통해 구조와 작동 원리를 쉽게 이해

## 시간표(요약)
### 1·3·5·7·9·11월
| 시간 | 화 | 수 | 목 | 금 | 토 | 일 |
| --- | --- | --- | --- | --- | --- | --- |
| 11:40 | 사이언스랩 | 사이언스랩 | 사이언스랩 | 사이언스랩 | 사이언스랩 | 사이언스랩 |
| 13:40 | 사이언스랩 | 사이언스랩 | 사이언스랩 | 사이언스랩 | 사이언스랩 | 사이언스랩 |

### 2·4·6·8·10·12월
| 시간 | 화 | 수 | 목 | 금 | 토 | 일 |
| --- | --- | --- | --- | --- | --- | --- |
| 11:40 | 로봇쇼 | 로봇쇼 | 로봇쇼 | 로봇쇼 | 로봇쇼 | 로봇쇼 |
| 13:40 | 로봇쇼 | 로봇쇼 | 로봇쇼 | 로봇쇼 | 로봇쇼 | 로봇쇼 |

추가 메모
- 5월: 비눗방울 / 7월: 공기 / 9월: 힘 / 11월: 공기
- 로봇쇼 예시: 축구로봇, 댄스로봇, 로봇개

* 프로그램 내용 및 시간표는 운영 상황에 따라 변동될 수 있어요. 공휴일은 주말 일정과 동일하게 운영합니다."""

            if ("천체투영관" in message) and ("자세" in message or "상세" in message):
                return """천체투영관 안내입니다. 🌙

- 어떤 곳인가요?
  - 돔 스크린으로 별자리/우주 영상을 관람하고 해설을 듣는 프로그램이에요.

- 예약/입장
  - 예약이 필요한 경우가 많고, 회차별 정원이 있어요.

원하시면 '오늘 천체투영관 시간' 또는 '천체투영관 예약 방법'이라고 물어보면 더 자세히 안내해드릴게요."""

            if ("빛놀이터" in message) and ("자세" in message or "상세" in message):
                return """빛놀이터 안내입니다. ✨ (포스터 기준)

## 운영 기간
- 2026.04.01 ~ 2026.04.30

## 회차(예시)
- 10:00 / 10:40 / 11:20
- 12:00 / 14:00 / 14:40
- 15:20 / 16:00 / 16:40

## 빛놀이터 이용안내(요약)
- 미디어 체험은 **회차별로 입장**할 수 있어요.
- 입장/퇴장 동선이 따로 있을 수 있으니, **현장 안내 표지(입구/출구)**를 따라 이동해 주세요.

※ 운영 회차/동선은 운영 상황에 따라 변동될 수 있어요."""

            science_show_times = ["11:40", "13:40"]

            explanation_times = []
            if month in [1, 2, 8]:
                explanation_times = ["14:40", "15:40"]
            else:
                if is_weekend:
                    explanation_times = ["10:40", "14:40", "15:40"]
                else:
                    explanation_times = ["10:40", "14:40", "15:40"]

            planetarium_rows = [
                ("10:00~10:40", "별자리 해설 + 코코몽 우주탐험"),
                ("11:00~11:40", "별자리 해설 + 길냥이 키츠 슈퍼문 대모험"),
                ("12:00~12:40", "바니 앤 비니"),
                ("14:00~14:40", "다이노소어"),
                ("15:00~15:40", "별자리 해설 + 길냥이 키츠 우주정거장의 비밀"),
                ("16:00~16:40", "바니 앤 비니"),
            ]

            planetarium_lines = "\n".join([f"- {t}: {p}" for t, p in planetarium_rows])

            return f"""**{now_kst.strftime('%Y년 %m월 %d일')} {weekday_kr}요일이에요!**

오늘(요약) 프로그램 시간표를 안내해드릴게요. *(운영 상황에 따라 변동될 수 있어요.)*

## 과학쇼 (1층 과학극장)
- 오늘 프로그램: **{science_show_type}** ({month_label} 기준)
- 시간: {", ".join(science_show_times)}

## 전시해설
- 오늘 프로그램: **{explanation_type}** ({month_label} 기준)
- 시간: {", ".join(explanation_times)}

## 천체투영관 (시간/프로그램)
{planetarium_lines}

원하시면 "과학쇼 자세히", "전시해설 자세히", "천체투영관 시간표"처럼 말해주면 안내 규정/입장방법까지 더 자세히 설명해줄게요."""

        if category == "reservation_guide":
            return """예약안내를 친절하게 정리해서 알려드릴게요! 😊

## 예약 기본 안내
- 하루 입장 인원은 **최대 1,600명**으로 제한됩니다.

## (중요) 어린이 동반 없는 성인/청소년 관람객 안내
- **어린이(신체연령 초등학생 이하)를 동반하지 않은 성인 및 청소년**은 관람을 위해 사전 협의가 필요합니다.
- 방문을 원하실 경우, **방문 3일 전까지** 방문신청서를 담당자 메일로 보내주세요.
  - 담당자 메일: **proxima11@korea.kr**
  - 방문 신청서 양식은 ‘성인 및 청소년 관람객 입장안내’ 게시글의 첨부파일을 확인해 주세요.

## 체험별 사전예약 비율(요약)
- **상설전시관**
  - 인터넷 예매: (3~6월/9~12월) 평일 50%, 주말 75% / (7~8월/1~2월) 평일 75%, 주말 75%
  - 현장 판매: (3~6월/9~12월) 평일 50%, 주말 25% / (7~8월/1~2월) 평일 25%, 주말 25%
  - 단체: **사전예약 필수**(단체 관람 이용안내 참조)
- **천체투영관**
  - 인터넷 예매: **100% (사전예약 필수)**

## 예약 가능 기간(개인/단체)
- 예약 판매를 우선으로 하며, **잔여석에 한해 현장판매**를 진행합니다.
- 인터넷 예매는 관람일 **2주(14일) 전부터** 가능합니다.
  - 예1) 오늘이 5월 8일이면, 5월 22일(14일 후)까지 예약 가능(하루씩 자동 연장)
  - 예2) 5월 22일 예약은 5월 8일 **00:00부터** 가능
- 예약 기간은 과학관 운영 사정에 따라 변경될 수 있습니다.(변동 시 별도 공지)
- **평일에 예약 가능**하며, **주말 및 공휴일에는 예약을 받지 않습니다.**

## 예약 시 유의사항
- 상설전시관/천체투영관은 **개인예약 선택 후** 예약할 수 있습니다.
- 예약 신청 시 받은 **문자 메시지(URL)**를 보관해 주세요. (신청 확인/취소에 사용)
- **5명 이상 어린이 기관**은 단체예약 페이지에서 예약해 주세요.
- 예약한 **입장시간을 지켜** 입장해 주세요.
- 예약 고객은 예약 완료 후 발권되는 **‘모바일 QR입장권’**으로 입장할 수 있습니다.
- **결제까지 완료**되어야 최종 예약 완료입니다.

원하시면 지금 상황(개인/단체/교육, 관람 날짜, 인원, 어린이 동반 여부)을 말해주시면 딱 맞게 안내해드릴게요!"""

        if category == "planetarium_timetable":
            timetable = """
| 회차 | 시간 | 프로그램 | 정원 | 권장연령 |
| --- | --- | --- | ---: | --- |
| 1 | 10:00 ~ 10:40 | 별자리 해설 + 코코몽 우주탐험 | 65명 | 유아 이상 |
| 2 | 11:00 ~ 11:40 | 별자리 해설 + 길냥이 키츠 슈퍼문 대모험 | 65명 | 유아 이상 |
| 3 | 12:00 ~ 12:40 | 바니 앤 비니 | 65명 | 유아 이상 |
| 4 | 14:00 ~ 14:40 | 다이노소어 | 65명 | 초등학생 이상 |
| 5 | 15:00 ~ 15:40 | 별자리 해설 + 길냥이 키츠 우주정거장의 비밀 | 65명 | 유아 이상 |
| 6 | 16:00 ~ 16:40 | 바니 앤 비니 | 65명 | 유아 이상 |
"""

            return f"""천체투영관 시간표를 정리해드릴게요! 🌙

## 오늘/상설 시간표(안내)
{timetable}

## 유의사항
- **연나이 4세 이상** 어린이부터 입장 가능합니다.
- 미취학 아동은 **보호자 동반 필수**(유아만 입장 불가)
- 상영 시작 이후에는 안전상 **입장/퇴장 불가**
- 환불은 현장에서 **상영 시작 30분 전까지** 가능
- 내부 음식물 섭취 금지, 휴대전화는 진동으로 설정
- 천체투영관 예약은 **2주 전 0시 오픈**됩니다.(상설전시장 예약도 필수)
"""

        if category == "admission_fee":
            exhibit_table = """
#### 상설전시관(연나이 기준)

| 구분 | 개인 | 단체 | 대상 |
| --- | ---: | ---: | --- |
| 성인 | 2,000원 | 이용불가 | 19세 이상 |
| 청소년 | 1,000원 | 이용불가 | 13~18세 |
| 초등학생 | 1,000원 | 500원 | 7~12세 |
| 유아 | 무료 | 무료 | 6세 이하 |
| 우대고객 | 무료 | 이용불가 | 경로우대자, 장애인 등(증빙 필요) |
"""
            planet_table = """
#### 천체투영관(연나이 기준)

| 구분 | 개인 | 단체 | 대상 |
| --- | ---: | ---: | --- |
| 성인 | 1,500원 | 이용불가 | 19세 이상 |
| 청소년 | 1,000원 | 이용불가 | 13~18세 |
| 초등학생 | 1,000원 | 1,000원 | 7~12세 |
| 유아 | 1,000원 | 1,000원 | 4~6세(성인 보호자 동반 및 결제 시) |
| 우대고객 | 1,000원 | 이용불가 | 경로우대자, 장애인 등(증빙 필요) |
"""
            notes = """
#### 요금 할인/면제(요약)
- 우대고객은 **신분증/증명서 지참 필수**
- 중증장애(1~3급): 본인 + 동반 보호자 1인 혜택(상설전시관 무료 등)
- 경증장애(4급 이상): 본인 혜택
- 다자녀카드: 상설전시관 개인요금 **50% 할인**(신분증 + 카드 지참)
"""
            prefix = "관람료를 보기 쉽게 정리해드릴게요! 💸\n" if mode == "어린이" else "관람료 안내입니다.\n"
            return f"{prefix}{exhibit_table}\n\n{planet_table}\n\n{notes}"

        if category == "operating_hours":
            status = get_today_status()
            if mode == "어린이":
                return f"""🕘 운영시간 안내

{status}

## ⏰ 기본 운영시간
- **관람시간**: 09:30 ~ 17:30
- **매표·입장 마감**: **16:30** (폐관 1시간 전)
- **추천 관람 시간**: 2~3시간

## 😴 쉬는 날 (휴관일)
- **매주 월요일** (월요일이 공휴일이면 개관하고, 다음 화요일에 대체 휴관할 수 있어요!)
- **1월 1일 (신정)**
- **설날 당일 · 추석 당일**
- 시설 점검·특별 행사 시에는 공식 홈페이지에 별도로 공지해요.

## 💡 과학관 방문 꿀팁
- 주말·공휴일에는 사람이 많으니 **평일 오전 10~11시**에 오면 여유롭게 체험할 수 있어요!
- 하루 입장 인원이 **최대 1,600명**으로 제한되어서, 붐비는 날에는 **예약하고 가는 게 안전**해요.
- **천체투영관은 100% 사전예약**이에요. 입장 14일 전 0시부터 예약할 수 있어!

## 📞 문의
- **대표전화**: 02-3668-3350
- **공식 홈페이지**: https://www.csc.go.kr

⚠️ 특별 연장·임시 휴관은 바뀔 수 있으니, 출발 전에 공식 홈페이지 공지사항을 꼭 확인해줘!"""
            return f"""🕘 운영시간 안내

{status}

## ⏰ 기본 운영시간
- **관람시간**: 09:30 ~ 17:30
- **매표 및 입장 마감**: **16:30** (폐관 1시간 전)
- **권장 관람 시간**: 2~3시간

## 😴 정기 휴관일
- **매주 월요일**
  - 월요일이 공휴일인 경우 **개관**하며, **다음 화요일에 대체 휴관**할 수 있습니다.
- **1월 1일** (신정)
- **설날 당일 · 추석 당일**
- **기타 임시 휴관**: 시설 점검·특별 행사 시 홈페이지 공지사항으로 별도 안내됩니다.

## 🎟️ 예약 및 입장 안내
- **하루 입장 인원 1,600명 제한** — 주말·공휴일은 조기 마감 가능
- 인터넷 예매는 관람일 **14일 전 0시부터** 가능
- 상설전시관은 일부 현장판매 병행, **천체투영관은 100% 사전예약 필수**
- **어린이 미동반 성인·청소년**은 방문 **3일 전까지** proxima11@korea.kr로 방문신청서 제출 필요

## 📞 문의
- **대표전화**: 02-3668-3350
- **공식 홈페이지**: https://www.csc.go.kr

⚠️ 임시 휴관·특별 연장 운영 등은 변경될 수 있으니, 방문 전 공식 홈페이지에서 꼭 확인해 주세요."""

        elif category == "parking":
            if mode == "어린이":
                return """🚗 주차 안내

### 핵심만 먼저!
- **국립어린이과학관 건물 안에는 주차장이 없어요.**
- 그래서 **지하철이나 버스**로 오는 게 제일 편해요!
- 차로 오게 되면 근처 유료 주차장을 써야 해요.

### 🚇 지하철로 오는 방법
- **4호선 혜화역 4번 출구** → 걸어서 약 10~15분
  - 혜화역에서 나오면 창경궁 방향(북쪽)으로 쭉 걸어오면 돼요!
- **1호선 종로5가역 2번 출구** → 걸어서 약 20분

### 🚌 버스로 오는 방법
- **창경궁 앞(홍화문)** 정류장에서 내려서 5분 정도 걸어오세요.
- 지나가는 버스 번호(예): 101, 102, 104, 106, 107, 108, 140, 150, 160 등

### 🅿️ 꼭 차로 와야 한다면
- **창경궁 주차장**(가장 가까워요) — 창경궁 홍화문 바로 옆
- **서울대학교병원 주차장** — 대학로 쪽
- **종로구청 주차장** — 종로3가 근처
- 주말이나 공휴일에는 주차장이 꽉 차 있을 수 있어요. 조금 일찍 오거나 대중교통을 추천해요!

### ♿ 도움이 필요하다면
- 혜화역 4번 출구에는 엘리베이터가 있어요.
- 과학관 1층 안내데스크에서 **유모차(5대)**, **휠체어(2대)**를 빌릴 수 있어요. (신분증 꼭 챙겨오기!)

### 📍 주소 & 연락처
- **주소**: 서울특별시 종로구 창경궁로 215 (와룡동 2-1)
- **전화**: 02-3668-3350

⚠️ 주차/교통 안내는 바뀔 수 있으니, 출발 전에 공식 홈페이지(www.csc.go.kr)에서 한 번 더 확인해줘!"""

            return """🚗 주차 안내

### 핵심 안내
- **국립어린이과학관은 전용 주차장이 없습니다.**
- 차량으로 오시는 경우 인근 유료 주차장을 이용하셔야 하며, **가능하면 대중교통 이용을 권장드립니다.**

### 🚇 지하철 (가장 빠르고 편리한 방법)
- **4호선 혜화역 4번 출구** → 도보 약 10~15분
  - 혜화역 4번 출구로 나와 창경궁로를 따라 북쪽(창경궁 방향)으로 직진하시면 됩니다.
- **1호선 종로5가역 2번 출구** → 도보 약 20분

### 🚌 버스
- **창경궁 정문(홍화문)** 정류장 하차 후 도보 약 5분
- 경유 노선(일부): 101, 102, 104, 106, 107, 108, 140, 150, 160 등
- 버스 도착 정보는 서울 버스 앱이나 정류장 전광판에서 확인 가능합니다.

### 🅿️ 차량 이용 시 (인근 유료 주차장)
- **창경궁 주차장** (가장 가까움) — 창경궁 홍화문 옆
  - 소액 단위 요금제 / **주말·공휴일에는 만차 가능성이 높습니다.**
- **서울대학교병원 주차장** — 대학로 방면
- **종로구청 주차장** — 종로3가 인근

※ 주차 요금·운영시간은 주차장별로 상이하니 방문 전 확인을 권장드립니다.

### ♿ 장애인·교통약자 편의
- **혜화역 4번 출구**에는 엘리베이터가 설치되어 있습니다.
- 창경궁 주차장에는 장애인 주차구역이 마련되어 있습니다.
- 과학관 1층 안내데스크에서 **유모차(5대)** 및 **휠체어(2대)** 대여 가능 (신분증 지참).
- 의무실은 1층에 있으며 일반의약품을 구비하고 있습니다.

### 📍 주소 및 문의
- **주소**: 서울특별시 종로구 창경궁로 215 (와룡동 2-1)
- **대표전화**: 02-3668-3350
- **공식 홈페이지**: https://www.csc.go.kr

⚠️ 주차장 및 교통편 관련 최신 안내는 방문 전 공식 홈페이지 '오시는 길' 페이지에서 꼭 확인해 주세요."""

        elif category == "directions":
            has_origin = (
                re.search(r"(에서|출발|출발지|역에서)", message)
                or re.search(r"[가-힣A-Za-z0-9]{2,}역", message)
                or "집" in message
            )
            if not has_origin:
                st.session_state["awaiting_directions_origin"] = True
                if mode == "어린이":
                    return """🧭 오시는 길 안내

국립어린이과학관은 **서울 종로구 창경궁 바로 옆**에 있어요!

## 🚇 지하철로 오는 방법 (제일 추천!)
- **4호선 혜화역 4번 출구** → 걸어서 약 10~15분 (가장 가까워요!)
  - 혜화역에서 나오면 창경궁 쪽(북쪽)으로 쭉 걸어오면 돼요.
- **1호선 종로5가역 2번 출구** → 걸어서 약 20분

## 🚌 버스로 오는 방법
- **창경궁 앞(홍화문)** 정류장에서 내려서 5분 정도 걸어오세요.
- 경유 노선은 서울 버스 앱이나 지도 앱에서 '창경궁' 정류장을 검색하면 실시간으로 확인할 수 있어요.

## 🚗 차로 오는 건?
- **과학관 안에는 주차장이 없어요.** 근처 창경궁 주차장, 서울대학교병원 주차장 등을 써야 해요.
- 그래서 **대중교통이 제일 편해요!**

## ♿ 도움이 필요할 때
- 혜화역 4번 출구에는 엘리베이터가 있어요.
- 과학관 1층 안내데스크에서 유모차(5대)와 휠체어(2대)를 빌릴 수 있어! (신분증 꼭 가져오기!)

## 📍 주소 & 연락처
- **주소**: 서울특별시 종로구 창경궁로 215
- **전화**: 02-3668-3350

😊 **어디에서 출발해?** 알려주면 더 딱 맞는 길을 안내해줄게!
(예: 강남역, 혜화역, 잠실, OO동/OO구)"""
                return """🧭 오시는 길 안내

국립어린이과학관은 서울 종로구 창경궁 인근에 위치하고 있습니다.

## 📍 주소
- **서울특별시 종로구 창경궁로 215** (와룡동 2-1)

## 🚇 지하철 (가장 빠르고 편리한 방법)
- **4호선 혜화역 4번 출구** → 도보 약 10~15분 (가장 가까움)
  - 혜화역 4번 출구로 나와 창경궁로를 따라 북쪽(창경궁 방향)으로 직진하세요.
- **1호선 종로5가역 2번 출구** → 도보 약 20분

## 🚌 버스
- **창경궁 정문(홍화문)** 정류장 하차 후 도보 약 5분
- 경유 노선 정보는 **서울 버스 앱 또는 지도 앱**에서 '창경궁' 정류장을 검색하시면 실시간으로 확인 가능합니다.

## 🚗 차량 이용
- **국립어린이과학관에는 전용 주차장이 없습니다.**
- 인근 유료 주차장 (창경궁 주차장·서울대학교병원 주차장·종로구청 주차장) 이용 필요.
- 주말·공휴일은 만차 가능성이 높으니 **대중교통 이용을 강력히 권장**합니다.

## ♿ 교통약자 편의
- **혜화역 4번 출구**에는 엘리베이터가 설치되어 있습니다.
- 과학관 1층 안내데스크에서 **유모차(5대)**·**휠체어(2대)** 대여 가능 (신분증 지참).
- 의무실은 1층에 있으며 일반의약품을 구비하고 있습니다.

## 📞 문의
- **대표전화**: 02-3668-3350
- **공식 홈페이지**: https://www.csc.go.kr

💬 **출발지를 알려주시면** (예: 강남역, 잠실, OO구) 가장 편한 환승 경로를 구체적으로 안내해드릴게요.

⚠️ 지하철 운행 정보·버스 노선은 변경될 수 있으니 공식 홈페이지 '오시는 길' 페이지도 참고해 주세요."""

            base = STATIC_FAQ.get("교통안내", "")
            verify = "\n\n오시는 길은 노선/출입구 변경이 있을 수 있어 정확성이 중요합니다.\n공식 홈페이지(www.csc.go.kr) '오시는 길' 페이지를 기준으로 확인해 주세요.\n추가로 02-3668-1500으로 문의하시면 가장 정확합니다."
            if mode == "어린이":
                return (base or "오시는 길을 알려드릴게요! 🧭") + verify
            return (base or "오시는 길 안내입니다.") + verify
            
    return ""


def answer_rule_based_localized(intent: str, message: str, mode: str, language: str) -> tuple[str, str]:
    """규칙 기반 답변을 사용자 언어로 반환.

    반환: (answer_in_target_language, ko_original)

    동작:
    1) 한국어 원문은 항상 answer_rule_based 로 생성
    2) language == "한국어" → 그대로 반환
    3) basic intent + 정적 번역 보유 → static_translations 즉시 사용 (LLM 우회)
    4) operating_hours → 동적 status 부분만 매핑 후 템플릿 주입
    5) 정적 번역 미보유 → translate_answer_cached (LLM, 24h 캐시) 폴백
    """
    ko_answer = answer_rule_based(intent, message, mode)
    if not ko_answer:
        return "", ""

    if language == "한국어":
        return ko_answer, ko_answer

    if intent == "basic":
        category = classify_basic_category(message)
        # 운영시간: 동적 status 부분 별도 매핑 (시간/휴관 메시지 그대로 보존)
        if category == "operating_hours":
            ko_status = get_today_status()
            translated = get_operating_hours_text(language, mode, ko_status)
            if translated:
                return translated, ko_answer
        else:
            static = get_static_answer(category, language, mode)
            if static:
                return static, ko_answer

    # 폴백: LLM 번역
    translated = translate_answer_cached(ko_answer, language)
    return translated, ko_answer

# ============================================================================
# RAG SYSTEM - Vector DB 및 데이터 로딩
# ============================================================================

def load_csv_data():
    """CSV files from data directory - real-time loading"""
    def load_csv_safe(path: str) -> pd.DataFrame:
        for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                continue
        return pd.read_csv(path)

    docs = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    pages_dir = os.path.join(base_dir, "data", "pages")
    
    csv_files = []
    if os.path.exists(data_dir):
        csv_files.extend(glob.glob(os.path.join(data_dir, "*.csv")))
    if os.path.exists(pages_dir):
        csv_files.extend(glob.glob(os.path.join(pages_dir, "*.csv")))
    
    print(f"Base dir: {base_dir}")
    print(f"Data dir exists: {os.path.exists(data_dir)}")
    print(f"Pages dir exists: {os.path.exists(pages_dir)}")
    print(f"Found CSV files: {csv_files}")
    
    for csv_file in csv_files:
        try:
            print(f"Loading CSV: {csv_file}")
            df = load_csv_safe(csv_file)

            df.columns = [str(c).strip() for c in df.columns]
            expected_cols = {"title", "content", "detail", "category"}
            has_expected = len(expected_cols.intersection(set(df.columns))) >= 2
            has_unnamed = any(str(c).startswith("Unnamed") for c in df.columns)

            if (not has_expected) and (has_unnamed or df.shape[1] >= 2):
                try:
                    df.columns = [str(v).strip() for v in df.iloc[0].tolist()]
                    df = df.iloc[1:].reset_index(drop=True)
                    df.columns = [str(c).strip() for c in df.columns]
                except Exception:
                    pass

            rename_map = {}
            synonyms = {
                "title": ["title", "전시물명", "전시물", "전시명", "제목", "명칭", "이름"],
                "content": ["content", "내용", "설명", "전시내용", "본문"],
                "detail": ["detail", "세부설명", "상세", "상세설명"],
                "category": ["category", "분류", "카테고리", "구분"],
            }

            cols_lower = {str(c).strip().lower(): str(c).strip() for c in df.columns}
            for target, candidates in synonyms.items():
                if target in df.columns:
                    continue
                found = None
                for cand in candidates:
                    key = str(cand).strip().lower()
                    if key in cols_lower:
                        found = cols_lower[key]
                        break
                if found is not None and found != target:
                    rename_map[found] = target
            if rename_map:
                df = df.rename(columns=rename_map)

            print(f"CSV shape: {df.shape}")
            print(f"CSV columns: {df.columns.tolist()}")
            
            for idx, row in df.iterrows():
                if pd.isna(row.get('title', '')):
                    continue
                    
                title = str(row.get('title', ''))
                content = str(row.get('content', ''))
                detail = str(row.get('detail', ''))
                category = str(row.get('category', ''))
                
                # Determine zone name from filename
                zone_name = os.path.splitext(os.path.basename(csv_file))[0]
                if "AI놀이터" in csv_file:
                    zone_name = "AI놀이터"
                elif "탐구놀이터" in csv_file or "탐구놀이터널" in csv_file:
                    zone_name = "탐구놀이터"
                elif "관찰놀이터" in csv_file:
                    zone_name = "관찰놀이터"
                elif "행동놀이터" in csv_file:
                    zone_name = "행동놀이터"
                
                text = f"[{zone_name}] {title}\nCategory: {category}\nContent: {content}\nDetails: {detail}"
                metadata = {
                    "source": f"csv_{zone_name}", 
                    "title": title, 
                    "category": zone_name,
                    "subcategory": category
                }
                docs.append(Document(page_content=text, metadata=metadata))
                
            print(f"Loaded {len(docs)} docs from {csv_file}")
                
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    print(f"Total CSV docs: {len(docs)}")
    print("=== End CSV Loading Debug ===")
    return docs

def load_multilingual_brochures():
    """Load multilingual brochure data for RAG system"""
    multilingual_dir = os.path.join(os.path.dirname(__file__), "multilingual")
    
    brochure_files = {
        "english": {
            "files": [
                os.path.join(multilingual_dir, "Science Center Information_ENG_250318.pdf"),
                os.path.join(multilingual_dir, "Science Center Information_ENG_250318.txt"),
                os.path.join(multilingual_dir, "Science Center Information_ENG_250318.csv")
            ],
            "language": "English"
        },
        "japanese": {
            "files": [
                os.path.join(multilingual_dir, "Science Center Information_JPN_250318.pdf"),
                os.path.join(multilingual_dir, "Science Center Information_JPN_250318.txt"),
                os.path.join(multilingual_dir, "Science Center Information_JPN_250318.csv")
            ],
            "language": "Japanese"
        },
        "chinese": {
            "files": [
                os.path.join(multilingual_dir, "Science Center Information_CHN_250318.pdf"),
                os.path.join(multilingual_dir, "Science Center Information_CHN_250318.txt"),
                os.path.join(multilingual_dir, "Science Center Information_CHN_250318.csv")
            ],
            "language": "Chinese"
        }
    }
    
    docs = []
    
    for lang_code, lang_info in brochure_files.items():
        for file_path in lang_info["files"]:
            if os.path.exists(file_path):
                try:
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(file_path, encoding='utf-8')
                        for idx, row in df.iterrows():
                            if pd.notna(row.iloc[0]):
                                content = ' '.join([str(val) for val in row if pd.notna(val)])
                                text = f"[{lang_info['language']}] {content}"
                                metadata = {
                                    "source": f"multilingual_{lang_code}",
                                    "language": lang_info["language"],
                                    "file_type": "csv"
                                }
                                docs.append(Document(page_content=text, metadata=metadata))
                    
                    elif file_path.endswith('.txt'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            text = f"[{lang_info['language']}] {content}"
                            metadata = {
                                "source": f"multilingual_{lang_code}",
                                "language": lang_info["language"],
                                "file_type": "txt"
                            }
                            docs.append(Document(page_content=text, metadata=metadata))
                    
                    elif file_path.endswith('.pdf'):
                        text = f"[{lang_info['language']}] Brochure content available in PDF format"
                        metadata = {
                            "source": f"multilingual_{lang_code}",
                            "language": lang_info["language"],
                            "file_type": "pdf"
                        }
                        docs.append(Document(page_content=text, metadata=metadata))
                        
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
    
    return docs

def initialize_vector_db():
    """Initialize vector database with exhibit information"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print("=== Vector DB Initialization ===")
    print("Creating new vector store (Streamlit Cloud - no persistence)...")
    
    docs = []
    
    # Add static exhibit info
    for name, desc in STATIC_EXHIBIT_INFO.items():
        url = CSC_URLS.get(name, "https://www.csc.go.kr")
        docs.append(Document(page_content=f"[{name}] {desc}", metadata={"source": url}))
    
    # Add CSV data
    csv_docs = load_csv_data()
    docs.extend(csv_docs)
    
    # Add multilingual brochures
    multilingual_docs = load_multilingual_brochures()
    docs.extend(multilingual_docs)
    
    print(f"Loaded {len(csv_docs)} CSV entries + {len(multilingual_docs)} multilingual entries + {len(STATIC_EXHIBIT_INFO)} static entries")
    print(f"Total documents: {len(docs)}")
    
    # Create new vector store (no persistence for Streamlit Cloud)
    vectorstore = Chroma.from_documents(
        docs, 
        embeddings
    )
    
    print("=== Vector DB Initialization Complete ===")
    return vectorstore

# ============================================================================
# LANGCHAIN TOOLS - 웹 크롤링 및 검색 도구
# ============================================================================

def parse_html_tables_to_markdown(soup: BeautifulSoup) -> str:
    """HTML Table을 마크다운으로 파싱"""
    markdown_text = ""
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for i, row in enumerate(rows):
            cols = row.find_all(["th", "td"])
            row_text = "| " + " | ".join(col.get_text(strip=True) for col in cols) + " |"
            markdown_text += row_text + "\n"
            if i == 0:
                markdown_text += "|" + "|".join(["---"] * len(cols)) + "|\n"
        markdown_text += "\n"
    return markdown_text

@tool
def check_museum_closed_date(date_str: str) -> str:
    """
    특정 날짜의 국립어린이과학관 휴관일 여부를 확인합니다.
    
    [언제 사용하는가]
    - 사용자가 "내일 가도 돼?", "다음주 월요일 열어?", "3월 24일 휴관이야?" 같은 질문을 할 때
    
    [입력 형식]
    - date_str: "2026-03-24" 또는 "내일" 또는 "다음주 월요일" 형태
    
    [무엇을 반환하는가]
    - 해당 날짜의 휴관 여부와 이유
    """
    now_utc = datetime.now(timezone.utc)
    now_kst = now_utc + timedelta(hours=9)
    
    if "내일" in date_str or "tomorrow" in date_str.lower():
        target_date = now_kst + timedelta(days=1)
    elif "모레" in date_str:
        target_date = now_kst + timedelta(days=2)
    else:
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
        except:
            return f"Observation: 날짜 형식을 인식할 수 없습니다. YYYY-MM-DD 형식으로 입력해주세요."
    
    month_day = target_date.strftime("%m-%d")
    weekday = target_date.weekday()
    weekday_kr = ["월", "화", "수", "목", "금", "토", "일"][weekday]
    
    monday_exceptions = {"02-16", "03-02", "05-25", "08-17", "10-05"}
    holiday_closed = {"01-01", "02-17", "09-25"}
    substitute_closed = {"02-19", "03-03", "05-26", "08-18", "10-06"}
    
    date_display = target_date.strftime("%Y년 %m월 %d일")
    
    if month_day in holiday_closed:
        return f"Observation: {date_display}({weekday_kr}요일)은 명절 정기 휴관일입니다."
    if month_day in substitute_closed:
        return f"Observation: {date_display}({weekday_kr}요일)은 대체 휴관일입니다."
    if weekday == 0 and month_day not in monday_exceptions:
        return f"Observation: {date_display}({weekday_kr}요일)은 정기휴관일입니다."
    
    return f"Observation: {date_display}({weekday_kr}요일)은 정상 운영일입니다."

@tool
def search_directions(origin: str, destination: str = "국립어린이과학관") -> str:
    """
    출발지에서 목적지까지의 대중교통 경로를 검색합니다.
    
    [언제 사용하는가]
    - 사용자가 "어떻게 가?", "길 알려줘", "교통편" 같은 질문을 할 때
    - 출발지와 목적지가 명확할 때만 사용
    
    [입력 형식]
    - origin: 출발지 (예: "강남역", "잠실", "종로구")
    - destination: 목적지 (기본값: "국립어린이과학관")
    
    [무엇을 반환하는가]
    - 지하철/버스 경로, 소요시간, 환승 정보
    """
    try:
        result = f"""Observation: {origin}에서 {destination}까지의 경로 안내

정확한 실시간 경로는 다음 방법으로 확인하세요:
1. 네이버 지도 앱/웹사이트에서 '{origin}'에서 '{destination}' 검색
2. 카카오맵에서 '{origin}'에서 '{destination}' 검색
3. 대중교통 앱 (지하철, 버스) 이용

일반적인 안내:
- {destination} 주소: 서울특별시 종로구 창경궁로 215
- 가까운 지하철역: 4호선 혜화역 4번 출구 (도보 15분)
- 국립어린이과학관은 전용 주차장이 없으므로 대중교통 이용을 권장합니다.

※ 정확한 버스 노선, 소요시간, 환승 정보는 위 지도 앱에서 실시간으로 확인해주세요.
※ 교통 상황에 따라 소요시간이 달라질 수 있습니다."""
        return result
    except Exception as e:
        return f"Observation: 경로 검색 중 오류가 발생했습니다: {str(e)}\n네이버 지도나 카카오맵에서 '{origin}'에서 '{destination}'을 직접 검색해주세요."

@tool
def search_csc_live_info(keyword: str) -> str:
    """
    국립어린이과학관 공식 홈페이지의 실시간 정보를 확인합니다.
    
    [언제 사용하는가]
    - 전시관 상세 안내, 프로그램 정보, 예약 방법 등을 확인할 때
    """
    target_url = CSC_URLS.get(keyword)
    if not target_url:
        for key, url in CSC_URLS.items():
            if keyword in key or key in keyword:
                target_url = url
                keyword = key
                break

    if not target_url:
        return f"Observation: '{keyword}' 페이지를 찾을 수 없습니다."

    try:
        response = requests.get(target_url, timeout=10, verify=False)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        for unwanted in soup(['script', 'style', 'nav', 'footer', 'header']):
            unwanted.decompose()

        main_content = soup.find('div', class_='content') or soup.find('main') or soup.find('body')

        if main_content:
            text_content = main_content.get_text(separator='\n', strip=True)
            table_content = parse_html_tables_to_markdown(main_content)
            combined = f"{text_content}\n\n{table_content}" if table_content else text_content
            return f"Observation: [{keyword}] 페이지 정보\n출처: {target_url}\n\n{combined[:3000]}"

        return f"Observation: '{keyword}' 페이지에서 콘텐츠를 찾을 수 없습니다."

    except Exception as e:
        return f"Observation: '{keyword}' 페이지 로드 중 오류 발생: {str(e)}"


def get_latest_notices_text(limit: int = 5) -> str:
    url = CSC_URLS.get("공지사항")
    if not url:
        return "공지사항 URL을 찾을 수 없습니다."

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "identity",
            "Connection": "close",
        }

        html = _fetch_html_bytes(url, headers=headers, must_contain=[b"goView", b"rbbs", b"boardList"])
        soup = BeautifulSoup(html, "html.parser")

        links = []

        notice_anchors = soup.select(
            "div.rbbs_list_sec a[onclick*='goView'], div.rbbs_list a[onclick*='goView'], table a[onclick*='goView']"
        )
        if not notice_anchors:
            notice_anchors = soup.select("a[onclick*='goView']")

        for a in notice_anchors:
            onclick = a.get("onclick", "")
            m_full = re.search(
                r"goView\(\s*'(?P<pkid>\d+)'\s*,\s*'(?P<num>\d+)'\s*,\s*'(?P<page>\d+)'\s*\)",
                onclick,
            )
            m = m_full or re.search(r"goView\(\s*'(?P<pkid>\d+)'", onclick)
            if not m:
                continue
            pkid = m.group("pkid")
            num = m.group("num") if m_full else "0"

            title_el = a.select_one("div.title_line div.title div.text")
            title = title_el.get_text(" ", strip=True) if title_el else a.get_text(" ", strip=True)
            if not title:
                continue

            if (
                "요청하신 페이지를 찾을 수 없습니다" in title
                or "죄송합니다" in title
                or "이전페이지" in title
                or "대표번호" in title
                or "Science Center Information" in title
                or "국립어린이과학관 메인" in title
            ):
                continue

            if len(title) > 120:
                continue

            href = f"{MUSEUM_BASE_URL}/boardView.do?bbspkid=22&pkid={pkid}&num={num}"
            links.append((title, href))

        if not links:
            for a in soup.select("a[href*='boardView.do']"):
                href = a.get("href", "").strip()
                title = a.get_text(" ", strip=True)
                if not href or not title:
                    continue
                if href.startswith("/"):
                    href = f"{MUSEUM_BASE_URL}{href}"
                elif href.startswith("boardView.do"):
                    href = f"{MUSEUM_BASE_URL}/{href}"
                links.append((title, href))

        uniq = []
        seen = set()
        for title, href in links:
            if href in seen:
                continue
            seen.add(href)
            uniq.append((title, href))
            if len(uniq) >= limit:
                break

        if not uniq:
            candidates = []
            for onclick in re.findall(r"goView\(\s*'\d+'\s*,\s*'\d+'\s*,\s*'\d+'\s*\)", soup.decode() if hasattr(soup, 'decode') else str(soup)):
                m = re.search(r"goView\(\s*'(?P<pkid>\d+)'\s*,\s*'(?P<num>\d+)'\s*,\s*'(?P<page>\d+)'\s*\)", onclick)
                if not m:
                    continue
                pkid = m.group("pkid")
                num = m.group("num")
                href = f"{MUSEUM_BASE_URL}/boardView.do?bbspkid=22&pkid={pkid}&num={num}"
                candidates.append((pkid, num, href))

            resolved = []
            for pkid, num, href in candidates:
                if href in seen:
                    continue
                seen.add(href)
                title = _resolve_notice_title(pkid=pkid, num=num)
                if not title:
                    continue
                resolved.append((title, href))
                if len(resolved) >= limit:
                    break
            uniq = resolved

        if not uniq:
            return (
                "현재 공지사항 목록을 자동으로 불러오지 못했어요.\n\n"
                "아래 공식 공지사항 페이지에서 최신 글을 확인해 주세요.\n"
                f"- {CSC_URLS.get('공지사항', MUSEUM_BASE_URL)}\n\n"
                "급한 문의는 대표번호로 연락해 주세요.\n"
                "- 02-3668-3350"
            )

        lines = ["최근 공지사항을 정리해드릴게요! (공식 홈페이지 기준)", ""]
        for i, (title, href) in enumerate(uniq, start=1):
            lines.append(f"{i}. {title}\n- {href}")

        lines.append("\n원하시면 \"공지 1번 자세히\"처럼 번호를 말해주면 본문도 요약해서 보여줄게요.")
        return "\n".join(lines)
    except Exception:
        return (
            "현재 공지사항 목록을 자동으로 불러오지 못했어요.\n\n"
            "아래 공식 공지사항 페이지에서 최신 글을 확인해 주세요.\n"
            f"- {CSC_URLS.get('공지사항', MUSEUM_BASE_URL)}\n\n"
            "급한 문의는 대표번호로 연락해 주세요.\n"
            "- 02-3668-3350"
        )


def _resolve_notice_title(pkid: str, num: str = "0") -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "identity",
        "Connection": "close",
    }
    url = f"{MUSEUM_BASE_URL}/boardView.do?bbspkid=22&pkid={pkid}&num={num}"

    try:
        html = _fetch_html_bytes(url, headers=headers, max_attempts=2)
        soup = BeautifulSoup(html, "html.parser")
        title = soup.select_one("div.sub_contents.sub_depth_content h3")
        if not title:
            title = soup.select_one("div.sub_contents.sub_depth_content .sub_tit")
        title_text = title.get_text(" ", strip=True) if title else ""
        if not title_text:
            return ""
        if (
            "요청하신 페이지를 찾을 수 없습니다" in title_text
            or "죄송합니다" in title_text
            or "Science Center Information" in title_text
        ):
            return ""
        return title_text
    except Exception:
        return ""


def _build_retry_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _read_response_bytes(resp: requests.Response, max_bytes: int = 2_000_000) -> bytes:
    data = bytearray()
    try:
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            data.extend(chunk)
            if len(data) >= max_bytes:
                break
    except Exception:
        pass
    return bytes(data)


def _fetch_html_bytes(
    url: str,
    headers: dict,
    max_attempts: int = 3,
    must_contain: list[bytes] | None = None,
) -> bytes:
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            session = _build_retry_session()
            resp = session.get(url, timeout=(10, 25), verify=False, headers=headers, stream=True)
            resp.raise_for_status()
            data = _read_response_bytes(resp)
            if data and (not must_contain or any(tok in data for tok in must_contain)):
                return data
        except Exception as e:
            last_err = e

        try:
            resp2 = requests.get(url, timeout=(10, 25), verify=False, headers=headers)
            resp2.raise_for_status()
            if resp2.content and (not must_contain or any(tok in resp2.content for tok in must_contain)):
                return resp2.content
        except Exception as e:
            last_err = e

        time.sleep(0.6 * attempt)

    raise RuntimeError(str(last_err) if last_err else "unknown error")


def get_notice_detail_text(pkid: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "identity",
        "Connection": "close",
    }
    url = f"{MUSEUM_BASE_URL}/boardView.do?bbspkid=22&pkid={pkid}&num=0"

    try:
        html = _fetch_html_bytes(url, headers=headers)
        soup = BeautifulSoup(html, "html.parser")

        title = soup.select_one("div.sub_contents.sub_depth_content h3")
        if not title:
            title = soup.select_one("div.sub_contents.sub_depth_content .sub_tit")
        title_text = title.get_text(" ", strip=True) if title else "공지사항"

        substance = soup.select_one("div.rbbs_read_sec div.substance")
        if not substance:
            substance = soup.select_one("div.sub_contents.sub_depth_content div.rbbs_read_sec")

        if not substance:
            return f"공지 본문을 찾지 못했어요.\n- {url}"

        lines = []
        for el in substance.select("p, span, li"):
            t = el.get_text(" ", strip=True)
            if not t:
                continue
            lines.append(t)

        merged = "\n".join(lines)
        merged = re.sub(r"\n{3,}", "\n\n", merged).strip()
        if not merged:
            merged = substance.get_text("\n", strip=True)

        merged = merged.strip()
        return f"{title_text}\n\n{merged[:2500]}\n\n출처: {url}"
    except Exception as e:
        return f"공지사항 본문을 불러오는 중 오류가 발생했어요: {str(e)}\n- {url}"


@tool
def fetch_latest_notices(limit: int = 5) -> str:
    """국립어린이과학관 공지사항(게시판) 최신 글 링크를 가져옵니다."""
    return f"Observation: {get_latest_notices_text(limit=limit)}"


def get_tools():
    """LangChain agent에서 사용할 도구 목록 반환"""
    return [
        check_museum_closed_date,
        search_directions,
        search_csc_live_info,
        fetch_latest_notices,
    ]

# ============================================================================
# UTILS - 유틸리티 함수
# ============================================================================

def get_dynamic_prompt(mode: str, language: str = "한국어") -> str:
    """LLM 시스템 프롬프트 생성"""
    now_utc = datetime.now(timezone.utc)
    now_kst = now_utc + timedelta(hours=9)
    today_kst = now_kst.strftime("%Y년 %m월 %d일 %H:%M")
    weekday_kr = ["월", "화", "수", "목", "금", "토", "일"][now_kst.weekday()]
    
    language_instruction = {
        "한국어": "**중요: 모든 답변은 한국어로 작성하세요.**",
        "English": (
            "**CRITICAL LANGUAGE RULE: You MUST respond ENTIRELY in English, "
            "even if the user's question, FAQ trigger text, or any retrieved RAG content is in Korean. "
            "Translate all Korean content into English before answering. "
            "NEVER output Korean text (except official place names inside parentheses as specified in the glossary below).**"
        ),
        "日本語": (
            "**最重要言語ルール：ユーザーの質問やFAQトリガー、RAG検索結果が韓国語であっても、"
            "必ず回答全体を日本語で書いてください。韓国語の内容は日本語に翻訳してから答えること。"
            "韓国語をそのまま出力してはいけません（下記グロッサリーで指定された括弧内の英語公式名称のみ例外）。**"
        ),
        "中文": (
            "**最重要语言规则：即使用户的问题、FAQ触发语或RAG检索内容是韩语，"
            "你也必须完全用中文回答。所有韩语内容必须先翻译成中文。"
            "绝对不要输出韩语原文（只有下方词汇表中指定的括号内英文官方名称可作为例外）。**"
        ),
    }

    safety_instruction = {
        "한국어": (
            "=== 안전/아동친화 가드레일 ===\n"
            "- 욕설/비속어/모욕/혐오표현/공격적인 말투를 절대 사용하지 마세요.\n"
            "- 어린이가 주 고객층입니다. 과격하거나 위협적이거나 폭력적인 표현을 하지 마세요.\n"
            "- 사용자가 욕설/폭력/혐오/괴롭힘을 요구하더라도 정중히 거절하고 안전한 대안으로 안내하세요.\n"
        ),
        "English": (
            "=== Safety & Kid-Friendly Guardrails ===\n"
            "- Never use profanity, slurs, insults, harassment, hate speech, or aggressive tone.\n"
            "- The primary audience includes children. Avoid violent, threatening, or graphic language.\n"
            "- If the user asks for harmful/abusive content, refuse politely and redirect to safe alternatives.\n"
        ),
        "日本語": (
            "=== 安全・子ども向けガードレール ===\n"
            "- 罵倒語/卑語/侮辱/差別/攻撃的な口調は絶対に使わないでください。\n"
            "- 子どもが主な利用者です。暴力的・脅迫的・過激な表現を避けてください。\n"
            "- 有害/攻撃的な内容の要求は丁寧に拒否し、安全な代替案に誘導してください。\n"
        ),
        "中文": (
            "=== 安全与儿童友好守则 ===\n"
            "- 绝不使用脏话、侮辱、歧视、仇恨言论或攻击性语气。\n"
            "- 主要受众包含儿童。避免暴力、威胁或过激表达。\n"
            "- 若用户要求有害/辱骂内容，请礼貌拒绝并提供安全替代方案。\n"
        ),
    }
    
    base_prompt = f"""
당신은 국립어린이과학관 전문 안내 어시스턴트입니다.
[오늘 날짜] {today_kst} ({weekday_kr}요일) KST

{language_instruction.get(language, language_instruction["한국어"])}

{safety_instruction.get(language, safety_instruction["한국어"])}

=== 핵심 임무 ===
국립어린이과학관의 모든 시설, 전시관, 프로그램, 운영 정보를 정확하고 친절하게 안내하는 것입니다.

=== 답변해야 할 주요 영역 ===
1. 운영 정보: 관람시간, 휴관일, 입장료, 교통안내
2. 상설전시관: 탐구놀이터, 관찰놀이터, 행동놀이터, 생각놀이터, AI놀이터
3. 천문우주: 천체투영관, 천체관측소
4. 과학교육: 교육프로그램, 창작교실
5. 예약: 개인예약, 단체예약

=== 환각 방지 가드레일 ===
- 운영시간, 입장료, 휴관일 → 반드시 RAG 또는 도구 결과 기반
- RAG/도구에 없는 정보 → "공식 홈페이지(www.csc.go.kr)에서 확인해주세요"
   - "정확한 정보는 02-3668-3350으로 문의해주세요."
- **주차 안내**: 국립어린이과학관은 **전용 주차장이 없습니다**. "주차 가능", "주차장 마련" 같은 말은 절대 하지 말 것. 자가용 이용 권장 금지. 대중교통 이용 안내 필수.

=== 국립어린이과학관 위치 정보 (고정값) ===
**주소**: 서울특별시 종로구 창경궁로 215 (와룡동 2-1)
**가까운 지하철역**: 
- 4호선 혜화역 4번 출구 (도보 15분)
- 1호선 종로5가역 2번 출구 (도보 20분)
**주요 버스 정류장**: 창경궁 앞 정류장
**대표 전화**: 02-3668-3350

=== 길찾기 응대 규칙 ===
1. **과학관에서 집으로 가는 경우**
   - 사용자가 "과학관에서 우리집으로", "여기서 집으로" 같은 표현을 쓰면
   - 출발지는 **국립어린이과학관(혜화역 근처)**으로 자동 설정
   - 도착지만 물어보세요: "어느 구/어느 동(또는 지하철역 이름)으로 가나요?"

2. **집에서 과학관으로 오는 경우**
   - 사용자가 "집에서 어떻게 가?", "우리집에서 과학관까지" 처럼 출발지가 불명확하면
   - 출발지를 물어보세요: "어느 구/어느 동(또는 지하철역 이름)에서 출발하나요?"
   - 도착지는 **국립어린이과학관**으로 자동 설정

3. **경로 안내 원칙 (매우 중요!)**
   - **정확한 집 주소/상세 위치는 입력하지 말아달라**고 안내하세요 (개인정보 보호)
   - **반드시 search_directions 도구를 사용**하여 경로 안내를 제공하세요
   - 도구 없이 추측으로 버스 노선이나 소요시간을 말하지 마세요
   - **자가용/주차 안내 금지** — 과학관에 주차장이 없음을 반드시 안내

4. **안내 방법**
   - search_directions(origin="출발지", destination="국립어린이과학관") 도구를 먼저 호출
   - 도구 결과에 나온 지하철/버스 안내를 기반으로 친절하게 정리
   - 출발지가 모호하면 도구를 부르지 말고 먼저 출발지를 물어볼 것

=== OFFICIAL PLACE NAMES (MANDATORY GLOSSARY for non-Korean answers) ===
FORMAT RULES by target language:
- English mode: write ONLY the official English name (e.g., "Thinking Zone").
- Japanese mode: write the Japanese name, then the official English name in parentheses, e.g., "考えるゾーン (Thinking Zone)".
- Chinese mode: write the Chinese name, then the official English name in parentheses, e.g., "思考区 (Thinking Zone)".
- NEVER use the raw Korean name (e.g., "생각놀이터") in non-Korean answers. Always replace with the target-language form.

Mapping table (Korean → Japanese | Chinese | English Official — ALWAYS use English exactly as shown, never invent variants):
- AI놀이터 → AIゾーン | AI区 | AI Zone
- 행동놀이터 → アクティブゾーン | 行动区 | Activity Zone
- 생각놀이터 → 考えるゾーン | 思考区 | Thinking Zone
- 탐구놀이터 → 探究ゾーン | 探究区 | Discovery Zone
- 관찰놀이터 → 観察ゾーン | 观察区 | Discovery Zone
- 과학극장 → 科学劇場 | 科学剧场 | Science Theater
- 빛놀이터 → ひかりシアター | 光影剧场 | Interactive Theater
- 어린이교실 → こども教室 | 儿童教室 | Kids Classroom
- 천체투영관 → プラネタリウム | 天体投影馆 | Planetarium
- 휴게실 → 休憩室 | 休息室 | Lounge

CRITICAL:
- The English Official name is FIXED — do NOT invent "Thought Playground", "Observation Zone", "Light Zone", "Exploration Zone" or any other variant.
- For Japanese/Chinese answers, always append the English Official name in parentheses right after the localized name.
- For English answers, do NOT append anything extra — just the English Official name.
"""
    
    if mode == "어린이":
        base_prompt += "\n\n어린이 모드: 쉽고 재미있게 설명하세요. 이모지를 활용하세요."
    
    return base_prompt

PLANETARIUM_VIDEO_INFO = {
    "코코몽 우주탐험": {
        "themes": "토성, 위성 타이탄, 태양계 행성, 우주여행, 모험",
        "fulldomedb_url": "https://www.fddb.org/fulldome-shows/cocomong-space-adventure/",
    },
    "길냥이 키츠 슈퍼문 대모험": {
        "themes": "달, 슈퍼문, 아폴로 미션, 달 기지, 미래 우주 탐사",
        "fulldomedb_url": "https://www.fddb.org/",
    },
    "바니 앤 비니": {
        "themes": "바다 생태계, 별과 별자리, 해양 생물, 자연의 신비",
        "fulldomedb_url": "https://www.fddb.org/",
    },
    "다이노소어": {
        "themes": "공룡, 중생대, 시간여행, 멸종, 지구의 역사",
        "fulldomedb_url": "https://www.fddb.org/",
    },
    "길냥이 키츠 우주정거장의 비밀": {
        "themes": "국제우주정거장(ISS), 무중력, 인공지능(A.I.), 우주생활",
        "fulldomedb_url": "https://www.fddb.org/",
    },
}


def _load_planetarium_videos():
    """천체투영관 CSV에서 상영 영상 5개를 표준 row 형식으로 반환"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    csv_path = os.path.join(data_dir, "천체투영관.csv")
    if not os.path.exists(csv_path):
        return []

    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        return []

    df.columns = [str(c).strip() for c in df.columns]
    rows = []
    seen_titles = set()
    for _, r in df.iterrows():
        cat = str(r.get("category", "")).strip()
        if not cat.startswith("프로그램_"):
            continue
        answer = str(r.get("answer", "")).strip()
        # 영상 제목은 PLANETARIUM_VIDEO_INFO 키 중 answer/category에 매칭되는 것을 찾음
        title = None
        for video_title in PLANETARIUM_VIDEO_INFO.keys():
            if video_title in answer or video_title.replace(" ", "") in cat.replace("_", "").replace(" ", ""):
                title = video_title
                break
        if not title:
            continue
        if title in seen_titles:
            continue
        seen_titles.add(title)

        info = PLANETARIUM_VIDEO_INFO.get(title, {})
        rows.append({
            "title": title,
            "content": answer,
            "detail": f"학습 주제: {info.get('themes', '')} | 참고: {info.get('fulldomedb_url', '')}",
            "category": "상영영상",
        })
    return rows


def load_zone_rows_from_csv(zone_name: str):
    # 천체투영관: 상영 영상을 전시물로 사용
    if zone_name == "천체투영관":
        return _load_planetarium_videos()

    def load_csv_safe(path: str) -> pd.DataFrame:
        for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                continue
        return pd.read_csv(path)

    # Use absolute path for Streamlit Cloud compatibility
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    target_files = [p for p in csv_files if zone_name in os.path.basename(p)]
    if not target_files:
        return []

    path = target_files[0]
    df = load_csv_safe(path)
    df.columns = [str(c).strip() for c in df.columns]

    expected_cols = {"title", "content", "detail", "category"}
    has_expected = len(expected_cols.intersection(set(df.columns))) >= 2
    has_unnamed = any(str(c).startswith("Unnamed") for c in df.columns)
    if (not has_expected) and (has_unnamed or df.shape[1] >= 2):
        try:
            df.columns = [str(v).strip() for v in df.iloc[0].tolist()]
            df = df.iloc[1:].reset_index(drop=True)
            df.columns = [str(c).strip() for c in df.columns]
        except Exception:
            pass

    rename_map = {}
    synonyms = {
        "title": ["title", "전시물명", "전시물", "전시명", "제목", "명칭", "이름"],
        "content": ["content", "내용", "설명", "전시내용", "본문"],
        "detail": ["detail", "세부설명", "상세", "상세설명"],
        "category": ["category", "분류", "카테고리", "구분"],
    }
    cols_lower = {str(c).strip().lower(): str(c).strip() for c in df.columns}
    for target, candidates in synonyms.items():
        if target in df.columns:
            continue
        found = None
        for cand in candidates:
            key = str(cand).strip().lower()
            if key in cols_lower:
                found = cols_lower[key]
                break
        if found is not None and found != target:
            rename_map[found] = target
    if rename_map:
        df = df.rename(columns=rename_map)

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "title": "" if pd.isna(r.get("title", "")) else str(r.get("title", "")),
            "content": "" if pd.isna(r.get("content", "")) else str(r.get("content", "")),
            "detail": "" if pd.isna(r.get("detail", "")) else str(r.get("detail", "")),
            "category": "" if pd.isna(r.get("category", "")) else str(r.get("category", "")),
        })
    rows = [x for x in rows if x.get("title")]
    return rows


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def translate_answer_cached(text: str, target_language: str) -> str:
    """규칙 기반 한국어 답변을 다른 언어로 번역 (캐시됨).
    마크다운/이모지/표 구조는 유지한다."""
    if not text or target_language == "한국어":
        return text
    lang_label = {
        "English": "natural English",
        "日本語": "natural Japanese (日本語)",
        "中文": "natural Simplified Chinese (中文)",
    }.get(target_language, "natural English")
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = (
            f"Translate the following Korean text into {lang_label}. "
            f"Strictly preserve markdown formatting (headings, bullets, tables, bold), "
            f"emojis, numbers, times, and proper nouns. Do NOT add explanations or notes. "
            f"Output only the translated text.\n\n---\n{text}"
        )
        resp = llm.invoke(prompt)
        out = (resp.content or "").strip()
        return out or text
    except Exception as e:
        print(f"답변 번역 실패: {e}")
        return text


def render_source_buttons(sources: list, language_mode: str = "한국어", key_suffix: str = ""):
    """출처(참고 홈페이지) 렌더링 — 기본은 접힘, '자세히 보기' 펼침 안에서 버튼 노출."""
    if not isinstance(sources, (list, tuple)):
        return
    sources = [s for s in sources if s]
    if not sources:
        return

    expander_label = {
        "한국어": "📚 참고 홈페이지 자세히 보기",
        "English": "📚 More info (reference websites)",
        "日本語": "📚 参考サイトを詳しく見る",
        "中文": "📚 查看参考网站",
    }.get(language_mode, "📚 More info (reference websites)")

    link_label = {
        "한국어": "🔗 참고 홈페이지",
        "English": "🔗 Reference site",
        "日本語": "🔗 参考サイト",
        "中文": "🔗 参考网站",
    }.get(language_mode, "🔗 Reference site")

    with st.expander(expander_label, expanded=False):
        for i, source in enumerate(sources[:5]):
            if isinstance(source, str) and source.startswith("http"):
                st.markdown(f"- [{link_label} {i+1}]({source})")
            else:
                st.markdown(f"- `{source}`")
