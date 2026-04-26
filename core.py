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
    """사용자 질문의 의도를 파악하여 라우팅"""
    lowered = text.lower().strip()

    if st.session_state.get("awaiting_directions_origin") and lowered:
        st.session_state["awaiting_directions_origin"] = False
        st.session_state["directions_origin"] = text.strip()
        return "llm_agent"

    if any(token in lowered for token in ["공지", "공지사항", "알림"]):
        return "notice"
    if ("집" in lowered or "우리집" in lowered or "집에서" in lowered) and any(token in lowered for token in ["어떻게 가", "가는", "가야", "가려고", "길", "오려", "가고"]):
        return "llm_agent"
    if any(token in lowered for token in ["층별", "1층", "2층", "층 안내", "동선", "연령", "나이", "프로그램", "오늘의 프로그램"]):
        return "llm_agent"
    if any(token in lowered for token in ["오시는길", "오는길", "교통", "길찾기", "주소", "어떻게 가", "어디", "위치"]):
        if re.search(r"(에서|출발|출발지|역에서)", text) or re.search(r"[가-힣A-Za-z0-9]{2,}역", text):
            return "llm_agent"
        return "llm_agent"
    if any(token in lowered for token in ["운영", "시간", "휴관", "입장료", "관람료", "주차"]):
        return "llm_agent"
    if any(token in lowered for token in ["예약", "예매", "방문신청", "방문 신청", "단체예약", "개인예약", "교육예약", "모바일 qr", "입장권", "정원", "1600"]):
        return "llm_agent"
    if any(token in lowered for token in ["시설", "편의시설", "의무실", "수유실", "유아휴게", "물품보관", "보관함", "락커", "유모차", "휠체어", "대여", "안내데스크", "매표소", "꿈트리", "휴게실", "영유아놀이터", "하늘마당", "옥상", "시간표", "상영", "회차"]):
        return "llm_agent"
    return "llm_agent"

def classify_basic_category(message: str) -> str:
    """기본 질문 카테고리 분류"""
    lowered = message.lower()
    rules = [
        ("floor_guide",     ["층별", "층 안내", "1층", "2층", "게이트", "입구", "출구"]),
        ("facility_amenities", ["시설", "편의시설", "의무실", "수유실", "유아휴게", "물품보관", "보관함", "락커", "유모차", "휠체어", "대여", "안내데스크", "매표소", "꿈트리", "휴게실", "영유아놀이터", "하늘마당", "옥상"]),
        ("exhibit_guide",   ["전시관", "전시관 안내", "놀이터 안내", "ai놀이터", "행동놀이터", "관찰놀이터", "탐구놀이터", "생각놀이터", "빛놀이터"]),
        ("route_by_age",    ["동선", "연령", "나이", "추천", "4~7", "유아", "초등", "저학년", "고학년"]),
        ("today_programs",  ["오늘의 프로그램", "오늘 프로그램", "프로그램", "과학쇼", "전시해설", "천체투영관", "빛놀이터"]),
        ("planetarium_timetable", ["천체투영관 시간표", "투영관 시간표", "시간표", "상영", "회차", "프로그램(투영관)", "코코몽", "키츠", "바니", "다이노"]),
        ("reservation_guide", ["예약", "예매", "방문신청", "방문 신청", "단체예약", "개인예약", "교육예약", "모바일 qr", "입장권", "정원", "1600"]),
        ("operating_hours", ["운영", "시간", "휴관", "몇 시", "마감"]),
        ("admission_fee",   ["관람료", "입장료", "요금", "가격", "얼마"]),
        ("parking",         ["주차", "주차장"]),
        ("directions",      ["오시는길", "오는길", "교통", "길찾기", "주소", "위치", "어떻게 가", "어디"]),
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
            return """전시관(놀이터)들을 짧게 소개해드릴게요! 😊

- **AI놀이터(1층)**: AI 미션을 해결하며 인공지능을 쉽고 재미있게 체험하는 공간이에요.
- **행동놀이터(1층)**: 몸을 움직이며 건강/운동 원리를 체험하는 활동형 전시관이에요.
- **생각놀이터(1층)**: 어린이들의 생각을 키우는 전시관(2026년 5월 개관 예정)입니다.
- **빛놀이터(2층)**: 빛/숲/생태 주제를 미디어 인터랙션으로 몰입 체험하는 공간이에요.
- **탐구놀이터(2층)**: 생활 속 도구·에너지·기계 원리를 직접 만지고 실험하며 탐구해요.
- **관찰놀이터(2층)**: 공룡/화석/표본 등을 관찰하며 과학적 사고력을 키우는 공간이에요.

원하시면 "AI놀이터 전시물 뭐가 있어?"처럼 **특정 놀이터 이름**을 말해주면 더 자세히도 찾아서 안내해줄게요!"""

        if category == "route_by_age":
            return """연령별로 추천 동선을 알려드릴게요! 😊

- 4~7세(유아)
  - 짧게 집중할 수 있는 체험 위주로, '빛놀이터'나 몸으로 움직이는 전시를 먼저 추천해요.
  - 중간중간 쉬는 시간(휴게실/수유실)도 꼭 챙겨주세요.

- 초등 저학년
  - '탐구놀이터/관찰놀이터'에서 직접 만지고 해보는 체험을 먼저 하고,
  - 관심이 생기면 '천체투영관'으로 마무리하면 좋아요.

- 초등 고학년
  - 'AI놀이터'에서 미션형 체험을 하고,
  - '탐구놀이터'에서 원리 탐색을 한 뒤,
  - 시간이 되면 '전시해설/과학쇼' 같은 프로그램도 추천해요.

원하시면 아이 나이(예: 6살, 초2, 초5)랑 지금 위치(1층/2층)를 말해주면 더 딱 맞게 짜줄게요!"""

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

        if category == "operating_hours":
            status = get_today_status()
            prefix = "오늘 어린이과학관은 어떨까요? 🚀\n" if mode == "어린이" else "운영 상태 안내입니다.\n"
            extra = "\n\n휴관일은 기본적으로 **매주 월요일**, **1월 1일**, **설날/추석 당일**이에요.\n(월요일이 공휴일이면 개관하고, 화요일에 대체 휴관할 수 있어요.)" if mode == "어린이" else "\n\n휴관일은 기본적으로 매주 월요일, 1월 1일, 설날/추석 당일입니다. (월요일 공휴일은 개관 후 화요일 대체 휴관 가능)"
            return f"{prefix}\n{status}{extra}"

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
        elif category == "parking":
            return ""

        elif category == "directions":
            has_origin = (
                re.search(r"(에서|출발|출발지|역에서)", message)
                or re.search(r"[가-힣A-Za-z0-9]{2,}역", message)
                or "집" in message
            )
            if not has_origin:
                st.session_state["awaiting_directions_origin"] = True
                if mode == "어린이":
                    return "좋아! 어디에서 출발해? 😊\n(예: 강남역, 혜화역, 잠실, OO동/OO구)"
                return "출발지를 알려주시면(예: 강남역/잠실/OO구) 그 기준으로 가장 쉬운 경로를 안내해드릴게요."

            base = STATIC_FAQ.get("교통안내", "")
            verify = "\n\n오시는 길은 노선/출입구 변경이 있을 수 있어 정확성이 중요합니다.\n공식 홈페이지(www.csc.go.kr) '오시는 길' 페이지를 기준으로 확인해 주세요.\n추가로 02-3668-1500으로 문의하시면 가장 정확합니다."
            if mode == "어린이":
                return (base or "오시는 길을 알려드릴게요! 🧭") + verify
            return (base or "오시는 길 안내입니다.") + verify
            
    return ""

# ============================================================================
# RAG SYSTEM - Vector DB 및 데이터 로딩
# ============================================================================

def load_csv_data():
    """CSV files from data directory - real-time loading"""
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    pages_dir = os.path.join(base_dir, "data", "pages")
    
    csv_files = []
    # Load from data/*.csv
    if os.path.exists(data_dir):
        csv_files.extend([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".csv") and not f.startswith("국립어린이과학관 전시물품 대장")
        ])
    # Load from data/pages/*.csv
    if os.path.exists(pages_dir):
        csv_files.extend([
            os.path.join(pages_dir, f)
            for f in os.listdir(pages_dir)
            if f.endswith(".csv")
        ])
    
    docs = []
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file, encoding='utf-8', skiprows=1)
                print(f"Loading {os.path.basename(csv_file)}: {len(df)} rows")
                
                for idx, row in df.iterrows():
                    category = str(row.get('분류', '')).strip()
                    title = str(row.get('제목', '')).strip()
                    content = str(row.get('내용', '')).strip()
                    detail = str(row.get('세부 설명', '')).strip()
                    
                    if title and title != 'nan' and len(title) > 0:
                        zone_name = ""
                        if "AI놀이터" in csv_file:
                            zone_name = "AI놀이터"
                        elif "탐구놀이터" in csv_file:
                            zone_name = "탐구놀이터"
                        elif "관찰놀이터" in csv_file:
                            zone_name = "관찰놀이터"
                        elif "행동놀이터" in csv_file:
                            zone_name = "행동놀이터"
                        
                        text = f"[{zone_name}] {title}\n분류: {category}\n내용: {content}\n세부설명: {detail}"
                        metadata = {
                            "source": f"csv_{zone_name}", 
                            "title": title, 
                            "category": zone_name,
                            "subcategory": category
                        }
                        docs.append(Document(page_content=text, metadata=metadata))
                        
            except Exception as e:
                print(f"CSV load error {csv_file}: {e}")
    
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
                    print(f"Error loading {file_path}: {e}")
    
    return docs

def initialize_vector_db():
    """정적 전시관 정보를 Chroma Vector DB로 구성합니다."""
    persist_directory = "./chroma_db"
    collection_name = "csc_exhibits"
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    try:
        vectorstore = Chroma(
            embedding_function=embeddings
        )
        return vectorstore
    except Exception:
        docs = []
        
        for name, desc in STATIC_EXHIBIT_INFO.items():
            url = CSC_URLS.get(name, "https://www.csc.go.kr")
            docs.append(Document(page_content=f"[{name}] {desc}", metadata={"source": url}))
        
        csv_docs = load_csv_data()
        docs.extend(csv_docs)
        
        multilingual_docs = load_multilingual_brochures()
        docs.extend(multilingual_docs)
        
        print(f"Loaded {len(csv_docs)} CSV entries + {len(multilingual_docs)} multilingual entries + {len(STATIC_EXHIBIT_INFO)} static entries")
        
        vectorstore = Chroma.from_documents(
            docs, 
            embeddings
        )
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
        "English": "**IMPORTANT: Respond in English.**",
        "日本語": "**重要：すべての回答は日本語で書いてください。**",
        "中文": "**重要：所有回答必须用中文书写。**"
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
"""
    
    if mode == "어린이":
        base_prompt += "\n\n어린이 모드: 쉽고 재미있게 설명하세요. 이모지를 활용하세요."
    
    return base_prompt

def load_zone_rows_from_csv(zone_name: str):
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

def render_source_buttons(sources: list):
    """출처 버튼 렌더링"""
    if not isinstance(sources, (list, tuple)):
        return
    if sources:
        st.markdown("**📚 참고 자료:**")
        for i, source in enumerate(sources[:3]):
            if source.startswith("http"):
                st.markdown(f"[🔗 출처 {i+1}]({source})")
