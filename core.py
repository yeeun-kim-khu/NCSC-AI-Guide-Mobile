# core.py - 핵심 시스템 통합
# config.py + rag.py + tools.py + utils.py + multilingual_loader.py 통합

import os
import re
import requests
import pandas as pd
import urllib3
import streamlit as st
from bs4 import BeautifulSoup
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
STATIC_EXHIBIT_INFO = {
    # Exhibition zones - detailed info from CSV files only
    "탐구놀이터": """탐구놀이터는 생활 속 도구, 에너지, 기계 등의 작동원리를 관찰하고 체험해 보면서 탐구해보는 체험 공간입니다.
위치: 2층
상세 전시물 정보는 CSV 데이터를 참조하세요.""",
    
    "관찰놀이터": """관찰놀이터는 공룡, 화석, 표본 등을 디지털 미디어를 통해 관찰해보며 과학적 사고력을 키워보는 공간입니다.
위치: 2층
상세 전시물 정보는 CSV 데이터를 참조하세요.""",
    
    "AI놀이터": """AI놀이터는 인공지능 "조이"를 도와 지구를 지키기 위한 활동을 체험하는 공간입니다.
위치: 1층
상세 전시물 정보는 CSV 데이터를 참조하세요.""",
    
    "행동놀이터": """행동놀이터는 다양한 신체 활동을 통해 내 몸을 알아보고 건강한 어린이가 되어보는 공간입니다.
위치: 1층
상세 전시물 정보는 CSV 데이터를 참조하세요.""",
    
    "생각놀이터": """생각놀이터는 어린이들의 생각을 키울 전시관으로, 2026년 5월 개관을 앞두고 있습니다.
위치: 1층""",
    
    "빛놀이터": """빛놀이터는 씨앗이 자라 나무가 되고, 나무들이 숲을 만드는 과정과 생태계 상호작용의 과학적 원리를 미디어 인터랙션을 통해 체험해 볼 수 있는 몰입형 실감 미디어 체험관입니다.
위치: 2층""",
    
    # 천문우주 시설
    "천체투영관": """돔 스크린에서 별자리와 우주를 관람하는 곳입니다. 계절별 별자리 해설과 천문 영상을 상영합니다.
상영 시간: 평일 11시, 14시, 16시 / 주말 11시, 13시, 15시, 17시
소요 시간: 약 40분
예약: 홈페이지 사전 예약 필수 (현장 예약 불가)
특징: 전문 해설사의 별자리 설명, 계절별 다른 프로그램""",
    
    "천체관측소": """망원경으로 태양, 달, 행성 등을 직접 관측하는 공간입니다. 날씨에 따라 관측 가능 여부가 달라집니다.
관측 시간: 주간(태양 관측) 10시~17시, 야간(별 관측) 19시~21시
예약: 홈페이지 사전 예약 필수
주의사항: 날씨에 따라 관측 불가능할 수 있음
특징: 실제 천체 관측, 전문 망원경 사용""",
    
    "메타버스과학관": """가상현실(VR)로 과학관을 체험할 수 있는 온라인 공간입니다. 집에서도 전시관을 둘러볼 수 있습니다.
접속 방법: 국립어린이과학관 홈페이지 접속
이용 시간: 24시간 언제나 가능
특징: VR 기기 없이도 PC/모바일로 체험 가능, 전시관 가상 투어""",
    
    # 운영 정보
    "운영시간": """국립어린이과학관 운영시간
관람시간: 오전 09:30 ~ 오후 17:30
입장마감: 오후 16:30
휴관일: 매주 월요일, 1월 1일, 설날/추석 당일
- 월요일이 공휴일인 경우 개관하며, 화요일에 대체 휴관
문의: 02-3668-1500

관람 시 유의사항:
- 전시관내 음식물 반입금지
- 애완동물 출입금지 (시각장애 안내견 제외)
- 바퀴달린 신발 착용금지
- 킥보드 탑승금지
- 뛰거나 큰소리로 떠들지 않기
- 체험시설물은 질서 있게 이용

관람방법:
- 어린이를 동반하지 않은 관람객 및 보호자를 동반하지 않은 어린이의 입장 제한
- 온라인 사전예약제 운영
- 과학관 입구(2층)에서 예약한 입장권(QR코드) 확인 후 관람
- 당일에 한해 입장권 소지 시 재입장 가능
- 상설전시관은 입장시간보다 늦을 경우에도 입장 가능""",
    
    "입장료": """국립어린이과학관 입장료
모든 나이는 '연나이' 기준 (연나이=현재년도-출생년도)

상설전시관:
- 성인(19세 이상): 개인 2,000원 (단체 이용불가)
- 청소년(13~18세): 개인 1,000원 (단체 이용불가)
- 초등학생(7~12세): 개인 1,000원 / 단체 500원
- 유아(6세 이하): 무료
- 우대고객: 무료 (65세 이상, 장애인, 과학기술유공자, 국가유공자, 기초생활수급자, 차상위계층, 한부모가족 지원대상자)

천체투영관:
- 성인(19세 이상): 1,500원
- 청소년(13~18세): 1,000원
- 초등학생(7~12세): 1,000원
- 유아(4~6세): 1,000원 (성인보호자 동반 및 결제시 이용 가능)
- 우대고객: 1,000원

할인 및 면제:
- 중증장애인(1~3급): 장애인과 동반보호자 1인 무료/우대요금
- 경증장애인(4급 이상): 장애인 본인만 무료/우대요금
- 다자녀카드 보유자: 상설전시관 개인요금의 50% 할인
- 단체 인솔자: 초등학생 20명당 1명, 유아 10명당 1명 무료

유의사항:
- 어린이를 동반하지 않은 성인 및 청소년, 보호자를 동반하지 않은 9세 이하 어린이 입장 제한
- 개인 관람객 환불은 관람 당일 오전 10시 전까지 신청콕에서 전체 취소만 가능
- 우대고객은 신분증과 증명서 지참 필수""",
    
    "교통안내": """국립어린이과학관 오시는 길
- 주소: 서울특별시 종로구 창경궁로 215
- 지하철: 4호선 혜화역 4번 출구 도보 10분
- 버스: 파랑(간선) 100, 102, 104, 106, 107, 108, 140, 143, 150, 160, 163, 172, 273, 710 / 초록(지선) 2112""",
    
    "예약안내": """국립어린이과학관 예약 방법
- 개인 예약: 홈페이지에서 관람일 7일 전부터 예약 가능
- 단체 예약: 20인 이상, 관람일 14일 전까지 예약 필수
- 교육 프로그램: 각 프로그램별 공지사항 확인 후 예약
- 천체투영관: 사전 예약 필수 (현장 예약 불가)
- 예약 취소: 관람일 1일 전까지 가능
- 문의: 02-3668-1500"""
}

# ============================================================================
# RULES - 규칙 및 로직 함수
# ============================================================================

def route_intent(text: str) -> str:
    """사용자 질문의 의도를 파악하여 라우팅"""
    lowered = text.lower().strip()
    if any(token in lowered for token in ["운영", "시간", "휴관", "입장료", "관람료", "주차"]):
        return "basic"
    return "llm_agent"

def classify_basic_category(message: str) -> str:
    """기본 질문 카테고리 분류"""
    lowered = message.lower()
    rules = [
        ("operating_hours", ["운영", "시간", "휴관", "몇 시", "마감"]),
        ("admission_fee",   ["관람료", "입장료", "요금", "가격", "얼마"]),
        ("parking",         ["주차", "주차장"]),
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
    
    monday_exceptions = {"02-16", "03-02", "05-25", "08-17", "10-05"}
    holiday_closed = {"01-01", "02-17", "09-25"}
    substitute_closed = {"02-19", "03-03", "05-26", "08-18", "10-06"}
    
    if month_day in holiday_closed:
        return (True, f"{target_date.strftime('%m월 %d일')}({weekday_kr}요일)은 명절 정기 휴관일입니다.")
    if month_day in substitute_closed:
        return (True, f"{target_date.strftime('%m월 %d일')}({weekday_kr}요일)은 대체 휴관일입니다.")
    if weekday == 0 and month_day not in monday_exceptions:
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
    if current.hour < 10:
        return "아직 개관 전이에요. (운영시간: 10:00~17:00)"
    if current.hour >= 17:
        return "오늘 운영 시간은 종료됐어요. (운영시간: 10:00~17:00)"
    return "현재 정상 운영 중입니다! (운영시간: 10:00~17:00)"

def answer_rule_based(intent: str, message: str, mode: str) -> str:
    """규칙 기반 답변 생성"""
    if intent == "basic":
        category = classify_basic_category(message)
        if category == "operating_hours":
            status = get_today_status()
            prefix = "오늘 어린이과학관은 어떨까요? 🚀\n" if mode == "어린이" else "운영 상태 안내입니다.\n"
            return f"{prefix}\n{status}"
        elif category == "admission_fee":
            fee_table = """
| 대상 | 연령/구분 | 개인 | 단체(20인 이상) |
| --- | --- | ---: | ---: |
| 어른 | 20~64세 | 2,000원 | 1,000원 |
| 청소년/어린이 | 7~19세 | 1,000원 | 500원 |
| 무료 | 6세 이하, 65세 이상 | 무료 | 무료 |
"""
            prefix = "상설전시관 관람료를 알려드릴게요! 💸\n" if mode == "어린이" else "상설전시관 관람료 안내입니다.\n"
            return f"{prefix}{fee_table}"
        elif category == "parking":
            return ""
            
    return ""

# ============================================================================
# RAG SYSTEM - Vector DB 및 데이터 로딩
# ============================================================================

def load_csv_data():
    """CSV files from data directory - real-time loading"""
    base_dir = os.path.dirname(__file__)
    csv_files = [
        os.path.join(base_dir, "data", "국립어린이과학관 전시물품 대장_260407 - AI놀이터.csv"),
        os.path.join(base_dir, "data", "국립어린이과학관 전시물품 대장_260407 - 관찰놀이터.csv"),
        os.path.join(base_dir, "data", "국립어린이과학관 전시물품 대장_260407 - 탐구놀이터.csv"),
        os.path.join(base_dir, "data", "국립어린이과학관 전시물품 대장_260407 - 행동놀이터.csv")
    ]
    
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

def get_tools():
    """LangChain agent에서 사용할 도구 목록 반환"""
    return [
        check_museum_closed_date,
        search_csc_live_info,
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
    
    base_prompt = f"""
당신은 국립어린이과학관 전문 안내 어시스턴트입니다.
[오늘 날짜] {today_kst} ({weekday_kr}요일) KST

{language_instruction.get(language, language_instruction["한국어"])}

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
- 불확실한 정보 → "02-3668-1500으로 문의해주세요"
"""
    
    if mode == "어린이":
        base_prompt += "\n\n어린이 모드: 쉽고 재미있게 설명하세요. 이모지를 활용하세요."
    
    return base_prompt

def render_source_buttons(sources: list):
    """출처 버튼 렌더링"""
    if sources:
        st.markdown("**📚 참고 자료:**")
        for i, source in enumerate(sources[:3]):
            if source.startswith("http"):
                st.markdown(f"[🔗 출처 {i+1}]({source})")
