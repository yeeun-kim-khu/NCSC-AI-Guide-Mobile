# learning.py - 또만나 놀이터 시스템 통합
# post_visit_learning.py + audiobook_generator.py + visualization.py 통합

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain.prompts import PromptTemplate
from openai import OpenAI
import os
import random
import re
import requests
from collections import Counter
from core import initialize_vector_db, load_zone_rows_from_csv

# 퀴즈 음성 출력 함수 가져오기
try:
    from voice import text_to_speech, get_language_code
except Exception:
    text_to_speech = None
    get_language_code = None

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def _queue_ga_event(event_name: str, params: dict | None = None) -> None:
    """Queue a GA event to be sent on the next render (safe before st.rerun)."""
    if "_ga_event_queue" not in st.session_state:
        st.session_state._ga_event_queue = []
    st.session_state._ga_event_queue.append({"name": event_name, "params": dict(params or {})})


def _safe_secret_get(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

# ============================================================================
# 놀이터 정보
# ============================================================================

ZONE_INFO = {
    "AI놀이터": {
        "floor": "1층",
        "description": "AI와 로봇 기술의 원리를 배워요",
        "has_data": True
    },
    "생각놀이터": {
        "floor": "1층",
        "description": "우리의 뇌는 특별할까요? 어떤 역할을 할까요? 뇌에 대해 알아보고 나의 뇌 능력도 테스트해 보아요!",
        "has_data": True
    },
    "행동놀이터": {
        "floor": "1층",
        "description": "몸을 움직이며 과학을 배워요",
        "has_data": True
    },
    "천체투영관": {
        "floor": "1층",
        "description": "우주와 별의 비밀을 알아봐요",
        "has_data": True
    },
    "탐구놀이터": {
        "floor": "2층",
        "description": "생활 속 과학원리를 탐구해요",
        "has_data": True
    },
    "관찰놀이터": {
        "floor": "2층",
        "description": "자연을 관찰하며 배워요",
        "has_data": True
    },
    "빛놀이터": {
        "floor": "2층",
        "description": "인터랙션으로 만나는 온대림! 에코크리에이터가 되어 아름다운 미래의 자연을 만들어가는 몰입형 실감 미디어 체험관",
        "has_data": True
    }
}

STORY_ZONE_WHITELIST = {
    "행동놀이터", "탐구놀이터", "관찰놀이터", "생각놀이터", "빛놀이터", "AI놀이터",
}

# 전시관별 갈등 템플릿 — 매칭되면 LLM 씨앗 생성 대신 직접 사용
STORY_CONFLICT_TEMPLATES = {
    "행동놀이터": [
        {
            "keywords": ["민첩성"],
            "갈등": "좁은 숲길에서 이쪽저쪽으로 굴러오는 돌멩이들을 피해야 하는데, 눈으로는 보이는데 몸이 따라가질 않아 계속 맞음",
            "실패": '"분명히 봤는데 왜 발이 안 움직이지?" — 눈과 발이 따로 노는 느낌',
            "아하": "치타를 보니 눈이 포착하는 순간 이미 온몸이 반응함. 눈→뇌→발이 하나로 연결되어야 한다는 걸 깨달음",
            "감각": "돌멩이가 발등에 탁 닿는 느낌, 치타 발이 땅을 박박 긁는 소리",
        },
        {
            "keywords": ["심폐지구력", "심폐"],
            "갈등": "긴 언덕을 뛰어 올라가야 하는데 절반쯤 가면 숨이 차고 다리가 무거워져 멈추게 됨",
            "실패": "짧게 쉬고 다시 뛰지만 또 금방 지침. 거북이는 느리지만 끝까지 멈추지 않고 올라감",
            "아하": "거북이는 심장이 튼튼해서 쿵쾅 뛴 후 금방 제자리로 돌아옴. 꾸준히 움직이면 심장도 폐도 단련된다는 걸 깨달음",
            "감각": "가슴이 터질 것 같은 두근거림, 귀에서 쿵쿵 소리, 거북이 등껍질의 묵직한 무게감",
        },
        {
            "keywords": ["근력", "근육"],
            "갈등": "길을 막은 커다란 바위를 밀어야 하는데 아무리 힘을 써도 꿈쩍도 안 함",
            "실패": "팔 하나로 밀기 → 안 됨. 온몸으로 밀기 → 조금 움직이나 금방 포기",
            "아하": "코끼리가 어깨와 다리 근육 전체를 한꺼번에 써서 밀어냄. 근육을 최대로 모아 쓰는 게 근력이라는 걸 깨달음",
            "감각": "팔뚝이 부들부들 떨리는 느낌, 바위가 스르르 밀리는 순간의 쾅 소리",
        },
        {
            "keywords": ["순발력"],
            "갈등": "갑자기 발밑에서 웅덩이가 나타나는데, 미리 알고 뛰면 넘을 수 있지만 갑자기 나타나면 너무 늦어버림",
            "실패": "보이고 나서 뛰려 하니 이미 빠짐. 준비하고 뛰면 되는데 갑작스러운 상황엔 속수무책",
            "아하": "순발력은 보는 순간 폭발적으로 힘을 내는 것. 짧은 시간에 강한 힘을 한꺼번에 쏟아야 함",
            "감각": "발바닥으로 땅을 탁 차는 느낌, 공중에 뜨는 찰나의 가벼움",
        },
        {
            "keywords": ["평형성", "균형"],
            "갈등": "흔들리는 외나무다리를 건너야 하는데 자꾸 한쪽으로 기울어 떨어짐",
            "실패": "빠르게 건너려 할수록 더 흔들림. 팔을 내리고 있으면 더 위험함",
            "아하": "고양이처럼 팔을 벌리고 무게 중심을 낮추면 흔들림에 맞게 몸을 조금씩 조절할 수 있음",
            "감각": "발바닥이 나무에 닿는 간지러운 느낌, 바람에 몸이 살짝 쏠리는 감각",
        },
        {
            "keywords": ["뇌와 신경", "신경"],
            "갈등": "뜨거운 돌을 밟았는데 발이 움직이질 않음. 뇌가 명령을 못 내리고 있는 것 같음",
            "실패": "직접 발을 들어올리려 해도 신호가 느리게 가서 계속 데임",
            "아하": "뇌는 신경으로 몸 전체와 연결되어 있고, 신호가 빠를수록 반응도 빠름. 뇌-신경-몸이 한 팀이라는 것을 깨달음",
            "감각": "발바닥이 따끔하게 데이는 느낌, 머릿속에서 번쩍 신호가 오는 느낌",
        },
    ],
    "탐구놀이터": [
        {
            "keywords": ["도르래"],
            "갈등": "높은 나뭇가지에 걸린 물건을 내려야 하는데 손이 닿지 않고 점프로도 안 됨",
            "실패": "긴 막대로 찔러보지만 물건이 더 높이 올라감",
            "아하": "나뭇가지에 줄을 걸고 도르래처럼 당기면 힘의 방향이 바뀌어 위에 있는 것을 아래로 내릴 수 있음",
            "감각": "줄이 팽팽해지는 느낌, 물건이 스르르 내려오는 모습",
        },
        {
            "keywords": ["빗면"],
            "갈등": "무거운 상자를 높은 선반 위에 올려야 하는데 들어올릴 힘이 부족함",
            "실패": "두 손으로 들어올리려다 중간에 떨어뜨림",
            "아하": "비스듬한 판을 대고 밀면 훨씬 적은 힘으로 올라감. 각도가 작을수록 힘이 덜 든다는 것을 발견",
            "감각": "판 위를 상자가 미끄러지듯 올라가는 느낌, 팔에서 힘이 쑥 빠지는 가벼움",
        },
        {
            "keywords": ["쐐기"],
            "갈등": "꽉 닫힌 문틈을 열어야 하는데 손가락이 들어가질 않음",
            "실패": "손톱으로 긁어보지만 문은 꿈쩍도 안 함",
            "아하": "뾰족하게 깎인 쐐기를 틈에 넣고 두드리면 힘이 양옆으로 퍼지면서 문이 쩍 벌어짐",
            "감각": "쐐기가 틈으로 파고드는 느낌, 나무가 쩍 갈라지는 소리",
        },
        {
            "keywords": ["부력"],
            "갈등": "물속에 가라앉은 보물 상자를 꺼내야 하는데 너무 무거워 들 수 없음",
            "실패": "밧줄로 당겨보지만 물 밖에서는 너무 무거움",
            "아하": "상자 안에 공기를 채우면 물이 밀어 올리는 힘(부력)이 생겨 저절로 떠오름",
            "감각": "공기 방울이 보글보글 올라오는 모습, 상자가 천천히 수면으로 떠오르는 느낌",
        },
        {
            "keywords": ["에너지 하베스팅", "에너지"],
            "갈등": "깜깜한 동굴에서 불빛이 꺼졌는데 전기가 없음",
            "실패": "소리쳐도, 손을 흔들어도 불이 안 켜짐",
            "아하": "발을 구르거나 뛰면 그 움직임이 전기로 바뀐다는 걸 발견. 온 힘을 다해 발을 구르자 불이 반짝반짝 켜짐",
            "감각": "발판이 쿵쿵 울리는 진동, 어둠 속에서 불빛이 하나둘 켜지는 순간",
        },
        {
            "keywords": ["사이클로이드"],
            "갈등": "두 갈래 길 중 어느 쪽이 더 빨리 도착하는지 선택해야 함. 직선이 더 짧아 보이는데 곡선 쪽이 더 빠르다고 함",
            "실패": "직선을 선택했다가 늦게 도착",
            "아하": "곡선을 따라 내려오면 처음에 빠르게 가속이 붙어서 결국 더 빨리 도착함. 짧은 길이 항상 빠른 게 아니라는 것을 깨달음",
            "감각": "경사를 내려갈 때 몸이 앞으로 쏠리는 가속감",
        },
    ],
    "관찰놀이터": [
        {
            "keywords": ["철새", "이동", "철새 이동"],
            "갈등": "혼자 남겨진 새 친구가 무리를 찾아가야 하는데 어느 방향인지 모름",
            "실패": "아무 방향이나 날아가다 길을 잃음",
            "아하": "철새들은 먹이와 따뜻한 날씨를 찾아 계절마다 방향을 정해 이동함. 바람과 해의 위치를 따라가면 무리를 찾을 수 있음",
            "감각": "날개 끝에 닿는 차가운 바람, 저 멀리 무리의 V자 대형이 보이는 순간",
        },
        {
            "keywords": ["위장", "보호색"],
            "갈등": "숲속에서 친구를 찾아야 하는데 아무리 봐도 보이지 않음",
            "실패": "소리를 질러도 반응이 없고, 눈앞에 있는데도 못 찾음",
            "아하": "친구가 주변 나뭇잎과 똑같은 색으로 몸을 숨기고 있었음. 보호색은 천적을 피하기 위한 생존 전략",
            "감각": "눈을 비비고 다시 봐도 안 보이는 답답함, 발견하는 순간의 놀라움",
        },
        {
            "keywords": ["공룡", "알", "부화"],
            "갈등": "차갑게 식어버린 알을 따뜻하게 해야 부화할 수 있는데 어떻게 해야 할지 모름",
            "실패": "알 옆에 불을 피워봤지만 한쪽만 뜨거워짐",
            "아하": "양손으로 감싸 체온으로 천천히 골고루 데워야 함. 어미 공룡도 그렇게 했다는 걸 발견",
            "감각": "손바닥으로 느끼는 알의 차가운 표면이 조금씩 따뜻해지는 감각",
        },
    ],
    "생각놀이터": [
        {
            "keywords": ["운동과 뇌", "브레인", "집중력"],
            "갈등": "어려운 문제를 풀어야 하는데 머리가 멍하고 생각이 안 남",
            "실패": "가만히 앉아서 계속 생각해봐도 안 풀림. 오히려 더 답답해짐",
            "아하": "잠깐 뛰고 나서 문제를 보니 갑자기 생각이 떠오름. 운동을 하면 뇌로 피와 산소가 더 많이 가서 집중력이 높아진다는 것",
            "감각": "달리고 난 후 뺨이 뜨거워지는 느낌, 머릿속이 맑아지는 느낌",
        },
        {
            "keywords": ["반응속도", "반응 속도", "뇌의 신호"],
            "갈등": "날아오는 공을 잡아야 하는데 자꾸 늦게 반응해서 못 잡음",
            "실패": "눈으로 보이는데 손이 따라가지 않음. '분명히 봤는데 왜 손이 늦지?'",
            "아하": "눈→뇌→손으로 신호가 전달되는 시간이 있음. 집중하면 이 신호가 빨라진다는 것",
            "감각": "공이 손바닥에 탁 잡히는 감각, 빗나갔을 때 허공을 가르는 허탈감",
        },
        {
            "keywords": ["기억력", "단기기억", "장기기억"],
            "갈등": "중요한 것의 위치를 기억해야 하는데 너무 많아서 다 잊어버림",
            "실패": "그냥 외우려 해도 금방 사라짐",
            "아하": "반복하거나 의미를 붙이면 단기기억이 장기기억으로 저장됨. 연결고리를 만들면 오래 기억할 수 있음",
            "감각": "기억이 뚝 끊기는 답답함, 연결고리를 찾는 순간 번쩍 떠오르는 느낌",
        },
        {
            "keywords": ["뇌 가소성", "가소성", "뉴런"],
            "갈등": "처음 해보는 것이라 자꾸 실수하고 못 함",
            "실패": "한 번 해보고 '난 못해'라고 포기",
            "아하": "뇌는 새로운 걸 배울 때마다 뉴런 연결이 새로 생김. 반복할수록 연결이 강해져서 잘하게 됨",
            "감각": "처음엔 삐뚤삐뚤하다가 점점 부드러워지는 손의 움직임",
        },
    ],
    "빛놀이터": [
        {
            "keywords": ["씨앗", "산포"],
            "갈등": "씨앗을 멀리 보내야 하는데 그냥 떨어뜨리면 바로 아래만 떨어짐",
            "실패": "손으로 던져봐도 멀리 안 감",
            "아하": "민들레처럼 솜털이 있으면 바람을 타고, 도깨비바늘처럼 가시가 있으면 동물 털에 붙어서 멀리 퍼짐. 식물마다 씨앗을 퍼뜨리는 전략이 다름",
            "감각": "홀씨가 바람에 하늘하늘 날아가는 모습, 손으로 후 부는 느낌",
        },
        {
            "keywords": ["광합성"],
            "갈등": "어두운 곳에 있는 식물이 시들어가고 있는데 어떻게 살려야 할지 모름",
            "실패": "물만 줘봐도 계속 시듦",
            "아하": "식물은 햇빛이 있어야 잎 속 초록 세포가 깨어나 에너지를 만들 수 있음. 빛이 없으면 아무리 물을 줘도 소용없음",
            "감각": "시들었던 잎이 햇빛을 받아 천천히 고개를 드는 모습, 잎맥을 따라 흐르는 수분의 느낌",
        },
    ],
    "AI놀이터": [
        {
            "keywords": ["기후변화", "물의 순환", "물방울"],
            "갈등": "메마른 땅에 물이 필요한데 어디서 와야 할지 모름",
            "실패": "바다의 물을 직접 가져오려 해도 너무 멀고 짬",
            "아하": "물은 증발해서 구름이 되고 비가 되어 돌아옴. 지금 당장 없어 보여도 물은 어딘가에서 여행 중",
            "감각": "구름이 뭉게뭉게 모이는 모습, 빗방울이 톡톡 땅에 닿는 소리",
        },
        {
            "keywords": ["인공지능", "AI", "학습"],
            "갈등": "아무것도 모르는 AI에게 지구를 구하는 방법을 가르쳐야 하는데 어떻게 말해야 할지 모름",
            "실패": "너무 어려운 말로 설명하니 AI가 이해 못 함",
            "아하": "인공지능은 사람이 가르쳐준 대로 배움. 쉽고 정확하게 반복해서 알려줄수록 점점 똑똑해짐",
            "감각": "AI의 눈이 반짝 빛나는 순간, 처음엔 틀리다가 점점 맞춰가는 과정",
        },
    ],
}


def _get_conflict_template(zone_name: str, principles_text: str) -> dict | None:
    """zone_name과 principles_text 키워드로 갈등 템플릿 매칭. 없으면 None."""
    p_lower = principles_text.lower()
    for zone_key, templates in STORY_CONFLICT_TEMPLATES.items():
        # 다중 전시관("행동놀이터, 탐구놀이터") 대응 — 어느 zone이든 포함되면 탐색
        if zone_key not in zone_name:
            continue
        for tmpl in templates:
            for kw in tmpl["keywords"]:
                if kw.lower() in p_lower:
                    return tmpl
    return None


ZONE_GROUPS = {
    "1층놀이터(AI·행동·생각 놀이터)": ["AI놀이터", "행동놀이터", "생각놀이터"],
    "2층(관찰·탐구 놀이터)": ["관찰놀이터", "탐구놀이터"],
    "천체투영관": ["천체투영관"],
    "빛놀이터": ["빛놀이터"],
}

ZONE_GROUP_LABELS = {
    "한국어": {
        "1층놀이터(AI·행동·생각 놀이터)": "1층놀이터(AI·행동·생각)",
        "2층(관찰·탐구 놀이터)": "2층(관찰·탐구)",
        "천체투영관": "천체투영관",
        "빛놀이터": "빛놀이터",
    },
    "English": {
        "1층놀이터(AI·행동·생각 놀이터)": "1F (AI / Activity / Thinking)",
        "2층(관찰·탐구 놀이터)": "2F (Discovery / Exploration)",
        "천체투영관": "Planetarium",
        "빛놀이터": "Interactive Theater",
    },
    "日本語": {
        "1층놀이터(AI·행동·생각 놀이터)": "1階 (AI・うごき・考える)",
        "2층(관찰·탐구 놀이터)": "2階 (しらべる・たんきゅう)",
        "천체투영관": "プラネタリウム",
        "빛놀이터": "ひかりゾーン",
    },
    "中文": {
        "1층놀이터(AI·행동·생각 놀이터)": "1层 (AI·行动·思考)",
        "2층(관찰·탐구 놀이터)": "2层 (观察·探究)",
        "천체투영관": "天体投影馆",
        "빛놀이터": "光区",
    },
}


def _select_zones_by_group(prefix_key: str, language_mode: str = "한국어") -> list[str]:
    selected = []
    label_map = ZONE_GROUP_LABELS.get(language_mode, ZONE_GROUP_LABELS["한국어"])
    for label, zones in ZONE_GROUPS.items():
        display_label = label_map.get(label, label)
        if st.checkbox(display_label, key=f"{prefix_key}_{label}"):
            # Only add zones that have data
            for zone in zones:
                if ZONE_INFO.get(zone, {}).get("has_data", False):
                    selected.append(zone)
    seen = set()
    uniq = []
    for z in selected:
        if z not in seen:
            seen.add(z)
            uniq.append(z)
    return uniq

def _render_zone_buttons(prefix_key: str, display_fn, language_mode: str = "한국어") -> list:
    """체크박스 대신 토글 버튼으로 존 선택 (세션 스테이트 기반)."""
    sel_key = f"zone_sel_{prefix_key}"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = []

    all_zones = (
        [z for z, info in ZONE_INFO.items() if info["floor"] == "1층"] +
        [z for z, info in ZONE_INFO.items() if info["floor"] == "2층"]
    )

    cols = st.columns(3)
    for i, zone in enumerate(all_zones):
        is_sel = zone in st.session_state[sel_key]
        zone_disp = display_fn(zone)
        label = f"✅ {zone_disp}" if is_sel else zone_disp
        with cols[i % 3]:
            if st.button(
                label,
                key=f"zonebtn_{prefix_key}_{zone}",
                type="primary" if is_sel else "secondary",
                use_container_width=True,
            ):
                sel = list(st.session_state[sel_key])
                if is_sel:
                    sel.remove(zone)
                else:
                    sel.append(zone)
                st.session_state[sel_key] = sel
                st.rerun()

    return list(st.session_state[sel_key])

# ============================================================================
# CSV 데이터 로딩
# ============================================================================

def _preload_all_zone_csv_rows():
    data = {}
    for zone, info in ZONE_INFO.items():
        if info.get("has_data"):
            try:
                rows = load_zone_rows_from_csv(zone)
                data[zone] = rows
                print(f"Loaded {len(rows)} rows for {zone}")
            except Exception as e:
                print(f"Error loading CSV for {zone}: {e}")
                data[zone] = []
    return data


def _csv_fingerprint() -> str:
    """data/ 폴더 CSV 파일들의 수정시간 합산 → 변경 감지용."""
    import glob as _glob
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    total = 0
    for p in sorted(_glob.glob(os.path.join(base, "*.csv"))):
        try:
            total += int(os.path.getmtime(p) * 1000)
        except Exception:
            pass
    return str(total)

# ============================================================================
# 키워드 추출 및 렌더링
# ============================================================================

def _extract_zone_keywords(zone_rows, top_n=12):
    text = " ".join(
        [
            str(r.get("title", "")) + " " + str(r.get("category", "")) + " " + str(r.get("content", ""))
            for r in (zone_rows or [])
        ]
    )
    words = re.findall(r"\b\w+\b", text)
    counter = Counter(words)
    keywords = [w for w, _ in counter.most_common(top_n * 3) if len(w) > 1]
    return keywords[:top_n]


def _split_title_ko_en(raw_title: str):
    """CSV 제목 셀이 '한국어\\n영문' 형태로 합쳐진 경우 분리.

    반환: (kr, en) — 영문이 없으면 en은 빈 문자열.
    한글이 없는 경우(영문 전용 전시) kr=영문, en="" 로 반환.
    """
    if not raw_title:
        return "", ""
    parts = [p.strip() for p in re.split(r"[\r\n]+", str(raw_title)) if p.strip()]
    if not parts:
        return "", ""

    KOR_RE = re.compile(r"[\uac00-\ud7a3]")
    kr = ""
    en = ""
    for p in parts:
        if KOR_RE.search(p):
            if not kr:
                kr = p
        else:
            if not en:
                en = p
    if not kr and parts:
        # 한글 없는 행 → 첫 줄을 kr 자리에 둔다 (다국어 번역의 소스 텍스트로 사용)
        kr = parts[0]
        if len(parts) > 1 and not en:
            en = parts[1]
    kr = re.sub(r"\s+", " ", kr).strip()
    en = re.sub(r"\s+", " ", en).strip()
    return kr, en


def _extract_zone_keywords_from_titles(zone_rows, top_n=20):
    """제목을 한/영 쌍으로 추출.

    반환: list of (kr, en) tuples — kr은 항상 채워져 있음, en은 비어있을 수 있음.
    """
    # 키워드 컬럼 값 분류: Y=제목 그대로, 그룹명=묶음, 빈칸=제외
    _has_any_flag = any(r.get("keyword_flag", "") for r in (zone_rows or []))

    pairs = []
    seen_kr = set()

    if _has_any_flag:
        # Y 행: 제목을 키워드로
        for r in (zone_rows or []):
            if r.get("keyword_flag", "") != "Y":
                continue
            raw = str(r.get("title", "")).strip()
            if not raw or len(raw) <= 1 or "체험방법" in raw:
                continue
            kr, en = _split_title_ko_en(raw)
            if not kr or len(kr) <= 1:
                continue
            kr_normalized = re.sub(r"\s*[-–]\s*체험\S*$", "", kr).strip() or kr
            if kr_normalized in seen_kr:
                continue
            seen_kr.add(kr_normalized)
            pairs.append((kr_normalized, en))
        # 그룹명 행: keyword_flag 값 자체를 키워드로
        for r in (zone_rows or []):
            flag = r.get("keyword_flag", "").strip()
            if not flag or flag == "Y":
                continue
            if flag in seen_kr:
                continue
            seen_kr.add(flag)
            pairs.append((flag, ""))
    else:
        # 자동 파싱 fallback: 체험/실감형 우선, 패널 후순위
        _PANEL_TYPES = {"패널", "도입부 패널", "패널 (게시판형)", "패널 (인터랙티브)"}
        def _sort_key(r):
            zt = str(r.get("zone_type", "")).strip()
            return 0 if zt and zt not in _PANEL_TYPES else 1
        for r in sorted(zone_rows or [], key=_sort_key):
            raw = str(r.get("title", "")).strip()
            if not raw or len(raw) <= 1 or "체험방법" in raw:
                continue
            kr, en = _split_title_ko_en(raw)
            if not kr or len(kr) <= 1:
                continue
            kr_normalized = re.sub(r"\s*[-–]\s*체험\S*$", "", kr).strip() or kr
            if kr_normalized in seen_kr:
                continue
            seen_kr.add(kr_normalized)
            pairs.append((kr_normalized, en))
    return pairs[:top_n]


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def _extract_zone_keywords_llm(zone_name: str, language_mode: str, csv_compact_text: str):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

    if language_mode == "한국어":
        prompt = f"""너는 초등 4~6학년 어린이(10~12세)와 학부모를 위한 전시관 키워드 편집자야.

아래는 '{zone_name}' 전시물 CSV에서 뽑은 제목/설명 일부야.
이 내용을 보고, 아이가 이해하기 쉬운 '굵직한 키워드'만 8~12개 뽑아줘.

규칙:
1) 조사/어미/추상어(예: 우리, 해요, 방법, 활동, 체험)는 제외.
2) 가능한 한 명사 위주.
3) 너무 전문적인 단어는 쉬운 말로 바꿔.
4) 결과는 쉼표로 구분한 한 줄.

CSV 요약:
{csv_compact_text}
"""
    else:
        prompt = f"""You are a keyword editor for young kids and parents.

From the exhibit CSV snippets for '{zone_name}', extract 8-12 big, easy keywords.
Include specific items (foods, objects, materials) mentioned in the exhibits.
Avoid particles/verbs/very generic words.
Return a single line, comma-separated.

CSV snippets:
{csv_compact_text}
"""

    resp = llm.invoke(prompt)
    line = (resp.content or "").strip().split("\n")[0]
    parts = [p.strip() for p in re.split(r"[,，]", line) if p.strip()]
    uniq = []
    seen = set()
    for p in parts:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq[:12]


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def _translate_keywords_cached(keywords_tuple: tuple, target_language: str) -> list:
    """한국어 키워드를 다른 언어로 번역 (캐시됨)"""
    if target_language == "한국어" or not keywords_tuple:
        return list(keywords_tuple)
    lang_label = {
        "English": "English",
        "日本語": "Japanese (in 日本語 / kana/kanji only)",
        "中文": "Simplified Chinese (中文 only)",
    }.get(target_language, "English")
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        joined = ", ".join(keywords_tuple)
        prompt = (
            f"Translate each Korean keyword into {lang_label}. "
            f"Keep them concise (1-3 words). Output ONLY a comma-separated single line, "
            f"in the same order, with the same number of items. Do not add explanations.\n\n"
            f"Keywords: {joined}"
        )
        resp = llm.invoke(prompt)
        line = (resp.content or "").strip().split("\n")[0]
        parts = [p.strip() for p in re.split(r"[,，、]", line) if p.strip()]
        if len(parts) == len(keywords_tuple):
            return parts
        # length mismatch: pad/truncate gracefully
        if parts:
            if len(parts) < len(keywords_tuple):
                parts = parts + list(keywords_tuple[len(parts):])
            return parts[:len(keywords_tuple)]
    except Exception as e:
        print(f"키워드 번역 실패: {e}")
    return list(keywords_tuple)


def _get_zone_keywords(zone_name: str, zone_rows, language_mode: str):
    """전시관 키워드를 (한국어 원본, 표시 문자열) 튜플 리스트로 반환.

    - 1순위: CSV 제목에서 한/영 분리 추출 → English 모드에서는 CSV 영문 그대로 사용
    - 2순위: LLM 키워드 추출(이미 언어 모드 반영) → kr 자리에도 동일 텍스트 사용 (퀴즈용)
    - 3순위: 단어 빈도 기반 폴백 → 필요 시 번역
    """
    pairs = _extract_zone_keywords_from_titles(zone_rows)  # list of (kr, en)
    if pairs:
        # 한국어
        if language_mode == "한국어":
            return [(kr, kr) for kr, _ in pairs]
        # English: CSV 영문이 있으면 그대로, 없으면 번역
        if language_mode == "English":
            need_translate_idx = [i for i, (_, en) in enumerate(pairs) if not en]
            if need_translate_idx:
                src = tuple(pairs[i][0] for i in need_translate_idx)
                translated = _translate_keywords_cached(src, "English")
                trans_map = dict(zip(need_translate_idx, translated))
            else:
                trans_map = {}
            out = []
            for i, (kr, en) in enumerate(pairs):
                disp = en if en else trans_map.get(i, kr)
                out.append((kr, disp))
            return out
        # 日本語 / 中文: 한국어 원본을 번역
        kr_list = [kr for kr, _ in pairs]
        translated = _translate_keywords_cached(tuple(kr_list), language_mode)
        return list(zip(kr_list, translated))

    # 폴백: LLM 키워드 추출 (언어 모드 반영하여 직접 생성)
    compact_lines = []
    for r in (zone_rows or [])[:40]:
        title = str(r.get("title", "")).strip()
        cat = str(r.get("category", "")).strip()
        content = str(r.get("content", "")).strip()
        if title or content:
            compact_lines.append(f"- {title} ({cat}) {content[:120]}")
    csv_compact_text = "\n".join(compact_lines)[:6000]

    kws: list = []
    try:
        kws = _extract_zone_keywords_llm(zone_name, language_mode, csv_compact_text)
    except Exception as e:
        print(f"키워드 LLM 추출 실패: {e}")

    if not kws:
        kws = _extract_zone_keywords(zone_rows)
        if language_mode != "한국어" and kws:
            translated = _translate_keywords_cached(tuple(kws), language_mode)
            return list(zip(kws, translated))

    return [(k, k) for k in kws]


def _render_keyword_tags(zone_name: str, keyword_pairs, zone_rows, language_mode: str = "한국어", mode: str = "exhibits", llm=None):
    """키워드 버튼 렌더링.
    keyword_pairs: list of (kr_keyword, display_keyword) tuples
    mode: 'exhibits' (기존: 관련 전시물 표시) | 'quiz' (바로 퀴즈 생성) | 'question' (질문 입력)
    """
    if not keyword_pairs:
        return None, None

    # 질문 모드에서는 키워드 태그를 렌더링하지 않음 (빈 공백 제거)
    if mode == "question":
        return None, None

    heading_text = {
        "한국어": "##### 🔑 키워드",
        "English": "##### 🔑 Keywords",
        "日本語": "##### 🔑 キーワード",
        "中文": "##### 🔑 关键词",
    }.get(language_mode, "##### 🔑 Keywords")
    st.markdown(heading_text)

    state_key = f"kw_selected_{zone_name}_{mode}"
    if state_key not in st.session_state:
        st.session_state[state_key] = ""

    num_cols = min(len(keyword_pairs), 4)
    cols = st.columns(num_cols)
    selected_kw = st.session_state.get(state_key, "")
    for i, (kw_kr, kw_disp) in enumerate(keyword_pairs):
        with cols[i % num_cols]:
            is_selected = (selected_kw == kw_kr)
            btn_type = "primary" if is_selected else "secondary"
            if st.button(kw_disp, key=f"kw_btn_{zone_name}_{mode}_{kw_kr}", type=btn_type):
                st.session_state[state_key] = kw_kr
                st.rerun()
    selected_disp = selected_kw
    for kw_kr, kw_disp in keyword_pairs:
        if kw_kr == selected_kw:
            selected_disp = kw_disp
            break

    clear_label = {
        "한국어": "키워드 선택 해제",
        "English": "Clear keyword",
        "日本語": "キーワード解除",
        "中文": "清除关键词",
    }.get(language_mode, "Clear keyword")
    selected_label = {
        "한국어": "선택한 키워드",
        "English": "Selected keyword",
        "日本語": "選んだキーワード",
        "中文": "已选关键词",
    }.get(language_mode, "Selected keyword")

    if selected_kw:
        st.caption(f"{selected_label}: {selected_disp}")
        if st.button(clear_label, key=f"kw_clear_{zone_name}_{mode}"):
            st.session_state[state_key] = ""
            selected_kw = ""
            st.rerun()

    return selected_kw, selected_disp


def _render_quiz_card(zone_name: str, keyword: str, quiz_obj, language_mode: str = "한국어"):
    """4지선다 퀴즈 카드 렌더링.

    - 정답은 expander 안에 숨김 → 사용자가 직접 펼쳐야 정답·해설 확인.
    - 문제 음성 듣기 / 정답 음성 듣기 별도 버튼 (정답 음성은 expander 안에 위치하여 자동 재생되지 않음).
    - quiz_obj 가 비어있거나 폴백(raw만 존재)이면 그대로 markdown 출력.
    """
    if not quiz_obj:
        return

    labels = {
        "한국어": {
            "question": "📘 문제",
            "listen_q": "🔊 문제 듣기",
            "reveal": "🎁 정답 보기",
            "answer": "✅ 정답",
            "explain": "💡 해설",
            "listen_a": "🔊 정답 듣기",
            "tts_fail": "음성 생성에 실패했어요.",
        },
        "English": {
            "question": "📘 Question",
            "listen_q": "🔊 Listen to question",
            "reveal": "🎁 Show answer",
            "answer": "✅ Answer",
            "explain": "💡 Explanation",
            "listen_a": "🔊 Listen to answer",
            "tts_fail": "TTS generation failed.",
        },
        "日本語": {
            "question": "📘 問題",
            "listen_q": "🔊 問題を聞く",
            "reveal": "🎁 答えを見る",
            "answer": "✅ 答え",
            "explain": "💡 解説",
            "listen_a": "🔊 答えを聞く",
            "tts_fail": "音声の生成に失敗しました。",
        },
        "中文": {
            "question": "📘 题目",
            "listen_q": "🔊 听题目",
            "reveal": "🎁 查看答案",
            "answer": "✅ 答案",
            "explain": "💡 解析",
            "listen_a": "🔊 听答案",
            "tts_fail": "语音生成失败。",
        },
    }
    L = labels.get(language_mode, labels["한국어"])

    # JSON 파싱 실패 폴백: raw만 출력
    if not isinstance(quiz_obj, dict) or "question" not in quiz_obj:
        raw = quiz_obj.get("raw") if isinstance(quiz_obj, dict) else None
        if raw:
            st.markdown(raw)
        return

    question = quiz_obj.get("question", "")
    options = quiz_obj.get("options", [])
    correct_index = quiz_obj.get("correct_index", 0)
    explanation = quiz_obj.get("explanation", "")

    # 문제 + 선택지 표시
    st.markdown(f"**{L['question']}**: {question}")
    options_md = "\n".join(f"{i + 1}. {opt}" for i, opt in enumerate(options))
    st.markdown(options_md)

    # 문제 TTS (정답 제외)
    try:
        from voice import text_to_speech, get_language_code
    except Exception:
        text_to_speech = None
        get_language_code = None

    q_audio_key = f"quiz_audio_q_{zone_name}_{keyword}"
    if text_to_speech is not None:
        if st.button(L["listen_q"], key=f"btn_q_tts_{zone_name}_{keyword}"):
            with st.spinner("..."):
                try:
                    lang_code = get_language_code(language_mode) if get_language_code else "ko"
                    tts_text = f"{question}. " + ". ".join(
                        f"{i + 1}번, {opt}" if language_mode == "한국어" else f"{i + 1}. {opt}"
                        for i, opt in enumerate(options)
                    )
                    audio = text_to_speech(tts_text, language=lang_code)
                    if audio:
                        st.session_state[q_audio_key] = audio
                    else:
                        st.warning(L["tts_fail"])
                except Exception as e:
                    print(f"문제 TTS 오류: {e}")
                    st.warning(L["tts_fail"])
        if st.session_state.get(q_audio_key):
            st.audio(st.session_state[q_audio_key], format="audio/mp3")

    # 정답: expander 로 숨김 (사용자가 펼쳐야 보임)
    # 정답 토글: 문제(question) 텍스트의 해시를 키에 포함 → 새 문제마다 자동으로 닫힘 상태로 시작
    import hashlib as _hashlib
    qid = _hashlib.md5(str(question).encode("utf-8")).hexdigest()[:8] if question else "0"
    reveal_key = f"quiz_reveal_{zone_name}_{keyword}_{qid}"
    if reveal_key not in st.session_state:
        st.session_state[reveal_key] = False

    hide_label = {
        "한국어": "🙈 정답 숨기기",
        "English": "🙈 Hide answer",
        "日本語": "🙈 答えを隠す",
        "中文": "🙈 隐藏答案",
    }.get(language_mode, "🙈 Hide answer")

    btn_label = hide_label if st.session_state[reveal_key] else L["reveal"]
    if st.button(btn_label, key=f"btn_reveal_{zone_name}_{keyword}_{qid}"):
        new_reveal_state = not st.session_state[reveal_key]
        st.session_state[reveal_key] = new_reveal_state
        # 처음 정답을 확인하면 스탬프 적립
        if new_reveal_state:
            stamps = list(st.session_state.get("science_stamps", []))
            if zone_name not in stamps:
                stamps.append(zone_name)
            st.session_state["science_stamps"] = stamps
        st.rerun()

    if st.session_state[reveal_key]:
        if 0 <= correct_index < len(options):
            st.success(f"**{L['answer']}**: {correct_index + 1}. {options[correct_index]}")
        if explanation:
            st.markdown(f"**{L['explain']}**: {explanation}")

        # 정답 음성: 정답이 펼쳐진 상태에서만 노출 → 자동 재생 없음, 사용자가 명시적으로 클릭해야 재생
        a_audio_key = f"quiz_audio_a_{zone_name}_{keyword}_{qid}"
        if text_to_speech is not None:
            if st.button(L["listen_a"], key=f"btn_a_tts_{zone_name}_{keyword}_{qid}"):
                with st.spinner("..."):
                    try:
                        lang_code = get_language_code(language_mode) if get_language_code else "ko"
                        ans_text = ""
                        if 0 <= correct_index < len(options):
                            if language_mode == "한국어":
                                ans_text = f"정답은 {correct_index + 1}번, {options[correct_index]} 입니다. "
                            elif language_mode == "日本語":
                                ans_text = f"答えは{correct_index + 1}番、{options[correct_index]} です。"
                            elif language_mode == "中文":
                                ans_text = f"答案是第{correct_index + 1}个，{options[correct_index]}。"
                            else:
                                ans_text = f"The answer is number {correct_index + 1}, {options[correct_index]}. "
                        ans_text += explanation
                        audio = text_to_speech(ans_text, language=lang_code)
                        if audio:
                            st.session_state[a_audio_key] = audio
                        else:
                            st.warning(L["tts_fail"])
                    except Exception as e:
                        print(f"정답 TTS 오류: {e}")
                        st.warning(L["tts_fail"])
            if st.session_state.get(a_audio_key):
                st.audio(st.session_state[a_audio_key], format="audio/mp3")


# ============================================================================
# RAG 검색 및 원리 추출
# ============================================================================

def _load_exhibits_from_csv_direct(zone_name):
    """CSV에서 직접 전시물 로드 (RAG fallback)"""
    try:
        rows = load_zone_rows_from_csv(zone_name)
        exhibits = []
        for r in rows:
            title = r.get("title", "")
            content = r.get("content", "")
            detail = r.get("detail", "")
            category = r.get("category", "")
            text = f"[{zone_name}] {title}\nCategory: {category}\nContent: {content}\nDetails: {detail}"
            exhibits.append({
                "content": text,
                "metadata": {
                    "source": f"csv_{zone_name}",
                    "title": title,
                    "category": zone_name,
                    "subcategory": category,
                    "detail": detail,
                }
            })
        print(f"CSV 직접 로드: {zone_name}에서 {len(exhibits)}개 전시물")
        return exhibits
    except Exception as e:
        print(f"CSV 직접 로드 오류: {e}")
        return []


def get_zone_exhibits_from_rag(zone_name, vector_db):
    """RAG에서 해당 놀이터의 전시물 정보 가져오기 (CSV fallback 포함)"""
    exhibits = []
    try:
        if vector_db is None:
            print(f"vector_db is None for {zone_name}, falling back to CSV")
            return _load_exhibits_from_csv_direct(zone_name)

        docs = []
        for q in (zone_name, f"[{zone_name}]", f"csv_{zone_name}"):
            try:
                results = vector_db.similarity_search(q, k=80)
                docs.extend(results)
                print(f"Query '{q}' returned {len(results)} docs")
            except Exception as e:
                print(f"RAG 검색 오류(쿼리={q}): {e}")

        print(f"Total docs retrieved: {len(docs)}")

        seen_keys = set()
        expected_source = f"csv_{zone_name}"

        for doc in docs:
            metadata = doc.metadata or {}
            category = metadata.get("category", "")
            source = metadata.get("source", "")
            title = metadata.get("title", "")
            content = doc.page_content or ""

            is_csv_doc_for_zone = (source == expected_source) or (category == zone_name)
            if not is_csv_doc_for_zone:
                continue

            dedup_key = (source, category, title, content[:200])
            if dedup_key in seen_keys:
                continue

            exhibits.append({
                "content": content,
                "metadata": metadata
            })
            seen_keys.add(dedup_key)

        print(f"최종 RAG 검색 결과: {zone_name}에서 {len(exhibits)}개 전시물 발견")
    except Exception as e:
        print(f"RAG 검색 오류: {e}")
        import traceback
        traceback.print_exc()

    # Fallback: RAG 결과가 없으면 CSV에서 직접 로드
    if not exhibits:
        print(f"RAG 결과 없음, CSV fallback 사용: {zone_name}")
        exhibits = _load_exhibits_from_csv_direct(zone_name)

    return exhibits

# ============================================================================
# 궁금해요 ReAct 도구
# ============================================================================

def _get_question_tools(zone_name, zone_rows, vector_db):
    """궁금해요 ReAct 에이전트용 도구 목록 반환"""
    
    def search_exhibits_rag(query: str) -> str:
        """전시물 RAG 검색 도구"""
        try:
            if vector_db:
                docs = vector_db.similarity_search(query, k=10)
                results = []
                for doc in docs:
                    metadata = doc.metadata or {}
                    title = metadata.get("title", "")
                    content = doc.page_content or ""
                    if title and content:
                        results.append(f"[{title}] {content}")
                return "\n\n".join(results[:5])
            else:
                # CSV fallback
                results = []
                for r in zone_rows[:5]:
                    title = r.get("title", "")
                    content = r.get("content", "")
                    if title and content:
                        results.append(f"[{title}] {content}")
                return "\n\n".join(results)
        except Exception as e:
            return f"검색 오류: {e}"
    
    def get_exhibit_detail(title: str) -> str:
        """특정 전시물 상세 정보 조회"""
        try:
            for r in zone_rows:
                if title.lower() in r.get("title", "").lower():
                    detail = r.get("detail", "")
                    content = r.get("content", "")
                    return f"[{r.get('title', '')}] {content}\n[세부 설명] {detail if detail else '없음'}"
            return f"'{title}'에 해당하는 전시물을 찾을 수 없습니다."
        except Exception as e:
            return f"조회 오류: {e}"
    
    return [search_exhibits_rag, get_exhibit_detail]

def _create_question_agent(llm, tools, language_mode, user_mode):
    """궁금해요 ReAct 에이전트 생성"""
    
    # ReAct 프롬프트 템플릿
    is_child = (user_mode == "어린이")
    
    if language_mode == "한국어":
        if is_child:
            answer_instruction = "초등 4~6학년(10~12세) 어린이가 이해할 수 있게, 쉬운 단어와 실생활 예시를 들어 한국어로 재미있게 설명해주세요. 3~4문장으로 마무리하고, 마지막에 '신기하지 않아?'처럼 호기심을 자극하는 한마디를 추가해주세요."
        else:
            answer_instruction = "과학적으로 정확하고 자세하게 한국어로 설명해주세요. 전시물과 관련된 원리 및 실생활 예시를 포함해 주세요."
    elif language_mode == "English":
        if is_child:
            answer_instruction = "Please explain in clear English for upper elementary students (ages 10-12), using everyday examples and relatable comparisons. Keep it to 3-4 sentences and end with a curious 'Did you know?' hook!"
        else:
            answer_instruction = "Please answer in clear, accurate English with scientific detail and real-life examples related to the exhibit."
    elif language_mode == "日本語":
        if is_child:
            answer_instruction = "小学4〜6年生（10〜12歳）にわかるように、身近な例えをつかって日本語で3〜4文で説明してください。最後に「不思議だと思わない？」など好奇心を刺激する一言を加えてください。"
        else:
            answer_instruction = "科学的に正確で詳しい日本語で、展示に関連した原理と実例を含めて説明してください。"
    elif language_mode == "中文":
        if is_child:
            answer_instruction = "请用通俗易懂的中文为小学高年级学生（10-12岁）解释，使用贴近生活的比喻，用3-4句话说明，最后加一句激发好奇心的话，例如：你有没有想过这个问题？"
        else:
            answer_instruction = "请用科学准确、详细的中文说明，并结合展品相关的原理和实际例子。"
    else:
        answer_instruction = "어린이가 이해하기 쉽게 답변해주세요."
    
    prompt = PromptTemplate.from_template(
        """너는 국립어린이과학관의 전시물에 대해 설명하는 친절한 안내원이야.

사용자의 질문에 답변할 때, 아래 도구들을 사용하여 정확한 전시물 정보를 찾아주세요.

도구:
{tools}

도구 이름: {tool_names}

사용자 질문: {input}

Thought: 사용자의 질문을 분석하고 필요한 도구를 선택하세요.
Action: 도구 이름을 입력하세요.
Action Input: 도구에 필요한 입력값을 입력하세요.
Observation: 도구의 결과를 확인하세요.
... (필요하면 반복) ...
Thought: 이제 최종 답변을 구성할 수 있습니다.
Final Answer: {answer_instruction}

중요:
- 반드시 도구를 사용하여 전시물 정보를 확인한 후 답변하세요.
- 전시물과 관련 없는 내용을 지어내지 마세요.
- 도구가 없는 정보는 "정보를 찾을 수 없습니다"라고 답변하세요."""
    )
    
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
    
    return agent_executor

def extract_principles_from_exhibits(exhibits, llm):
    """전시물에서 과학원리 추출"""
    if not exhibits:
        return [], ""
    
    exhibit_text = "\n\n".join([ex["content"] for ex in exhibits[:10]])
    
    prompt = f"""다음 전시물들에서 핵심 과학원리와 구체적인 항목(음식, 물건, 재료 등)을 추출해주세요.

전시물 정보:
{exhibit_text}

**응답 형식:**
1. 먼저 과학원리와 구체적인 항목 목록을 쉼표로 구분하여 한 줄로 작성하세요.
   예: 운동과 뇌, 달걀, 브로콜리, 시큼치, 혈류 증가

2. 그 다음 각 원리에 대한 설명을 작성하세요.
   - 원리명: 간단한 설명 (1-2문장)

최대 5-7개의 핵심 원리와 구체적인 항목을 추출하세요."""

    try:
        response = llm.invoke(prompt)
        content = response.content
        
        lines = content.strip().split('\n')
        principles_line = lines[0] if lines else ""
        
        principles = [p.strip() for p in principles_line.split(',') if p.strip()]
        principles = [p.split('.')[-1].strip() if '.' in p else p for p in principles]
        
        return principles, content
    except Exception as e:
        print(f"원리 추출 오류: {e}")
        return [], "원리를 추출할 수 없습니다."

# ============================================================================
# 퀴즈 생성
# ============================================================================

def generate_quiz(zone_name, principle, llm, language="한국어", variation_seed: int = 0, exhibit_detail: str = "", prev_questions: list = None, difficulty: str = "초등", quiz_count: int = 0, user_mode: str = "어린이"):
    """과학원리 기반 4지선다 퀴즈 생성.

    LLM에게는 JSON 형태(question, options, correct_index, explanation)를 받고,
    클라이언트에서 옵션을 **항상** 무작위로 셔플 → 정답 위치 편향(LLM의 1번 선호)을 원천 제거.
    반환: dict {question, options[4], correct_index(0-3), explanation, raw}
    실패 시: {raw: 원문 텍스트} 만 담긴 dict (호출 측에서 markdown으로 폴백 표시)
    """
    import random
    import json as _json

    rng = random.Random(variation_seed if variation_seed else random.randint(1, 10**9))

    # 질문 접근 각도: 콘텐츠 관점
    angles = [
        "일상생활 상황(요리, 스포츠, 날씨, 게임)에 연결해 직관적으로 묻기",
        "자연 현상(동물, 식물, 우주)에서 이 원리를 발견하게 묻기",
        "실제 실험 장면을 짧게 묘사하고 결과를 예측하게 묻기",
        "이 원리가 없다면 어떤 일이 벌어질지 상상하게 묻기",
        "원인을 찾는 형태 — 왜 이런 일이 일어날까?",
        "결과/예측 형태 — 이렇게 하면 어떻게 될까?",
        "두 가지 현상을 비교해 차이점을 묻기",
        "이 원리가 실제로 쓰이는 예시를 고르는 형태",
        "전시물 체험 장면을 시나리오로 만들어 직접 연결하기",
        "역할 전환 — 주인공이 직접 실험하는 장면에서 묻기",
        "숨겨진 관계 찾기 — 겉보기엔 달라 보이지만 같은 원리를 공유하는 것 고르기",
        "오개념 교정 — 흔히 잘못 알고 있는 것을 찾아내게 묻기",
    ]
    # 질문 형식: 답 구조 관점
    q_formats = [
        "정답형: 가장 옳은 것을 하나 고르는 형식",
        "부정형: 보기 중 틀린 것을 하나 고르는 형식('다음 중 틀린 것은?')",
        "빈칸형: 문장의 (  ) 안에 들어갈 말을 고르는 형식",
        "시나리오형: 짧은 상황 설명(2~3문장) 뒤에 질문하는 형식",
    ]
    angle = angles[quiz_count % len(angles)]
    q_format = q_formats[quiz_count % len(q_formats)]

    glossary_rules = _get_ui_glossary_rules(language)

    # 출력 언어 강제 (principle 이 한국어여도 답변 언어를 지정)
    output_lang_instruction = {
        "한국어": "[출력 언어: 한국어] question, options, explanation 모든 텍스트는 반드시 한국어로 작성.",
        "English": "[OUTPUT LANGUAGE: English] question, all 4 options, and explanation MUST be written in English. Do NOT use Korean. Translate any Korean topic into English. ALL TEXT IN ENGLISH ONLY.",
        "日本語": "[出力言語: 日本語] question, options, explanation のすべてを日本語（漢字・かな）で記述。韓国語禁止。トピックが韓国語でも日本語に翻訳すること。",
        "中文": "[输出语言: 简体中文] question, options, explanation 必须全部使用简体中文。禁止使用韩文。即使主题是韩文，也必须翻译成中文。",
    }.get(language, "")

    if difficulty == "유아":
        quality_rules_ko = """
[문제 품질 규칙 — 유치~초등 저학년 (5~8세)]
1) 어휘: 5~8세가 일상에서 쓰는 아주 쉬운 단어만. 한자어·학술어 금지.
2) 문장 길이: 질문 10단어 이내, 선택지 5단어 이내.
3) 보기 형태: 단어 또는 2~3단어 짧은 구. 의성어·의태어 환영.
4) 소재: 동물, 색깔, 음식, 놀이 등 직관적인 일상 소재 활용.
5) 이야기형 질문: "토끼가 언덕에서 굴러 내려와요. 왜 빠르게 굴러갈까요?" 같은 짧은 장면으로 시작.
6) 정답 1개, 오답 3개. 어린이가 쉽게 구별할 수 있을 정도로 명확하게 틀려야 함.
7) 재미 요소: 끝에 "왜 그럴까요? 실험해봐요!" 같은 호기심 자극 한마디 포함.
"""
    else:
        quality_rules_ko = """
[문제 품질 규칙 — 초등 중·고학년 (3~6학년, 9~12세)]
1) 사실 검증: 과학적으로 명백히 참인 정답 1개, 명백히 거짓인 오답 3개. 애매하거나 둘 다 맞을 수 있는 표현 금지.
2) 구체성: "맞다/아니다"처럼 추상적인 선택지 금지. 각 선택지는 명사구 또는 짧은 문장으로 의미가 분명해야 함.
3) 어휘 수준: 초등 3~6학년(9~12세)이 이해할 수 있는 단어. 학술 용어는 풀어서 설명.
4) 일관성: 4개 선택지의 문법 형태/길이를 비슷하게 맞추기 (정답만 길거나 짧으면 안 됨).
5) 함정 주의: 오답은 흔한 오개념이나 비슷한 다른 현상에서 가져오기 (무관한 단어 나열 금지).
6) 질문은 한 가지만 묻기. 이중 부정, 복수 조건 금지.
7) 실제 전시 체험과 연관된 장면을 1개 이상 사용.
"""
    quality_rules_en = """
[Quality rules — MUST follow]
1) Factual: exactly 1 clearly correct answer; 3 clearly wrong distractors. No ambiguous wording.
2) Concrete: each option must be a meaningful phrase, not vague yes/no.
3) Vocabulary for ages 10–12 (upper elementary). Avoid heavy jargon.
4) Consistent length/form across the 4 options.
5) Distractors should be common misconceptions, not random unrelated words.
6) Single, clear question. Avoid double negatives or compound conditions.
7) Tie at least one element to actual exhibit experience.
"""

    # 전시물 세부설명이 있으면 프롬프트에 추가
    detail_section = ""
    if exhibit_detail and str(exhibit_detail).strip() and str(exhibit_detail).strip().lower() != "nan":
        detail_section = f"\n[전시물 상세 설명]\n{exhibit_detail}\n"

    # 이전 문제 회피 섹션
    prev_section_ko = ""
    prev_section_en = ""
    if prev_questions:
        prev_list = "\n".join(f"- {q}" for q in prev_questions[-7:])
        prev_section_ko = f"\n[이전에 생성된 문제 — 반드시 피할 것]\n{prev_list}\n위 문제들과 같거나 유사한 질문, 같은 과학 개념·정답·선택지를 절대 반복하지 마세요. 완전히 다른 측면에서 출제하세요.\n"
        prev_section_en = f"\n[Previously generated questions — MUST avoid]\n{prev_list}\nDo NOT repeat the same question, concept, answer, or similar options. Choose a completely different angle.\n"

    if user_mode == "어린이":
        explanation_tone = (
            "어린이 친구에게 설명하듯 따뜻하고 친절하게 3~4문장으로 작성하세요. "
            "먼저 핵심 과학 용어를 '~란 ~이에요!' 형식으로 정의하고, "
            "왜 이 답이 맞는지 쉽게 설명한 뒤, "
            "헷갈리는 오답 1개를 '~처럼 보이지만 사실은 ~이랍니다!'로 교정하세요. "
            "말투는 '~에요', '~이랍니다' 같은 친근한 존댓말을 쓰고, 이모지 1~2개를 포함하세요."
        )
        explanation_tone_en = (
            "Write 3-4 warm, child-friendly sentences: "
            "① Define the key concept simply ('___ is ___!') "
            "② Explain why the answer is correct in easy words "
            "③ Correct the most tempting wrong choice. "
            "Tone: encouraging, simple vocabulary for ages 6-10. Include 1-2 emojis."
        )
        explanation_tone_ja = (
            "子どもに話しかけるように温かく3〜4文で書いてください: "
            "①「〜とは〜です！」の形で重要な用語を定義 "
            "②なぜその答えが正しいかを簡単に説明 "
            "③間違いやすい選択肢を「〜に見えますが、実は〜なんです！」で訂正。"
            "口調：「〜だよ」「〜だね」など優しい口調。絵文字1〜2個。"
        )
        explanation_tone_zh = (
            "用温暖、亲切的语气为儿童写3-4句话: "
            "①先用'___就是___！'定义核心概念 "
            "②解释为什么答案正确（简单易懂） "
            "③纠正最容易混淆的错误选项。"
            "语气：亲切活泼，适合6-10岁。包含1-2个表情符号。"
        )
    else:
        explanation_tone = (
            "과학적으로 정확하고 간결하게 3~4문장으로 작성:\n"
            "• 핵심 개념 정의 및 정답이 맞는 이유\n"
            "• 관련 원리나 실생활 응용 예시 1가지\n"
            "• 오답 중 가장 혼동하기 쉬운 선택지의 오류 설명\n"
            "말투: 정중하고 명확한 존댓말 (~습니다, ~입니다). 이모지 없음."
        )
        explanation_tone_en = (
            "Write 3-4 precise, informative sentences: "
            "① Define the key concept accurately "
            "② Explain the scientific reasoning behind the correct answer "
            "③ Clarify why the most tempting wrong choice is incorrect. "
            "Tone: clear, adult-appropriate. No emojis."
        )
        explanation_tone_ja = (
            "科学的に正確で簡潔な3〜4文で書いてください: "
            "①重要な概念を正確に定義 "
            "②正解が正しい科学的理由を説明 "
            "③最も間違えやすい選択肢の誤りを指摘。"
            "口調：丁寧で明確な敬語（〜です、〜ます）。絵文字なし。"
        )
        explanation_tone_zh = (
            "用准确、简洁的语气写3-4句话: "
            "①准确定义核心概念 "
            "②说明正确答案背后的科学原理 "
            "③指出最容易混淆的错误选项的问题。"
            "语气：正式、准确。不使用表情符号。"
        )

    language_prompts = {
        "한국어": f"""{output_lang_instruction}

'{zone_name}'의 '{principle}' 주제로 4지선다 퀴즈를 만들어주세요.
{detail_section}
▶ 이번 질문 각도: {angle}
▶ 이번 질문 형식: {q_format}
{prev_section_ko}
[전시물 정보 활용 규칙]
- 제공된 전시물 정보를 최대한 활용하세요.
- 같은 전시물이라도 매번 다른 측면(원인, 결과, 비교, 예시, 응용 등)에서 질문하세요.
- 위에 지정된 각도와 형식을 반드시 따르세요. CSV 정보만 사용하세요.
⛔ **절대 금지**: 위 [전시물 상세 설명]과 전혀 관련 없는 엉뚱한 소재로 문제를 만들지 말 것.
✅ **허용**: 전시물 설명에 등장하는 용어·개념은 교과서 수준의 추가 정의를 해설에 포함해도 됨.

{quality_rules_ko}

[출력 형식 — JSON만, 다른 텍스트 금지]
다음 스키마의 JSON 객체 한 개를 출력하세요. 코드블록(```)도 붙이지 마세요.
{{
  "question": "초등 4~6학년이 이해할 수 있는 질문 (1문장)",
  "options": ["선택지1", "선택지2", "선택지3", "선택지4"],
  "correct_index": 0,
  "explanation": "{explanation_tone}"
}}
- correct_index 는 0~3 정수, options 배열에서 정답 위치.
- options 는 정확히 4개.
- 따옴표/JSON 문법 정확히 지킬 것.""",

        "English": f"""{output_lang_instruction}

Create a 4-choice quiz about '{principle}' from '{zone_name}'.{glossary_rules}
{detail_section}
▶ Question angle this time: {angle}
▶ Question format this time: {q_format}
{prev_section_en}
[Exhibit usage rules]
- Use ALL provided exhibit information.
- Ask from a different aspect each time (cause, result, comparison, example, application).
- Follow the angle and format above strictly. Use ONLY CSV information.
⛔ STRICTLY FORBIDDEN: Do NOT introduce science concepts, experiments, or examples that are NOT found in the exhibit description above. Stay within the provided data only.

{quality_rules_en}

[Output format — JSON only, no other text]
Output a single JSON object with this schema. No code fences.
{{
  "question": "Single, clear question for ages 10-12",
  "options": ["option 1", "option 2", "option 3", "option 4"],
  "correct_index": 0,
  "explanation": "{explanation_tone_en}"
}}
- correct_index is an integer 0-3 indexing the options array.
- Exactly 4 options.
- Strict JSON syntax."""
    }

    # 日本語 / 中文 prompts
    if language == "日本語":
        language_prompts[language] = f"""{output_lang_instruction}

'{zone_name}'の'{principle}'をテーマに、子ども向け4択クイズを作ってください。
{detail_section}
▶ 今回の質問の角度: {angle}
▶ 今回の質問の形式: {q_format}
{prev_section_en}
{quality_rules_en}

[出力形式 — JSONのみ、他のテキスト禁止]
{{
  "question": "10〜12歳の子どもが分かる1文の質問",
  "options": ["選択肢1", "選択肢2", "選択肢3", "選択肢4"],
  "correct_index": 0,
  "explanation": "{explanation_tone_ja}"
}}
"""
    elif language == "中文":
        language_prompts[language] = f"""{output_lang_instruction}

请围绕'{zone_name}'中的'{principle}'，为10-12岁儿童设计一道四选一测验。
{detail_section}
▶ 本次提问角度: {angle}
▶ 本次题型: {q_format}
{prev_section_en}
{quality_rules_en}

[输出格式 — 只输出JSON，不要任何其他文本]
{{
  "question": "10-12岁儿童能理解的一句话提问",
  "options": ["选项1", "选项2", "选项3", "选项4"],
  "correct_index": 0,
  "explanation": "{explanation_tone_zh}"
}}
"""

    prompt = language_prompts.get(language, language_prompts["한국어"])

    def _parse_json_relaxed(s: str):
        if not s:
            return None
        s = s.strip()
        # 코드펜스 제거
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        # 첫 { 부터 마지막 } 까지 추출
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = s[start:end + 1]
        try:
            return _json.loads(candidate)
        except Exception:
            # trailing comma 등 흔한 오류 보정
            try:
                fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                return _json.loads(fixed)
            except Exception:
                return None

    try:
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        data = _parse_json_relaxed(raw)
        if (
            data
            and isinstance(data.get("options"), list)
            and len(data["options"]) == 4
            and isinstance(data.get("correct_index"), int)
            and 0 <= data["correct_index"] <= 3
            and data.get("question")
        ):
            # 클라이언트 셔플 — 정답 위치 편향 제거
            options = [str(o) for o in data["options"]]
            correct_text = options[data["correct_index"]]
            shuffled = list(options)
            # 시스템 random 사용 (시드 비종속) → 정답 위치 진짜 무작위화
            random.SystemRandom().shuffle(shuffled)
            new_idx = shuffled.index(correct_text)
            return {
                "question": str(data["question"]).strip(),
                "options": shuffled,
                "correct_index": new_idx,
                "explanation": str(data.get("explanation", "")).strip(),
                "raw": raw,
            }
        # 폴백: JSON 파싱 실패
        return {"raw": raw}
    except Exception as e:
        print(f"퀴즈 생성 오류: {e}")
        return None

# ============================================================================
# 오디오북 생성
# ============================================================================

def _get_ui_glossary_rules(language_mode: str) -> str:
    glossary = {
        "English": {
            "놀이터": "Zone",
            "전시물": "Exhibit",
            "과학원리": "Science principle",
            "오디오북": "Audiobook",
            "AI놀이터": "AI Zone",
            "행동놀이터": "Activity Zone",
            "생각놀이터": "Thinking Zone",
            "탐구놀이터": "Discovery Zone",
            "관찰놀이터": "Discovery Zone",
            "과학극장": "Science Theater",
            "빛놀이터": "Interactive Theater",
            "어린이교실": "Kids Classroom",
            "천체투영관": "Planetarium",
            "휴게실": "Lounge",
        }
    }
    if language_mode == "한국어":
        return ""
    lang_terms = glossary.get(language_mode, glossary["English"])
    rule_lines = [f"- '{ko}' -> '{lang}'" for ko, lang in lang_terms.items()]
    return (
        "\n\nGLOSSARY (must follow EXACTLY — these are fixed official names, never translate differently):\n"
        + "\n".join(rule_lines)
        + "\n- Use these terms consistently. Do not mix languages.\n"
        + "- CRITICAL: place/zone names above are OFFICIAL and FIXED — do NOT invent or alter them.\n"
    )


def get_random_story_dna(language="한국어"):
    """매 호출마다 완전히 다른 이야기 DNA 생성"""
    import random

    genres = {
        "한국어": [
            {"name": "미스터리 탐정",  "desc": "수수께끼를 푸는 탐정 어드벤처. 단서를 하나씩 모아 진실에 다가간다."},
            {"name": "우정 성장담",    "desc": "혼자서는 못 하지만 함께라면 가능한 이야기. 우정이 과학적 발견을 이끈다."},
            {"name": "구출 작전",      "desc": "소중한 누군가(또는 무언가)를 구하기 위해 모험을 떠난다."},
            {"name": "발명가 이야기",  "desc": "실패를 거듭하며 결국 놀라운 발명을 완성하는 이야기."},
            {"name": "시간 여행",      "desc": "과거 또는 미래로 이동해 과학 원리로 문제를 해결한다."},
            {"name": "작은 영웅",      "desc": "아무도 주목하지 않던 평범한 아이가 세상을 구한다."},
            {"name": "꿈속 모험",      "desc": "잠들었다가 깨어난 꿈속 세계. 현실과 꿈의 경계가 흐릿하다."},
            {"name": "변신 이야기",    "desc": "주인공이 다른 존재로 변해 완전히 다른 시각으로 세상을 본다."},
        ],
        "English": [
            {"name": "Mystery Detective",  "desc": "A detective adventure solving a puzzle, gathering clues one by one to reach the truth."},
            {"name": "Friendship Quest",   "desc": "Impossible alone, but achievable together — friendship drives the scientific discovery."},
            {"name": "Rescue Mission",     "desc": "An adventure to save someone (or something) precious before it's too late."},
            {"name": "Young Inventor",     "desc": "Failing again and again, the hero finally completes a remarkable invention."},
            {"name": "Time Travel",        "desc": "Journeying to the past or future, using science to fix a broken timeline."},
            {"name": "Unlikely Hero",      "desc": "An ordinary kid nobody noticed ends up saving the world with science."},
            {"name": "Dream Adventure",    "desc": "A dream world where the boundary between reality and imagination blurs."},
            {"name": "Transformation",     "desc": "The hero transforms into another being and sees the world from a completely new perspective."},
        ],
        "日本語": [
            {"name": "ミステリー探偵",    "desc": "謎を解く探偵アドベンチャー。手がかりを一つずつ集めて真実に近づいていく。"},
            {"name": "友情の成長物語",    "desc": "一人ではできないけど、一緒ならできる。友情が科学的な発見を導く。"},
            {"name": "救出作戦",          "desc": "大切な誰か（または何か）を救うために冒険に出かける。"},
            {"name": "発明家の物語",      "desc": "何度も失敗しながら、ついに驚くべき発明を完成させる物語。"},
            {"name": "タイムトラベル",    "desc": "過去または未来へ移動し、科学の原理で問題を解決する。"},
            {"name": "ちいさな英雄",      "desc": "誰も注目しなかった普通の子どもが世界を救う。"},
            {"name": "夢の冒険",          "desc": "眠りについた夢の世界。現実と夢の境界がぼんやりしている。"},
            {"name": "変身の物語",        "desc": "主人公が別の存在に変わり、まったく違う視点から世界を見る。"},
        ],
        "中文": [
            {"name": "神秘侦探",    "desc": "解谜侦探冒险。一点一点收集线索，逐渐接近真相。"},
            {"name": "友情成长",    "desc": "一个人做不到，但一起就能做到。友情引领科学发现。"},
            {"name": "救援行动",    "desc": "为了拯救珍贵的某人（或某物）而踏上冒险之旅。"},
            {"name": "小发明家",    "desc": "屡次失败，最终完成惊人发明的故事。"},
            {"name": "时间旅行",    "desc": "穿越过去或未来，用科学原理解决问题。"},
            {"name": "无名英雄",    "desc": "没人注意的普通孩子，用科学拯救了世界。"},
            {"name": "梦境冒险",    "desc": "进入梦中世界，现实与梦境的边界模糊不清。"},
            {"name": "变身故事",    "desc": "主人公变成另一种存在，以全新的视角看待世界。"},
        ],
    }

    conflicts = {
        "한국어": [
            "소중한 것이 사라졌다 (친구, 색깔, 소리, 빛)",
            "세상이 멈춰버렸다 (시간, 움직임, 계절)",
            "누군가 거짓말을 하고 있다 (진실을 밝혀야 한다)",
            "둘 중 하나를 선택해야 한다 (포기할 수 없는 두 가지)",
            "점점 작아지거나 커지고 있다 (몸이 변하고 있다)",
            "혼자 남겨졌다 (동반자를 찾아야 한다)",
            "기억이 사라지고 있다 (원리를 기억해내야만 한다)",
        ],
        "English": [
            "Something precious has vanished (a friend, a color, a sound, the light)",
            "The world has frozen still (time, movement, seasons have stopped)",
            "Someone is lying — the truth must be uncovered",
            "A choice must be made between two things that can't be given up",
            "The hero is slowly shrinking or growing (the body is changing)",
            "Left completely alone — must find the companion again",
            "Memories are fading — the science principle must be remembered to survive",
        ],
        "日本語": [
            "大切なものが消えてしまった（友だち、色、音、光）",
            "世界が止まってしまった（時間、動き、季節）",
            "誰かが嘘をついている（真実を明らかにしなければならない）",
            "二つのうちどちらかを選ばなければならない（どちらもあきらめられない）",
            "だんだん小さくなっている、または大きくなっている（体が変わっている）",
            "一人ぼっちになってしまった（仲間を見つけなければならない）",
            "記憶が消えていく（科学の原理を思い出さなければ助からない）",
        ],
        "中文": [
            "珍贵的东西消失了（朋友、颜色、声音、光）",
            "世界停止了（时间、运动、季节都静止了）",
            "有人在撒谎——必须揭开真相",
            "必须在两件都无法放弃的事情中做出选择",
            "身体越来越小或越来越大（正在发生变化）",
            "被独自留下了——必须找到同伴",
            "记忆正在消失——必须想起科学原理才能得救",
        ],
    }

    world_textures = {
        "한국어": [
            "모든 것이 거꾸로인 세상 (중력, 색깔, 소리가 뒤집혀 있다)",
            "밤에만 보이는 세상 (낮에는 투명해진다)",
            "감정이 날씨로 나타나는 세상 (기쁘면 햇살, 슬프면 비)",
            "크기가 없는 세상 (모든 것이 주인공 마음대로 커지고 작아진다)",
            "소리로 만들어진 세상 (건물도 나무도 모두 음악으로 되어있다)",
            "그림자가 살아있는 세상 (그림자가 따로 움직인다)",
        ],
        "English": [
            "A world where everything is upside down (gravity, colors, and sounds are all reversed)",
            "A world visible only at night (everything turns invisible in daylight)",
            "A world where emotions become weather (joy brings sunshine, sadness brings rain)",
            "A world without fixed size (everything grows or shrinks at the hero's will)",
            "A world made entirely of sound (buildings, trees — everything is music)",
            "A world where shadows are alive (shadows move on their own)",
        ],
        "日本語": [
            "すべてが逆さまの世界（重力、色、音がひっくり返っている）",
            "夜しか見えない世界（昼間は透明になってしまう）",
            "感情が天気になる世界（うれしいと晴れ、悲しいと雨）",
            "大きさのない世界（主人公の気持ちで何でも大きくなったり小さくなったりする）",
            "音でできた世界（建物も木もすべて音楽でできている）",
            "影が生きている世界（影が自分で動く）",
        ],
        "中文": [
            "一切都颠倒的世界（重力、颜色、声音全都反过来了）",
            "只在夜晚才能看见的世界（白天一切都变得透明）",
            "情感变成天气的世界（高兴就晴天，悲伤就下雨）",
            "没有固定大小的世界（一切都随主人公的意念变大变小）",
            "由声音构成的世界（建筑、树木——一切都是音乐）",
            "影子有生命的世界（影子自己会动）",
        ],
    }

    lang = language if language in genres else "한국어"
    chosen_genre    = random.choice(genres[lang])
    chosen_conflict = random.choice(conflicts[lang])
    chosen_texture  = random.choice(world_textures[lang])

    return chosen_genre, chosen_conflict, chosen_texture


def generate_science_story(zone_name, exhibits, principles, language="한국어"):
    """방문한 놀이터 기반 과학동화 생성 (상상력 강화 버전)"""
    import random
    genre, conflict, texture = get_random_story_dna(language)

    # 랜덤 주인공 이름 선택
    protagonist_names = {
        "한국어": ["지우", "서연", "민준", "하은", "도윤", "수아", "예준", "시우"],
        "English": ["Alex", "Emma", "Noah", "Olivia", "Liam", "Sophia", "Lucas", "Mia"],
        "日本語": ["ゆうと", "さくら", "はると", "ひまり", "そうた", "あおい"],
        "中文": ["小明", "小华", "小芳", "小杰", "小美", "小强"]
    }
    protagonist = random.choice(protagonist_names.get(language, protagonist_names["한국어"]))

    # ---- CSV 활용 3단 구조: zone 정체성 + 분위기 재료 5개 + 핵심 아이템 2개(설명포함) ----
    def _short_desc_from_content(text: str, limit: int = 70) -> str:
        """page_content('[zone] title\\nCategory:..\\nContent:..\\nDetails:..')에서 Content 한 줄을 짧게 추출"""
        if not text:
            return ""
        for line in text.splitlines():
            line = line.strip()
            if line.lower().startswith("content:"):
                desc = line.split(":", 1)[1].strip()
                if desc and desc.lower() != "nan":
                    return (desc[:limit] + "…") if len(desc) > limit else desc
        parts = [p.strip() for p in text.splitlines() if p.strip()]
        if len(parts) >= 2:
            return (parts[1][:limit] + "…") if len(parts[1]) > limit else parts[1]
        return ""

    # zone 정체성 한 줄 — exhibit_summary 폴백보다 먼저 계산
    all_titles = []
    for ex in exhibits[:10]:
        t = ex.get("metadata", {}).get("title", "") or ""
        if t:
            all_titles.append(t)
    zone_identity_line = ", ".join(all_titles[:8]) if all_titles else ""

    # 핵심 마법 아이템 2개 (제목 + 짧은 설명 + 세부설명) — 갈등을 해결하는 키
    core_lines = []
    for ex in exhibits[:2]:
        t = ex.get("metadata", {}).get("title", "") or ""
        c = _short_desc_from_content(ex.get("content", ""))
        d = ex.get("metadata", {}).get("detail", "")
        if d and str(d).strip() and str(d).strip().lower() != "nan":
            d_short = (str(d).strip()[:120] + "…") if len(str(d).strip()) > 120 else str(d).strip()
        else:
            d_short = ""
        parts = []
        if c:
            parts.append(f"설명: {c}")
        if d_short:
            parts.append(f"세부: {d_short}")
        if parts:
            core_lines.append(f"- {t} ({'; '.join(parts)})")
        elif t:
            core_lines.append(f"- {t}")
    exhibit_summary = "\n".join(core_lines)
    if not exhibit_summary.strip() and all_titles:
        for _t in all_titles[:2]:
            core_lines.append(f"- {_t}")
        exhibit_summary = "\n".join(core_lines)

    # 분위기 재료 (다음 5개 전시물 title) — 세계관에 자연스럽게 흩뿌릴 풍경/소품
    atmosphere_titles = []
    for ex in exhibits[2:7]:
        t = ex.get("metadata", {}).get("title", "") or ""
        if t:
            atmosphere_titles.append(t)
    atmosphere_summary = ", ".join(atmosphere_titles) if atmosphere_titles else ""

    # ---- 텍스트 패널(title 없는 설명 행) 로드 — 다중 전시관 대응 ----
    _all_csv_rows = []
    try:
        from core import load_zone_rows_from_csv as _load_csv_all
        for _zn in [z.strip() for z in zone_name.split(",")]:
            if _zn:
                _all_csv_rows.extend(_load_csv_all(_zn, include_text_panels=True))
    except Exception:
        pass
    _text_panels = [
        r for r in _all_csv_rows
        if not r.get("title") and (
            (r.get("content") and str(r.get("content")).strip().lower() not in ("", "nan")) or
            (r.get("detail") and str(r.get("detail")).strip().lower() not in ("", "nan"))
        )
    ]

    # ---- CSV content 컬럼 전체 설명 수집 (씨앗 생성 + 원리 보강용) ----
    _content_descs = []
    for ex in exhibits[:8]:
        for line in ex.get("content", "").splitlines():
            line = line.strip()
            if line.lower().startswith("content:"):
                desc = line.split(":", 1)[1].strip()
                if desc and desc.lower() != "nan":
                    _content_descs.append(desc)
                    break
    for r in _text_panels[:4]:
        c = str(r.get("content", "")).strip()
        d = str(r.get("detail", "")).strip()
        panel_text = c if (c and c.lower() != "nan") else (d if (d and d.lower() != "nan") else "")
        if panel_text:
            _content_descs.append(f"[설명판] {panel_text[:150]}")
    principles_descriptions_text = (
        "\n".join(f"- {d}" for d in _content_descs[:8])
        if _content_descs else ", ".join(principles[:3])
    )

    # principles_text: 원리 이름 (아하 순간 명명용) — 설명은 _principles_context 로 별도 제공
    principles_text = principles[0] if principles else (all_titles[0] if all_titles else zone_name)
    _principles_context = (
        f"   (전시관 과학 설명 — LLM 참고용, 이야기 속 직접 인용 금지):\n"
        + principles_descriptions_text
        if principles_descriptions_text else ""
    )

    glossary_rules = _get_ui_glossary_rules(language)

    # ---- 동반자 풀 (매번 다른 단짝) ----
    companion_pool = {
        "한국어": ["작은 로봇 '삐삐'", "은빛 여우 요정", "꼬마 공룡 친구", "말하는 별똥별", "미니 우주비행사 고양이"],
        "English": ["a tiny robot named Beep", "a silver fox spirit", "a small talking dinosaur", "a shooting star that can speak", "a mini astronaut cat"],
        "日本語": ["小さなロボット『ピピ』", "銀色のキツネの妖精", "おしゃべりな子恐竜", "話す流れ星", "ミニ宇宙飛行士のネコ"],
        "中文": ["小机器人『叮叮』", "银色狐狸精灵", "会说话的小恐龙", "会讲话的流星", "迷你宇航员小猫"],
    }
    companion = random.choice(companion_pool.get(language, companion_pool["한국어"]))

    # ---- zone과 어울리는 세계관 매칭 (충돌 방지) ----
    zone_world_map = {
        "한국어": {
            "AI놀이터": ["반짝이는 회로로 가득한 비밀 연구소", "구름 위에 숨은 작은 로봇 마을"],
            "행동놀이터": ["바람이 살아있는 모험의 숲", "거대한 놀이 기구가 움직이는 마법 공원"],
            "생각놀이터": ["수수께끼가 떠다니는 별빛 도서관", "거울로 만든 신비한 탑"],
            "탐구놀이터": ["지하 깊숙이 빛나는 보석 동굴", "낡은 지도로만 갈 수 있는 잊힌 섬"],
            "관찰놀이터": ["커다란 망원경이 서 있는 언덕 위 정원", "작은 생물들이 노래하는 안개 숲"],
            "과학극장": ["무대 뒤편의 비밀 무대 마을", "커튼이 살아 움직이는 환상의 극장"],
            "빛놀이터": ["일곱 빛깔이 흐르는 무지개 궁전", "그림자가 춤추는 빛의 미로"],
            "어린이교실": ["분필이 스스로 그림을 그리는 작은 마법 학교"],
            "천체투영관": ["별과 별 사이를 떠다니는 우주 정거장", "달빛 위에 떠 있는 은하 마을"],
            "휴게실": ["구름 위 포근한 쉼터 정원"],
        },
        "English": {
            "AI놀이터": ["a secret lab full of glowing circuits", "a tiny robot village hidden above the clouds"],
            "행동놀이터": ["an adventure forest where the wind is alive", "a magical park where giant rides move on their own"],
            "생각놀이터": ["a starlit library where riddles float in the air", "a mysterious tower made of mirrors"],
            "탐구놀이터": ["a glittering gem cave deep underground", "a forgotten island reachable only by an old map"],
            "관찰놀이터": ["a hilltop garden with a giant telescope", "a misty forest where tiny creatures sing"],
            "과학극장": ["a secret stage village behind the curtains", "an enchanted theater whose curtains dance"],
            "빛놀이터": ["a rainbow palace flowing with seven colors", "a maze of light where shadows dance"],
            "어린이교실": ["a tiny magic school where the chalk draws by itself"],
            "천체투영관": ["a space station drifting between stars", "a galaxy village floating on moonlight"],
            "휴게실": ["a cozy rest garden above the clouds"],
        },
        "日本語": {
            "AI놀이터": ["きらめく回路でいっぱいの秘密研究所", "雲の上に隠れた小さなロボット村"],
            "행동놀이터": ["風が生きている冒険の森", "巨大な遊具がひとりでに動く魔法の公園"],
            "생각놀이터": ["なぞなぞが空に浮かぶ星明かりの図書館", "鏡でできた不思議な塔"],
            "탐구놀이터": ["深い地下に輝く宝石の洞窟", "古い地図でしか行けない忘れられた島"],
            "관찰놀이터": ["大きな望遠鏡が立つ丘の上の庭", "小さな生き物たちが歌う霧の森"],
            "과학극장": ["舞台裏の秘密の村", "カーテンが踊る幻の劇場"],
            "빛놀이터": ["七色が流れる虹の宮殿", "影が踊る光の迷路"],
            "어린이교실": ["チョークがひとりでに絵を描く小さな魔法学校"],
            "천체투영관": ["星と星の間を漂う宇宙ステーション", "月明かりの上に浮かぶ銀河の村"],
            "휴게실": ["雲の上のあたたかな休息の庭"],
        },
        "中文": {
            "AI놀이터": ["布满闪亮电路的秘密研究所", "藏在云端的小机器人村庄"],
            "행동놀이터": ["风都活着的冒险森林", "巨大游乐设施自动运转的魔法乐园"],
            "생각놀이터": ["谜题漂浮在空中的星光图书馆", "用镜子建成的神秘高塔"],
            "탐구놀이터": ["地下深处闪烁的宝石洞窟", "只能靠古老地图到达的被遗忘之岛"],
            "관찰놀이터": ["立着大望远镜的山丘花园", "小生物歌唱的雾之森林"],
            "과학극장": ["幕后的秘密小镇", "幕布翩翩起舞的梦幻剧场"],
            "빛놀이터": ["流淌着七色光的彩虹宫殿", "影子起舞的光之迷宫"],
            "어린이교실": ["粉笔会自己画画的小魔法学校"],
            "천체투영관": ["漂浮在星辰之间的太空站", "悬于月光之上的银河小镇"],
            "휴게실": ["云端上温暖的休憩花园"],
        },
    }
    fallback_worlds = {
        "한국어": ["구름 위에 숨겨진 하늘 정원", "별과 별 사이를 떠다니는 도서관"],
        "English": ["a hidden sky garden above the clouds", "a library drifting between stars"],
        "日本語": ["雲の上に隠された空の庭園", "星と星の間を漂う図書館"],
        "中文": ["藏在云上的空中花园", "漂浮在星辰之间的图书馆"],
    }
    _zone_map = zone_world_map.get(language, zone_world_map["한국어"])
    _world_candidates = _zone_map.get(zone_name) or fallback_worlds.get(language, fallback_worlds["한국어"])
    world = random.choice(_world_candidates)

    # ---- 1단계: 동화 씨앗 생성 (LLM — 갈등 템플릿 있으면 참고 구조로 주입) ----
    _story_seed = ""
    _conflict_tmpl = _get_conflict_template(zone_name, principles_text)
    _tmpl_section = ""
    if _conflict_tmpl:
        _tmpl_section = (
            f"\n[참고할 갈등 구조 — 그대로 쓰지 말고 창의적으로 변형할 것]\n"
            f"발단 사건 참고: {_conflict_tmpl['갈등']}\n"
            f"실패 장면 참고: {_conflict_tmpl['실패']}\n"
            f"아하 힌트 참고: {_conflict_tmpl['아하']}\n"
            f"핵심 감각 참고: {_conflict_tmpl['감각']}\n"
        )
        print(f"[과학동화 씨앗] 갈등 템플릿 매칭: {principles_text}")
    if principles_descriptions_text or _conflict_tmpl:
        _seed_prompt = f"""너는 초등 어린이 과학동화 편집자야.
아래 전시관 과학 내용을 읽고, 자연스러운 어린이 과학동화(SF 아님, 일상·판타지 배경)에 쓸 '이야기 씨앗'을 정확히 3줄로 만들어줘.

[전시관]: {zone_name}
[핵심 과학 현상]: {principles_text}
[전시관 과학 내용]:
{principles_descriptions_text}{_tmpl_section}
규칙:
- 주인공이 직접 겪는 구체적이고 감각적인 장면으로 쓸 것 (SF 장비·초능력·마법 장치 금지)
- 원리 이름은 절대 쓰지 말 것 — 현상만 감각(소리/촉감/시각)으로 묘사
- 평범한 사물(돌멩이, 물, 나뭇잎, 빛 등)을 이용한 장면으로
- 정확히 아래 형식 3줄로만 출력, 다른 말 없이

발단 사건: 발단 사건: (주인공이 이 과학 원리가 꼭 필요한 현실적인 상황 1문장. 
             크기 변화·마법·물체가 사라지는 현상 절대 금지.
             반드시 참고 갈등 구조를 기반으로 할 것)
실패 장면: (주인공이 이유를 모른 채 해결하려다 실패하는 1문장, 감각 묘사 포함)
아하 힌트: (같은 현상이 반복되어 패턴이 보이는 순간 1문장, 원리명 없이)"""
        try:
            _seed_llm = ChatOpenAI(model="gpt-4o", temperature=0.9)
            _seed_resp = _seed_llm.invoke(_seed_prompt)
            _story_seed = _seed_resp.content.strip()
            print(f"[과학동화 씨앗] LLM 생성\n{_story_seed}")
        except Exception as _se:
            print(f"씨앗 생성 오류: {_se}")

    _seed_inject = {
        "한국어": (
            f"\n▶ ★ 동화 씨앗 (아래 흐름을 이야기에 반드시 녹여낼 것. 그대로 베끼지 말고 장면으로 구체화할 것):\n{_story_seed}\n"
            if _story_seed else ""
        ),
        "English": (
            f"\n▶ ★ Story Seed (weave this arc into the story — do NOT copy verbatim, expand each moment into vivid scenes):\n{_story_seed}\n"
            if _story_seed else ""
        ),
        "日本語": (
            f"\n▶ ★ 物語の種 (この流れを必ず物語に溶け込ませること。そのまま写さず、場面として具体的に展開すること):\n{_story_seed}\n"
            if _story_seed else ""
        ),
        "中文": (
            f"\n▶ ★ 故事种子 (必须将以下流程融入故事中，不要照抄，要展开成具体的场景):\n{_story_seed}\n"
            if _story_seed else ""
        ),
    }

    # 프롬프트 변수 실제 값 로그
    print(f"[과학동화 프롬프트 변수] zone_name={zone_name}, world={world}")
    print(f"[과학동화 프롬프트 변수] exhibit_summary={exhibit_summary[:200]}...")
    print(f"[과학동화 프롬프트 변수] principles_text={principles_text}")
    print(f"[과학동화 프롬프트 변수] atmosphere_summary={atmosphere_summary}")
    print(f"[과학동화 프롬프트 변수] zone_identity_line={zone_identity_line[:200]}...")
    print(f"[과학동화 프롬프트 변수] text_panels={len(_text_panels)}개, content_descs={len(_content_descs)}개")

    language_prompts = {
        "한국어": f"""너는 초등 4~6학년(10~12세) 어린이를 위한 감성적이고 재미있는 과학동화 작가야.

[재료 — 모두 CSV 실제 데이터 기반. 반드시 활용할 것]
※ 이 동화는 실재하는 전시관('{zone_name}')을 모티브로 한다. 아래 재료를 무시하고 무관한 설정을 만들지 말 것.

▶ 이 전시관의 정체성 (전시물 목록 — 분위기를 즉시 파악하라):
   {zone_identity_line}
   (예: 새, 공룡, 암석이 있다면 → 자연 관찰관. 회로/로봇이 있다면 → 미래 연구소.)

▶ 배경 분위기(직접 이름은 쓰지 말고 위 정체성을 살린 무대로 변형): {world}

▶ 주인공: 호기심 많은 어린이 '{protagonist}'
▶ 동반자(주인공과 대화하는 단짝): {companion}
   ※ 단순한 설명 도우미가 아닌 고유한 개성과 역할을 가진 캐릭터로 (예: 겁쟁이지만 냄새를 잘 맡는 여우, 말이 없지만 길을 잘 찾는 거북이).

▶ ★ 핵심 과학 도구(아래 전시물 2개를 과학 도구/비밀 장치로 변형해서만 사용. 다른 도구 발명 금지):
{exhibit_summary}

▶ 분위기 재료(이야기 곳곳에 풍경/소품/등장 생물로 자연스럽게 흩뿌려 등장시킬 것 — 최소 2개 이상 본문에 포함):
   {atmosphere_summary}

▶ 이야기의 갈등을 해결하는 단 하나의 과학 현상: {principles_text}
{_principles_context}{_seed_inject["한국어"]}
[개연성 규칙 — 매우 중요]
1) **간결한 3막 구조 (총 6~8문단) — 과학 현상이 이야기의 굵직한 축**:
   - 1막(2문단): {protagonist}의 평범한 순간 → **'{principles_text}'와 직접 관련된 이상한 사건** 발생 → "왜 이런 일이?"라는 명확한 하나의 목표.
     (갈등과 과학 원리는 시각적으로 연결된 물리적 상황으로 — 예: 무거운 돌이 굴러 내려온다 → 지레를 발견한다.)
   - 2막(3~4문단): [핵심 과학 도구]를 시도 → **현상이 작게 한 번 일어남** (감각 묘사) → 한 번 실패 → 동반자와 함께 같은 현상이 반복되는 걸 관찰하며 **"어? 항상 이렇게 되네?"라는 패턴을 발견**.
   - 3막(1~2문단): **★ 아하 순간**: 주인공이 큰 소리로 깨달음 — "아, 이게 바로 **{principles_text}**(이)구나!" 그 원리를 이용해 위기를 해결 → 1막의 수수께끼도 같은 원리로 설명 → 따뜻한 마무리.
2) **인과 사슬**: 모든 장면은 "~ 때문에 → ~이 일어났다" 순서. 갑자기 새 도구·새 능력 등장 금지.
3) **아이템 제한**: 위에 적힌 [핵심 과학 도구]만으로 위기를 해결. 새로운 도구를 즉석에서 만들지 말 것.
4) **목표·이름 일관성**: 1막의 목표는 끝까지 유지, 주인공 '{protagonist}'와 동반자 이름은 절대 바뀌지 않음.
5) **★ 과학 표현 규칙 (가장 중요) — "흘려들어도 원리가 박히게"**:
   - **1~2막에서는 용어 사용 금지**. 현상만 감각으로 묘사: "밀자 거꾸로 튕겨 나왔어요", "빛이 둥근 물방울을 지나자 무지개로 흩어졌어요".
   - **3막의 '아하 순간'에서 단 한 번** 원리명('{principles_text}')을 큰따옴표 대사로 명명할 것. 이때 한 줄짜리 쉬운 설명 추가 (예: "물건을 밀면 그 물건도 똑같은 힘으로 나를 밀어내는 거였어!").
   - 결말 부근에서 그 원리명을 **한 번 더 짧게 회상**하면서 위기를 해결 (총 명명 횟수: 2~3회).
   - 강의·백과사전 톤은 절대 금지. 동반자도 같이 깨닫는 친구.
   - **Show, Don't Tell**: 과학 원리는 대사로 설명하지 말고 감각(소리/촉감/시각)으로 먼저 보여줄 것 (예: "힘이 세졌어"가 아니라 "쿵! 땅이 울렸어요").
   - **과학 개념명 혼용 금지**: 같은 현상을 이야기 전체에서 하나의 이름으로만 부를 것 (예: '단순기계'와 '지레'를 섞어 쓰지 말고 하나만 선택).
6) **★ 과학 활용 방식 (매우 중요)**:
   - ❌ 금지: 과학 원리가 마법처럼 물리 현상을 일으키는 것 (예: "민첩성 때문에 몸이 작아진다")
   - ✅ 허용: 그 원리를 이미 잘 쓰는 동물·자연을 보고, 주인공이 그 원리를 깨닫는 구조
7) **문체 (초등 4~6학년 톤)**:
   - 의성어·의태어를 최소 3번 사용 (예: 폴짝폴짝, 윙윙, 반짝반짝, 살랑살랑, 또르르).
   - 짧은 문장 위주, 대사 비중 40% 이상.
   - 감각 묘사(소리/빛/냄새/촉감) 2개 이상 포함.
   - **받침 있는 이름 + 은/는**: 반드시 "이름+이+는" 형태로 (도윤이는, 민준이는, 하은이는). "도윤은", "민준은" 금지.
8) **금지 표현**: "놀이터", "전시물", "체험", "박물관" 같은 단어 절대 금지. 완전한 판타지 모험으로.
⛔ **전시관 일탈 금지 (매우 중요)**: 위 [재료] 목록에 없는 전시물·과학 개념·현상을 새로 지어내지 말 것. 반드시 위 재료만 사용. 위 목록에 없는 과학 원리를 절대 설명하지 말 것.
9) **결말**: 따뜻하고 희망적, 마지막 한 줄은 잠자리에 어울리는 다정한 인사.

[출력 형식]
- 첫 줄: 제목 (**굵게**)
- 빈 줄
- 본문: 6~8개 문단, 각 문단 2~3문장
- 총 분량: 약 1000~1400자

[분위기 태그]
동화 본문의 마지막 줄에 반드시 아래 형식으로 분위기 태그를 출력하세요:
MOOD_TAG: [wonder|adventure|mystery|cozy|exciting|melancholy] 중 하나만
""",
        "English": f"""You are a tender, imaginative science story writer for upper elementary children aged 10–12.{glossary_rules}

[Ingredients — all from REAL CSV data. You MUST use them; do not ignore.]
This story is inspired by a real exhibit zone ('{zone_name}'). Do not invent unrelated settings.

▶ Zone identity (full exhibit list — read the vibe at a glance):
   {zone_identity_line}
   (e.g. birds + dinosaurs + rocks → a nature observation hall. Circuits + robots → a future lab.)

▶ Setting atmosphere (don't name it literally; transform it into a stage that REFLECTS the identity above): {world}

▶ Protagonist: a curious child named '{protagonist}'
▶ Companion (talks with the hero, NOT an encyclopedia): {companion}
   ※ Must have a distinct personality and role — NOT just a helper who explains things (e.g., a cowardly fox with a keen nose; a quiet turtle who always finds the way).

▶ ★ Core science tools (use ONLY these two; transform them into science tools — DO NOT invent other tools):
{exhibit_summary}

▶ Atmosphere ingredients (sprinkle these as scenery / creatures / props throughout the story — include at least 2 in the body):
   {atmosphere_summary}

▶ The single natural phenomenon that resolves the conflict: {principles_text}
{_principles_context}{_seed_inject["English"]}
[Coherence Rules — CRITICAL]
1) **Compact 3-act structure (6–8 paragraphs total) — the phenomenon is the BACKBONE of the plot**:
   - Act 1 (2 paragraphs): '{protagonist}'s ordinary moment → a strange event **directly tied to '{principles_text}'** → ONE clear goal ("I must find out why…").
     (The conflict and the science principle must be linked through a visually concrete physical situation — e.g., a heavy boulder rolling down → the hero finds a lever.)
   - Act 2 (3–4 paragraphs): try the science tool → **the phenomenon happens in a small way** (sensory description) → fail once → observe the SAME phenomenon repeating with the companion → "Huh, it always happens this way!" — a clear PATTERN.
   - Act 3 (1–2 paragraphs): **★ Aha moment**: the hero exclaims aloud — "Oh! This is **{principles_text}**!" Use that idea to solve the crisis → the Act-1 mystery is explained by the same idea → warm wrap-up.
2) **Cause-and-effect**: every scene "because of X → Y happened". No sudden new tools or powers.
3) **Item discipline**: only the listed science tools solve the crisis. No improvising new tools.
4) **Goal & name consistency**: Act-1 goal persists; '{protagonist}' and the companion's name NEVER change.
5) **★ Science visibility (most important) — "even a half-listening child must catch it"**:
   - **In Acts 1–2 do NOT use the term**. Show the phenomenon through senses only ("when she pushed it, it bounced back the other way").
   - **At the Act-3 aha moment, name '{principles_text}' EXACTLY ONCE in dialogue**, followed by a one-sentence kid-friendly explanation (e.g., "When you push something, it pushes you back just as hard!").
   - Mention the term ONE more time near the resolution as the hero applies it. (Total namings: 2–3.)
   - Never lecture. The companion discovers WITH the hero, not as a teacher.
   - **Show, Don't Tell**: Never explain the principle through dialogue — show it first through senses (sound, touch, sight). ("The ground boomed" not "the force grew stronger").
   - **No concept-name mixing**: Use only ONE name for the phenomenon throughout (e.g., don't alternate "lever" and "simple machine" — pick one and keep it).
6) **★ How science must work (critical)**:
   - ❌ Forbidden: the science principle directly causing magical physical changes (e.g., "agility makes the body shrink")
   - ✅ Required: the hero observes an animal or natural phenomenon that ALREADY uses the principle → and learns from it
7) **Style (ages 10–12)**:
   - Use at least 3 onomatopoeia / mimetic words (whoosh, sparkle-sparkle, plip-plop, thump-thump).
   - Short sentences, dialogue ≥ 40%.
   - At least 2 sensory details (sound, light, smell, texture).
8) **Forbidden words**: "playground", "exhibit", "field trip", "museum" — write it as a true fantasy adventure.
9) **Ending**: warm, hopeful, final line suitable for bedtime.

[Output format]
- Line 1: **Bold title**
- Blank line
- Body: 6–8 paragraphs, each 2–3 sentences
- Length: about 1000–1400 characters total, child-friendly.

[Mood Tag]
At the very last line of the story, output a mood tag in this exact format:
MOOD_TAG: [wonder|adventure|mystery|cozy|exciting|melancholy] — choose only one
""",
        "日本語": f"""あなたは小学4〜6年生（10〜12歳）向けに、感動的で楽しい科学の物語を書く作家です。{glossary_rules}

[素材 — すべて実在のCSVデータ。必ず活用すること]
この物語は実在の展示館（『{zone_name}』）をモチーフにする。下の素材を無視して無関係な設定を作らない。

▶ 展示館の正体（展示物リスト — 雰囲気をひと目で把握）:
   {zone_identity_line}
   （例：鳥・恐竜・岩なら自然観察館。回路・ロボットなら未来の研究所。）

▶ 舞台の雰囲気（言葉自体は使わず、上の正体を活かした舞台に変形）: {world}

▶ 主人公: 好奇心いっぱいの子ども『{protagonist}』
▶ 相棒（主人公と話す友だち。百科事典ではない）: {companion}
   ※ 単なる説明役ではなく、固有の個性と役割を持つキャラクターとして描くこと（例：臆病だが鼻が利くキツネ、無口だが道を知っているカメ）。

▶ ★ 中心となる科学の道具（下の2点だけを科学の道具に変えて使う。他の道具は作らない）:
{exhibit_summary}

▶ 雰囲気の素材（物語の風景・生き物・小道具として自然に散りばめる — 本文に最低2つ以上登場させる）:
   {atmosphere_summary}

▶ 物語の事件を解く、たったひとつの自然現象: {principles_text}
{_principles_context}{_seed_inject["日本語"]}
[筋の通った物語ルール — 最重要]
1) **コンパクトな3幕構成（全6〜8段落）— 科学現象が物語の太い背骨になる**:
   - 第1幕（2段落）: 『{protagonist}』のふつうの瞬間 → **『{principles_text}』に直接かかわる不思議な出来事** → 「どうして？」というひとつの明確な目的。
     （事件と科学の原理は、視覚的に結びついた物理的な状況で設定すること — 例：重い岩が転がり落ちてくる → てこを発見する。）
   - 第2幕（3〜4段落）: 科学の道具を試す → **現象が小さく一度起きる**（五感で描写） → 一度失敗 → 相棒と一緒に同じ現象が繰り返されるのを観察 → 「あれ？いつもこうなる！」と **パターンに気づく**。
   - 第3幕（1〜2段落）: **★ アハ体験**: 主人公が声をあげて気づく — 「あっ、これって **{principles_text}** だ！」その考えで危機を解決 → 1幕の謎も同じ考えで説明 → あたたかい締めくくり。
2) **因果のつながり**: すべての場面は「〜だから → 〜になった」の順。突然の新しい道具・能力は禁止。
3) **アイテム制限**: 上に挙げた科学の道具だけで危機を解決すること。即興で別の道具を作らない。
4) **目的と名前の一貫性**: 1幕の目的は最後まで保たれ、『{protagonist}』と相棒の名前は最後まで変えない。
5) **★ 科学の見える化（最重要）— 「聞き流しても原理が頭に残るように」**:
   - **第1〜2幕では用語を使わない**。現象だけを五感で描写（「押すと、ぽいんと逆にはねかえった」など）。
   - **第3幕のアハの瞬間でちょうど一度だけ** 用語『{principles_text}』をセリフで名づける。続けて子ども向けの一文説明（例：「ものを押すと、そのものも同じ強さで自分を押しかえすんだ！」）。
   - 結末近くでもう一度だけ、主人公がその用語を使って危機を解く（合計命名2〜3回）。
   - 講義・百科事典口調は厳禁。相棒は先生ではなく、いっしょに発見する友だち。
   - **見せて、語るな**: 科学の原理をセリフで説明せず、まず感覚（音・触感・視覚）で見せること（例：「力が強くなった」ではなく「ドン！と地面が揺れた」）。
   - **科学用語の混用禁止**: 同じ現象を物語全体で一つの名前だけで呼ぶこと（例：「単純機械」と「てこ」を混ぜて使わない——どちらか一方だけ）。
6) **★ 科学の活用の仕方（重要）**:
   - ❌ 禁止: 科学の原理が魔法のように物理現象を引き起こすこと（例：「敏捷性のせいで体が小さくなる」）
   - ✅ 許可: その原理をすでに上手に使っている動物・自然を観察して、主人公が原理を学ぶ構成
7) **文体（小学4〜6年生向け）**:
   - 擬音語・擬態語を3回以上使う（ぴょんぴょん、ぴかぴか、ふわふわ、ころころ、ぽとんなど）。
   - 短い文中心、会話の割合は40%以上。
   - 五感の描写（音・光・におい・感触）を2つ以上入れる。
8) **禁句**: 「遊び場」「展示」「体験」「博物館」などは禁止。本物のファンタジー冒険として書く。
9) **結末**: あたたかく希望的、最後の一行は寝かしつけにふさわしいやさしい言葉。

[出力形式]
- 1行目: **太字のタイトル**
- 空行
- 本文: 6〜8段落、各段落2〜3文
- 分量: 全体で約1000〜1400字

[雰囲気タグ]
物語の最後の行に、必ず以下の形式で雰囲気タグを出力してください:
MOOD_TAG: [wonder|adventure|mystery|cozy|exciting|melancholy] から一つだけ
""",
        "中文": f"""你是一位为6〜8岁儿童写作的温柔而充满想象力的科学故事作家。{glossary_rules}

[素材 — 全部来自真实CSV数据，必须使用]
本童话以真实存在的展馆（『{zone_name}』）为蓝本。不要忽略以下素材去编造无关设定。

▶ 展馆身份（展品清单——一眼看清氛围）:
   {zone_identity_line}
   （例：有鸟、恐龙、岩石 → 自然观察馆。有电路、机器人 → 未来研究所。）

▶ 场景氛围（不直接写词，把上述身份活成舞台）: {world}

▶ 主人公: 好奇心旺盛的孩子『{protagonist}』
▶ 伙伴（与主人公对话的朋友，不是百科全书）: {companion}
   ※ 必须有独特的个性和作用，不只是解说助手（例：胆小但嗅觉灵敏的狐狸；沉默寡言但总能找到路的乌龟）。

▶ ★ 核心科学道具（仅用以下两件展品改写成科学道具，不要发明其他道具）:
{exhibit_summary}

▶ 氛围素材（作为风景／生物／道具散布于故事中——正文里至少出现2个以上）:
   {atmosphere_summary}

▶ 推动并解决故事冲突的唯一自然现象: {principles_text}
{_principles_context}{_seed_inject["中文"]}
[开展规则 — 至关重要]
1) **紧凑的三幕结构（共6〜8段）— 科学现象是故事的主干脊梁**:
   - 第一幕（2段）: 『{protagonist}』的平凡时刻 → **直接与『{principles_text}』相关的奇怪事件** → 一个明确目标（"我要弄清楚为什么…"）。
     （冲突与科学原理必须通过视觉上相连的物理情境来呈现 — 例：一块大石头滚下来 → 主人公发现了杠杆。）
   - 第二幕（3〜4段）: 摆弄科学道具 → **现象小小地发生一次**（用五感描写）→ 失败一次 → 与伙伴一起观察同一现象反复出现 → "咦？怎么每次都这样！"——发现 **规律**。
   - 第三幕（1〜2段）: **★ 顿悟时刻**: 主人公大声领悟——"啊，原来这就是 **{principles_text}**！"用这个原理化解危机 → 第一幕的谜团也用同一个原理解释 → 温馨收尾。
2) **因果链条**: 所有情节按"因为……所以……"顺序推进。不可突然出现新道具或新能力。
3) **道具限制**: 仅用上面列出的科学道具来解决危机，不要临时发明新的道具。
4) **目标与名字一致**: 第一幕设定的目标贯穿到底；『{protagonist}』与伙伴的名字自始至终不变。
5) **★ 让科学"看得见"（最重要）— "就算听漏也能记住原理"**:
   - **第一、二幕中绝不使用术语**，只用五感描写现象（如"她一推，它就反方向弹了回去"）。
   - **第三幕的顿悟瞬间，恰好命名一次** 术语『{principles_text}』（用对话），紧跟一句儿童化解释（例："推一下东西，那东西也会用一样的力气把你推回来！"）。
   - 接近结尾再让主人公简短复述一次该术语来解决危机（合计命名2〜3次）。
   - 严禁讲课口吻或百科全书腔调。伙伴不是老师，是和主人公一起发现的朋友。
   - **展示，不要解说**: 不要用对话解释科学原理，先用感官（声音/触感/视觉）展示它（例：不说"力气变大了"，而是"轰！地面震动了"）。
   - **科学概念名称不混用**: 整个故事中同一现象只用一个名称（例：不能"简单机械"和"杠杆"混用——选一个）。
6) **★ 科学的呈现方式（非常重要）**:
   - ❌ 禁止: 科学原理像魔法一样引发物理变化（例："因为敏捷性，身体变小了"）
   - ✅ 允许: 主人公观察已经善用该原理的动物或自然现象，从而领悟原理的结构
7) **文体（6〜8岁口吻）**:
   - 至少使用3个拟声词或叠词（蹦蹦跳跳、闪闪、咕噜咕噜、扑通、轻飘飘）。
   - 以短句为主，对话占比≥40%。
   - 至少加入2处感官描写（声音、光、气味、触感）。
8) **禁用词**: "游乐场""展品""体验""博物馆"等绝对不写。要写成真正的奇妙冒险。
9) **结尾**: 温暖且充满希望，最后一句是适合睡前读的温柔话语。

[输出格式]
- 第1行: **加粗标题**
- 空行
- 正文: 6〜8段，每段2〜3句
- 全文约1000〜1400字

[氛围标签]
在故事正文的最后一行，必须按以下格式输出氛围标签：
MOOD_TAG: [wonder|adventure|mystery|cozy|exciting|melancholy] 只能选一个
""",
    }

    _dna_header_templates = {
        "한국어": (
            f"[이번 동화의 DNA — 매번 다름]\n"
            f"▶ 장르: {genre['name']} — {genre['desc']}\n"
            f"▶ 위기: {conflict}\n"
            f"▶ 세계관 질감: {texture}\n\n"
            f"위 DNA와 과학 재료를 조합해서, 매번 완전히 다른 이야기를 써주세요. "
            f"장르와 위기가 이야기 구조를 결정하고, 과학 현상은 그 위기를 해결하는 열쇠가 됩니다.\n\n"
        ),
        "English": (
            f"[Story DNA — different every time]\n"
            f"▶ Genre: {genre['name']} — {genre['desc']}\n"
            f"▶ Conflict: {conflict}\n"
            f"▶ World texture: {texture}\n\n"
            f"Combine this DNA with the science ingredients to write a completely different story each time. "
            f"Genre and conflict determine the story structure; the science phenomenon is the key that resolves the crisis.\n\n"
        ),
        "日本語": (
            f"[この物語のDNA — 毎回ちがう]\n"
            f"▶ ジャンル: {genre['name']} — {genre['desc']}\n"
            f"▶ 危機: {conflict}\n"
            f"▶ 世界観の質感: {texture}\n\n"
            f"このDNAと科学の素材を組み合わせて、毎回まったく違う物語を書いてください。"
            f"ジャンルと危機が物語の構造を決め、科学現象はその危機を解決する鍵になります。\n\n"
        ),
        "中文": (
            f"[本次故事的DNA——每次都不同]\n"
            f"▶ 类型: {genre['name']} — {genre['desc']}\n"
            f"▶ 冲突: {conflict}\n"
            f"▶ 世界观质感: {texture}\n\n"
            f"将以上DNA与科学素材结合，每次写出完全不同的故事。"
            f"类型和冲突决定故事结构，科学现象是化解危机的关键。\n\n"
        ),
    }
    dna_header = _dna_header_templates.get(language, _dna_header_templates["한국어"])
    prompt = dna_header + language_prompts.get(language, language_prompts["한국어"])

    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.8)
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"동화 생성 오류: {e}")
        return None


BGM_MAP = {
    "wonder":     ["bgm/magical_wonderland.mp3",  "bgm/happy_happy_background.mp3"],
    "adventure":  ["bgm/epic_adventure_theme.mp3", "bgm/cartoon_funny_music.mp3"],
    "mystery":    ["bgm/magical_christmas.mp3",   "bgm/soft_piano_dream.mp3"],
    "cozy":       ["bgm/happy_happy_background.mp3", "bgm/soft_piano_dream.mp3"],
    "exciting":   ["bgm/energetic_run_play.mp3",   "bgm/cartoon_funny_music.mp3"],
    "melancholy": ["bgm/soft_piano_dream.mp3",    "bgm/magical_christmas.mp3"],
}


def parse_mood_and_bgm(llm_response: str) -> tuple[str, str, str]:
    """응답에서 MOOD_TAG를 파싱하고 스토리+BGM 경로 반환"""
    import random
    story = llm_response
    mood = "wonder"
    bgm_path = random.choice(BGM_MAP["wonder"])

    for line in llm_response.splitlines():
        if line.strip().startswith("MOOD_TAG:"):
            raw_mood = line.split(":", 1)[1].strip().lower()
            # 괄호나 기타 문자 제거
            raw_mood = raw_mood.strip(r"[]—\- ")
            if raw_mood in BGM_MAP:
                mood = raw_mood
                bgm_path = random.choice(BGM_MAP[mood])
            story = llm_response.replace(line, "").strip()
            break

    return story, mood, bgm_path


def text_to_audiobook(story_text, language="한국어", voice_override=None, speed_override=None):
    """텍스트를 오디오북으로 변환 (ElevenLabs)"""

    eleven_key = os.environ.get("ELEVENLABS_API_KEY")
    if (not eleven_key) and hasattr(st, "secrets"):
        eleven_key = _safe_secret_get("ELEVENLABS_API_KEY", "")

    # 언어별 Voice ID 매핑
    voice_id_env_map = {
        "한국어": ["ELEVENLABS_VOICE_ID_KO", "ELEVENLABS_VOICE_ID"],
        "English": ["ELEVENLABS_VOICE_ID_EN", "ELEVENLABS_VOICE_ID"],
        "日本語": ["ELEVENLABS_VOICE_ID_JA", "ELEVENLABS_VOICE_ID"],
        "中文": ["ELEVENLABS_VOICE_ID_ZH", "ELEVENLABS_VOICE_ID"],
    }
    env_keys = voice_id_env_map.get(language, ["ELEVENLABS_VOICE_ID_KO", "ELEVENLABS_VOICE_ID"])
    
    eleven_voice_id = voice_override
    if not eleven_voice_id:
        for env_key in env_keys:
            eleven_voice_id = os.environ.get(env_key)
            if eleven_voice_id:
                print(f"[TTS] Using voice ID from env: {env_key} = {eleven_voice_id}")
                break
    if (not eleven_voice_id) and hasattr(st, "secrets"):
        for env_key in env_keys:
            eleven_voice_id = _safe_secret_get(env_key, "")
            if eleven_voice_id:
                print(f"[TTS] Using voice ID from secrets: {env_key} = {eleven_voice_id}")
                break
    if not eleven_voice_id:
        eleven_voice_id = "21m00Tcm4TlvDq8ikWAM"
        print(f"[TTS] Using default voice ID: {eleven_voice_id}")

    eleven_model_id = os.environ.get("ELEVENLABS_MODEL_ID")
    if (not eleven_model_id) and hasattr(st, "secrets"):
        eleven_model_id = _safe_secret_get("ELEVENLABS_MODEL_ID", "")
    if not eleven_model_id:
        eleven_model_id = "eleven_multilingual_v2"

    if eleven_key:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{eleven_voice_id}"
        headers = {
            "xi-api-key": eleven_key,
            "accept": "audio/mpeg",
            "content-type": "application/json",
        }

        stability = 0.45
        similarity_boost = 0.75
        style = 0.35
        if isinstance(speed_override, (int, float)):
            stability = max(0.1, min(0.9, 0.65 - (float(speed_override) - 1.0) * 0.2))

        payload = {
            "text": story_text,
            "model_id": eleven_model_id,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
                "style": style,
                "use_speaker_boost": True,
            },
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=90)
            if resp.status_code == 200 and resp.content:
                return resp.content
            print(f"ElevenLabs TTS 오류: status={resp.status_code}, body={resp.text[:200]}")
            return None
        except Exception as e:
            print(f"ElevenLabs TTS 호출 오류: {e}")
            return None

    return None

# ============================================================================
# Streamlit UI
# ============================================================================

@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def _backtranslate_to_korean_cached(text: str, source_language: str) -> str:
    if not text or source_language == "한국어":
        return ""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = (
        "You are a precise translator. Translate the following UI text into Korean. "
        "Keep it concise and natural. Do not add extra explanations.\n\n"
        f"Source language: {source_language}\n"
        f"Text: {text}"
    )
    try:
        resp = llm.invoke(prompt)
        return (resp.content or "").strip()
    except Exception:
        return ""


def render_post_visit_learning(
    vector_db,
    language_mode="한국어",
    debug_show_korean: bool = False,
    debug_backtranslate: bool = False,
    user_mode: str = "성인",
):
    """사후 학습 시스템 메인 UI"""

    def _display_zone_name(zone: str) -> str:
        if language_mode == "한국어":
            return zone
        # 공식 영어 명칭 (고정 — LLM 번역 금지)
        official = {
            "AI놀이터": "AI Zone",
            "행동놀이터": "Activity Zone",
            "생각놀이터": "Thinking Zone",
            "탐구놀이터": "Discovery Zone",
            "관찰놀이터": "Discovery Zone",
            "과학극장": "Science Theater",
            "빛놀이터": "Interactive Theater",
            "어린이교실": "Kids Classroom",
            "천체투영관": "Planetarium",
            "휴게실": "Lounge",
        }
        if zone in official:
            return official[zone]
        return zone.replace("놀이터", "ZONE")
    
    texts = {
        "한국어": {
            "title": "🥰 또만나 놀이터",
            "subtitle": "다시 만나 반가워요! 즐거웠던 놀이터에서의 추억을 함께 나누어 보아요!",
            "floor1": "1층 놀이터",
            "floor2": "2층 놀이터",
            "tab_quiz": "퀴즈타임 & 궁금해요",
            "tab_story": "과학동화",
            "tab1": "퀴즈/질문",
            "tab2": "과학동화",
            "select_zone": "체험한 놀이터를 선택하세요",
            "no_data": "(준비 중)",
            "generating": "과학원리 분석 중...",
            "quiz_mode": "퀴즈 모드",
            "chat_mode": "질문 모드",
            "select_principle": "퀴즈 주제 선택",
            "make_quiz": "퀴즈 생성",
            "quiz_generating": "퀴즈 생성 중...",
            "ask_question": "질문하기",
            "question_prompt": "에 대해 궁금한 점을 물어보세요",
            "answer_prefix": "답변",
            "pick_zone_hint": "체험한 놀이터를 선택해주세요!",
            "exhibits_not_found": "의 전시물 정보를 찾을 수 없습니다.",
            "answer_error": "답변 생성 중 오류가 발생했습니다. 다시 시도해주세요.",
            "principles_not_found": "과학원리를 추출할 수 없습니다.",
            "csv_not_found": "CSV 전시물 정보를 찾을 수 없습니다.",
            "expander_parent": "보호자용: 전시물 전체보기",
            "story_intro": "오늘 체험한 놀이터를 바탕으로 나만의 과학동화를 만들어보세요!",
            "story_select_heading": "### 동화에 포함할 놀이터 선택",
            "story_generated": "### 📖 생성된 동화",
            "to_audiobook": "🎧 오디오북으로 변환",
            "audiobook_download": "💾 오디오북 다운로드",
            "story_fail": "동화 생성에 실패했습니다.",
            "audiobook_fail": "오디오북 생성에 실패했습니다.",
            "generate_story": "과학동화 만들기",
            "story_generating": "동화 생성 중...",
            "audiobook_generating": "오디오북 생성 중..."
        },
        "English": {
            "title": "🥰 Again Zone",
            "subtitle": "Select the zones you visited and review the science!",
            "floor1": "1st Floor",
            "floor2": "2nd Floor",
            "tab_quiz": "Quiz time",
            "tab_question": "I'm curious!",
            "tab_story": "Science story",
            "select_zone": "Select visited zones",
            "no_data": "(Coming soon)",
            "generating": "Analyzing principles...",
            "quiz_mode": "Quiz Mode",
            "chat_mode": "Q&A Mode",
            "select_principle": "Choose a quiz topic",
            "make_quiz": "Generate quiz",
            "quiz_generating": "Generating quiz...",
            "ask_question": "Ask",
            "question_prompt": ": ask what you're curious about",
            "answer_prefix": "Answer",
            "pick_zone_hint": "Please select the zones you visited!",
            "exhibits_not_found": ": exhibit information not found.",
            "answer_error": "An error occurred while generating the answer. Please try again.",
            "principles_not_found": "Unable to extract science principles.",
            "csv_not_found": "CSV exhibit information not found.",
            "expander_parent": "For parents: View all exhibits",
            "story_intro": "Create your own science story based on the zones you visited today!",
            "story_select_heading": "### Select zones to include in the story",
            "story_generated": "### 📖 Generated story",
            "to_audiobook": "🎧 Convert to audiobook",
            "audiobook_download": "💾 Download audiobook",
            "story_fail": "Failed to generate the story.",
            "audiobook_fail": "Failed to generate the audiobook.",
            "generate_story": "Create Story",
            "story_generating": "Generating story...",
            "audiobook_generating": "Creating audiobook..."
        },
        "日本語": {
            "title": "🥰 またねゾーン",
            "subtitle": "体験したゾーンを選んで、科学をふりかえってみよう！",
            "floor1": "1階",
            "floor2": "2階",
            "tab_quiz": "クイズタイム",
            "tab_question": "ききたい！",
            "tab_story": "かがくどうわ",
            "select_zone": "体験したゾーンを選んでください",
            "no_data": "(準備中)",
            "generating": "科学のポイントを分析中...",
            "quiz_mode": "クイズ",
            "chat_mode": "しつもん",
            "select_principle": "クイズのテーマを選択",
            "make_quiz": "クイズを作る",
            "quiz_generating": "クイズ作成中...",
            "ask_question": "質問する",
            "question_prompt": "について、気になることを聞いてみよう",
            "answer_prefix": "答え",
            "pick_zone_hint": "体験したゾーンを選んでください！",
            "exhibits_not_found": "の展示情報が見つかりませんでした。",
            "answer_error": "回答の生成中にエラーが発生しました。もう一度お試しください。",
            "principles_not_found": "科学のポイントを抽出できませんでした。",
            "csv_not_found": "CSVの展示情報が見つかりませんでした。",
            "expander_parent": "保護者向け：展示一覧を見る",
            "story_intro": "今日体験したゾーンをもとに、自分だけの科学どうわを作ってみよう！",
            "story_select_heading": "### どうわに入れるゾーンを選ぶ",
            "story_generated": "### 📖 作成したどうわ",
            "to_audiobook": "🎧 オーディオブックにする",
            "audiobook_download": "💾 オーディオブックを保存",
            "story_fail": "どうわの作成に失敗しました。",
            "audiobook_fail": "オーディオブックの作成に失敗しました。",
            "generate_story": "どうわをつくる",
            "story_generating": "どうわを作成中...",
            "audiobook_generating": "オーディオブック作成中..."
        },
        "中文": {
            "title": "🥰 再次乐园",
            "subtitle": "选择你体验过的区域，一起回顾科学吧！",
            "floor1": "1层",
            "floor2": "2层",
            "tab_quiz": "测验时间",
            "tab_question": "我很好奇！",
            "tab_story": "科学故事",
            "select_zone": "请选择体验过的区域",
            "no_data": "(准备中)",
            "generating": "正在分析科学要点...",
            "quiz_mode": "测验",
            "chat_mode": "问答",
            "select_principle": "选择测验主题",
            "make_quiz": "生成测验",
            "quiz_generating": "正在生成测验...",
            "ask_question": "提问",
            "question_prompt": "：请输入你想了解的问题",
            "answer_prefix": "回答",
            "pick_zone_hint": "请选择你体验过的区域！",
            "exhibits_not_found": "：未找到展品信息。",
            "answer_error": "回答生成时出错了，请重试。",
            "principles_not_found": "无法提取科学要点。",
            "csv_not_found": "未找到CSV展品信息。",
            "expander_parent": "给家长：查看全部展品",
            "story_intro": "根据你今天体验的区域，创作属于你的科学故事吧！",
            "story_select_heading": "### 选择要写进故事的区域",
            "story_generated": "### 📖 生成的故事",
            "to_audiobook": "🎧 转为有声书",
            "audiobook_download": "💾 下载有声书",
            "story_fail": "故事生成失败。",
            "audiobook_fail": "有声书生成失败。",
            "generate_story": "生成故事",
            "story_generating": "正在生成故事...",
            "audiobook_generating": "正在生成有声书..."
        }
    }
    
    text = texts.get(language_mode, texts["한국어"])

    st.subheader(text["title"])
    st.markdown(text["subtitle"])

    # Load CSV data once with session state persistence — CSV 변경 시 자동 재로드
    _fp = _csv_fingerprint()
    if "all_zone_rows" not in st.session_state or st.session_state.get("_csv_fp") != _fp:
        st.session_state.all_zone_rows = _preload_all_zone_csv_rows()
        st.session_state["_csv_fp"] = _fp
    all_zone_rows = st.session_state.all_zone_rows

    if "learning_sub_tab" not in st.session_state:
        st.session_state.learning_sub_tab = "quiz_question"

    sub_cols = st.columns(2)
    with sub_cols[0]:
        tq_type = "primary" if st.session_state.learning_sub_tab == "quiz_question" else "secondary"
        if st.button(text["tab_quiz"], key="btn_sub_quiz", use_container_width=True, type=tq_type):
            st.session_state.learning_sub_tab = "quiz_question"
            st.rerun()
    with sub_cols[1]:
        ts_type = "primary" if st.session_state.learning_sub_tab == "story" else "secondary"
        if st.button(text["tab_story"], key="btn_sub_story", use_container_width=True, type=ts_type):
            st.session_state.learning_sub_tab = "story"
            st.rerun()

    def _render_zone_selector(key_prefix: str, whitelist: set | None = None):
        st.markdown(f"#### {text['select_zone']}")

        selected = []

        st.markdown(f"##### {text['floor1']}")
        col1, col2 = st.columns(2)

        floor1_zones = [
            z for z, info in ZONE_INFO.items()
            if info["floor"] == "1층" and (whitelist is None or z in whitelist)
        ]
        for i, zone in enumerate(floor1_zones):
            col = col1 if i % 2 == 0 else col2
            with col:
                disabled = not ZONE_INFO[zone]["has_data"]
                zone_disp = _display_zone_name(zone)
                label = f"{zone_disp} {text['no_data']}" if disabled else zone_disp
                if st.checkbox(label, key=f"{key_prefix}_zone_{zone}", disabled=disabled):
                    selected.append(zone)

        st.markdown(f"##### {text['floor2']}")
        col3, col4 = st.columns(2)

        floor2_zones = [
            z for z, info in ZONE_INFO.items()
            if info["floor"] == "2층" and (whitelist is None or z in whitelist)
        ]
        for i, zone in enumerate(floor2_zones):
            col = col3 if i % 2 == 0 else col4
            with col:
                disabled = not ZONE_INFO[zone]["has_data"]
                zone_disp = _display_zone_name(zone)
                label = f"{zone_disp} {text['no_data']}" if disabled else zone_disp
                if st.checkbox(label, key=f"{key_prefix}_zone_{zone}", disabled=disabled):
                    selected.append(zone)

        return selected

    def _render_zone_header(zone: str, zone_rows, mode: str = "exhibits", llm=None):
        st.markdown(f"#### 🎯 {_display_zone_name(zone)}")
        exhibit_label = {
            "한국어": f"전시물 {len(zone_rows)}개",
            "English": f"{len(zone_rows)} exhibits",
            "日本語": f"展示 {len(zone_rows)}件",
            "中文": f"展品 {len(zone_rows)}件",
        }.get(language_mode, f"{len(zone_rows)} exhibits")
        st.caption(exhibit_label)
        keyword_pairs = _get_zone_keywords(zone, zone_rows, language_mode)
        selected_kw, selected_disp = _render_keyword_tags(
            zone, keyword_pairs, zone_rows, language_mode=language_mode, mode=mode, llm=llm
        )
        with st.expander(text["expander_parent"], expanded=False):
            if zone_rows:
                st.dataframe(zone_rows, use_container_width=True, hide_index=True)
            else:
                st.info(text["csv_not_found"])
        return selected_kw, selected_disp

    if st.session_state.learning_sub_tab == "quiz_question":
        # 스탬프 진행 현황
        try:
            earned = set(st.session_state.get("science_stamps", []))
            all_zone_names = list(ZONE_INFO.keys())
            if earned:
                stamp_title = {
                    "한국어": f"🏅 과학 스탬프 모아요! {len(earned)}/{len(all_zone_names)}",
                    "English": f"🏅 Science Stamps! {len(earned)}/{len(all_zone_names)}",
                    "日本語": f"🏅 スタンプ集め! {len(earned)}/{len(all_zone_names)}",
                    "中文": f"🏅 集科学印章! {len(earned)}/{len(all_zone_names)}",
                }.get(language_mode, f"🏅 Science Stamps! {len(earned)}/{len(all_zone_names)}")
                st.markdown(stamp_title)
                stamp_row = " ".join(
                    f"⭐ {_display_zone_name(z)}" if z in earned else f"○ {_display_zone_name(z)}"
                    for z in all_zone_names
                )
                st.caption(stamp_row)
                st.progress(len(earned) / len(all_zone_names))
                if len(earned) >= len(all_zone_names):
                    congrats = {
                        "한국어": "🎉 모든 놀이터 탐험 완료! 너는 진짜 과학 탐험가야! 🔬",
                        "English": "🎉 All zones explored! You're a true Science Explorer! 🔬",
                        "日本語": "🎉 全ゾーン制覇！きみは本物の科学探検家だ！ 🔬",
                        "中文": "🎉 全部区域探索完成！你是真正的科学探险家！🔬",
                    }.get(language_mode, "🎉 All zones explored!")
                    st.success(congrats)
                st.markdown("---")
        except Exception as _stamp_err:
            print(f"[STAMP] render error: {_stamp_err}")

        selected_zones = _render_zone_selector("quiz_question")

        if selected_zones:
            st.markdown("---")
            llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
            for zone in selected_zones:
                zone_rows = all_zone_rows.get(zone, [])
                
                # CSV 데이터가 없으면 직접 다시 로드
                if not zone_rows:
                    try:
                        from core import load_zone_rows_from_csv
                        zone_rows = load_zone_rows_from_csv(zone)
                        all_zone_rows[zone] = zone_rows
                    except Exception as e:
                        print(f"CSV 직접 로드 오류: {e}")
                        zone_rows = []
                
                selected_kw, selected_disp = _render_zone_header(zone, zone_rows, mode="quiz", llm=llm)

                if selected_kw:
                    # 퀴즈와 질문 선택 버튼
                    quiz_button_label = {
                        "한국어": "🎯 퀴즈타임!",
                        "English": "🎯 Quiz Time!",
                        "日本語": "🎯 クイズタイム!",
                        "中文": "🎯 测验时间!",
                    }.get(language_mode, "🎯 Quiz Time!")
                    
                    question_button_label = {
                        "한국어": "❓ 궁금해요!",
                        "English": "❓ I'm curious!",
                        "日本語": "❓ 気になる!",
                        "中文": "❓ 我很好奇!",
                    }.get(language_mode, "❓ I'm curious!")
                    
                    current_mode = st.session_state.get(f"mode_{zone}_{selected_kw}", "quiz")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        quiz_type = "primary" if current_mode == "quiz" else "secondary"
                        if st.button(quiz_button_label, key=f"btn_quiz_mode_{zone}_{selected_kw}", type=quiz_type, use_container_width=True):
                            st.session_state[f"mode_{zone}_{selected_kw}"] = "quiz"
                            st.rerun()
                    with col2:
                        question_type = "primary" if current_mode == "question" else "secondary"
                        if st.button(question_button_label, key=f"btn_question_mode_{zone}_{selected_kw}", type=question_type, use_container_width=True):
                            st.session_state[f"mode_{zone}_{selected_kw}"] = "question"
                            st.rerun()
                    
                    if current_mode == "quiz":
                        seed_key = f"quiz_seed_{zone}_{selected_kw}"
                        quiz_cache_key = f"quiz_cache_{zone}_{selected_kw}"
                        if seed_key not in st.session_state:
                            import random as _rnd
                            st.session_state[seed_key] = _rnd.randint(1, 10**9)

                        # 천체투영관: 영상 제목을 키워드로 받아 영상 내용을 원리로 변환
                        quiz_principle = selected_kw
                        if zone == "천체투영관":
                            try:
                                from core import PLANETARIUM_VIDEO_INFO
                                info = PLANETARIUM_VIDEO_INFO.get(selected_kw)
                                if info:
                                    vid_row = next(
                                        (r for r in zone_rows if r.get("title") == selected_kw),
                                        None,
                                    )
                                    desc = vid_row.get("content", "") if vid_row else ""
                                    quiz_principle = (
                                        f"천체투영관 상영 영상 '{selected_kw}'에서 배우는 내용\n"
                                        f"줄거리: {desc}\n"
                                        f"학습 주제: {info.get('themes', '')}\n"
                                    )
                            except Exception as e:
                                print(f"천투 영상 컨텍스트 조회 실패: {e}")

                        # 선택된 키워드(전시물)의 세부설명 찾기 — title 매칭 OR 그룹명(keyword_flag) 매칭
                        quiz_detail = ""
                        _selected_category = ""
                        _all_parts = []
                        _total_len = 0
                        _MAX_DETAIL = 2000
                        for r in zone_rows:
                            _match = (r.get("title") == selected_kw) or (r.get("keyword_flag", "").strip() == selected_kw)
                            if not _match:
                                continue
                            if not _selected_category:
                                _selected_category = r.get("category", "")
                            _t = str(r.get("title", "") or "").strip()
                            _c = str(r.get("content", "") or "").strip()
                            _d = str(r.get("detail", "") or "").strip()
                            _label = f"[{_t}]" if _t and _t != selected_kw else "[전시물 설명]"
                            if _c and _c.lower() != "nan" and _total_len < _MAX_DETAIL:
                                _all_parts.append(f"{_label} {_c}")
                                _total_len += len(_c)
                            if _d and _d.lower() != "nan" and _total_len < _MAX_DETAIL:
                                _all_parts.append(f"[체험 방법] {_d}")
                                _total_len += len(_d)
                        quiz_detail = "\n".join(_all_parts)
                        _matched_flags = [r.get("keyword_flag","") for r in zone_rows if (r.get("title")==selected_kw or r.get("keyword_flag","").strip()==selected_kw)]
                        print(f"[QUIZ_DETAIL DEBUG] zone={zone}, kw={selected_kw!r}, matched_rows={len(_matched_flags)}, detail_len={len(quiz_detail)}, flags={_matched_flags[:5]}")

                        # 데이터 부실(200자 미만) 시 같은 분류(category) 이웃 전시물로 자동 보충
                        if len(quiz_detail) < 200 and _selected_category:
                            _extra_parts = []
                            _extra_len = 0
                            for r in zone_rows:
                                if r.get("title") == selected_kw:
                                    continue
                                if r.get("category") != _selected_category:
                                    continue
                                _t = str(r.get("title", "") or "").strip()
                                _ec = str(r.get("content", "") or "").strip()
                                _ed = str(r.get("detail", "") or "").strip()
                                _added = []
                                if _ec and _ec.lower() != "nan":
                                    _added.append(f"[관련: {_t}] {_ec}")
                                if _ed and _ed.lower() != "nan":
                                    _added.append(f"[관련 체험: {_t}] {_ed}")
                                for line in _added:
                                    if _extra_len + len(line) > 500:
                                        break
                                    _extra_parts.append(line)
                                    _extra_len += len(line)
                                if _extra_len >= 500:
                                    break
                            if _extra_parts:
                                quiz_detail = (quiz_detail + "\n" if quiz_detail else "") + "\n".join(_extra_parts)

                        # 이전 문제 이력 키 / 퀴즈 카운터
                        prev_q_key = f"quiz_prev_qs_{zone}_{selected_kw}"
                        count_key = f"quiz_count_{zone}_{selected_kw}"
                        if prev_q_key not in st.session_state:
                            st.session_state[prev_q_key] = []
                        if count_key not in st.session_state:
                            st.session_state[count_key] = 0

                        # 난이도 선택 — 항상 표시 (퀴즈 생성 후에도 변경 가능)
                        _diff_labels = {
                            "한국어": {"유아": "유치~초등 저학년", "초등": "초등 중·고학년 (3~6학년)"},
                            "English": {"유아": "Preschool ~ Lower Elem.", "초등": "Upper Elem. (Gr.3-6)"},
                            "日本語": {"유아": "幼児〜小低学년", "초등": "小中〜高学年（3〜6年）"},
                            "中文": {"유아": "幼儿~小学低年级", "초등": "小学中高年级（3-6年级）"},
                        }.get(language_mode, {"유아": "유치~초등 저학년", "초등": "초등 중·고학년"})
                        _diff_title = {"한국어": "난이도", "English": "Difficulty",
                                       "日本語": "レベル", "中文": "难度"}.get(language_mode, "난이도")
                        quiz_difficulty = st.radio(
                            _diff_title,
                            options=["유아", "초등"],
                            format_func=lambda x: _diff_labels[x],
                            horizontal=True,
                            key=f"diff_{zone}_{selected_kw}",
                        )

                        if quiz_cache_key not in st.session_state:
                            if st.button(text["make_quiz"], key=f"btn_make_quiz_{zone}_{selected_kw}"):
                                _queue_ga_event("quiz_generated", {"zone": zone, "language": language_mode})
                                with st.spinner(text["quiz_generating"]):
                                    _qcount = st.session_state.get(count_key, 0)
                                    quiz = generate_quiz(
                                        zone, selected_kw, llm, language_mode,
                                        variation_seed=st.session_state[seed_key],
                                        exhibit_detail=quiz_detail,
                                        prev_questions=st.session_state[prev_q_key],
                                        difficulty=quiz_difficulty,
                                        quiz_count=_qcount,
                                        user_mode=user_mode,
                                    )
                                    st.session_state[quiz_cache_key] = quiz or {}
                                    if quiz and quiz.get("question"):
                                        prev = st.session_state[prev_q_key]
                                        prev.append(quiz["question"])
                                        st.session_state[prev_q_key] = prev[-10:]
                                        st.session_state[count_key] = _qcount + 1
                                st.rerun()

                        if quiz_cache_key in st.session_state:
                            quiz_obj = st.session_state[quiz_cache_key]
                            if quiz_obj and isinstance(quiz_obj, dict) and quiz_obj.get("question"):
                                _render_quiz_card(zone, selected_kw, quiz_obj, language_mode)

                                new_quiz_label = {
                                    "한국어": "🔄 다른 문제 만들기",
                                    "English": "🔄 Generate another question",
                                    "日本語": "🔄 別の問題をつくる",
                                    "中文": "🔄 换一道题",
                                }.get(language_mode, "🔄 Generate another question")
                                if st.button(new_quiz_label, key=f"quiz_refresh_{zone}_{selected_kw}"):
                                    import random as _rnd
                                    st.session_state[seed_key] = _rnd.randint(1, 10**9)
                                    st.session_state.pop(quiz_cache_key, None)
                                    for k in list(st.session_state.keys()):
                                        if k.startswith(f"quiz_reveal_{zone}_{selected_kw}") or \
                                           k.startswith(f"quiz_audio_{zone}_{selected_kw}"):
                                            st.session_state.pop(k, None)
                                    _queue_ga_event("quiz_generated", {"zone": zone, "language": language_mode})
                                    with st.spinner(text["quiz_generating"]):
                                        _qcount = st.session_state.get(count_key, 0)
                                        _cur_diff = st.session_state.get(f"diff_{zone}_{selected_kw}", quiz_difficulty)
                                        quiz = generate_quiz(
                                            zone, selected_kw, llm, language_mode,
                                            variation_seed=st.session_state[seed_key],
                                            exhibit_detail=quiz_detail,
                                            prev_questions=st.session_state.get(prev_q_key, []),
                                            difficulty=_cur_diff,
                                            quiz_count=_qcount,
                                            user_mode=user_mode,
                                        )
                                        st.session_state[quiz_cache_key] = quiz or {}
                                        if quiz and quiz.get("question"):
                                            prev = st.session_state.get(prev_q_key, [])
                                            prev.append(quiz["question"])
                                            st.session_state[prev_q_key] = prev[-10:]
                                            st.session_state[count_key] = _qcount + 1
                                    st.rerun()
                            else:
                                quiz_fail_msg = {
                                    "한국어": "퀴즈 생성에 실패했습니다.",
                                    "English": "Failed to generate quiz.",
                                    "日本語": "クイズの生成に失敗しました。",
                                    "中文": "测验生成失败。",
                                }.get(language_mode, "퀴즈 생성에 실패했습니다.")
                                st.warning(quiz_fail_msg)
                                if st.button(f"🔄 {text['make_quiz']}", key=f"btn_retry_quiz_{zone}_{selected_kw}"):
                                    st.session_state.pop(quiz_cache_key, None)
                                    st.rerun()
                    
                    elif current_mode == "question":
                        # 질문 모드
                        question_mode_label = {
                            "한국어": "#### ❓ 질문하기",
                            "English": "#### ❓ Ask a question",
                            "日本語": "#### ❓ 質問する",
                            "中文": "#### ❓ 提问",
                        }.get(language_mode, "#### ❓ Ask a question")
                        st.markdown(question_mode_label)
                        question_placeholder = {
                            "한국어": f"{_display_zone_name(zone)}의 {selected_disp}에 대해 질문하세요",
                            "English": f"Ask a question about {selected_disp} in {_display_zone_name(zone)}",
                            "日本語": f"{_display_zone_name(zone)}の{selected_disp}について質問してください",
                            "中文": f"请询问关于{_display_zone_name(zone)}的{selected_disp}的问题",
                        }.get(language_mode, f"Ask a question about {selected_disp} in {_display_zone_name(zone)}")
                        user_question = st.text_input(
                            question_placeholder,
                            key=f"question_input_{zone}_{selected_kw}"
                        )

                        if user_question:
                            _queue_ga_event("question_asked", {"zone": zone, "language": language_mode})
                            
                            # 답변 캐시 키
                            answer_cache_key = f"answer_cache_{zone}_{selected_kw}_{hash(user_question)}"
                            
                            # 답변이 이미 캐시되어 있으면 사용
                            if answer_cache_key not in st.session_state:
                                try:
                                    # ReAct 에이전트 생성
                                    tools = _get_question_tools(zone, zone_rows, vector_db)
                                    agent_executor = _create_question_agent(llm, tools, language_mode, user_mode)
                                    
                                    # ReAct 에이전트 실행
                                    with st.spinner(text.get("answer_generating", "답변 생성 중...")):
                                        result = agent_executor.invoke({"input": user_question})
                                        answer_text = result.get("output", "")
                                        st.session_state[answer_cache_key] = answer_text
                                        st.markdown(f"**{text['answer_prefix']}:** {answer_text}")
                                except Exception as e:
                                    print(f"학습 질문 ReAct 답변 오류: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    st.error(text.get("answer_error", "답변 생성 중 오류가 발생했습니다. 다시 시도해주세요."))
                            else:
                                # 캐시된 답변 사용
                                answer_text = st.session_state[answer_cache_key]
                                st.markdown(f"**{text['answer_prefix']}:** {answer_text}")
                            
                            # 음성 듣기 버튼
                            listen_answer_label = {
                                "한국어": "🔊 답변 듣기",
                                "English": "🔊 Listen to answer",
                                "日本語": "🔊 答えを聞く",
                                "中文": "🔊 听答案",
                            }.get(language_mode, "🔊 Listen to answer")
                            
                            answer_audio_key = f"answer_audio_{zone}_{selected_kw}_{hash(user_question)}"
                            tts_fail_msg = {
                                "한국어": "음성 생성 실패",
                                "English": "Voice generation failed",
                                "日本語": "音声の生成に失敗しました",
                                "中文": "语音生成失败",
                            }.get(language_mode, "음성 생성 실패")
                            if st.button(listen_answer_label, key=f"btn_answer_audio_{zone}_{selected_kw}_{hash(user_question)}"):
                                try:
                                    if text_to_speech is not None:
                                        lang_code = get_language_code(language_mode) if get_language_code else "ko"
                                        audio = text_to_speech(answer_text, language=lang_code)
                                        if audio:
                                            st.session_state[answer_audio_key] = audio
                                        else:
                                            st.warning(tts_fail_msg)
                                    else:
                                        audio = client.audio.speech.create(
                                            model="tts-1",
                                            voice="alloy",
                                            input=answer_text
                                        )
                                        st.session_state[answer_audio_key] = audio.content
                                except Exception as e:
                                    print(f"답변 TTS 오류: {e}")
                                    st.warning(tts_fail_msg)
                            
                            if answer_audio_key in st.session_state:
                                st.audio(st.session_state[answer_audio_key], format="audio/mp3")
        else:
            st.info(text["pick_zone_hint"])
    
    else:
        st.subheader(text["tab_story"])
        st.markdown(text["story_intro"])

        if language_mode != "한국어" and debug_show_korean:
            st.caption(f"KO: {texts['한국어']['story_intro']}")
        if language_mode != "한국어" and debug_backtranslate:
            bt = _backtranslate_to_korean_cached(text["story_intro"], language_mode)
            if bt:
                st.caption(f"BT: {bt}")

        story_state_key = "post_learning_story"
        story_zones_key = "post_learning_story_zones"
        audio_state_key = "post_learning_story_audio"
        
        selected_zones_story = _render_zone_selector("story", whitelist=STORY_ZONE_WHITELIST)

        if selected_zones_story and st.button(text["generate_story"]):
            _queue_ga_event("story_generated", {"zone_count": len(selected_zones_story), "language": language_mode})
            with st.spinner(text["story_generating"]):
                llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
                
                all_exhibits = []
                all_principles = []
                
                for zone in selected_zones_story:
                    exhibits = get_zone_exhibits_from_rag(zone, vector_db)
                    if exhibits:
                        all_exhibits.extend(exhibits)
                        principles, _ = extract_principles_from_exhibits(exhibits, llm)
                        all_principles.extend(principles)
                
                if all_exhibits:
                    zone_names = ", ".join(selected_zones_story)
                    story = generate_science_story(zone_names, all_exhibits, all_principles, language_mode)
                    
                    if story:
                        st.session_state[story_state_key] = story
                        st.session_state[story_zones_key] = selected_zones_story
                        if audio_state_key in st.session_state:
                            del st.session_state[audio_state_key]
                    else:
                        st.error(text["story_fail"])
                else:
                    st.warning(text["pick_zone_hint"])

        if story_state_key in st.session_state and st.session_state.get(story_state_key):
            st.markdown(text["story_generated"])

            if language_mode != "한국어" and debug_show_korean:
                st.caption(f"KO: {texts['한국어']['story_generated']}")
            if language_mode != "한국어" and debug_backtranslate:
                bt = _backtranslate_to_korean_cached(text["story_generated"], language_mode)
                if bt:
                    st.caption(f"BT: {bt}")
            raw_story = st.session_state[story_state_key]
            story_text, mood, bgm_path = parse_mood_and_bgm(raw_story)
            st.markdown(story_text)
            if os.path.exists(bgm_path):
                with open(bgm_path, "rb") as f:
                    st.audio(f.read(), format="audio/mp3")
                st.caption(f"🎵 {mood} 테마 배경음악")
            else:
                st.caption(f"🎵 {mood} 테마 배경음악 ({os.path.basename(bgm_path)} 파일 없음 — bgm/ 폴더에 음악 파일을 넣어주세요)")
            # 동화는 외국어 모드에서 직접 그 언어로 생성되므로 별도의 KO 원문이 없음.
            # → debug_show_korean 또는 debug_backtranslate 가 켜지면 한국어 역번역본을 노출 (사실상 동일한 자료).
            if language_mode != "한국어" and (debug_show_korean or debug_backtranslate):
                bt_story = _backtranslate_to_korean_cached(st.session_state[story_state_key], language_mode)
                if bt_story:
                    label = "🇰🇷 한국어로 보기 (디버그)" if debug_show_korean else "BT (동화 본문 역번역)"
                    with st.expander(label, expanded=False):
                        st.markdown(bt_story)

            if st.button(text["to_audiobook"]):
                _queue_ga_event("audiobook_converted", {"language": language_mode})
                with st.spinner(text["audiobook_generating"]):
                    # 사용자가 설정한 음성 아이디가 있으면 전달
                    custom_voice_id = os.environ.get("ELEVENLABS_VOICE_ID")
                    if (not custom_voice_id) and hasattr(st, "secrets"):
                        custom_voice_id = _safe_secret_get("ELEVENLABS_VOICE_ID", "")
                    audio_bytes = text_to_audiobook(
                        st.session_state[story_state_key],
                        language_mode,
                        voice_override=custom_voice_id,
                    )
                    if audio_bytes:
                        st.session_state[audio_state_key] = audio_bytes
                    else:
                        st.error(text["audiobook_fail"])

            if audio_state_key in st.session_state and st.session_state.get(audio_state_key):
                st.audio(st.session_state[audio_state_key], format="audio/mp3")
                st.download_button(
                    label=text["audiobook_download"],
                    data=st.session_state[audio_state_key],
                    file_name="my_science_story.mp3",
                    mime="audio/mp3"
                )
