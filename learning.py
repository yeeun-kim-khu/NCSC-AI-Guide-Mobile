# learning.py - 또만나 놀이터 시스템 통합
# post_visit_learning.py + audiobook_generator.py + visualization.py 통합

import streamlit as st
from langchain_openai import ChatOpenAI
from openai import OpenAI
import os
import random
import re
import requests
from collections import Counter
from core import initialize_vector_db, load_zone_rows_from_csv

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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
        "description": "생각하는 힘을 키워요",
        "has_data": False
    },
    "행동놀이터": {
        "floor": "1층",
        "description": "몸을 움직이며 과학을 배워요",
        "has_data": True
    },
    "천체투영관": {
        "floor": "1층",
        "description": "우주와 별의 비밀을 알아봐요",
        "has_data": False
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
        "description": "빛의 신비를 체험해요",
        "has_data": False
    }
}

ZONE_GROUPS = {
    "1층놀이터(AI·행동·생각 놀이터)": ["AI놀이터", "행동놀이터", "생각놀이터"],
    "2층(관찰·탐구 놀이터)": ["관찰놀이터", "탐구놀이터"],
    "천체투영관": ["천체투영관"],
    "빛놀이터": ["빛놀이터"],
}


def _select_zones_by_group(prefix_key: str) -> list[str]:
    selected = []
    for label, zones in ZONE_GROUPS.items():
        if st.checkbox(label, key=f"{prefix_key}_{label}"):
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

# ============================================================================
# CSV 데이터 로딩
# ============================================================================

@st.cache_data(show_spinner=False)
def _preload_all_zone_csv_rows():
    """모든 놀이터의 CSV 데이터를 미리 로드"""
    import os
    import glob
    
    # 디버깅: 현재 디렉토리와 data 폴더 확인
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    
    st.write(f"🔍 **디버깅 정보:**")
    st.write(f"- 현재 디렉토리: `{current_dir}`")
    st.write(f"- Data 폴더: `{data_dir}`")
    st.write(f"- Data 폴더 존재: {os.path.exists(data_dir)}")
    
    if os.path.exists(data_dir):
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        st.write(f"- 발견된 CSV 파일 수: {len(csv_files)}")
        st.write(f"- CSV 파일 목록: {[os.path.basename(f) for f in csv_files[:5]]}")
    
    data = {}
    for zone, info in ZONE_INFO.items():
        if info.get("has_data"):
            try:
                rows = load_zone_rows_from_csv(zone)
                data[zone] = rows
                st.write(f"✅ **{zone}**: {len(rows)}개 전시물 로드 성공")
            except Exception as e:
                st.error(f"❌ **{zone}** 로드 실패: {str(e)}")
                data[zone] = []
    
    return data

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


def _extract_zone_keywords_from_titles(zone_rows, top_n=12):
    titles = []
    for r in (zone_rows or []):
        t = str(r.get("title", "")).strip()
        if not t or len(t) <= 1:
            continue
        # "체험방법" 제외
        if "체험방법" in t:
            continue
        t = re.sub(r"\s+", " ", t)
        titles.append(t)
    titles = list(dict.fromkeys(titles))
    return titles[:top_n]


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def _extract_zone_keywords_llm(zone_name: str, language_mode: str, csv_compact_text: str):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    if language_mode == "한국어":
        prompt = f"""너는 4세~초등 저학년 어린이와 학부모를 위한 전시관 키워드 편집자야.

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


def _get_zone_keywords(zone_name: str, zone_rows, language_mode: str):
    kws = _extract_zone_keywords_from_titles(zone_rows)
    if kws:
        return kws

    compact_lines = []
    for r in (zone_rows or [])[:40]:
        title = str(r.get("title", "")).strip()
        cat = str(r.get("category", "")).strip()
        content = str(r.get("content", "")).strip()
        if title or content:
            compact_lines.append(f"- {title} ({cat}) {content[:120]}")
    csv_compact_text = "\n".join(compact_lines)[:6000]

    try:
        kws = _extract_zone_keywords_llm(zone_name, language_mode, csv_compact_text)
        if kws:
            return kws
    except Exception as e:
        print(f"키워드 LLM 추출 실패: {e}")

    return _extract_zone_keywords(zone_rows)


def _render_keyword_tags(zone_name: str, keywords, zone_rows):
    if not keywords:
        return

    st.markdown("### 🔑 오늘의 키워드")

    state_key = f"kw_selected_{zone_name}"
    if state_key not in st.session_state:
        st.session_state[state_key] = ""

    cols = st.columns(4)
    for i, kw in enumerate(keywords):
        with cols[i % 4]:
            if st.button(kw, key=f"kw_btn_{zone_name}_{kw}"):
                st.session_state[state_key] = kw

    selected_kw = st.session_state.get(state_key, "")
    if selected_kw:
        st.caption(f"선택한 키워드: {selected_kw}")
        matched = []
        for r in (zone_rows or []):
            title = str(r.get("title", ""))
            content = str(r.get("content", ""))
            detail = str(r.get("detail", ""))
            category = str(r.get("category", ""))
            if selected_kw in (title + " " + category + " " + content + " " + detail):
                if title:
                    matched.append(title)
        matched = list(dict.fromkeys(matched))
        if matched:
            st.markdown("**관련 전시물**")
            st.markdown("\n".join([f"- {t}" for t in matched[:10]]))
            if len(matched) > 10:
                st.caption(f"+ {len(matched) - 10}개 더 있음")
        if st.button("키워드 선택 해제", key=f"kw_clear_{zone_name}"):
            st.session_state[state_key] = ""


# ============================================================================
# RAG 검색 및 원리 추출
# ============================================================================

def get_zone_exhibits_from_rag(zone_name, vector_db):
    """RAG에서 해당 놀이터의 전시물 정보 가져오기"""
    try:
        docs = []
        for q in (zone_name, f"[{zone_name}]", f"csv_{zone_name}"):
            try:
                docs.extend(vector_db.similarity_search(q, k=80))
            except Exception as e:
                print(f"RAG 검색 오류(쿼리={q}): {e}")

        exhibits = []
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

        print(f"최종 검색 결과: {zone_name}에서 {len(exhibits)}개 전시물 발견")
        return exhibits
    except Exception as e:
        print(f"RAG 검색 오류: {e}")
        return []

def extract_principles_from_exhibits(exhibits, llm):
    """전시물에서 과학원리 추출"""
    if not exhibits:
        return [], ""
    
    exhibit_text = "\n\n".join([ex["content"] for ex in exhibits[:10]])
    
    prompt = f"""다음 전시물들에서 핵심 과학원리를 추출해주세요.

전시물 정보:
{exhibit_text}

**응답 형식:**
1. 먼저 과학원리 목록을 쉼표로 구분하여 한 줄로 작성하세요.
   예: 빛의 굴절, 소리의 진동, 전기회로, 자기장, 에너지 변환

2. 그 다음 각 원리에 대한 설명을 작성하세요.
   - 원리명: 간단한 설명 (1-2문장)

최대 5-7개의 핵심 원리를 추출하세요."""

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

def generate_quiz(zone_name, principle, llm, language="한국어"):
    """과학원리 기반 퀴즈 생성"""

    def _get_ui_glossary_rules(language_mode: str) -> str:
        glossary = {
            "English": {
                "놀이터": "Zone",
                "전시물": "Exhibit",
                "과학원리": "Science principle",
                "오디오북": "Audiobook",
            }
        }
        if language_mode == "한국어":
            return ""
        lang_terms = glossary.get(language_mode, glossary["English"])
        rule_lines = [f"- '{ko}' -> '{lang}'" for ko, lang in lang_terms.items()]
        return (
            "\n\nGLOSSARY (must follow exactly):\n"
            + "\n".join(rule_lines)
            + "\n- Use these terms consistently. Do not mix languages.\n"
        )

    glossary_rules = _get_ui_glossary_rules(language)

    language_prompts = {
        "한국어": f"""'{zone_name}'의 '{principle}' 원리에 대한 퀴즈를 만들어주세요.

퀴즈 형식:
**질문**: [어린이가 이해하기 쉬운 질문]

**선택지**:
1. [정답]
2. [오답1]
3. [오답2]
4. [오답3]

**정답**: 1번

**해설**: [정답인 이유를 쉽게 설명]

어린이 눈높이에 맞춰 재미있고 교육적인 퀴즈를 만들어주세요!""",
        
        "English": f"""Create a quiz about '{principle}' from '{zone_name}'.{glossary_rules}

Quiz format:
**Question**: [Easy-to-understand question for children]

**Options**:
1. [Correct answer]
2. [Wrong answer 1]
3. [Wrong answer 2]
4. [Wrong answer 3]

**Answer**: 1

**Explanation**: [Why this is correct, explained simply]

Make it fun and educational for children!"""
    }
    
    prompt = language_prompts.get(language, language_prompts["한국어"])
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"퀴즈 생성 오류: {e}")
        return None

# ============================================================================
# 오디오북 생성
# ============================================================================

def generate_science_story(zone_name, exhibits, principles, language="한국어"):
    """방문한 놀이터 기반 과학동화 생성"""
    
    exhibit_summary = "\n".join([f"- {ex['metadata'].get('title', '')}" for ex in exhibits[:5]])
    principles_text = ", ".join(principles[:3])

    def _get_ui_glossary_rules(language_mode: str) -> str:
        glossary = {
            "English": {
                "놀이터": "Zone",
                "전시물": "Exhibit",
                "과학원리": "Science principle",
                "오디오북": "Audiobook",
            }
        }
        if language_mode == "한국어":
            return ""
        lang_terms = glossary.get(language_mode, glossary["English"])
        rule_lines = [f"- '{ko}' -> '{lang}'" for ko, lang in lang_terms.items()]
        return (
            "\n\nGLOSSARY (must follow exactly):\n"
            + "\n".join(rule_lines)
            + "\n- Use these terms consistently. Do not mix languages.\n"
        )

    glossary_rules = _get_ui_glossary_rules(language)
    
    language_prompts = {
        "한국어": f"""당신은 어린이를 위한 과학동화 작가입니다.

**배경:**
오늘 어린이가 '{zone_name}'에서 다음 전시물들을 체험했습니다:
{exhibit_summary}

이 전시물들에는 다음과 같은 과학원리가 담겨 있습니다:
{principles_text}

**요청:**
이 체험을 바탕으로 5-7분 분량의 과학동화를 만들어주세요.

**동화 구성:**
1. 주인공: 호기심 많은 어린이 (이름: 지우)
2. 스토리: 지우가 '{zone_name}'에서 체험한 내용을 모험 이야기로 구성
3. 과학원리: 자연스럽게 녹여서 설명
4. 톤: 따뜻하고 재미있게, 잠들기 전 듣기 좋은 분위기
5. 길이: 약 1000-1500자

**중요:**
- 어린이가 이해하기 쉬운 단어 사용
- 과학원리를 억지로 설명하지 말고 이야기 속에 자연스럽게 녹이기
- 긍정적이고 희망찬 결말
- 잠들기 전 듣기 좋은 차분한 분위기""",

        "English": f"""You are a children's science storyteller.{glossary_rules}

**Background:**
Today, a child visited '{zone_name}' and experienced these exhibits:
{exhibit_summary}

These exhibits contain the following scientific principles:
{principles_text}

**Request:**
Create a 5-7 minute science bedtime story based on this experience.

**Story Structure:**
1. Protagonist: A curious child (Name: Jiwoo)
2. Story: Turn Jiwoo's experience at '{zone_name}' into an adventure
3. Science: Naturally weave in the scientific principles
4. Tone: Warm, fun, perfect for bedtime
5. Length: About 1000-1500 characters

**Important:**
- Use simple, child-friendly language
- Don't force science explanations - make them natural
- Positive, hopeful ending
- Calm atmosphere suitable for bedtime listening"""
    }
    
    prompt = language_prompts.get(language, language_prompts["한국어"])
    
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.8)
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"동화 생성 오류: {e}")
        return None

def text_to_audiobook(story_text, language="한국어"):
    """텍스트를 오디오북으로 변환"""
    
    voice_map = {
        "한국어": "nova",
        "English": "alloy",
        "日本語": "shimmer",
        "中文": "fable"
    }
    
    voice = voice_map.get(language, "nova")
    
    try:
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input=story_text,
            speed=0.9
        )
        
        return response.content
    except Exception as e:
        print(f"오디오 생성 오류: {e}")
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
):
    """사후 학습 시스템 메인 UI"""

    def _display_zone_name(zone: str) -> str:
        if language_mode == "한국어":
            return zone
        official = {
            "AI놀이터": "AI Zone",
            "행동놀이터": "Activity Zone",
            "관찰놀이터": "Discovery Zone",
            "탐구놀이터": "Exploration Zone",
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
            "tab_quiz": "퀴즈타임",
            "tab_question": "궁금해요!",
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
            "principles_not_found": "과학원리를 추출할 수 없습니다.",
            "csv_not_found": "CSV 전시물 정보를 찾을 수 없습니다.",
            "expander_parent": "학부모용: 전시물 전체보기",
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
    
    st.title(text["title"])
    st.markdown(text["subtitle"])
    
    # CSV 데이터 미리 로드
    all_zone_rows = _preload_all_zone_csv_rows()

    tab_quiz, tab_question, tab_story = st.tabs([text["tab_quiz"], text["tab_question"], text["tab_story"]])

    def _render_zone_selector(key_prefix: str):
        st.subheader(text["select_zone"])

        selected = []

        st.markdown(f"### {text['floor1']}")
        col1, col2 = st.columns(2)

        floor1_zones = [z for z, info in ZONE_INFO.items() if info["floor"] == "1층"]
        for i, zone in enumerate(floor1_zones):
            col = col1 if i % 2 == 0 else col2
            with col:
                disabled = not ZONE_INFO[zone]["has_data"]
                zone_disp = _display_zone_name(zone)
                label = f"{zone_disp} {text['no_data']}" if disabled else zone_disp
                if st.checkbox(label, key=f"{key_prefix}_zone_{zone}", disabled=disabled):
                    selected.append(zone)

        st.markdown(f"### {text['floor2']}")
        col3, col4 = st.columns(2)

        floor2_zones = [z for z, info in ZONE_INFO.items() if info["floor"] == "2층"]
        for i, zone in enumerate(floor2_zones):
            col = col3 if i % 2 == 0 else col4
            with col:
                disabled = not ZONE_INFO[zone]["has_data"]
                zone_disp = _display_zone_name(zone)
                label = f"{zone_disp} {text['no_data']}" if disabled else zone_disp
                if st.checkbox(label, key=f"{key_prefix}_zone_{zone}", disabled=disabled):
                    selected.append(zone)

        return selected

    def _render_zone_header(zone: str, zone_rows):
        st.markdown(f"## 🎯 {_display_zone_name(zone)}")
        st.caption(f"전시물 {len(zone_rows)}개")
        keywords = _get_zone_keywords(zone, zone_rows, language_mode)
        _render_keyword_tags(zone, keywords, zone_rows)
        with st.expander(text["expander_parent"], expanded=False):
            if zone_rows:
                st.dataframe(zone_rows, use_container_width=True, hide_index=True)
            else:
                st.info(text["csv_not_found"])

    with tab_quiz:
        selected_zones = _render_zone_selector("quiz")

        if selected_zones:
            st.markdown("---")
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
            for zone in selected_zones:
                zone_rows = all_zone_rows.get(zone, [])
                _render_zone_header(zone, zone_rows)

                with st.spinner(text["generating"]):
                    exhibits = get_zone_exhibits_from_rag(zone, vector_db)
                    if exhibits:
                        principles, principles_text = extract_principles_from_exhibits(exhibits, llm)
                        if principles:
                            selected_principle = st.selectbox(
                                text["select_principle"],
                                principles,
                                key=f"principle_{zone}"
                            )

                            if st.button(text["make_quiz"], key=f"quiz_{zone}"):
                                with st.spinner(text["quiz_generating"]):
                                    quiz = generate_quiz(zone, selected_principle, llm, language_mode)
                                    if quiz:
                                        st.markdown(quiz)
                        else:
                            st.info(text["principles_not_found"])
                    else:
                        st.warning(f"{_display_zone_name(zone)}{text['exhibits_not_found']}")
        else:
            st.info(text["pick_zone_hint"])

    with tab_question:
        selected_zones = _render_zone_selector("question")

        if selected_zones:
            st.markdown("---")
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
            for zone in selected_zones:
                zone_rows = all_zone_rows.get(zone, [])
                _render_zone_header(zone, zone_rows)

                with st.spinner(text["generating"]):
                    exhibits = get_zone_exhibits_from_rag(zone, vector_db)
                    if exhibits:
                        user_question = st.text_input(
                            f"{_display_zone_name(zone)}{text['question_prompt']}",
                            key=f"question_input_{zone}"
                        )

                        if st.button(text["ask_question"], key=f"question_btn_{zone}") and user_question:
                            context = "\n".join([ex["content"] for ex in exhibits[:5]])
                            prompt = f"""다음은 '{zone}'의 전시물 정보입니다:
{context}

사용자 질문: {user_question}

어린이가 이해하기 쉽게 답변해주세요."""
                            response = llm.invoke(prompt)
                            st.markdown(f"**{text['answer_prefix']}:** {response.content}")
                    else:
                        st.warning(f"{_display_zone_name(zone)}{text['exhibits_not_found']}")
        else:
            st.info(text["pick_zone_hint"])
    
    with tab_story:
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
        
        selected_zones_story = []
        
        st.markdown(text["story_select_heading"])

        if language_mode != "한국어" and debug_show_korean:
            st.caption(f"KO: {texts['한국어']['story_select_heading']}")
        if language_mode != "한국어" and debug_backtranslate:
            bt = _backtranslate_to_korean_cached(text["story_select_heading"], language_mode)
            if bt:
                st.caption(f"BT: {bt}")
        selected_zones_story = _select_zones_by_group("story")
        
        # Debug: show selected zones
        if selected_zones_story:
            st.info(f"선택된 놀이터: {', '.join(selected_zones_story)}")
        else:
            st.warning("놀이터를 선택해주세요")

        if selected_zones_story and st.button(text["generate_story"]):
            with st.spinner(text["story_generating"]):
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
                
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
            st.markdown(st.session_state[story_state_key])

            if st.button(text["to_audiobook"]):
                with st.spinner(text["audiobook_generating"]):
                    audio_bytes = text_to_audiobook(
                        st.session_state[story_state_key],
                        language_mode,
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
