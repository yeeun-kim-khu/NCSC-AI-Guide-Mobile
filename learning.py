# learning.py - 사후 학습 시스템 통합
# post_visit_learning.py + audiobook_generator.py + visualization.py 통합

import streamlit as st
from langchain_openai import ChatOpenAI
from openai import OpenAI
import os
import random
from core import initialize_vector_db

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

# ============================================================================
# RAG 검색 및 원리 추출
# ============================================================================

def get_zone_exhibits_from_rag(zone_name, vector_db):
    """RAG DB: zone_name(='AI/Thinking/Action/Explore/Observe/Light') -> list of exhibits"""
    try:
        print(f"=== RAG Search Debug ===")
        print(f"Input zone_name: {zone_name}")
        
        # zone_name based search terms - match actual CSV data
        search_terms = []
        
        # Map zone names to actual CSV categories
        csv_zone_mapping = {
            "AI": "AI놀이터",
            "Action": "행동놀이터",
            "Explore": "탐구놀이터",
            "Observe": "관찰놀이터",
            "Thinking": "",  # 데이터 없음
            "Light": ""      # 데이터 없음
        }
        
        mapped_zone = csv_zone_mapping.get(zone_name, zone_name)
        print(f"Mapped zone: {mapped_zone}")
        
        # Skip zones without data
        if not mapped_zone:
            print(f"No data available for zone: {zone_name}")
            return []
            
        search_terms.extend([mapped_zone, zone_name])
        print(f"Search terms: {search_terms}")
        
        # Search with multiple terms
        all_docs = []
        for term in search_terms:
            try:
                docs = vector_db.similarity_search(term, k=30)
                print(f"Term '{term}' found {len(docs)} docs")
                all_docs.extend(docs)
            except Exception as e:
                print(f"Search error for term '{term}': {e}")
                continue
        
        print(f"Total docs before filtering: {len(all_docs)}")
        
        # Remove duplicates and filter by zone relevance
        exhibits = []
        seen_content = set()
        
        for doc in all_docs:
            content = doc.page_content
            category = doc.metadata.get("category", "")
            
            # Check if content is relevant to zone
            is_relevant = (
                mapped_zone in content or 
                zone_name in content or 
                mapped_zone in category or
                zone_name in category
            )
            
            print(f"Doc check - Category: '{category}', Relevant: {is_relevant}")
            
            if is_relevant and content not in seen_content:
                exhibits.append({
                    "content": content,
                    "metadata": doc.metadata
                })
                seen_content.add(content)
        
        print(f"Final result: {len(exhibits)} exhibits found in {zone_name}")
        print(f"=== End RAG Search Debug ===")
        return exhibits
    except Exception as e:
        print(f"RAG search error: {e}")
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
            "tab1": "퀴즈 & 질문",
            "tab2": "나만의 과학동화",
            "select_zone": "체험한 놀이터를 선택하세요",
            "no_data": "(준비 중)",
            "generating": "과학원리 분석 중...",
            "quiz_mode": "퀴즈 모드",
            "chat_mode": "질문 모드",
            "select_principle": "퀴즈 주제 선택",
            "make_quiz": "퀴즈 생성",
            "quiz_generating": "퀴즈 생성 중...",
            "question_prompt": "에 대해 궁금한 점을 물어보세요",
            "answer_prefix": "답변",
            "pick_zone_hint": "체험한 놀이터를 선택해주세요!",
            "exhibits_not_found": "의 전시물 정보를 찾을 수 없습니다.",
            "principles_not_found": "과학원리를 추출할 수 없습니다.",
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
            "tab1": "Quiz & Questions",
            "tab2": "My Science Story",
            "select_zone": "Select visited zones",
            "no_data": "(Coming soon)",
            "generating": "Analyzing principles...",
            "quiz_mode": "Quiz Mode",
            "chat_mode": "Q&A Mode",
            "select_principle": "Choose a quiz topic",
            "make_quiz": "Generate quiz",
            "quiz_generating": "Generating quiz...",
            "question_prompt": ": ask what you're curious about",
            "answer_prefix": "Answer",
            "pick_zone_hint": "Please select the zones you visited!",
            "exhibits_not_found": ": exhibit information not found.",
            "principles_not_found": "Unable to extract science principles.",
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
        }
    }
    
    text = texts.get(language_mode, texts["한국어"])
    
    st.title(text["title"])
    st.markdown(text["subtitle"])
    
    learning_tab1, learning_tab2 = st.tabs([text["tab1"], text["tab2"]])
    
    with learning_tab1:
        st.subheader(text["select_zone"])
        
        selected_zones = []
        
        st.markdown(f"### {text['floor1']}")
        col1, col2 = st.columns(2)
        
        floor1_zones = [z for z, info in ZONE_INFO.items() if info["floor"] == "1층"]
        for i, zone in enumerate(floor1_zones):
            col = col1 if i % 2 == 0 else col2
            with col:
                disabled = not ZONE_INFO[zone]["has_data"]
                zone_disp = _display_zone_name(zone)
                label = f"{zone_disp} {text['no_data']}" if disabled else zone_disp
                if st.checkbox(label, key=f"zone_{zone}", disabled=disabled):
                    selected_zones.append(zone)
        
        st.markdown(f"### {text['floor2']}")
        col3, col4 = st.columns(2)
        
        floor2_zones = [z for z, info in ZONE_INFO.items() if info["floor"] == "2층"]
        for i, zone in enumerate(floor2_zones):
            col = col3 if i % 2 == 0 else col4
            with col:
                disabled = not ZONE_INFO[zone]["has_data"]
                zone_disp = _display_zone_name(zone)
                label = f"{zone_disp} {text['no_data']}" if disabled else zone_disp
                if st.checkbox(label, key=f"zone_{zone}", disabled=disabled):
                    selected_zones.append(zone)
        
        if selected_zones:
            st.markdown("---")
            
            mode = st.radio("학습 모드 선택", [text["quiz_mode"], text["chat_mode"]], horizontal=True)
            
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
            
            for zone in selected_zones:
                st.markdown(f"## 🎯 {_display_zone_name(zone)}")
                
                with st.spinner(text["generating"]):
                    exhibits = get_zone_exhibits_from_rag(zone, vector_db)
                    
                    if exhibits:
                        principles, principles_text = extract_principles_from_exhibits(exhibits, llm)
                        
                        if principles:
                            st.markdown("**발견한 과학원리:**")
                            st.markdown(principles_text)
                            
                            if mode == text["quiz_mode"]:
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
                            
                            else:  # 질문 모드
                                user_question = st.text_input(
                                    f"{_display_zone_name(zone)}{text['question_prompt']}",
                                    key=f"question_{zone}"
                                )
                                
                                if user_question:
                                    context = "\n".join([ex["content"] for ex in exhibits[:5]])
                                    prompt = f"""다음은 '{zone}'의 전시물 정보입니다:
{context}

사용자 질문: {user_question}

어린이가 이해하기 쉽게 답변해주세요."""
                                    
                                    response = llm.invoke(prompt)
                                    st.markdown(f"**{text['answer_prefix']}:** {response.content}")
                        else:
                            st.info(text["principles_not_found"])
                    else:
                        st.warning(f"{_display_zone_name(zone)}{text['exhibits_not_found']}")
        else:
            st.info(text["pick_zone_hint"])
    
    with learning_tab2:
        st.subheader(text["tab2"])
        st.markdown(text["story_intro"])

        if language_mode != "한국어" and debug_show_korean:
            st.caption(f"KO: {texts['한국어']['story_intro']}")
        if language_mode != "한국어" and debug_backtranslate:
            bt = _backtranslate_to_korean_cached(text["story_intro"], language_mode)
            if bt:
                st.caption(f"BT: {bt}")
        
        selected_zones_story = []
        
        st.markdown(text["story_select_heading"])

        if language_mode != "한국어" and debug_show_korean:
            st.caption(f"KO: {texts['한국어']['story_select_heading']}")
        if language_mode != "한국어" and debug_backtranslate:
            bt = _backtranslate_to_korean_cached(text["story_select_heading"], language_mode)
            if bt:
                st.caption(f"BT: {bt}")
        for zone, info in ZONE_INFO.items():
            if info["has_data"]:
                if st.checkbox(zone, key=f"story_{zone}"):
                    selected_zones_story.append(zone)
        
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
                        st.markdown(text["story_generated"])

                        if language_mode != "한국어" and debug_show_korean:
                            st.caption(f"KO: {texts['한국어']['story_generated']}")
                        if language_mode != "한국어" and debug_backtranslate:
                            bt = _backtranslate_to_korean_cached(text["story_generated"], language_mode)
                            if bt:
                                st.caption(f"BT: {bt}")
                        st.markdown(story)
                        
                        if st.button(text["to_audiobook"]):
                            with st.spinner(text["audiobook_generating"]):
                                audio_bytes = text_to_audiobook(story, language_mode)
                                
                                if audio_bytes:
                                    st.audio(audio_bytes, format="audio/mp3")
                                    st.download_button(
                                        label=text["audiobook_download"],
                                        data=audio_bytes,
                                        file_name="my_science_story.mp3",
                                        mime="audio/mp3"
                                    )
                                else:
                                    st.error(text["audiobook_fail"])
                    else:
                        st.error(text["story_fail"])
                else:
                    st.warning(text["pick_zone_hint"])
