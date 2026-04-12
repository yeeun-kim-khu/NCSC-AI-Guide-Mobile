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
    """RAG에서 해당 놀이터의 전시물 정보 가져오기"""
    try:
        try:
            docs = vector_db.similarity_search(
                zone_name,
                k=50,
                filter={"category": zone_name}
            )
        except:
            docs = vector_db.similarity_search(zone_name, k=50)
        
        exhibits = []
        seen_titles = set()
        
        for doc in docs:
            category = doc.metadata.get("category", "")
            title = doc.metadata.get("title", "")
            content = doc.page_content
            
            if zone_name in category and title not in seen_titles:
                exhibits.append({
                    "content": content,
                    "metadata": doc.metadata
                })
                seen_titles.add(title)
        
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
        
        "English": f"""Create a quiz about '{principle}' from '{zone_name}'.

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

        "English": f"""You are a children's science storyteller.

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

def render_post_visit_learning(vector_db, language_mode="한국어"):
    """사후 학습 시스템 메인 UI"""
    
    texts = {
        "한국어": {
            "title": "🎓 사후 학습",
            "subtitle": "오늘 체험한 놀이터를 선택하고 과학원리를 복습해보세요!",
            "floor1": "1층 전시관",
            "floor2": "2층 전시관",
            "tab1": "퀴즈 & 질문",
            "tab2": "나만의 과학동화",
            "select_zone": "체험한 놀이터를 선택하세요",
            "no_data": "(준비 중)",
            "generating": "과학원리 분석 중...",
            "quiz_mode": "퀴즈 모드",
            "chat_mode": "질문 모드",
            "generate_story": "과학동화 만들기",
            "story_generating": "동화 생성 중...",
            "audiobook_generating": "오디오북 생성 중..."
        },
        "English": {
            "title": "🎓 Post-Visit Learning",
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
                label = f"{zone} {text['no_data']}" if disabled else zone
                if st.checkbox(label, key=f"zone_{zone}", disabled=disabled):
                    selected_zones.append(zone)
        
        st.markdown(f"### {text['floor2']}")
        col3, col4 = st.columns(2)
        
        floor2_zones = [z for z, info in ZONE_INFO.items() if info["floor"] == "2층"]
        for i, zone in enumerate(floor2_zones):
            col = col3 if i % 2 == 0 else col4
            with col:
                disabled = not ZONE_INFO[zone]["has_data"]
                label = f"{zone} {text['no_data']}" if disabled else zone
                if st.checkbox(label, key=f"zone_{zone}", disabled=disabled):
                    selected_zones.append(zone)
        
        if selected_zones:
            st.markdown("---")
            
            mode = st.radio("학습 모드 선택", [text["quiz_mode"], text["chat_mode"]], horizontal=True)
            
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
            
            for zone in selected_zones:
                st.markdown(f"## 🎯 {zone}")
                
                with st.spinner(text["generating"]):
                    exhibits = get_zone_exhibits_from_rag(zone, vector_db)
                    
                    if exhibits:
                        principles, principles_text = extract_principles_from_exhibits(exhibits, llm)
                        
                        if principles:
                            st.markdown("**발견한 과학원리:**")
                            st.markdown(principles_text)
                            
                            if mode == text["quiz_mode"]:
                                selected_principle = st.selectbox(
                                    "퀴즈 주제 선택",
                                    principles,
                                    key=f"principle_{zone}"
                                )
                                
                                if st.button(f"퀴즈 생성", key=f"quiz_{zone}"):
                                    with st.spinner("퀴즈 생성 중..."):
                                        quiz = generate_quiz(zone, selected_principle, llm, language_mode)
                                        if quiz:
                                            st.markdown(quiz)
                            
                            else:  # 질문 모드
                                user_question = st.text_input(
                                    f"{zone}에 대해 궁금한 점을 물어보세요",
                                    key=f"question_{zone}"
                                )
                                
                                if user_question:
                                    context = "\n".join([ex["content"] for ex in exhibits[:5]])
                                    prompt = f"""다음은 '{zone}'의 전시물 정보입니다:
{context}

사용자 질문: {user_question}

어린이가 이해하기 쉽게 답변해주세요."""
                                    
                                    response = llm.invoke(prompt)
                                    st.markdown(f"**답변:** {response.content}")
                        else:
                            st.info("과학원리를 추출할 수 없습니다.")
                    else:
                        st.warning(f"{zone}의 전시물 정보를 찾을 수 없습니다.")
        else:
            st.info("체험한 놀이터를 선택해주세요!")
    
    with learning_tab2:
        st.subheader(text["tab2"])
        st.markdown("오늘 체험한 놀이터를 바탕으로 나만의 과학동화를 만들어보세요!")
        
        selected_zones_story = []
        
        st.markdown("### 동화에 포함할 놀이터 선택")
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
                        st.markdown("### 📖 생성된 동화")
                        st.markdown(story)
                        
                        if st.button("🎧 오디오북으로 변환"):
                            with st.spinner(text["audiobook_generating"]):
                                audio_bytes = text_to_audiobook(story, language_mode)
                                
                                if audio_bytes:
                                    st.audio(audio_bytes, format="audio/mp3")
                                    st.download_button(
                                        label="💾 오디오북 다운로드",
                                        data=audio_bytes,
                                        file_name="my_science_story.mp3",
                                        mime="audio/mp3"
                                    )
                                else:
                                    st.error("오디오북 생성에 실패했습니다.")
                    else:
                        st.error("동화 생성에 실패했습니다.")
                else:
                    st.warning("선택한 놀이터의 정보를 찾을 수 없습니다.")
