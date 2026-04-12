# app_with_voice.py - Museum guide with voice features
import os
import uuid
import streamlit as st
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from audio_recorder_streamlit import audio_recorder
import base64

from core import get_tools, route_intent, answer_rule_based, get_dynamic_prompt, render_source_buttons, initialize_vector_db
from voice import speech_to_text, text_to_speech, get_language_code, autoplay_audio
from learning import render_post_visit_learning

# Optimized RAG loading with progress indication
@st.cache_resource
def load_rag_db():
    """Load RAG database with caching"""
    with st.spinner("RAG database loading..."):
        from core import initialize_vector_db
        vector_db = initialize_vector_db()
        st.success("RAG database ready!")
    return vector_db

def main():
    st.set_page_config(page_title="국립어린이과학관 AI 가이드", page_icon="🐣", layout="centered")

    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "")

    # RAG 시스템 로드
    vector_db = load_rag_db()

    with st.sidebar:
        st.title("⚙️ 안내 모드")
        user_mode = st.selectbox("사용자 모드 선택:", options=["어린이", "청소년/성인"], index=1)
        
        # Language mode selection
        language_mode = st.selectbox(
            "언어 모드 선택:",
            options=["한국어", "English", "日本語", "中文"],
            index=0
        )
        
        # Voice features toggle
        st.markdown("---")
        st.subheader("🎤 음성 기능")
        enable_voice_input = st.checkbox("음성 입력 활성화", value=True)
        enable_voice_output = st.checkbox("음성 출력 활성화", value=True)
        
        if st.button("대화 새로고침 🔄"):
            st.session_state.messages = []
            st.session_state.thread_id = uuid.uuid4().hex
            st.session_state.debug_logs = []
            st.rerun()

    st.title("국립어린이과학관 AI 가이드🤖")
    
    # Tab navigation
    tab1, tab2 = st.tabs(["💬 과학관 안내", "🎓 사후 학습"])
    
    with tab1:
        # Architecture Overview Section
        with st.expander("LLM-based Active Scientific Principle Exploration Architecture", expanded=False):
            st.markdown("Architecture visualization temporarily disabled due to graphviz dependency.")
            st.markdown("Key components working: LLM Agent, RAG System, Real-time Tools, Voice I/O")

        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "thread_id" not in st.session_state:
            st.session_state.thread_id = uuid.uuid4().hex
        if "debug_logs" not in st.session_state:
            st.session_state.debug_logs = []

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        memory = MemorySaver()
        agent = create_react_agent(
            model=llm,
            tools=get_tools(),
            prompt=get_dynamic_prompt(user_mode, language_mode),
            checkpointer=memory,
        )

        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "debug":
                with st.expander("🔍 디버깅 정보(도구 호출 내역)"):
                    with st.container(height=400):
                        st.text(msg["content"])
            else:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Voice input
        user_input = None
        
        if enable_voice_input:
            st.markdown("### 🎤 음성으로 질문하기")
            audio_bytes = audio_recorder(
                text="녹음 시작",
                recording_color="#e74c3c",
                neutral_color="#3498db",
                icon_name="microphone",
                icon_size="2x"
            )
            
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                with st.spinner("음성을 텍스트로 변환 중..."):
                    user_input = speech_to_text(audio_bytes)
                    if user_input:
                        st.success(f"인식된 질문: {user_input}")
                    else:
                        st.error("음성 인식에 실패했습니다. 다시 시도해주세요.")
        
        # Text input (fallback)
        if not user_input:
            user_input = st.chat_input("질문을 입력해주세요!")
        
        # Recommended Questions
        if not st.session_state.messages:
            st.markdown("### 💡 추천 질문")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔬 과학 원리", key="science"):
                    user_input = "빛이 굽는 원리가 궁금해요!"
            with col2:
                if st.button("🏛️ 전시관 안내", key="exhibit"):
                    user_input = "탐구놀이터에서 뭐 할 수 있어요?"
            with col3:
                if st.button("📅 운영 정보", key="info"):
                    user_input = "내일 가도 돼요?"
        
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            intent = route_intent(user_input)
            
            with st.chat_message("assistant"):
                if intent in ["notice", "basic"]:
                    # 규칙 기반 엔진 동작 (RAG/LLM 미사용, 속도 최적화)
                    with st.spinner("(규칙기반)확인 중입니다..."):
                        answer = answer_rule_based(intent, user_input, user_mode)
                    st.markdown(answer)
                    render_source_buttons(answer)
                else:
                    # LLM + RAG + Crawling 엔진 동작
                    with st.spinner("(LLM)잠시만 기다려 주세요..."):
                        # FAISS RAG에서 관련 정보 사전 검색하여 컨텍스트 주입
                        retrieved_docs = vector_db.similarity_search(user_input, k=3)
                        rag_context = "\n\n".join([f"[{doc.metadata.get('source', 'N/A')}]\n{doc.page_content}" for doc in retrieved_docs])
                        
                        # RAG 컨텍스트를 시스템 메시지로 추가 (사용자 메시지와 분리)
                        config = {"configurable": {"thread_id": st.session_state.thread_id}}
                        messages = [
                            {"role": "system", "content": f"[RAG 배경지식]\n{rag_context}"},
                            {"role": "user", "content": user_input}
                        ]
                        result = agent.invoke({"messages": messages}, config=config)
                        answer = result["messages"][-1].content
                        
                        # 디버깅 로그 수집 (RAG 검색 결과 포함)
                        debug_info = f"=== RAG 검색 결과 (k=3) ===\n{rag_context}\n\n{'='*50}\n\n"
                        for msg in result["messages"][:-1]:  # 마지막 답변 제외
                            if hasattr(msg, 'pretty_repr'):
                                debug_info += msg.pretty_repr() + "\n\n"
                            elif hasattr(msg, 'content'):
                                debug_info += str(msg.content) + "\n\n"
                        
                    st.markdown(answer)
                    render_source_buttons(answer)
                    
                    # 디버깅 정보 표시 (답변 뒤)
                    if debug_info.strip():
                        with st.expander("🔍 디버깅 정보 (도구 호출 내역)"):
                            with st.container(height=400):
                                st.text(debug_info)
                        st.session_state.messages.append({"role": "debug", "content": debug_info})

            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Voice output
            if enable_voice_output and answer:
                with st.spinner("음성으로 변환 중..."):
                    lang_code = get_language_code(language_mode)
                    audio_bytes = text_to_speech(answer, language=lang_code)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")
                        autoplay_audio(audio_bytes)
    
    with tab2:
        # Post-visit learning system
        render_post_visit_learning(vector_db, language_mode)

if __name__ == "__main__":
    main()
