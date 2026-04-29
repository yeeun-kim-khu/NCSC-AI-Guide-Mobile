# app_with_voice.py - Museum guide with voice features
import os
import uuid
import streamlit as st
import urllib.parse
import warnings
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from audio_recorder_streamlit import audio_recorder
import base64

from core import get_tools, route_intent, answer_rule_based, answer_rule_based_localized, get_dynamic_prompt, render_source_buttons, initialize_vector_db, CSC_URLS, translate_answer_cached
from voice import speech_to_text, text_to_speech, get_language_code, autoplay_audio, get_tts_cache_namespace
from learning import render_post_visit_learning, _backtranslate_to_korean_cached

# RAG loading with session state persistence
@st.cache_resource
def load_rag_db():
    """Load RAG database with caching"""
    from core import initialize_vector_db
    with st.spinner("RAG database loading..."):
        vector_db = initialize_vector_db()
    return vector_db

def _render_mascot_animation() -> None:
    """어린이 모드용 마스코트를 본문 글자 뒤 워터마크 배경으로 렌더링.

    - 파일: assets/NCSC_character.png
    - 효과: 화면 중앙에 고정된 반투명 배경 (스크롤해도 따라옴)
    - 가독성을 위해 OPACITY는 0.22 권장 (0.6~0.7은 글자 읽기 어려움)
    """
    from pathlib import Path
    # 워터마크 불투명도 — 글자 가독성과 캐릭터 존재감 사이 균형
    # 살짝 더 진하게: 0.30 / 더 옅게: 0.15 / 클로드 추천 0.22 
    OPACITY = 0.15
    SIZE_VMIN = 80  # 화면 짧은 변의 80% 크기 (클로드 추천 60%)

    mascot_path = Path(__file__).parent / "assets" / "NCSC_character.png"
    if not mascot_path.exists():
        return
    try:
        b64 = base64.b64encode(mascot_path.read_bytes()).decode("ascii")
    except Exception:
        return
    html = f"""
    <style>
      @keyframes ncsc-mascot-bg-float {{
        0%   {{ transform: translate(-50%, -50%) translateY(0px); }}
        50%  {{ transform: translate(-50%, -50%) translateY(-12px); }}
        100% {{ transform: translate(-50%, -50%) translateY(0px); }}
      }}
      .ncsc-mascot-bg {{
        position: fixed;
        top: 55%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: {SIZE_VMIN}vmin;
        height: {SIZE_VMIN}vmin;
        max-width: 600px;
        max-height: 600px;
        background-image: url("data:image/png;base64,{b64}");
        background-size: contain;
        background-repeat: no-repeat;
        background-position: center;
        opacity: {OPACITY};
        pointer-events: none;
        z-index: 0;
        animation: ncsc-mascot-bg-float 5s ease-in-out infinite;
      }}
      /* 본문이 워터마크 위에 오도록 */
      [data-testid="stAppViewContainer"] .main .block-container {{
        position: relative;
        z-index: 1;
      }}
    </style>
    <div class="ncsc-mascot-bg" aria-hidden="true"></div>
    """
    st.markdown(html, unsafe_allow_html=True)


def _render_privacy_notice_gate() -> None:
    """첫 진입 시 개인정보 처리 안내 팝업.

    - 세션당 1회만 표시 (확인 누르면 다시 안 뜸)
    - 동의 전에는 본문 렌더링 차단 (st.stop)
    - st.dialog 사용 (Streamlit 1.31+); 미지원 환경에서는 본문 상단 배너로 폴백
    """
    if st.session_state.get("privacy_notice_acknowledged"):
        return

    notice_md = """
**🛡️ 서비스 이용 안내 (개인정보 처리)**
안녕하세요! 이용 전에 짧은 안내 드립니다. 확인해주세요!

**✔️ 입력하신 내용 활용**
- 적어주신 글·말하신 음성은 **답변을 만들 때만 잠깐** 사용되고, **답변이 끝나면 바로 사라져요.**
- 저희 서버·관리자가 그 내용을 **따로 저장하거나 보관하지 않습니다.**
- 화면에 보이는 대화 내용도 **페이지를 새로고침하면 모두 지워져요.**

**🙅 입력 자제 권장**
- 정확한 집 주소·전화번호·주민번호 등 **민감한 개인정보는 입력하지 마세요.**
- 길찾기는 "○○역 근처", "○○동" 정도로 충분히 안내됩니다.

**👶 어린이 관람객에게**
- 어린이 친구들은 **성인 보호자와 함께** 이용해 주세요.
- 음성 기능 사용 전 보호자 동의를 권장합니다.

확인 후 서비스를 이용하실 수 있어요.
"""

    def _ack() -> None:
        st.session_state["privacy_notice_acknowledged"] = True
        st.rerun()

    # st.dialog (Streamlit 1.31+) 우선 사용
    if hasattr(st, "dialog"):
        @st.dialog("이용 안내", width="large")
        def _privacy_dialog():
            st.markdown(notice_md)
            if st.button("확인하고 시작하기 ✨", type="primary", use_container_width=True):
                _ack()

        _privacy_dialog()
        # dialog가 닫히기 전엔 본문 렌더링 차단
        st.stop()
    else:
        # 폴백: 페이지 상단 카드 형태
        st.warning(notice_md)
        if st.button("확인하고 시작하기 ✨", type="primary"):
            _ack()
        st.stop()


def main():
    warnings.filterwarnings("ignore", message=r".*create_react_agent has been moved to `langchain\.agents`\..*")

    # 첫 진입 시 개인정보 안내 팝업 (동의 전 본문 차단)
    _render_privacy_notice_gate()

    if "language_mode" not in st.session_state:
        st.session_state["language_mode"] = "한국어"

    prev_language_for_page = st.session_state.get("language_mode") or "한국어"
    ui_text = {
        "한국어": {
            "page_title": "국립어린이과학관 AI 가이드",
            "app_title": "국립어린이과학관 AI 가이드🤖",
            "sidebar_title": "⚙️ 안내 모드",
            "user_mode_label": "사용자 모드 선택:",
            "user_mode_child": "어린이",
            "user_mode_adult": "청소년/성인",
            "language_label": "언어/Language",
            "voice_section": "🎤 음성 기능",
            "voice_in": "음성 입력 활성화",
            "voice_out": "음성 출력 활성화",
            "voice_ask": "### 🎤 음성으로 질문하기",
            "voice_rec_fail": "음성 인식에 실패했습니다. 다시 시도해주세요.",
            "refresh": "대화 새로고침 🔄",
            "faq_header": "#### ❓ 자주 묻는 질문",
            "faq_floor": "🏢 층별 안내",
            "faq_programs": "🎭 오늘의 프로그램",
            "faq_route": "👶 연령별 동선",
            "faq_exhibits": "🧩 전시관 안내",
            "quick_menu": "⚡ 빠른 메뉴(모바일 추천)",
            "quick_floor": "🏢 층별",
            "quick_route": "👶 동선",
            "quick_programs": "🎭 프로그램",
            "quick_exhibits": "🧩 전시관",
            "tab_guide": "🏙️ 과학관 안내",
            "tab_learning": "🥰 또만나 놀이터",
            "chat_placeholder": "질문을 입력해주세요!",
            "mode_lang_changed": "사용자 모드/언어 설정이 변경되었어요. 다음 답변부터 새 설정으로 안내할게요.",
            "program_explain": "전시해설",
            "program_show": "과학쇼",
            "program_planet": "천체투영관",
            "program_light": "빛놀이터",
            "tts_listen": "🔊 음성으로 듣기",
            "tts_rendering": "음성으로 변환 중...",
            "record_start": "녹음 시작",
            "spinner_rule": "(규칙기반)확인 중입니다...",
            "spinner_llm": "(LLM)잠시만 기다려 주세요...",
            "debug_tool_calls": "🔍 디버깅 정보(도구 호출 내역)",
            "debug_tool_calls_after": "🔍 디버깅 정보 (도구 호출 내역)",
            "reservation_person": "개인 예약",
            "reservation_group": "단체 예약",
            "reservation_edu": "교육 예약",
            "debug_section": "🧪 디버그",
            "debug_show_ko": "한국어 원문 보기(디버그)",
            "debug_backtranslate": "역번역 결과 보기(디버그)",
        },
        "English": {
            "page_title": "NCSC AI Guide",
            "app_title": "NCSC AI Guide 🤖",
            "sidebar_title": "⚙️ Guide Mode",
            "user_mode_label": "Visitor type:",
            "user_mode_child": "Child",
            "user_mode_adult": "Teen/Adult",
            "language_label": "Language",
            "voice_section": "🎤 Voice",
            "voice_in": "Enable voice input",
            "voice_out": "Enable voice output",
            "voice_ask": "### 🎤 Ask by voice",
            "voice_rec_fail": "Voice recognition failed. Please try again.",
            "refresh": "Reset chat 🔄",
            "faq_header": "#### ❓ FAQ",
            "faq_floor": "🏢 Floor guide",
            "faq_programs": "🎭 Today's programs",
            "faq_route": "👶 Recommended route",
            "faq_exhibits": "🧩 Exhibits",
            "quick_menu": "⚡ Quick menu (mobile)",
            "quick_floor": "🏢 Floors",
            "quick_route": "👶 Route",
            "quick_programs": "🎭 Programs",
            "quick_exhibits": "🧩 Exhibits",
            "tab_guide": "🏙️ Guide",
            "tab_learning": "🥰 Playground",
            "chat_placeholder": "Type your question",
            "mode_lang_changed": "Your mode/language has changed. I will answer with the new settings from now on.",
            "program_explain": "Exhibit tour",
            "program_show": "Science show",
            "program_planet": "Planetarium",
            "program_light": "Light play",
            "tts_listen": "🔊 Listen",
            "tts_rendering": "Generating audio...",
            "record_start": "Start recording",
            "spinner_rule": "Checking (rule-based)...",
            "spinner_llm": "Please wait (LLM)...",
            "debug_tool_calls": "🔍 Debug (tool calls)",
            "debug_tool_calls_after": "🔍 Debug (tool calls)",
            "reservation_person": "Personal",
            "reservation_group": "Group",
            "reservation_edu": "Education",
            "debug_section": "🧪 Debug",
            "debug_show_ko": "Show Korean original (debug)",
            "debug_backtranslate": "Show back-translation (debug)",
        },
        "日本語": {
            "page_title": "AIガイド",
            "app_title": "AIガイド 🤖",
            "sidebar_title": "⚙️ 案内モード",
            "user_mode_label": "利用者:",
            "user_mode_child": "こども",
            "user_mode_adult": "中高生/大人",
            "language_label": "言語/Language",
            "voice_section": "🎤 音声",
            "voice_in": "音声入力を有効化",
            "voice_out": "音声出力を有効化",
            "voice_ask": "### 🎤 音声で質問",
            "voice_rec_fail": "音声認識に失敗しました。もう一度お試しください。",
            "refresh": "会話をリセット 🔄",
            "faq_header": "#### ❓ よくある質問",
            "faq_floor": "🏢 フロア案内",
            "faq_programs": "🎭 本日のプログラム",
            "faq_route": "👶 おすすめ動線",
            "faq_exhibits": "🧩 展示案内",
            "quick_menu": "⚡ クイックメニュー(モバイル)",
            "quick_floor": "🏢 フロア",
            "quick_route": "👶 動線",
            "quick_programs": "🎭 プログラム",
            "quick_exhibits": "🧩 展示",
            "tab_guide": "🏙️ 科学館案内",
            "tab_learning": "🥰 あそび",
            "chat_placeholder": "質問を入力してください",
            "mode_lang_changed": "モード/言語が変更されました。次の回答から新しい設定で案内します。",
            "program_explain": "展示解説",
            "program_show": "サイエンスショー",
            "program_planet": "プラネタリウム",
            "program_light": "光あそび",
            "tts_listen": "🔊 音声で聞く",
            "tts_rendering": "音声を生成中...",
            "record_start": "録音開始",
            "spinner_rule": "（ルール）確認中...",
            "spinner_llm": "少々お待ちください（LLM）...",
            "debug_tool_calls": "🔍 デバッグ（ツール呼び出し）",
            "debug_tool_calls_after": "🔍 デバッグ（ツール呼び出し）",
            "reservation_person": "個人予約",
            "reservation_group": "団体予約",
            "reservation_edu": "教育予約",
            "debug_section": "🧪 デバッグ",
            "debug_show_ko": "韓国語の原文を表示（デバッグ）",
            "debug_backtranslate": "逆翻訳を表示（デバッグ）",
        },
        "中文": {
            "page_title": "AI 导览",
            "app_title": "AI 导览 🤖",
            "sidebar_title": "⚙️ 导览模式",
            "user_mode_label": "访客类型:",
            "user_mode_child": "儿童",
            "user_mode_adult": "青少年/成人",
            "language_label": "语言/Language",
            "voice_section": "🎤 语音",
            "voice_in": "启用语音输入",
            "voice_out": "启用语音输出",
            "voice_ask": "### 🎤 语音提问",
            "voice_rec_fail": "语音识别失败，请重试。",
            "refresh": "重置对话 🔄",
            "faq_header": "#### ❓ 常见问题",
            "faq_floor": "🏢 楼层导览",
            "faq_programs": "🎭 今日节目",
            "faq_route": "👶 推荐动线",
            "faq_exhibits": "🧩 展馆导览",
            "quick_menu": "⚡ 快捷菜单(移动端)",
            "quick_floor": "🏢 楼层",
            "quick_route": "👶 动线",
            "quick_programs": "🎭 节目",
            "quick_exhibits": "🧩 展馆",
            "tab_guide": "🏙️ 参观导览",
            "tab_learning": "🥰 游乐",
            "chat_placeholder": "请输入你的问题",
            "mode_lang_changed": "模式/语言已更改。从下一次回答开始将使用新设置。",
            "program_explain": "展览讲解",
            "program_show": "科学秀",
            "program_planet": "天象馆",
            "program_light": "光乐园",
            "tts_listen": "🔊 收听",
            "tts_rendering": "正在生成语音...",
            "record_start": "开始录音",
            "spinner_rule": "正在确认（规则）...",
            "spinner_llm": "请稍候（LLM）...",
            "debug_tool_calls": "🔍 调试（工具调用）",
            "debug_tool_calls_after": "🔍 调试（工具调用）",
            "reservation_person": "个人预约",
            "reservation_group": "团体预约",
            "reservation_edu": "教育预约",
            "debug_section": "🧪 调试",
            "debug_show_ko": "显示韩文原文（调试）",
            "debug_backtranslate": "显示回译结果（调试）",
        },
    }

    def t(key: str) -> str:
        lang = st.session_state.get("language_mode") or prev_language_for_page
        return ui_text.get(lang, ui_text["한국어"]).get(key, ui_text["한국어"].get(key, key))

    st.set_page_config(
        page_title=ui_text.get(st.session_state.get("language_mode"), ui_text["한국어"])["page_title"],
        page_icon="🐣",
        layout="centered",
    )

    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "")

    # RAG 시스템 로드
    vector_db = load_rag_db()

    with st.sidebar:
        st.header(t("sidebar_title"))

        user_mode_display = st.selectbox(
            t("user_mode_label"),
            options=[t("user_mode_child"), t("user_mode_adult")],
            index=1,
        )
        user_mode = "어린이" if user_mode_display == t("user_mode_child") else "청소년/성인"
        
        # Language mode selection
        language_mode = st.selectbox(
            t("language_label"),
            options=["한국어", "English", "日本語", "中文"],
            index=["한국어", "English", "日本語", "中文"].index(st.session_state.get("language_mode", "한국어")),
        )

        if language_mode != st.session_state.get("language_mode"):
            st.session_state["language_mode"] = language_mode
            st.session_state["thread_id"] = uuid.uuid4().hex
            st.session_state["mode_language_changed"] = True
            st.session_state["messages"] = []
            if "tts_cache" in st.session_state:
                del st.session_state["tts_cache"]
            st.rerun()
        
        # Voice features toggle
        st.markdown("---")
        st.subheader(t("voice_section"))
        enable_voice_input = st.checkbox(t("voice_in"), value=True)
        enable_voice_output = st.checkbox(t("voice_out"), value=True)

        st.markdown("---")
        st.subheader(t("debug_section"))
        debug_show_ko = st.checkbox(t("debug_show_ko"), value=False)
        debug_backtranslate = st.checkbox(t("debug_backtranslate"), value=False)

        auto_clear_on_change = True

        prev_user_mode = st.session_state.get("_prev_user_mode")
        prev_language_mode = st.session_state.get("_prev_language_mode")
        if (prev_user_mode is not None and prev_user_mode != user_mode) or (
            prev_language_mode is not None and prev_language_mode != language_mode
        ):
            st.session_state["thread_id"] = uuid.uuid4().hex
            st.session_state["mode_language_changed"] = True
            if auto_clear_on_change:
                st.session_state["messages"] = []
                if "tts_cache" in st.session_state:
                    del st.session_state["tts_cache"]
        st.session_state["_prev_user_mode"] = user_mode
        st.session_state["_prev_language_mode"] = language_mode

        st.markdown("---")
        st.markdown(ui_text.get(language_mode, ui_text["한국어"])["faq_header"])
        s1, s2 = st.columns(2)
        with s1:
            if st.button(ui_text.get(language_mode, ui_text["한국어"])["faq_floor"], key="faq_floor_sidebar"):
                st.session_state["pending_user_input"] = "층별 안내"
                st.rerun()
            if st.button(ui_text.get(language_mode, ui_text["한국어"])["faq_programs"], key="faq_programs_sidebar"):
                st.session_state["pending_user_input"] = "오늘의 프로그램"
                st.session_state["pending_ui_program_buttons"] = True
                st.rerun()
        with s2:
            if st.button(ui_text.get(language_mode, ui_text["한국어"])["faq_route"], key="faq_route_sidebar"):
                st.session_state["pending_user_input"] = "연령별 맞춤 동선 추천"
                st.rerun()
            if st.button(ui_text.get(language_mode, ui_text["한국어"])["faq_exhibits"], key="faq_exhibits_sidebar"):
                st.session_state["pending_user_input"] = "전시관 안내"
                st.rerun()

        if enable_voice_input:
            st.markdown(t("voice_ask"))

            if "audio_recorder_key" not in st.session_state:
                st.session_state["audio_recorder_key"] = uuid.uuid4().hex

            audio_bytes = audio_recorder(
                text=ui_text.get(language_mode, ui_text["한국어"])["record_start"],
                recording_color="#e74c3c",
                neutral_color="#3498db",
                icon_name="microphone",
                icon_size="2x",
                key=st.session_state["audio_recorder_key"],
                # 모바일에서 1초 컷 방지:
                # - pause_threshold: 무음 감지 자동종료까지의 시간 (기본 2.0s, 모바일에선 너무 짧음)
                # - energy_threshold: 침묵 판단 임계값 (낮출수록 작은 소리도 발화로 인식)
                # - sample_rate: 모바일 호환성 위해 41100 → 16000으로 낮춤 (Whisper에 충분)
                pause_threshold=4.0,
                energy_threshold=(-1.0, 1.0),
                sample_rate=16000,
            )

            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")

                audio_sig = str(hash(audio_bytes))
                already_processed = st.session_state.get("_last_audio_sig") == audio_sig

                if not already_processed:
                    st.session_state["_last_audio_sig"] = audio_sig
                    with st.spinner("음성을 텍스트로 변환 중..."):
                        recognized = speech_to_text(audio_bytes)
                        if recognized:
                            st.session_state["pending_user_input"] = recognized
                            st.session_state["audio_recorder_key"] = uuid.uuid4().hex
                            st.rerun()
                        else:
                            st.error(t("voice_rec_fail"))
                            st.session_state["audio_recorder_key"] = uuid.uuid4().hex
                            st.rerun()
        
        if st.button(t("refresh")):
            st.session_state.messages = []
            st.session_state.thread_id = uuid.uuid4().hex
            st.session_state.debug_logs = []
            if "tts_cache" in st.session_state:
                del st.session_state["tts_cache"]
            st.rerun()

    st.title(ui_text.get(st.session_state.get("language_mode"), ui_text["한국어"])["app_title"])

    # 🎨 마스코트 워터마크 배경 (모든 모드에서 표시)
    _render_mascot_animation()

    intro_text_map = {
        "한국어": (
            "과학이 기쁨이 되는 어린이과학관에 오신 여러분들 환영합니다!\n\n"
            "저는 **국립어린이과학관 AI 가이드**예요. 어린이과학관에 대해서 궁금한 점이 있나요?\n"
            "아래 채팅으로 질문하거나, 왼쪽 사이드바에서 **음성으로 질문**해도 좋아요.\n"
            "필요하면 전시/프로그램/이용 안내까지 제가 차근차근 도와드릴게요~"
        ),
        "English": (
            "Welcome to the National Children's Science Center!\n\n"
            "I'm your **AI Guide**. Have any questions about exhibits, programs, or visiting tips?\n"
            "Ask in the chat below, or use **voice input** from the left sidebar."
        ),
        "日本語": (
            "国立こども科学館へようこそ！\n\n"
            "私は **AIガイド** です。展示・プログラム・利用案内など、気になることはありますか？\n"
            "下のチャット、または左のサイドバーの **音声入力** で質問できます。"
        ),
        "中文": (
            "欢迎来到国立儿童科学馆！\n\n"
            "我是你的 **AI 导览**。关于展览、节目或参观方式，有什么想了解的吗？\n"
            "你可以在下方聊天提问，也可以在左侧栏使用 **语音提问**。"
        ),
    }
    language_mode = st.session_state.get("language_mode", "한국어")
    
    # Enhanced app description
    intro_enhanced = {
        "한국어": (
            "과학이 기쁨이 되는 어린이과학관에 오신 여러분들 환영합니다! 🎉\n\n"
            "저는 **국립어린이과학관 AI 가이드**예요. 궁금한 게 있으면 언제든 물어보세요!\n\n"
            "**💬 이런 걸 도와드려요:**\n"
            "- 🏢 층별 안내 및 전시관 소개\n"
            "- 🎭 오늘의 프로그램 및 천체투영관 시간표\n"
            "- 🎫 예약 방법, 관람료, 운영시간\n"
            "- 🚇 오시는 길 (지하철/버스 경로 안내)\n"
            "- 👶 연령별 맞춤 동선 추천\n\n"
            "**📝 사용 방법:**\n"
            "- 아래 채팅창에 질문을 입력하거나\n"
            "- 왼쪽 사이드바에서 🎤 음성으로 질문하세요\n"
            "- 자주 묻는 질문 버튼을 눌러도 돼요!\n\n"
            "편하게 질문해주세요~ 😊"
        ),
        "English": (
            "Welcome to the National Children's Science Center! 🎉\n\n"
            "I'm your **AI Guide**. Feel free to ask me anything!\n\n"
            "**💬 I can help with:**\n"
            "- 🏢 Floor guide & exhibit info\n"
            "- 🎭 Today's programs & planetarium schedule\n"
            "- 🎫 Reservations, fees, hours\n"
            "- 🚇 Directions (subway/bus routes)\n"
            "- 👶 Age-appropriate tour routes\n\n"
            "**📝 How to use:**\n"
            "- Type in the chat below, or\n"
            "- Use 🎤 voice input from the sidebar\n"
            "- Click FAQ buttons!\n\n"
            "Ask away! 😊"
        ),
        "日本語": (
            "国立こども科学館へようこそ！🎉\n\n"
            "私は **AIガイド** です。何でも聞いてください！\n\n"
            "**💬 お手伝いできること:**\n"
            "- 🏢 フロア案内・展示情報\n"
            "- 🎭 本日のプログラム・プラネタリウム時間\n"
            "- 🎫 予約方法・料金・営業時間\n"
            "- 🚇 アクセス（地下鉄・バス）\n"
            "- 👶 年齢別おすすめ動線\n\n"
            "**📝 使い方:**\n"
            "- 下のチャットで質問、または\n"
            "- サイドバーで 🎤 音声質問\n"
            "- よくある質問ボタンをクリック！\n\n"
            "お気軽にどうぞ！😊"
        ),
        "中文": (
            "欢迎来到国立儿童科学馆！🎉\n\n"
            "我是你的 **AI 导览**。有问题随时问我！\n\n"
            "**💬 我能帮你:**\n"
            "- 🏢 楼层导览·展馆介绍\n"
            "- 🎭 今日节目·天文馆时间表\n"
            "- 🎫 预约方法·门票·营业时间\n"
            "- 🚇 交通指南（地铁/公交）\n"
            "- 👶 年龄推荐路线\n\n"
            "**📝 使用方法:**\n"
            "- 在下方聊天框输入问题，或\n"
            "- 在侧边栏使用 🎤 语音提问\n"
            "- 点击常见问题按钮！\n\n"
            "欢迎提问！😊"
        ),
    }
    st.markdown(intro_enhanced.get(language_mode, intro_enhanced["한국어"]))

    if st.session_state.get("mode_language_changed"):
        st.info(ui_text.get(language_mode, ui_text["한국어"])["mode_lang_changed"])
        del st.session_state["mode_language_changed"]

    with st.expander(ui_text.get(language_mode, ui_text["한국어"])["quick_menu"], expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button(ui_text.get(language_mode, ui_text["한국어"])["quick_floor"], key="quick_floor"):
                st.session_state["pending_user_input"] = "층별 안내"
                st.rerun()
        with c2:
            if st.button(ui_text.get(language_mode, ui_text["한국어"])["quick_route"], key="quick_route"):
                st.session_state["pending_user_input"] = "연령별 맞춤 동선 추천"
                st.rerun()
        with c3:
            if st.button(ui_text.get(language_mode, ui_text["한국어"])["quick_programs"], key="quick_programs"):
                st.session_state["pending_user_input"] = "오늘의 프로그램"
                st.session_state["pending_ui_program_buttons"] = True
                st.rerun()
        with c4:
            if st.button(ui_text.get(language_mode, ui_text["한국어"])["quick_exhibits"], key="quick_exhibits"):
                st.session_state["pending_user_input"] = "전시관 안내"
                st.rerun()
    
    # Tab navigation
    tab1, tab2 = st.tabs([
        ui_text.get(language_mode, ui_text["한국어"])["tab_guide"],
        ui_text.get(language_mode, ui_text["한국어"])["tab_learning"],
    ])
    
    with tab1:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "thread_id" not in st.session_state:
            st.session_state.thread_id = uuid.uuid4().hex
        if "debug_logs" not in st.session_state:
            st.session_state.debug_logs = []
        if "tts_cache" not in st.session_state:
            st.session_state.tts_cache = {}

        system_prompt = get_dynamic_prompt(user_mode, language_mode)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        memory = MemorySaver()
        agent = create_react_agent(
            model=llm,
            tools=get_tools(),
            checkpointer=memory,
        )

        def render_tts_for_answer(answer_text: str):
            if not enable_voice_output or not answer_text:
                return
            lang_code = get_language_code(language_mode)
            tts_ns = get_tts_cache_namespace(language=lang_code)

            tts_text = answer_text
            if len(tts_text) > 1200:
                tts_text = tts_text[:1200]

            cache_key = f"{language_mode}::{tts_ns}::" + str(hash(tts_text))
            if cache_key not in st.session_state.tts_cache:
                with st.spinner(ui_text.get(language_mode, ui_text["한국어"])["tts_rendering"]):
                    audio_bytes = text_to_speech(tts_text, language=lang_code)
                    if audio_bytes:
                        st.session_state.tts_cache[cache_key] = audio_bytes
            audio_bytes = st.session_state.tts_cache.get(cache_key)
            if audio_bytes:
                if st.button(ui_text.get(language_mode, ui_text["한국어"])["tts_listen"], key=f"tts_play_inline_{cache_key}"):
                    autoplay_audio(audio_bytes)
                st.audio(audio_bytes, format="audio/mp3")

        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "debug":
                with st.expander(ui_text.get(language_mode, ui_text["한국어"])["debug_tool_calls"]):
                    with st.container(height=400):
                        st.text(msg["content"])
            else:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

                    # 디버그: 어시스턴트 답변에 KO 원문 / 역번역 캡션 (외국어 모드 전용)
                    if msg["role"] == "assistant" and language_mode != "한국어" and msg.get("content"):
                        if debug_show_ko and msg.get("ko_original"):
                            st.caption(f"KO: {msg['ko_original']}")
                        if debug_backtranslate:
                            bt = _backtranslate_to_korean_cached(msg["content"], language_mode)
                            if bt:
                                st.caption(f"BT: {bt}")

                    if msg.get("ui") == "program_buttons":
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            if st.button(ui_text.get(language_mode, ui_text["한국어"])["program_explain"], key=f"prog_explain_{i}"):
                                st.session_state["pending_user_input"] = "전시해설 자세히 알려줘"
                                st.rerun()
                        with col2:
                            if st.button(ui_text.get(language_mode, ui_text["한국어"])["program_show"], key=f"prog_show_{i}"):
                                st.session_state["pending_user_input"] = "과학쇼 자세히 알려줘"
                                st.rerun()
                        with col3:
                            if st.button(ui_text.get(language_mode, ui_text["한국어"])["program_planet"], key=f"prog_planet_{i}"):
                                st.session_state["pending_user_input"] = "천체투영관 자세히 알려줘"
                                st.rerun()
                        with col4:
                            if st.button(ui_text.get(language_mode, ui_text["한국어"])["program_light"], key=f"prog_light_{i}"):
                                st.session_state["pending_user_input"] = "빛놀이터 자세히 알려줘"
                                st.rerun()

                    if msg.get("ui") == "reservation_links":
                        col1, col2, col3 = st.columns(3)
                        if hasattr(st, "link_button"):
                            with col1:
                                st.link_button(ui_text.get(language_mode, ui_text["한국어"])["reservation_person"], "https://www.csc.go.kr/new1/reservation/reservation_person.jsp")
                            with col2:
                                st.link_button(ui_text.get(language_mode, ui_text["한국어"])["reservation_group"], "https://www.csc.go.kr/new1/reservation/reservation_group.jsp")
                            with col3:
                                st.link_button(ui_text.get(language_mode, ui_text["한국어"])["reservation_edu"], "https://www.csc.go.kr/new1/reservation/education_creation.jsp")
                        else:
                            with col1:
                                st.markdown(f"[{ui_text.get(language_mode, ui_text['한국어'])['reservation_person']}](https://www.csc.go.kr/new1/reservation/reservation_person.jsp)")
                            with col2:
                                st.markdown(f"[{ui_text.get(language_mode, ui_text['한국어'])['reservation_group']}](https://www.csc.go.kr/new1/reservation/reservation_group.jsp)")
                            with col3:
                                st.markdown(f"[{ui_text.get(language_mode, ui_text['한국어'])['reservation_edu']}](https://www.csc.go.kr/new1/reservation/education_creation.jsp)")

                    if enable_voice_output and msg["role"] == "assistant" and msg.get("content"):
                        lang_code = get_language_code(language_mode)
                        tts_ns = get_tts_cache_namespace(language=lang_code)

                        tts_text = msg["content"]
                        if len(tts_text) > 1200:
                            tts_text = tts_text[:1200]

                        cache_key = f"{language_mode}::{tts_ns}::" + str(hash(tts_text))
                        if cache_key not in st.session_state.tts_cache:
                            with st.spinner(ui_text.get(language_mode, ui_text["한국어"])["tts_rendering"]):
                                audio_bytes = text_to_speech(tts_text, language=lang_code)
                                if audio_bytes:
                                    st.session_state.tts_cache[cache_key] = audio_bytes

                        audio_bytes = st.session_state.tts_cache.get(cache_key)
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/mp3")

                            if st.button(ui_text.get(language_mode, ui_text["한국어"])["tts_listen"], key=f"tts_play_msg_{i}_{cache_key}"):
                                autoplay_audio(audio_bytes)

        user_input = None

        if "pending_user_input" in st.session_state and st.session_state.get("pending_user_input"):
            user_input = st.session_state.get("pending_user_input")
            del st.session_state["pending_user_input"]
        
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            intent = route_intent(user_input)
            lowered_input = user_input.lower()
            if any(token in lowered_input for token in ["예약", "예매", "방문신청", "방문 신청", "단체예약", "개인예약", "교육예약", "입장권", "qr", "정원", "1600"]):
                st.session_state["pending_ui_reservation_links"] = True
            
            with st.chat_message("assistant"):
                if intent in ["notice", "basic"]:
                    # 규칙 기반 엔진 동작 (RAG/LLM 미사용, 속도 최적화)
                    # answer_rule_based_localized: 정적 사전 번역 우선 → 폴백 LLM 번역
                    with st.spinner(ui_text.get(language_mode, ui_text["한국어"])["spinner_rule"]):
                        answer, ko_original = answer_rule_based_localized(
                            intent, user_input, user_mode, language_mode
                        )
                    if language_mode == "한국어":
                        ko_original = ""
                    st.markdown(answer)
                    if language_mode != "한국어" and debug_show_ko and ko_original:
                        st.caption(f"KO: {ko_original}")
                    if language_mode != "한국어" and debug_backtranslate:
                        bt = _backtranslate_to_korean_cached(answer, language_mode)
                        if bt:
                            st.caption(f"BT: {bt}")
                    rule_sources = []
                    lowered = user_input.lower()
                    if intent == "notice":
                        rule_sources = [CSC_URLS.get("공지사항")]
                    else:
                        if any(k in lowered for k in ["오시는길", "오는길", "교통", "길찾기", "주소", "위치"]):
                            rule_sources = [CSC_URLS.get("오시는길")]
                        elif any(k in lowered for k in ["예약", "예매", "단체", "개인", "교육"]):
                            rule_sources = [CSC_URLS.get("예약안내"), CSC_URLS.get("개인예약"), CSC_URLS.get("단체예약"), CSC_URLS.get("교육예약")]
                        elif any(k in lowered for k in ["천체투영관"]):
                            rule_sources = [CSC_URLS.get("천체투영관")]
                        else:
                            rule_sources = [CSC_URLS.get("이용안내")]
                    rule_sources = [s for s in dict.fromkeys([s for s in rule_sources if s])]
                    render_source_buttons(rule_sources, language_mode=language_mode)
                    render_tts_for_answer(answer)
                else:
                    # LLM + RAG + Crawling 엔진 동작
                    with st.spinner(ui_text.get(language_mode, ui_text["한국어"])['spinner_llm']):
                        if st.session_state.get("directions_origin"):
                            origin = st.session_state.get("directions_origin")
                            del st.session_state["directions_origin"]
                            user_input = (
                                f"출발지: {origin}\n"
                                "목적지: 국립어린이과학관(국립어린이과학관, 서울 종로구 창경궁로 215)\n"
                                "요청: 대중교통(지하철/버스) 기준으로 가장 쉬운 경로를 단계별로 자세하고 친절하게 안내해줘. "
                                "출입구/도보 이동/환승 포인트가 있으면 같이 알려줘. "
                                "마지막에 노선/출입구는 변동될 수 있으니 공식 홈페이지(www.csc.go.kr) '오시는 길' 확인과 02-3668-1500 문의를 덧붙여줘."
                            )
                        # FAISS RAG에서 관련 정보 사전 검색하여 컨텍스트 주입
                        retrieved_docs = vector_db.similarity_search(user_input, k=3)
                        rag_context = "\n\n".join([f"[{doc.metadata.get('source', 'N/A')}]\n{doc.page_content}" for doc in retrieved_docs])
                        rag_sources = [doc.metadata.get("source", "N/A") for doc in retrieved_docs if getattr(doc, "metadata", None)]
                        rag_sources = [s for s in dict.fromkeys([s for s in rag_sources if s])]
                        
                        # 시스템 프롬프트와 RAG 컨텍스트를 시스템 메시지로 추가
                        config = {"configurable": {"thread_id": st.session_state.thread_id}}
                        # 외국어 모드에서 FAQ 트리거가 한국어일 경우에도 LLM이 반드시 대상 언어로 답하도록 강제 프리픽스 추가
                        llm_user_input = user_input
                        if language_mode != "한국어":
                            _lang_override = {
                                "English": "[REQUIRED OUTPUT LANGUAGE: English] You MUST answer ENTIRELY in English, even though the question above may be in Korean. Translate place names using the official glossary in the system prompt (e.g., AI놀이터 → AI Zone). Do NOT output Korean text.",
                                "日本語": "[出力言語指定: 日本語] 上の質問が韓国語であっても、必ず日本語だけで答えてください。場所名はシステムプロンプトのグロッサリーに従い、「日本語名称 (English Official Name)」の形式で記してください（例: 考えるゾーン (Thinking Zone)）。韓国語文字をそのまま出力しないこと。",
                                "中文": "[输出语言要求: 中文] 即使以上问题是韩语，你也必须完全用中文回答。地点名称请依照系统提示词中的词汇表，以“中文名称 (English Official Name)”的格式书写（例：思考区 (Thinking Zone)）。不要直接输出韩文。",
                            }.get(language_mode, "")
                            if _lang_override:
                                llm_user_input = f"{user_input}\n\n---\n{_lang_override}"
                        messages = [
                            {"role": "system", "content": f"{system_prompt}\n\n[RAG 배경지식]\n{rag_context}"},
                            {"role": "user", "content": llm_user_input}
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
                    if language_mode != "한국어" and debug_backtranslate:
                        bt = _backtranslate_to_korean_cached(answer, language_mode)
                        if bt:
                            st.caption(f"BT: {bt}")
                    render_source_buttons(rag_sources, language_mode=language_mode)
                    render_tts_for_answer(answer)
                    
                    # 디버깅 정보 표시 (답변 뒤)
                    if debug_info.strip():
                        with st.expander(ui_text.get(language_mode, ui_text["한국어"])["debug_tool_calls_after"]):
                            with st.container(height=400):
                                st.text(debug_info)
                        st.session_state.messages.append({"role": "debug", "content": debug_info})

            assistant_msg = {"role": "assistant", "content": answer}
            assistant_msg["tts_autoplayed"] = False
            # 디버그용 KO 원문 캐시 (rule-based 경로에서만 채워짐)
            if intent in ["notice", "basic"] and language_mode != "한국어":
                try:
                    if ko_original:
                        assistant_msg["ko_original"] = ko_original
                except NameError:
                    pass
            if st.session_state.get("pending_ui_program_buttons"):
                assistant_msg["ui"] = "program_buttons"
                del st.session_state["pending_ui_program_buttons"]
            if st.session_state.get("pending_ui_reservation_links"):
                assistant_msg["ui"] = "reservation_links"
                del st.session_state["pending_ui_reservation_links"]
            st.session_state.messages.append(assistant_msg)

            st.session_state["scroll_to_bottom"] = True

        if st.session_state.get("scroll_to_bottom"):
            scroll_html = """
            <script>
              const parentDoc = window.parent.document;
              const main = parentDoc.querySelector('section.main') || parentDoc.body;
              if (main && main.scrollHeight) {
                main.scrollTo({ top: main.scrollHeight, behavior: 'smooth' });
              } else {
                window.parent.scrollTo({ top: parentDoc.documentElement.scrollHeight, behavior: 'smooth' });
              }
            </script>
            """
            data_url = "data:text/html," + urllib.parse.quote(scroll_html)
            st.iframe(data_url, height=1)
            del st.session_state["scroll_to_bottom"]
            
            # Voice output is rendered alongside assistant messages above (stable across reruns)
    
    with tab2:
        # Post-visit learning system
        render_post_visit_learning(
            vector_db,
            st.session_state.get("language_mode", "한국어"),
            debug_show_korean=debug_show_ko,
            debug_backtranslate=debug_backtranslate,
        )
    
    # Chat input at page bottom (outside tabs for stable positioning)
    typed_input = st.chat_input(
        ui_text.get(language_mode, ui_text["한국어"])["chat_placeholder"],
        key="main_chat_input"
    )
    if typed_input and not st.session_state.get("pending_user_input"):
        st.session_state["pending_user_input"] = typed_input
        st.rerun()

if __name__ == "__main__":
    main()
