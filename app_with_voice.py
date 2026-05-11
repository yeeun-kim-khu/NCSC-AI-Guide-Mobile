# app_with_voice.py - Museum guide with voice features
import os
import uuid
import streamlit as st
import streamlit.components.v1 as components
import urllib.parse
import warnings
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from audio_recorder_streamlit import audio_recorder
import base64
import json
import time
from datetime import datetime, timezone, timedelta

from core import get_tools, route_intent, answer_rule_based, answer_rule_based_localized, get_dynamic_prompt, render_source_buttons, initialize_vector_db, CSC_URLS, translate_answer_cached
from voice import speech_to_text, text_to_speech, get_language_code, autoplay_audio, get_tts_cache_namespace
from learning import render_post_visit_learning, _backtranslate_to_korean_cached

# Google Form URLs for feedback per language
# Tip: You can either create separate forms per language, or use one form with Google Forms' built-in translation.
# To get a link: Open your Google Form → Send → copy the "Link" tab URL
GOOGLE_FORM_URLS = {
    "한국어": "https://forms.gle/UvRfnMEwjUEZgFJJ8",
    "English": "https://forms.gle/UvRfnMEwjUEZgFJJ8",
    "日本語": "https://forms.gle/UvRfnMEwjUEZgFJJ8",
    "中文": "https://forms.gle/UvRfnMEwjUEZgFJJ8",
}

# RAG loading with session state persistence
@st.cache_resource(ttl=3600)
def load_rag_db():
    """Load RAG database with caching"""
    from core import initialize_vector_db
    with st.spinner("RAG database loading..."):
        vector_db = initialize_vector_db()
    return vector_db


# ---- Google Analytics 4 ----
GA_MEASUREMENT_ID = "G-7VS14G0T7P"

# Google Analytics 코드 삽입 (스트림릿 iframe 우회 적용)
ga_code = f"""
<script async src="https://www.googletagmanager.com/gtag/js?id={GA_MEASUREMENT_ID}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());

  // 구글에게 진짜 앱 주소와 이름을 강제로 인식시킵니다
  gtag('config', '{GA_MEASUREMENT_ID}', {{
    page_location: 'https://ncsc-ai-guide-mobile.streamlit.app/',
    page_title: '국립어린이과학관 AI 가이드'
  }});
</script>
"""
components.html(ga_code, width=1, height=1)


def _track_ga_event(event_name: str, params: dict | None = None) -> None:
    """Send a custom event to GA4."""
    safe_params = dict(params or {})
    for key in list(safe_params.keys()):
        if key.lower() in ("user_id", "email", "name", "content", "message", "query"):
            del safe_params[key]
    params_json = json.dumps(safe_params, ensure_ascii=False)
    script = f"""
    <script>
      if (typeof gtag !== 'undefined') {{
        gtag('event', '{event_name}', {params_json});
      }}
    </script>
    """
    st.markdown(script, unsafe_allow_html=True)


def _queue_ga_event(event_name: str, params: dict | None = None) -> None:
    """Queue a GA event to be sent on the next render (safe before st.rerun)."""
    if "_ga_event_queue" not in st.session_state:
        st.session_state._ga_event_queue = []
    st.session_state._ga_event_queue.append({"name": event_name, "params": dict(params or {})})


def _flush_ga_events() -> None:
    """Send all queued GA events. Call at the top of main()."""
    events = st.session_state.get("_ga_event_queue", [])
    for ev in events:
        _track_ga_event(ev["name"], ev["params"])
    st.session_state._ga_event_queue = []


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
**서비스 이용 전, 소중한 개인정보 보호를 위해 아래 내용을 꼭 확인해주세요!**

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

※ 서비스 개선을 위한 익명 통계 데이터(접속 시간, 클릭 등)는 수집될 수 있습니다.
"""

    def _ack() -> None:
        st.session_state["privacy_notice_acknowledged"] = True
        _queue_ga_event("privacy_consent", {"language": st.session_state.get("language_mode", "한국어")})
        st.rerun()

    # st.dialog (Streamlit 1.31+) 우선 사용
    if hasattr(st, "dialog"):
        @st.dialog("AI 가이드 이용 안내", width="large")
        def _privacy_dialog():
            st.markdown(notice_md)
            agreed = st.checkbox("위 내용을 확인했으며 이에 동의합니다.")
            if agreed and st.button("시작하기", type="primary", use_container_width=True):
                _ack()

        _privacy_dialog()
        # dialog가 닫히기 전엔 본문 렌더링 차단
        st.stop()
    else:
        # 폴백: 페이지 상단 카드 형태
        st.warning(notice_md)
        agreed = st.checkbox("위 내용을 확인했으며 이에 동의합니다.")
        if agreed and st.button("시작하기", type="primary"):
            _ack()
        st.stop()


def save_feedback(data: dict, language_mode: str, user_mode: str):
    kst = datetime.now(timezone.utc) + timedelta(hours=9)
    data["timestamp"] = kst.isoformat()
    data["language"] = language_mode
    data["mode"] = user_mode
    data["msg_count"] = len(st.session_state.get("messages", []))
    print(f"[FEEDBACK] {json.dumps(data, ensure_ascii=False)}")
    # Google Sheets 연동 (선택사항 — 아래 가이드 참고)
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        sheet_id = st.secrets.get("app", {}).get("feedback_sheet_id")
        if sheet_id:
            creds = Credentials.from_service_account_info(
                st.secrets["gcp_service_account"],
                scopes=["https://www.googleapis.com/auth/spreadsheets"],
            )
            gc = gspread.authorize(creds)
            sheet = gc.open_by_key(sheet_id).worksheet("feedback")
            row = [data.get("timestamp"), data.get("type"), data.get("language"), data.get("mode"), data.get("msg_count"), json.dumps(data, ensure_ascii=False)]
            sheet.append_row(row)
    except Exception:
        pass


def log_monitoring(intent: str, rule_based: bool, latency_ms: float, error: bool = False):
    """개발자용 모니터링 로그 (PII 없이 메타데이터만 기록)"""
    print(f"[MONITOR] intent={intent} rule_based={rule_based} latency_ms={latency_ms:.0f} error={error}")


GOOGLE_FORM_I18N = {
    "한국어": {
        "children_label": "💬 한마디 남기기 🎤",
        "children_msg": "소중한 의견을 들려주세요",
        "children_guardian": "* 어린이는 보호자와 함께 작성해 주세요",
        "parent_label": "💬 서비스 만족도 남기기",
        "parent_msg": "소중한 의견을 들려주세요",
        "btn_text": "📄 설문조사 참여!",
        "done_msg": "✅ 의견 감사합니다!",
    },
    "English": {
        "children_label": "💬 Leave a message 🎤",
        "children_msg": "We'd love to hear your thoughts",
        "children_guardian": "* Children, please fill this out with your parent or guardian",
        "parent_label": "💬 Service Feedback",
        "parent_msg": "We'd love to hear your thoughts",
        "btn_text": "📄 Take survey!",
        "done_msg": "✅ Thank you for your feedback!",
    },
    "日本語": {
        "children_label": "💬 メッセージを残す 🎤",
        "children_msg": "貴重なご意見をお聞かせください",
        "children_guardian": "* お子様は保護者の方と一緒にお書きください",
        "parent_label": "💬 サービスフィードバック",
        "parent_msg": "貴重なご意見をお聞かせください",
        "btn_text": "📄 アンケートに参加!",
        "done_msg": "✅ ご意見ありがとうございます!",
    },
    "中文": {
        "children_label": "💬 留言 🎤",
        "children_msg": "期待您的宝贵意见",
        "children_guardian": "* 儿童请在家长陪同下填写",
        "parent_label": "💬 服务反馈",
        "parent_msg": "期待您的宝贵意见",
        "btn_text": "📄 参与调查!",
        "done_msg": "✅ 感谢您的反馈!",
    },
}


def render_children_feedback(language_mode: str = "한국어", user_mode: str = "기본"):
    ft = GOOGLE_FORM_I18N.get(language_mode, GOOGLE_FORM_I18N["한국어"])
    st.caption(ft["children_msg"])
    st.caption(f"\n{ft['children_guardian']}")
    form_url = GOOGLE_FORM_URLS.get(language_mode, GOOGLE_FORM_URLS["한국어"])
    st.link_button(ft["btn_text"], form_url, use_container_width=True, type="primary")


def render_parent_feedback(language_mode: str = "한국어", user_mode: str = "기본"):
    ft = GOOGLE_FORM_I18N.get(language_mode, GOOGLE_FORM_I18N["한국어"])
    st.caption(ft["parent_msg"])
    form_url = GOOGLE_FORM_URLS.get(language_mode, GOOGLE_FORM_URLS["한국어"])
    st.link_button(ft["btn_text"], form_url, use_container_width=True, type="primary")


def main():
    warnings.filterwarnings("ignore", message=r".*create_react_agent has been moved to `langchain\.agents`\..*")

    # 첫 진입 시 개인정보 안내 팝업 (동의 전 본문 차단)
    _render_privacy_notice_gate()

    # Google Analytics 이벤트 플러시
    _flush_ga_events()

    if "language_mode" not in st.session_state:
        st.session_state["language_mode"] = "한국어"

    prev_language_for_page = st.session_state.get("language_mode") or "한국어"
    ui_text = {
        "한국어": {
            "page_title": "국립어린이과학관 AI 가이드",
            "app_title": "국립어린이과학관 AI 가이드",
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
            "refresh": "🔄 대화 새로고침",
            "refresh_hint": "사용 중 문제가 발생했다면?",
            "faq_header": "### ❓ 자주 묻는 질문",
            "faq_floor": "🏢 층별",
            "faq_programs": "🎭 프로그램",
            "faq_route": "👶 동선",
            "faq_exhibits": "🧩 전시관",
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
            "app_title": "NCSC AI Guide",
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
            "refresh": "🔄 Reset chat",
            "refresh_hint": "Start a fresh conversation",
            "faq_header": "### ❓ FAQ",
            "faq_floor": "🏢 Floors",
            "faq_programs": "🎭 Programs",
            "faq_route": "👶 Route",
            "faq_exhibits": "🧩 Exhibits",
            "quick_menu": "⚡ Quick menu (mobile)",
            "quick_floor": "🏢 Floors",
            "quick_route": "👶 Route",
            "quick_programs": "🎭 Programs",
            "quick_exhibits": "🧩 Exhibits",
            "tab_guide": "🏙️ Guide",
            "tab_learning": "🥰 Again Zone",
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
            "page_title": "国立子ども科学館 AIガイド",
            "app_title": "国立子ども科学館 AIガイド",
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
            "refresh": "🔄 会話をリセット",
            "refresh_hint": "最初からやり直す時",
            "faq_header": "### ❓ よくある質問",
            "faq_floor": "🏢 フロア",
            "faq_programs": "🎭 プログラム",
            "faq_route": "👶 動線",
            "faq_exhibits": "🧩 展示",
            "quick_menu": "⚡ クイックメニュー(モバイル)",
            "quick_floor": "🏢 フロア",
            "quick_route": "👶 動線",
            "quick_programs": "🎭 プログラム",
            "quick_exhibits": "🧩 展示",
            "tab_guide": "🏙️ 科学館案内",
            "tab_learning": "🥰 またねゾーン",
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
            "page_title": "国立儿童科学馆 AI 导览",
            "app_title": "国立儿童科学馆 AI 导览",
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
            "refresh": "🔄 重置对话",
            "refresh_hint": "重新开始对话时",
            "faq_header": "### ❓ 常见问题",
            "faq_floor": "🏢 楼层",
            "faq_programs": "🎭 节目",
            "faq_route": "👶 动线",
            "faq_exhibits": "🧩 展馆",
            "quick_menu": "⚡ 快捷菜单(移动端)",
            "quick_floor": "🏢 楼层",
            "quick_route": "👶 动线",
            "quick_programs": "🎭 节目",
            "quick_exhibits": "🧩 展馆",
            "tab_guide": "🏙️ 参观导览",
            "tab_learning": "🥰 再次乐园",
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

    # 실행 중 화면 흐려짐 방지
    st.markdown("""
    <style>
    .stApp.running { opacity: 1 !important; }
    .stApp.running * { opacity: 1 !important; }
    </style>
    """, unsafe_allow_html=True)

    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "")

    # RAG 시스템 로드
    vector_db = load_rag_db()

    with st.sidebar:
        st.subheader(t("sidebar_title"))

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
        
        # 음성 기능은 항상 활성화 (사이드바 UI 단순화)
        enable_voice_input = True
        enable_voice_output = True

        # 디버그 섹션 숨김 (출력만 비활성화, 변수는 False로 유지해 하위 코드 호환성 보장)
        # st.markdown("---")
        # st.subheader(t("debug_section"))
        # debug_show_ko = st.checkbox(t("debug_show_ko"), value=False)
        # debug_backtranslate = st.checkbox(t("debug_backtranslate"), value=False)
        debug_show_ko = False
        debug_backtranslate = False

        st.markdown("---")

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

        st.markdown(ui_text.get(language_mode, ui_text["한국어"])["faq_header"])
        s1, s2 = st.columns(2)
        # FAQ 질문 번역 매핑
        faq_inputs = {
            "한국어": {
                "floor": "층별 안내",
                "programs": "오늘의 프로그램",
                "route": "연령별 맞춤 동선 추천",
                "exhibits": "전시관 안내",
            },
            "English": {
                "floor": "Floor guide",
                "programs": "Today's programs",
                "route": "Recommended route by age",
                "exhibits": "Exhibition guide",
            },
            "日本語": {
                "floor": "フロア案内",
                "programs": "今日のプログラム",
                "route": "年齢別おすすめルート",
                "exhibits": "展示館案内",
            },
            "中文": {
                "floor": "楼层导览",
                "programs": "今日节目",
                "route": "按年龄推荐路线",
                "exhibits": "展馆导览",
            },
        }.get(language_mode, {
            "floor": "층별 안내",
            "programs": "오늘의 프로그램",
            "route": "연령별 맞춤 동선 추천",
            "exhibits": "전시관 안내",
        })

        with s1:
            if st.button(ui_text.get(language_mode, ui_text["한국어"])["faq_floor"], key="faq_floor_sidebar"):
                _queue_ga_event("faq_button_click", {"category": "floor", "language": language_mode})
                st.session_state["pending_user_input"] = faq_inputs["floor"]
                st.session_state["switch_to_guide_tab"] = True
                st.rerun()
            if st.button(ui_text.get(language_mode, ui_text["한국어"])["faq_programs"], key="faq_programs_sidebar"):
                _queue_ga_event("faq_button_click", {"category": "programs", "language": language_mode})
                st.session_state["pending_user_input"] = faq_inputs["programs"]
                st.session_state["pending_ui_program_buttons"] = True
                st.session_state["switch_to_guide_tab"] = True
                st.rerun()
        with s2:
            if st.button(ui_text.get(language_mode, ui_text["한국어"])["faq_route"], key="faq_route_sidebar"):
                _queue_ga_event("faq_button_click", {"category": "route", "language": language_mode})
                st.session_state["pending_user_input"] = faq_inputs["route"]
                st.session_state["switch_to_guide_tab"] = True
                st.rerun()
            if st.button(ui_text.get(language_mode, ui_text["한국어"])["faq_exhibits"], key="faq_exhibits_sidebar"):
                _queue_ga_event("faq_button_click", {"category": "exhibits", "language": language_mode})
                st.session_state["pending_user_input"] = faq_inputs["exhibits"]
                st.session_state["switch_to_guide_tab"] = True
                st.rerun()

        st.markdown("---")

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
                            _queue_ga_event("voice_input_used", {"language": language_mode})
                            st.session_state["pending_user_input"] = recognized
                            st.session_state["audio_recorder_key"] = uuid.uuid4().hex
                            st.rerun()
                        else:
                            st.error(t("voice_rec_fail"))
                            st.session_state["audio_recorder_key"] = uuid.uuid4().hex
                            st.rerun()

        st.markdown("---")
        st.caption(t("refresh_hint"))

        if st.button(t("refresh"), use_container_width=True, type="primary"):
            _queue_ga_event("chat_reset", {"language": language_mode, "user_mode": user_mode})
            # 세션 스테이트 전체 초기화
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        # 설문조사를 사이드바 맨 하단으로 이동 (대화 새로고침과 같은 그룹)
        if user_mode == "어린이":
            render_children_feedback(language_mode, user_mode)
        else:
            render_parent_feedback(language_mode, user_mode)

    st.title(ui_text.get(st.session_state.get("language_mode"), ui_text["한국어"])["app_title"])

    # 🎨 마스코트 워터마크 배경 (모든 모드에서 표시)
    _render_mascot_animation()

    language_mode = st.session_state.get("language_mode", "한국어")

    # 메인 화면 앱 소개 (탭 위에 표시) — 사용자 모드(어린이/청소년·성인)별로 톤 분기
    intro_enhanced = {
        "한국어": {
            "어린이": (
                "**국립어린이과학관**에 와줘서 정말 반가워! 🎉<br>"
                "**📅 AI 가이드 사용기간:** 5.22.(금) ~ 5.31.(일)\n\n"
                "**🏙️ 과학관 안내** — 어디로 갈지 모르겠어? 무슨 프로그램이 있는지 궁금해? 아래 채팅에 물어봐 줘! 🎤 사이드바에서 말로도 물어볼 수 있어.\n\n"
                "**🥰 또만나 놀이터** — 오늘 본 전시물, 다시 만나러 가볼까?! 재밌는 **퀴즈**도 풀고, 궁금한 거 **질문**도 하고, 인공지능이 만들어주는 신기한 **과학동화**까지 들어볼 수 있어~ 밑에 탭을 눌러봐!\n\n"
                "**💡 팁:** 화면 왼쪽 위 **>>** 를 누르면 언어·모드 변경, 음성 질문, 설문조사까지 할 수 있어!"
            ),
            "청소년/성인": (
                "과학이 기쁨이 되는 **국립어린이과학관**에 오신 걸 환영해요! 🎉<br>"
                "편리하고 풍성한 관람을 제공해 드리기 위해, 실시간 안내 AI 가이드 서비스를 시범 운영합니다.<br>"
                "**📅 AI 가이드 시범운영 기간:** 5.22.(금) ~ 5.31.(일)\n\n"
                "**🏙️ 과학관 안내** — 층별 안내·프로그램·관람료·예약·길찾기 등 방문 전후 궁금증을 답해드려요. "
                "아래 채팅창에 입력하거나, 왼쪽 사이드바에서 🎤 음성으로도 질문할 수 있어요.\n\n"
                "**🥰 또만나 놀이터** — 재미있었던 과학관 놀이터, 다시 즐겨볼까?! 과학전시물에 담긴 원리를 바탕으로 **퀴즈**를 풀고, 궁금한 내용은 **질문**하고, 인공지능이 실시간으로 만들어주는 신비한 **과학동화**까지 들어보자구요~ "
                "밑에 탭을 전환해보세요!\n\n"
                "**💡 팁:** 왼쪽 상단 **>>** 를 누르면 언어(Language)·모드 변경, 음성 질문, 설문조사까지 이용할 수 있어요!"
            ),
        },
        "English": {
            "어린이": (
                "Welcome to the **National Children's Science Center**! 🎉<br>"
                "**📅 AI Guide Period:** May 22 (Fri) ~ May 31 (Sun)\n\n"
                "**🏙️ Museum Guide** — Don't know where to go? Wondering what's on today? Just ask me "
                "in the chat below! 🎤 You can also ask out loud using the sidebar.\n\n"
                "**🥰 Again Zone** — Want to play with today's exhibits again?! Take fun "
                "**quizzes**, ask **questions** about anything you're curious about, and listen to magical "
                "AI-made **science stories**~ Tap the tab below!\n\n"
                "**💡 Tip:** Tap **>>** on the top left to change language, ask by voice, and take a survey!"
            ),
            "청소년/성인": (
                "Welcome to the **National Children's Science Center**! 🎉<br>"
                "To provide you with a convenient and enriching visit, we are piloting a real-time AI guide service.<br>"
                "**📅 AI Guide Pilot Period:** May 22 (Fri) ~ May 31 (Sun)\n\n"
                "**🏙️ Museum Guide** — I'll answer questions about floors, programs, fees, reservations, "
                "and directions before or after your visit. Type in the chat below, or use 🎤 voice input "
                "from the left sidebar.\n\n"
                "**🥰 Again Zone** — Want to relive the fun?! Take **quizzes** on the "
                "science behind the exhibits, ask **questions** about anything you're still curious about, "
                "and listen to magical AI-generated **science stories** in real time~ Switch using the tab below!\n\n"
                "** Tip:** Tap **>>** on the top left to change Language·mode, ask by voice, and take a survey!"
            ),
        },
        "日本語": {
            "어린이": (
                "**国立こども科学館**に来てくれてうれしい！🎉<br>"
                "**📅 AIガイド使用期間：** 5月22日(金) 〜 5月31日(日)\n\n"
                "**🏙️ 科学館案内** — どこに行けばいいかわからない？今日のプログラムが気になる？下のチャットで聞いてね！🎤 サイドバーからも音声で聞けるよ。\n\n"
                "**🥰 またねゾーン** — 今日見た展示物、もう一度会いに行こう！おもしろい **クイズ** を解いて、気になることは **質問** して、人工知能が作る不思議な **サイエンス童話** も聴いてみよう〜 下のタブを押してみてね！\n\n"
                "**💡 ヒント：** 画面左上の **>>** を押すと、言語・モード変更、音声質問、アンケートができるよ！"
            ),
            "청소년/성인": (
                "便利で充実した見学を提供するため、リアルタイム案内AIガイドサービスを試験運用しています。<br>"
                "**📅 AIガイド試験運用期間：** 5月22日(金) 〜 5月31日(日)\n\n"
                "**🏙️ 科学館案内** — フロア案内・プログラム・料金・予約・アクセスなど、来館前後の疑問にお答えします。"
                "下のチャットに入力、または左サイドバーから 🎤 音声でどうぞ。\n\n"
                "**🥰 またねゾーン** — 楽しかった科学館、もう一度楽しもう！展示に込められた科学原理をもとに "
                "**クイズ** を解いて、気になることを **質問** して、AIがリアルタイムで作る不思議な **サイエンス童話** "
                "まで聴いてみよう〜 下のタブで切り替えてみてね！\n\n"
                "**💡 ヒント：** 左上の **>>** を押すと、言語(Language)・モード変更、音声質問、アンケートができます！"
            ),
        },
        "中文": {
            "어린이": (
                "欢迎来到 **国立儿童科学馆**！🎉\n"
                "**📅 AI导览使用期间：** 5月22日(周五) 〜 5月31日(周日)\n\n"
                "**🏙️ 科学馆导览** — 不知道去哪儿？想知道今天有什么节目？在下方聊天框问我吧！"
                "🎤 也可以从侧边栏用语音问哦。\n\n"
                "**🥰 再次乐园** — 今天看到的展品，再一起玩一次吧！来做有趣的 **测验**、"
                "**提问** 感兴趣的内容、还能听AI做的奇妙 **科学故事** 哦~ 点点下面的标签试试！\n\n"
                "**💡 提示：** 点击左上角的 **>>**，可以切换语言、语音提问和填写问卷哦！"
            ),
            "청소년/성인": (
                "欢迎来到 **国立儿童科学馆**！🎉\n"
                "为了提供您便捷丰富的参观体验，我们正在进行实时AI导览服务的试运行。"
                "**📅 AI导览试运行期间：** 5月22日(周五) 〜 5月31日(周日)\n\n"
                "**🏙️ 科学馆导览** — 楼层、节目、门票、预约、交通等参观前后的问题随时为你解答。"
                "在下方聊天框输入，或在侧边栏使用 🎤 语音提问。\n\n"
                "**🥰 再次乐园** — 想再次回味乐趣吗？！基于展品中蕴含的科学原理来做 **测验**、"
                "自由 **提问** 感兴趣的内容、还能听到AI实时创作的奇妙 **科学故事** 哦~ 请切换下方标签试试！\n\n"
                "** 提示：** 点击左上角的 **>>**，可以切换语言(Language)、模式、语音提问和填写问卷！"
            ),
        },
    }
    intro_dict = intro_enhanced.get(language_mode, intro_enhanced["한국어"])
    if isinstance(intro_dict, dict):
        st.markdown(intro_dict.get(user_mode, intro_dict["청소년/성인"]), unsafe_allow_html=True)
    else:
        st.markdown(intro_dict, unsafe_allow_html=True)

    if st.session_state.get("mode_language_changed"):
        st.info(ui_text.get(language_mode, ui_text["한국어"])["mode_lang_changed"])
        del st.session_state["mode_language_changed"]

    with st.expander(ui_text.get(language_mode, ui_text["한국어"])["quick_menu"], expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button(ui_text.get(language_mode, ui_text["한국어"])["quick_floor"], key="quick_floor"):
                _queue_ga_event("quick_menu_click", {"category": "floor", "language": language_mode})
                st.session_state["pending_user_input"] = faq_inputs["floor"]
                st.rerun()
        with c2:
            if st.button(ui_text.get(language_mode, ui_text["한국어"])["quick_route"], key="quick_route"):
                _queue_ga_event("quick_menu_click", {"category": "route", "language": language_mode})
                st.session_state["pending_user_input"] = faq_inputs["route"]
                st.rerun()
        with c3:
            if st.button(ui_text.get(language_mode, ui_text["한국어"])["quick_programs"], key="quick_programs"):
                _queue_ga_event("quick_menu_click", {"category": "programs", "language": language_mode})
                st.session_state["pending_user_input"] = faq_inputs["programs"]
                st.session_state["pending_ui_program_buttons"] = True
                st.rerun()
        with c4:
            if st.button(ui_text.get(language_mode, ui_text["한국어"])["quick_exhibits"], key="quick_exhibits"):
                _queue_ga_event("quick_menu_click", {"category": "exhibits", "language": language_mode})
                st.session_state["pending_user_input"] = faq_inputs["exhibits"]
                st.rerun()
    
    # Tab navigation
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "guide"
    if st.session_state.get("switch_to_guide_tab"):
        st.session_state.active_tab = "guide"
        try:
            del st.session_state["switch_to_guide_tab"]
        except Exception:
            pass

    tab_labels = [
        ui_text.get(language_mode, ui_text["한국어"])["tab_guide"],
        ui_text.get(language_mode, ui_text["한국어"])["tab_learning"],
    ]
    tab_cols = st.columns(2)
    with tab_cols[0]:
        tab0_type = "primary" if st.session_state.active_tab == "guide" else "secondary"
        if st.button(tab_labels[0], key="tab_btn_guide", use_container_width=True, type=tab0_type):
            st.session_state.active_tab = "guide"
            st.rerun()
    with tab_cols[1]:
        tab1_type = "primary" if st.session_state.active_tab == "learning" else "secondary"
        if st.button(tab_labels[1], key="tab_btn_learning", use_container_width=True, type=tab1_type):
            st.session_state.active_tab = "learning"
            st.rerun()

    # Notify users to switch to guide tab when sidebar FAQ buttons are clicked
    if st.session_state.get("active_tab") != "guide" and st.session_state.get("pending_user_input"):
        try:
            st.toast("과학관 안내 탭에서 답변을 확인하세요!", icon="🔔")
        except Exception:
            pass
    
    if st.session_state.active_tab == "guide":
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
                    _queue_ga_event("tts_played", {"language": language_mode})
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
                                _queue_ga_event("program_detail_click", {"category": "explanation", "language": language_mode})
                                st.session_state["pending_user_input"] = "전시해설 자세히 알려줘"
                                st.rerun()
                        with col2:
                            if st.button(ui_text.get(language_mode, ui_text["한국어"])["program_show"], key=f"prog_show_{i}"):
                                _queue_ga_event("program_detail_click", {"category": "science_show", "language": language_mode})
                                st.session_state["pending_user_input"] = "과학쇼 자세히 알려줘"
                                st.rerun()
                        with col3:
                            if st.button(ui_text.get(language_mode, ui_text["한국어"])["program_planet"], key=f"prog_planet_{i}"):
                                _queue_ga_event("program_detail_click", {"category": "planetarium", "language": language_mode})
                                st.session_state["pending_user_input"] = "천체투영관 자세히 알려줘"
                                st.rerun()
                        with col4:
                            if st.button(ui_text.get(language_mode, ui_text["한국어"])["program_light"], key=f"prog_light_{i}"):
                                _queue_ga_event("program_detail_click", {"category": "light_zone", "language": language_mode})
                                st.session_state["pending_user_input"] = "빛놀이터 자세히 알려줘"
                                st.rerun()

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
                            if st.button(ui_text.get(language_mode, ui_text["한국어"])["tts_listen"], key=f"tts_play_msg_{i}_{cache_key}"):
                                _queue_ga_event("tts_played", {"language": language_mode})
                                autoplay_audio(audio_bytes)
                            st.audio(audio_bytes, format="audio/mp3")

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


        user_input = None

        if "pending_user_input" in st.session_state and st.session_state.get("pending_user_input"):
            user_input = st.session_state.get("pending_user_input")
            del st.session_state["pending_user_input"]
        
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            intent = route_intent(user_input)
            _track_ga_event("send_message", {
                "intent": intent,
                "language": language_mode,
                "user_mode": user_mode
            })
            lowered_input = user_input.lower()
            if any(token in lowered_input for token in ["예약", "예매", "방문신청", "방문 신청", "단체예약", "개인예약", "교육예약", "입장권", "qr", "정원", "1600"]):
                st.session_state["pending_ui_reservation_links"] = True
            
            with st.chat_message("assistant"):
                if intent in ["notice", "basic"]:
                    # 규칙 기반 엔진 동작 (RAG/LLM 미사용, 속도 최적화)
                    # answer_rule_based_localized: 정적 사전 번역 우선 → 폴백 LLM 번역
                    _t0 = time.time()
                    with st.spinner(ui_text.get(language_mode, ui_text["한국어"])["spinner_rule"]):
                        answer, ko_original = answer_rule_based_localized(
                            intent, user_input, user_mode, language_mode
                        )
                    log_monitoring(intent=intent, rule_based=True, latency_ms=(time.time()-_t0)*1000)
                    _track_ga_event("answer_delivered", {
                        "intent": intent,
                        "answer_type": "rule_based",
                        "language": language_mode,
                        "user_mode": user_mode
                    })
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
                    _t0 = time.time()
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
                        messages = [{"role": "system", "content": f"{system_prompt}\n\n[RAG 배경지식]\n{rag_context}"}]
                        # 이전 대화 내용 포함 (최근 10개 메시지, user/assistant만)
                        for hist_msg in st.session_state.messages[-10:]:
                            if hist_msg["role"] in ("user", "assistant"):
                                messages.append({"role": hist_msg["role"], "content": hist_msg["content"]})
                        messages.append({"role": "user", "content": llm_user_input})
                        result = agent.invoke({"messages": messages}, config=config)
                        answer = result["messages"][-1].content
                    log_monitoring(intent=intent, rule_based=False, latency_ms=(time.time()-_t0)*1000)
                    _track_ga_event("answer_delivered", {
                        "intent": intent,
                        "answer_type": "llm_rag",
                        "language": language_mode,
                        "user_mode": user_mode
                    })

                    st.markdown(answer)
                    if language_mode != "한국어" and debug_backtranslate:
                        bt = _backtranslate_to_korean_cached(answer, language_mode)
                        if bt:
                            st.caption(f"BT: {bt}")
                    render_source_buttons(rag_sources, language_mode=language_mode)
                    render_tts_for_answer(answer)
                    
                    # 디버깅 정보 표시 (답변 뒤)
                    debug_info = f"=== RAG 검색 결과 (k=3) ===\n{rag_context}\n\n{'='*50}\n\n"
                    for msg in result["messages"][:-1]:  # 마지막 답변 제외
                        if hasattr(msg, 'pretty_repr'):
                            debug_info += msg.pretty_repr() + "\n\n"
                        elif hasattr(msg, 'content'):
                            debug_info += str(msg.content) + "\n\n"
                    if debug_info.strip():
                        with st.expander(ui_text.get(language_mode, ui_text["한국어"])["debug_tool_calls_after"]):
                            with st.container(height=400):
                                st.text(debug_info)
                        st.session_state.messages.append({"role": "debug", "content": debug_info})

            answer_type = "rule_based" if intent in ["notice", "basic"] else "llm_rag"
            assistant_msg = {"role": "assistant", "content": answer, "intent": intent, "answer_type": answer_type}
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

            
            # Voice output is rendered alongside assistant messages above (stable across reruns)
    
    else:
        # Post-visit learning system
        render_post_visit_learning(
            vector_db,
            st.session_state.get("language_mode", "한국어"),
            debug_show_korean=debug_show_ko,
            debug_backtranslate=debug_backtranslate,
        )
    
    # Chat input at page bottom (outside tabs for stable positioning)
    if st.session_state.active_tab == "guide":
        typed_input = st.chat_input(
            ui_text.get(language_mode, ui_text["한국어"])["chat_placeholder"],
            key="main_chat_input"
        )
        if typed_input and not st.session_state.get("pending_user_input"):
            st.session_state["pending_user_input"] = typed_input
    else:
        typed_input = None


if __name__ == "__main__":
    main()
