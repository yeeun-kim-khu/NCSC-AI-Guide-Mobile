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
import requests as _requests
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
    with st.spinner("RAG database loading..."):
        vector_db = initialize_vector_db()
    return vector_db


# ---- Google Analytics 4 (Measurement Protocol, 서버사이드) ----
GA_MEASUREMENT_ID = "G-7VS14G0T7P"


def _track_ga_event(event_name: str, params: dict | None = None) -> None:
    """GA4 Measurement Protocol로 이벤트 전송 (서버사이드, 브라우저 무관)."""
    try:
        api_secret = st.secrets["GA4_API_SECRET"]
        measurement_id = st.secrets.get("GA4_MEASUREMENT_ID", GA_MEASUREMENT_ID)
    except Exception as e:
        print(f"[GA4] secrets 읽기 실패: {e}")
        return
    if not api_secret:
        print("[GA4] API secret이 비어 있음")
        return
    safe_params = dict(params or {})
    for key in list(safe_params.keys()):
        if key.lower() in ("user_id", "email", "name", "content", "message", "query"):
            del safe_params[key]
    if "ga_client_id" not in st.session_state:
        st.session_state["ga_client_id"] = str(uuid.uuid4())
    endpoint = (
        f"https://www.google-analytics.com/mp/collect"
        f"?measurement_id={measurement_id}&api_secret={api_secret}"
    )
    payload = {
        "client_id": st.session_state["ga_client_id"],
        "events": [{"name": event_name, "params": safe_params}],
    }
    try:
        resp = _requests.post(endpoint, json=payload, timeout=3)
        if resp.status_code != 204:
            print(f"[GA4] 응답 오류 {resp.status_code}: {resp.text[:200]}")
        else:
            print(f"[GA4] 이벤트 전송 성공: {event_name}")
    except Exception as e:
        print(f"[GA4] 요청 실패: {e}")


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
    """모든 모드에서 마스코트를 본문 글자 뒤 워터마크 배경으로 렌더링.

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

    - 언어 선택 포함: 선택한 언어로 안내문 표시, 시작 후 앱 언어로 적용
    - 세션당 1회만 표시 (확인 누르면 다시 안 뜸)
    - 동의 전에는 본문 렌더링 차단 (st.stop)
    - st.dialog 사용 (Streamlit 1.31+); 미지원 환경에서는 본문 상단 배너로 폴백
    """
    if st.session_state.get("privacy_notice_acknowledged"):
        return

    _LANG_OPTIONS = {
        "🇰🇷 한국어":  "한국어",
        "🇺🇸 English": "English",
        "🇯🇵 日本語":  "日本語",
        "🇨🇳 中文":    "中文",
    }

    _NOTICE = {
        "한국어": {
            "title":   "AI 가이드 이용 안내",
            "body": (
                "<b>이용 전, 아래 내용을 꼭 확인해주세요.</b><br>"
                "1. 입력한 글·음성은 <b>답변 생성에만 사용되며 저장되지 않습니다.</b> 새로고침하면 대화가 지워져요.<br>"
                "2. 주소·전화번호 등 <b>민감한 개인정보는 입력하지 마세요.</b><br>"
                "3. 어린이는 <b>보호자와 함께</b> 이용해 주세요.<br>"
                "<small>※ 서비스 개선을 위한 익명 통계(접속 시간, 클릭 수)가 수집될 수 있습니다.</small>"
            ),
            "checkbox": "위 내용을 확인했으며 동의합니다.",
            "button":   "시작하기",
        },
        "English": {
            "title":   "AI Guide — Notice",
            "body": (
                "<b>Please read before using this service.</b><br>"
                "1. Your text and voice are used <b>only to generate responses and are not stored.</b> Refreshing clears all conversation.<br>"
                "2. Do <b>not</b> enter sensitive personal information (address, phone number, etc.).<br>"
                "3. Children must use this service <b>with a guardian.</b><br>"
                "<small>※ Anonymous usage statistics (session time, clicks) may be collected for service improvement.</small>"
            ),
            "checkbox": "I have read and agree to the above.",
            "button":   "Start",
        },
        "日本語": {
            "title":   "AIガイド ご利用案内",
            "body": (
                "<b>ご利用前に以下をご確認ください。</b><br>"
                "1. 入力したテキスト・音声は<b>回答生成のみに使用され、保存されません。</b> 再読み込みすると会話が消えます。<br>"
                "2. 住所・電話番号などの<b>個人情報は入力しないでください。</b><br>"
                "3. お子様は<b>保護者の方と一緒に</b>ご利用ください。<br>"
                "<small>※ サービス改善のため、匿名の利用統計（接続時間、クリック数）が収集される場合があります。</small>"
            ),
            "checkbox": "上記の内容を確認し、同意します。",
            "button":   "スタート",
        },
        "中文": {
            "title":   "AI导览 使用须知",
            "body": (
                "<b>使用前请阅读以下内容。</b><br>"
                "1. 您输入的文字和语音<b>仅用于生成回答，不会被存储。</b> 刷新页面后对话将被清除。<br>"
                "2. 请<b>勿输入</b>地址、电话号码等敏感个人信息。<br>"
                "3. 儿童请在<b>监护人陪同下</b>使用。<br>"
                "<small>※ 为改善服务，可能会收集匿名使用统计信息（连接时间、点击次数）。</small>"
            ),
            "checkbox": "我已阅读并同意以上内容。",
            "button":   "开始",
        },
    }

    def _ack(lang: str) -> None:
        st.session_state["privacy_notice_acknowledged"] = True
        st.session_state["language_mode"] = lang
        _queue_ga_event("privacy_consent", {"language": lang})
        st.rerun()

    def _render_body():
        # 언어 선택 라디오
        lang_label = st.radio(
            "🌐 언어 / Language / 言語 / 语言",
            options=list(_LANG_OPTIONS.keys()),
            horizontal=True,
            key="popup_lang_select",
        )
        chosen_lang = _LANG_OPTIONS[lang_label]
        notice = _NOTICE[chosen_lang]

        st.markdown("<hr style='margin-top: -10px; border: 1px solid #ccc;'>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='line-height:2.0; font-size: 16px;'>{notice['body']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)
        agreed = st.checkbox(notice["checkbox"], key="popup_agreed")
        if agreed and st.button(notice["button"], type="primary", use_container_width=True):
            _ack(chosen_lang)

    if hasattr(st, "dialog"):
        @st.dialog("AI 가이드 이용 안내 (AI Guide Usage Notice)", width="large")
        def _privacy_dialog():
            _render_body()

        _privacy_dialog()
        st.stop()
    else:
        _render_body()
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


def render_feedback(language_mode: str = "한국어", user_mode: str = "기본"):
    ft = GOOGLE_FORM_I18N.get(language_mode, GOOGLE_FORM_I18N["한국어"])
    is_child = (user_mode == "어린이")
    msg_key = "children_msg" if is_child else "parent_msg"
    st.caption(ft[msg_key])
    if is_child:
        st.caption(ft["children_guardian"])
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
            "app_title": "국립어린이과학관 AI 가이드 (5.22.~5.31.)",
            "sidebar_title": "⚙️ 안내 모드",
            "user_mode_label": "사용자 모드 선택:",
            "user_mode_child": "어린이",
            "user_mode_adult": "성인",
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
            "chat_placeholder": "예) 오늘의 프로그램은 무엇인가요?",
            "mode_lang_changed": "사용자 모드/언어 설정이 변경되었어요. 다음 답변부터 새 설정으로 안내할게요.",
            "settings_label": "⚙️ 설정",
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
            "app_title": "NCSC AI Guide (5.22.~5.31.)",
            "sidebar_title": "⚙️ Guide Mode",
            "user_mode_label": "Visitor type:",
            "user_mode_child": "Child",
            "user_mode_adult": "Adult",
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
            "chat_placeholder": "e.g., What programs are available today?",
            "mode_lang_changed": "User mode/language settings changed. Responses will use the new settings from now on.",
            "settings_label": "⚙️ Settings",
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
            "app_title": "国立子ども科学館 AIガイド (5.22.~5.31.)",
            "sidebar_title": "⚙️ 案内モード",
            "user_mode_label": "利用者:",
            "user_mode_child": "こども",
            "user_mode_adult": "大人",
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
            "chat_placeholder": "例）今日のプログラムは何ですか？",
            "mode_lang_changed": "モード/言語が変更されました。次の回答から新しい設定で案内します。",
            "settings_label": "⚙️ 設定",
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
            "app_title": "国立儿童科学馆 AI 导览 (5.22.~5.31.)",
            "sidebar_title": "⚙️ 导览模式",
            "user_mode_label": "访客类型:",
            "user_mode_child": "儿童",
            "user_mode_adult": "成人",
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
            "chat_placeholder": "例) 今天有什么节目?",
            "mode_lang_changed": "用户模式/语言设置已更改。从下一个回答开始，将使用新设置进行引导。",
            "settings_label": "⚙️ 设置",
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
    [data-testid="stChatMessageAvatarAssistant"] {
        display: none !important;
        width: 0 !important;
        min-width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    [data-testid="stChatMessage"] { padding-top: 4px !important; padding-bottom: 4px !important; }
    .stChatMessage { margin-top: 4px !important; margin-bottom: 4px !important; }
    footer { visibility: hidden !important; display: none !important; height: 0 !important; }
    [data-testid="stFooter"] { display: none !important; }
    #MainMenu { visibility: hidden !important; display: none !important; }
    .main .block-container { padding-top: 0.5rem !important; padding-bottom: 0.5rem !important; }
    .block-container { padding-top: 0.5rem !important; padding-bottom: 0.5rem !important; }
    section.main > div.block-container { padding-top: 0.5rem !important; padding-bottom: 0.5rem !important; }
    .stApp > footer { display: none !important; }
    .appview-container > footer { display: none !important; }
    /* 메인 제목 굵게 */
    h2 { font-weight: 900 !important; }
    /* 구분선 두껍게, 위아래 여백 최소화 */
    hr { border-width: 3px !important; margin: 4px 0 !important; }
    /* 채팅 입력창 — 주황 테두리, 항상 글로우, overflow:hidden으로 끊김 방지 */
    [data-testid="stChatInput"] {
        border: 2px solid #ff6b35 !important;
        border-radius: 16px !important;
        overflow: hidden !important;
        box-shadow: 0 0 0 3px rgba(255,107,53,0.18) !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #e85520 !important;
        box-shadow: 0 0 0 4px rgba(255,107,53,0.32) !important;
    }
    [data-testid="stChatInput"] textarea {
        border: none !important;
        outline: none !important;
    }
    /* st.pills 크기 확대 */
    [data-testid="stPillsButton"] {
        padding: 14px 20px !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # JavaScript로 Streamlit footer 직접 숨김 (CSS 선택자가 버전마다 달라서 JS가 더 안정적)
    components.html("""<script>
    (function hideStreamlitFooter() {
        const p = window.parent;
        if (!p) return;
        function hide() {
            // footer 태그 전체 숨김
            p.document.querySelectorAll('footer').forEach(function(el) {
                el.style.setProperty('display', 'none', 'important');
            });
            // block-container 하단 패딩 제거
            p.document.querySelectorAll('.block-container').forEach(function(el) {
                el.style.setProperty('padding-bottom', '0.5rem', 'important');
            });
        }
        hide();
        setTimeout(hide, 300);
        setTimeout(hide, 1000);
    })();
    </script>""", height=0)

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
        user_mode = "어린이" if user_mode_display == t("user_mode_child") else "성인"
        
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

        # 설문조사를 사이드바 맨 하단으로 이동
        render_feedback(language_mode, user_mode)

    language_mode = st.session_state.get("language_mode", "한국어")

    st.header(ui_text.get(language_mode, ui_text["한국어"])["app_title"])

    # 🎨 마스코트 워터마크 배경 (모든 모드에서 표시)
    _render_mascot_animation()

    # 메인 화면 앱 소개 (탭 위에 표시) — 사용자 모드(어린이/성인)별로 톤 분기
    intro_enhanced = {
        "한국어": {
            "어린이": (
                "<strong>과학관 안내부터 퀴즈·과학동화 생성까지, AI가 함께합니다!</strong><br><br>"
                "🏙️ <strong>과학관 안내</strong> - 층별·프로그램·관람료·예약·길찾기<br>"
                "🥰 <strong>또만나 놀이터</strong> - 전시물 퀴즈, 질문, AI 과학동화<br>"
                "⚙️ <strong>설정</strong> - 언어·사용자 모드·음성·설문조사"
            ),
            "성인": (
                "<strong>과학관 안내부터 퀴즈·과학동화 생성까지, AI가 함께합니다!</strong><br><br>"
                "🏙️ <strong>과학관 안내</strong> - 층별·프로그램·관람료·예약·길찾기<br>"
                "🥰 <strong>또만나 놀이터</strong> - 전시물 퀴즈, 질문, AI 과학동화<br>"
                "⚙️ <strong>설정</strong> - 언어·사용자 모드·음성·설문조사"
            ),
        },
        "English": {
            "어린이": (
                "<strong>Your AI guide for museum visits and exhibit experiences.</strong><br><br>"
                "🏙️ <strong>Museum Guide</strong> - Floors · Programs · Fees · Reservations · Directions<br>"
                "🥰 <strong>Again Zone</strong> - Exhibit quizzes, Q&A, AI science stories<br>"
                "⚙️ <strong>Settings</strong> - Language · User mode · Voice · Survey"
            ),
            "성인": (
                "<strong>Your AI guide for museum visits and exhibit experiences.</strong><br><br>"
                "🏙️ <strong>Museum Guide</strong> - Floors · Programs · Fees · Reservations · Directions<br>"
                "🥰 <strong>Again Zone</strong> - Exhibit quizzes, Q&A, AI science stories<br>"
                "⚙️ <strong>Settings</strong> - Language · User mode · Voice · Survey"
            ),
        },
        "日本語": {
            "어린이": (
                "<strong>科学館の観覧と体験をAIがご案内します。</strong><br><br>"
                "🏙️ <strong>科学館案内</strong> - フロア・プログラム・料金・予約・アクセス<br>"
                "🥰 <strong>またねゾーン</strong> - 展示クイズ、質問、AIサイエンス童話<br>"
                "⚙️ <strong>設定</strong> - 言語・ユーザーモード・音声・アンケート"
            ),
            "성인": (
                "<strong>科学館の観覧と体験をAIがご案内します。</strong><br><br>"
                "🏙️ <strong>科学館案内</strong> - フロア・プログラム・料金・予約・アクセス<br>"
                "🥰 <strong>またねゾーン</strong> - 展示クイズ、質問、AIサイエンス童話<br>"
                "⚙️ <strong>設定</strong> - 言語・ユーザーモード・音声・アンケート"
            ),
        },
        "中文": {
            "어린이": (
                "<strong>AI为您提供科学馆参观及体验导览。</strong><br><br>"
                "🏙️ <strong>科学馆导览</strong> - 楼层·节目·门票·预约·交通<br>"
                "🥰 <strong>再次乐园</strong> - 展品测验、提问、AI科学故事<br>"
                "⚙️ <strong>设置</strong> - 语言·用户模式·语音提问·问卷调查"
            ),
            "성인": (
                "<strong>AI为您提供科学馆参观及体验导览。</strong><br><br>"
                "🏙️ <strong>科学馆导览</strong> - 楼层·节目·门票·预约·交通<br>"
                "🥰 <strong>再次乐园</strong> - 展品测验、提问、AI科学故事<br>"
                "⚙️ <strong>设置</strong> - 语言·用户模式·语音提问·问卷调查"
            ),
        },
    }
    intro_dict = intro_enhanced.get(language_mode, intro_enhanced["한국어"])
    _intro_text = intro_dict.get(user_mode, intro_dict["성인"]) if isinstance(intro_dict, dict) else intro_dict
    st.markdown(
        f'<div style="font-size:17px; line-height:1.8;">{_intro_text}</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.get("mode_language_changed"):
        st.info(ui_text.get(language_mode, ui_text["한국어"])["mode_lang_changed"])
        del st.session_state["mode_language_changed"]

    # Tab navigation
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "guide"
    if st.session_state.get("switch_to_guide_tab"):
        st.session_state.active_tab = "guide"
        try:
            del st.session_state["switch_to_guide_tab"]
        except Exception:
            pass

    _tab_guide_label = ui_text.get(language_mode, ui_text["한국어"])["tab_guide"]
    _tab_learn_label = ui_text.get(language_mode, ui_text["한국어"])["tab_learning"]
    _settings_label = ui_text.get(language_mode, ui_text["한국어"])["settings_label"]
    _tab_options = [_tab_guide_label, _tab_learn_label, _settings_label]

    # Apply pending pills reset BEFORE widget renders (Streamlit forbids setting widget key after render)
    if st.session_state.get("_reset_pills_to"):
        st.session_state["tab_pills_widget"] = st.session_state.pop("_reset_pills_to")

    _tab_default = _tab_guide_label if st.session_state.active_tab == "guide" else _tab_learn_label

    _tab_hint = {
        "한국어": "👇 원하는 기능을 선택해주세요!",
        "English": "👇 Select a feature to get started!",
        "日本語": "👇 ご利用になる機能を選んでください！",
        "中文": "👇 请选择您需要的功能！",
    }.get(language_mode, "👇 원하는 기능을 선택해주세요!")

    st.divider()
    st.markdown(f"<div style='text-align:center; font-size:15px; font-weight:600; margin-bottom:6px;'>{_tab_hint}</div>", unsafe_allow_html=True)
    _selected_pill = st.pills(
        "탭",
        options=_tab_options,
        default=_tab_default,
        key="tab_pills_widget",
        label_visibility="collapsed",
    )
    st.divider()

    if _selected_pill == _settings_label:
        # 설정 선택 시: 다음 run에서 pills를 현재 탭으로 되돌리고 사이드바 토글
        _active_label = _tab_guide_label if st.session_state.active_tab == "guide" else _tab_learn_label
        st.session_state["_reset_pills_to"] = _active_label
        st.session_state["_toggle_sidebar"] = True
        st.rerun()
    elif _selected_pill == _tab_guide_label and st.session_state.active_tab != "guide":
        st.session_state.active_tab = "guide"
        st.rerun()
    elif _selected_pill == _tab_learn_label and st.session_state.active_tab != "learning":
        st.session_state.active_tab = "learning"
        st.rerun()

    # 사이드바 토글 JS (설정 버튼 클릭 후 rerun 시 실행)
    if st.session_state.get("_toggle_sidebar"):
        del st.session_state["_toggle_sidebar"]
        import random as _random
        _nonce = _random.randint(0, 999999)
        components.html(f"""<script>
/* {_nonce} */
setTimeout(function(){{
    var p = window.parent;
    var openBtn = p.document.querySelector('[data-testid="stSidebarCollapsedControl"] button');
    if (openBtn) {{ openBtn.click(); return; }}
    var closeBtn = p.document.querySelector('[data-testid="stSidebarHeader"] button')
                || p.document.querySelector('section[data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"]')
                || p.document.querySelector('section[data-testid="stSidebar"] > div > div > button');
    if (closeBtn) closeBtn.click();
}}, 300);
</script>""", height=0)

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
                # 캐시 최대 20개 유지 (세션 메모리 누적 방지)
                if len(st.session_state.tts_cache) >= 20:
                    oldest_key = next(iter(st.session_state.tts_cache))
                    del st.session_state.tts_cache[oldest_key]
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

        last_assistant_idx = max(
            (j for j, m in enumerate(st.session_state.messages) if m["role"] == "assistant"),
            default=-1
        )
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "debug":
                continue  # 운영 중 내부 RAG/프롬프트 정보 노출 방지
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

                    if enable_voice_output and msg["role"] == "assistant" and msg.get("content") and i == last_assistant_idx:
                        lang_code = get_language_code(language_mode)
                        tts_ns = get_tts_cache_namespace(language=lang_code)
                        tts_text = msg["content"]
                        if len(tts_text) > 1200:
                            tts_text = tts_text[:1200]
                        cache_key = f"{language_mode}::{tts_ns}::" + str(hash(tts_text))
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
                                st.link_button(ui_text.get(language_mode, ui_text["한국어"])["reservation_person"], "https://www.sciencecenter.go.kr/csc/new1/reservation/reservation_person.jsp")
                            with col2:
                                st.link_button(ui_text.get(language_mode, ui_text["한국어"])["reservation_group"], "https://www.sciencecenter.go.kr/csc/new1/reservation/reservation_group.jsp")
                            with col3:
                                st.link_button(ui_text.get(language_mode, ui_text["한국어"])["reservation_edu"], "https://www.sciencecenter.go.kr/csc/new1/reservation/education_creation.jsp")
                        else:
                            with col1:
                                st.markdown(f"[{ui_text.get(language_mode, ui_text['한국어'])['reservation_person']}](https://www.sciencecenter.go.kr/csc/new1/reservation/reservation_person.jsp)")
                            with col2:
                                st.markdown(f"[{ui_text.get(language_mode, ui_text['한국어'])['reservation_group']}](https://www.sciencecenter.go.kr/csc/new1/reservation/reservation_group.jsp)")
                            with col3:
                                st.markdown(f"[{ui_text.get(language_mode, ui_text['한국어'])['reservation_edu']}](https://www.sciencecenter.go.kr/csc/new1/reservation/education_creation.jsp)")


        user_input = None

        if "pending_user_input" in st.session_state and st.session_state.get("pending_user_input"):
            user_input = st.session_state.get("pending_user_input")
            del st.session_state["pending_user_input"]
        
        if user_input:
            # ── 의도 확인(Clarification) 응답 처리 ────────────────────────────
            _skip = False
            if st.session_state.get("_awaiting_clarification"):
                clar = st.session_state.pop("_awaiting_clarification")
                _pos = ["네", "응", "맞아", "맞아요", "그래요", "맞습니다", "예", "ㅇㅇ", "yes", "はい", "是", "是的", "맞어", "맞음"]
                if any(w in user_input for w in _pos):
                    # 사용자 확인 → 원래 쿼리 + RAG 컨텍스트 주입
                    st.session_state["_clarification_rag_ctx"] = clar["rag_context"]
                    user_input = clar["original_query"]
                else:
                    # 사용자 부정 → 더 자세히 입력 요청
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    with st.chat_message("user"):
                        st.markdown(user_input)
                    _no_hint = {
                        "한국어": "더 구체적으로 입력해 주시면 정확하게 안내해 드릴게요! 😊\n\n예) '5월 과학교실 신청 방법', '교육프로그램 목록', '얼음공 프로그램 상세'",
                        "English": "Please provide more details so I can help you accurately! 😊\n\nEx) 'How to register for science class', 'Education program schedule'",
                        "日本語": "もう少し詳しく入力していただけますか？😊",
                        "中文": "请提供更多详情，这样我能更好地帮助您！😊",
                    }
                    _no_answer = _no_hint.get(language_mode, _no_hint["한국어"])
                    st.session_state.messages.append({"role": "assistant", "content": _no_answer})
                    with st.chat_message("assistant"):
                        st.markdown(_no_answer)
                    _skip = True

            if not _skip:
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

                _t0 = time.time()
                ko_original = ""
                rule_sources = []
                rag_sources = []
                rag_context = ""
                result = None
                answer = ""
                _stream_messages = None

                with st.chat_message("assistant"):
                    if intent in ["notice", "basic"]:
                        # 규칙 기반 엔진 동작 (RAG/LLM 미사용, 속도 최적화)
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
                        lowered = user_input.lower()
                        if intent == "notice":
                            rule_sources = [CSC_URLS.get("공지사항")]
                        elif intent == "science_show":
                            rule_sources = [CSC_URLS.get("과학쇼")]
                        else:
                            if any(k in lowered for k in ["오시는길", "오는길", "교통", "길찾기", "주소", "위치"]):
                                rule_sources = [CSC_URLS.get("오시는길")]
                            elif "천체투영관" in lowered:
                                rule_sources = [CSC_URLS.get("천체투영관")]
                            elif any(k in lowered for k in ["예약", "예매", "단체", "개인", "교육"]):
                                rule_sources = [CSC_URLS.get("예약안내"), CSC_URLS.get("개인예약"), CSC_URLS.get("단체예약"), CSC_URLS.get("교육예약")]
                            else:
                                rule_sources = [CSC_URLS.get("이용안내")]
                        rule_sources = [s for s in dict.fromkeys([s for s in rule_sources if s])]
                    else:
                        # LLM + RAG + Crawling 엔진 동작
                        _stream_messages = None
                        _stream_config = None
                        with st.spinner(ui_text.get(language_mode, ui_text["한국어"])['spinner_llm']):
                            if st.session_state.get("directions_origin"):
                                origin = st.session_state.get("directions_origin")
                                del st.session_state["directions_origin"]
                                user_input = (
                                    f"출발지: {origin}\n"
                                    "목적지: 국립어린이과학관(국립어린이과학관, 서울 종로구 창경궁로 215)\n"
                                    "요청: 대중교통(지하철/버스) 기준으로 가장 쉬운 경로를 단계별로 자세하고 친절하게 안내해줘. "
                                    "출입구/도보 이동/환승 포인트가 있으면 같이 알려줘. "
                                    "마지막에 노선/출입구는 변동될 수 있으니 공식 홈페이지(www.sciencecenter.go.kr/csc) '오시는 길' 확인과 02-3668-1500 문의를 덧붙여줘."
                                )
                            # RAG 검색
                            retrieved_docs = vector_db.similarity_search(user_input, k=3)
                            rag_context = "\n\n".join([f"[{doc.metadata.get('source', 'N/A')}]\n{doc.page_content}" for doc in retrieved_docs])
                            rag_sources = [doc.metadata.get("source", "N/A") for doc in retrieved_docs if getattr(doc, "metadata", None)]
                            rag_sources = [s for s in dict.fromkeys([s for s in rag_sources if s])]

                            # ── 의도 확인(Clarification) 질문 생성 ──────────────────
                            _asked_clarification = False
                            _clar_ctx = st.session_state.pop("_clarification_rag_ctx", None)
                            if _clar_ctx:
                                # 사용자가 확인한 경우 → 확인된 RAG 컨텍스트 앞에 주입
                                rag_context = _clar_ctx + "\n\n" + rag_context
                            elif (
                                len(user_input.strip()) <= 12
                                and len(user_input.strip().split()) <= 2
                                and retrieved_docs
                                and retrieved_docs[0].metadata.get("title", "").strip() not in ("", "nan")
                                and not any(t in user_input for t in ["오늘", "이번", "내일", "어제", "지금", "요즘", "최근", "언제", "today", "now"])
                            ):
                                _top = retrieved_docs[0]
                                _title = _top.metadata.get("title", "")
                                _cat = _top.metadata.get("category", "")
                                _cat_label = {
                                    "한국어": " 교육 프로그램" if _cat == "교육프로그램" else "",
                                    "English": " education program" if _cat == "교육프로그램" else "",
                                    "日本語": " 教育プログラム" if _cat == "교육프로그램" else "",
                                    "中文": " 教育课程" if _cat == "교육프로그램" else "",
                                }.get(language_mode, "")
                                _suffix = {
                                    "한국어": "\n\n맞으시면 **네**, 아니라면 좀 더 자세히 입력해 주세요 😊",
                                    "English": "\n\nSay **Yes** to confirm, or describe more specifically 😊",
                                    "日本語": "\n\n**はい**で確認、または詳しく入力してください 😊",
                                    "中文": "\n\n说 **是** 来确认，或提供更多详情 😊",
                                }.get(language_mode, "\n\n맞으시면 **네**, 아니라면 좀 더 자세히 입력해 주세요 😊")
                                _clar_q = {
                                    "한국어": f"혹시 **{_title}**{_cat_label}에 대해 질문하시는 건가요?{_suffix}",
                                    "English": f"Are you asking about **{_title}**{_cat_label}?{_suffix}",
                                    "日本語": f"**{_title}**{_cat_label}についてのご質問ですか？{_suffix}",
                                    "中文": f"您是在询问关于**{_title}**{_cat_label}的问题吗？{_suffix}",
                                }.get(language_mode, f"혹시 **{_title}**{_cat_label}에 대해 질문하시는 건가요?{_suffix}")
                                st.session_state["_awaiting_clarification"] = {
                                    "original_query": user_input,
                                    "rag_context": "\n\n".join([d.page_content for d in retrieved_docs[:2]])
                                }
                                answer = _clar_q
                                _asked_clarification = True

                            if not _asked_clarification:
                                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                                llm_user_input = user_input
                                if language_mode != "한국어":
                                    _lang_override = {
                                        "English": "[REQUIRED OUTPUT LANGUAGE: English] You MUST answer ENTIRELY in English, even though the question above may be in Korean. Translate place names using the official glossary in the system prompt (e.g., AI놀이터 → AI Zone). Do NOT output Korean text.",
                                        "日本語": "[出力言語指定: 日本語] 上の質問が韓国語であっても、必ず日本語だけで答えてください。場所名はシステムプロンプトのグロッサリーに従い、「日本語名称 (English Official Name)」の形式で記してください（例: 考えるゾーン (Thinking Zone)）。韓国語文字をそのまま出力しないこと。",
                                        "中文": "[输出语言要求: 中文] 即使以上问题是韩语，你也必须完全用中文回答。地点名称请依照系统提示词中的词汇表，以\"中文名称 (English Official Name)\"的格式书写（例：思考区 (Thinking Zone)）。不要直接输出韩文。",
                                    }.get(language_mode, "")
                                    if _lang_override:
                                        llm_user_input = f"{user_input}\n\n---\n{_lang_override}"
                                messages = [{"role": "system", "content": f"{system_prompt}\n\n[RAG 배경지식]\n{rag_context}"}]
                                for hist_msg in st.session_state.messages[-10:]:
                                    if hist_msg["role"] in ("user", "assistant"):
                                        messages.append({"role": hist_msg["role"], "content": hist_msg["content"]})
                                messages.append({"role": "user", "content": llm_user_input})
                                _stream_messages = messages
                                _stream_config = config

                        log_monitoring(intent=intent, rule_based=False, latency_ms=(time.time()-_t0)*1000)
                        _track_ga_event("answer_delivered", {
                            "intent": intent,
                            "answer_type": "llm_rag",
                            "language": language_mode,
                            "user_mode": user_mode
                        })

                        # 스트리밍 출력 (RAG 검색 완료 후 spinner 없이 즉시 토큰 표시)
                        if _stream_messages is not None:
                            def _llm_stream():
                                try:
                                    for msg, metadata in agent.stream(
                                        {"messages": _stream_messages},
                                        config=_stream_config,
                                        stream_mode="messages"
                                    ):
                                        if (
                                            hasattr(msg, "content")
                                            and isinstance(msg.content, str)
                                            and msg.content
                                            and metadata.get("langgraph_node") == "agent"
                                            and not getattr(msg, "tool_calls", None)
                                        ):
                                            yield msg.content
                                except Exception as _e:
                                    print(f"Streaming error, fallback to invoke: {_e}")
                                    try:
                                        _fb = agent.invoke({"messages": _stream_messages}, config=_stream_config)
                                        yield _fb["messages"][-1].content
                                    except Exception as _e2:
                                        print(f"Invoke fallback failed: {_e2}")
                                        yield "죄송해요, 일시적인 오류가 발생했어요. 다시 질문해 주세요. 😔"
                            answer = st.write_stream(_llm_stream())

                    if _stream_messages is None:
                        st.markdown(answer)
                    if language_mode != "한국어" and debug_show_ko and ko_original:
                        st.caption(f"KO: {ko_original}")
                    if language_mode != "한국어" and debug_backtranslate:
                        bt = _backtranslate_to_korean_cached(answer, language_mode)
                        if bt:
                            st.caption(f"BT: {bt}")
                    if intent in ["notice", "basic"]:
                        render_source_buttons(rule_sources, language_mode=language_mode)
                    else:
                        render_source_buttons(rag_sources, language_mode=language_mode)
                        # 디버그 정보는 서버 로그에만 기록 (UI 노출 제거)
                        if result is not None and (debug_show_ko or debug_backtranslate):
                            debug_info = f"=== RAG 검색 결과 (k=3) ===\n{rag_context}\n\n{'='*50}\n\n"
                            for msg in result["messages"][:-1]:
                                if hasattr(msg, 'pretty_repr'):
                                    debug_info += msg.pretty_repr() + "\n\n"
                                elif hasattr(msg, 'content'):
                                    debug_info += str(msg.content) + "\n\n"
                            if debug_info.strip():
                                with st.expander(ui_text.get(language_mode, ui_text["한국어"])["debug_tool_calls_after"]):
                                    with st.container(height=400):
                                        st.text(debug_info)
                                st.session_state.messages.append({"role": "debug", "content": debug_info})
                    render_tts_for_answer(answer)

                components.html("""<script>
                (function() {
                    const p = window.parent;
                    if (!p) return;
                    setTimeout(function() {
                        const candidates = [
                            p.document.querySelector('[data-testid="stAppViewBlockContainer"]'),
                            p.document.querySelector('section[data-testid="stMain"] > div'),
                            p.document.querySelector('[data-testid="stMain"]'),
                            p.document.querySelector('.main'),
                        ];
                        const el = candidates.find(function(e) { return e && e.scrollHeight > e.clientHeight + 10; });
                        if (el) {
                            el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
                        } else {
                            p.scrollTo({ top: p.document.body.scrollHeight, behavior: 'smooth' });
                        }
                    }, 300);
                })();
                </script>""", height=0)

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
            user_mode=user_mode,
        )
    
    # Footer — 시범운영 기간 문의처 (작고 눈에 띄지 않게)
    st.markdown("""
<div style="text-align:center; padding:10px 0 12px; margin-top:12px;
            font-size:11px; color:#aaa; line-height:1.8;">
  🛠️ 앱 사용 중 문제가 생기면 알려주세요!&nbsp;&nbsp;
  <a href="https://open.kakao.com/o/gk8Bgjvi" target="_blank"
     style="color:#999; text-decoration:none; margin:0 6px;">
    💬 카카오톡 문의
  </a>
</div>
""", unsafe_allow_html=True)

    # Chat input at page bottom (outside tabs for stable positioning)
    if st.session_state.active_tab == "guide":
        typed_input = st.chat_input(
            ui_text.get(language_mode, ui_text["한국어"])["chat_placeholder"],
            key="main_chat_input"
        )
        if typed_input and not st.session_state.get("pending_user_input"):
            st.session_state["pending_user_input"] = typed_input
            st.session_state["_scroll_to_input"] = True
            st.rerun()
        
        # 질문 입력 후에만 자동 스크롤 (components.html 사용 — st.markdown의 <script>는 브라우저가 실행 안 함)
        if st.session_state.get("_scroll_to_input"):
            components.html("""<script>
            (function() {
                const p = window.parent;
                if (!p) return;
                setTimeout(function() {
                    const el = p.document.querySelector('[data-testid="stChatInput"]')
                            || p.document.querySelector('textarea');
                    if (el) {
                        el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                    }
                }, 500);
            })();
            </script>""", height=0)
            st.session_state["_scroll_to_input"] = False
    else:
        typed_input = None


if __name__ == "__main__":
    main()
