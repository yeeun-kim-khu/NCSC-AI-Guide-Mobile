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
        "빛놀이터": "Light Zone",
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

# ============================================================================
# CSV 데이터 로딩
# ============================================================================

@st.cache_data(show_spinner=False)
def _preload_all_zone_csv_rows():
    data = {}
    for zone, info in ZONE_INFO.items():
        if info.get("has_data"):
            data[zone] = load_zone_rows_from_csv(zone)
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


def _extract_zone_keywords_from_titles(zone_rows, top_n=12):
    """제목을 한/영 쌍으로 추출.

    반환: list of (kr, en) tuples — kr은 항상 채워져 있음, en은 비어있을 수 있음.
    """
    pairs = []
    seen_kr = set()
    for r in (zone_rows or []):
        raw = str(r.get("title", "")).strip()
        if not raw or len(raw) <= 1:
            continue
        if "체험방법" in raw:
            continue
        kr, en = _split_title_ko_en(raw)
        if not kr or len(kr) <= 1:
            continue
        if kr in seen_kr:
            continue
        seen_kr.add(kr)
        pairs.append((kr, en))
    return pairs[:top_n]


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
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
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

    cols = st.columns(4)
    for i, (kw_kr, kw_disp) in enumerate(keyword_pairs):
        with cols[i % 4]:
            if st.button(kw_disp, key=f"kw_btn_{zone_name}_{mode}_{kw_kr}"):
                st.session_state[state_key] = kw_kr

    selected_kw = st.session_state.get(state_key, "")
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
        st.session_state[reveal_key] = not st.session_state[reveal_key]
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

def generate_quiz(zone_name, principle, llm, language="한국어", variation_seed: int = 0):
    """과학원리 기반 4지선다 퀴즈 생성.

    LLM에게는 JSON 형태(question, options, correct_index, explanation)를 받고,
    클라이언트에서 옵션을 **항상** 무작위로 셔플 → 정답 위치 편향(LLM의 1번 선호)을 원천 제거.
    반환: dict {question, options[4], correct_index(0-3), explanation, raw}
    실패 시: {raw: 원문 텍스트} 만 담긴 dict (호출 측에서 markdown으로 폴백 표시)
    """
    import random
    import json as _json

    rng = random.Random(variation_seed if variation_seed else random.randint(1, 10**9))
    angles = [
        "일상 생활의 구체적 상황(놀이, 음식, 날씨, 동물)으로 직관적으로 묻기",
        "원인을 묻는 형태",
        "결과/예측을 묻는 형태",
        "유사한 현상을 비교/대조",
        "관찰·실험 상황을 상상하게 하는 형태",
        "전시물 체험과 직접 연결된 시나리오",
    ]
    angle = rng.choice(angles)

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

    glossary_rules = _get_ui_glossary_rules(language)

    # 출력 언어 강제 (principle 이 한국어여도 답변 언어를 지정)
    output_lang_instruction = {
        "한국어": "[출력 언어: 한국어] question, options, explanation 모든 텍스트는 반드시 한국어로 작성.",
        "English": "[OUTPUT LANGUAGE: English] question, all 4 options, and explanation MUST be written in English. Do NOT use Korean. Translate any Korean topic into English. ALL TEXT IN ENGLISH ONLY.",
        "日本語": "[出力言語: 日本語] question, options, explanation のすべてを日本語（漢字・かな）で記述。韓国語禁止。トピックが韓国語でも日本語に翻訳すること。",
        "中文": "[输出语言: 简体中文] question, options, explanation 必须全部使用简体中文。禁止使用韩文。即使主题是韩文，也必须翻译成中文。",
    }.get(language, "")

    quality_rules_ko = """
[문제 품질 규칙 — 반드시 지킬 것]
1) 사실 검증: 과학적으로 명백히 참인 정답 1개, 명백히 거짓인 오답 3개. 애매하거나 둘 다 맞을 수 있는 표현 금지.
2) 구체성: "맞다/아니다"처럼 추상적인 선택지 금지. 각 선택지는 명사구 또는 짧은 문장으로 의미가 분명해야 함.
3) 어휘 수준: 6~10세 어린이가 이해할 수 있는 단어. 학술 용어는 풀어서 설명.
4) 일관성: 4개 선택지의 문법 형태/길이를 비슷하게 맞추기 (정답만 길거나 짧으면 안 됨).
5) 함정 주의: 오답은 흔한 오개념이나 비슷한 다른 현상에서 가져오기 (무관한 단어 나열 금지).
6) 질문은 한 가지만 묻기. 이중 부정, 복수 조건 금지.
7) 실제 전시 체험과 연관된 장면을 1개 이상 사용.
"""
    quality_rules_en = """
[Quality rules — MUST follow]
1) Factual: exactly 1 clearly correct answer; 3 clearly wrong distractors. No ambiguous wording.
2) Concrete: each option must be a meaningful phrase, not vague yes/no.
3) Vocabulary for ages 6–10. Avoid jargon.
4) Consistent length/form across the 4 options.
5) Distractors should be common misconceptions, not random unrelated words.
6) Single, clear question. Avoid double negatives or compound conditions.
7) Tie at least one element to actual exhibit experience.
"""

    language_prompts = {
        "한국어": f"""{output_lang_instruction}

'{zone_name}'의 '{principle}' 주제로 4지선다 퀴즈를 만들어주세요.

이번 스타일: {angle}
랜덤 시드: {variation_seed} (매번 다른 질문!)

{quality_rules_ko}

[출력 형식 — JSON만, 다른 텍스트 금지]
다음 스키마의 JSON 객체 한 개를 출력하세요. 코드블록(```)도 붙이지 마세요.
{{
  "question": "어린이가 이해할 수 있는 질문 (1문장)",
  "options": ["선택지1", "선택지2", "선택지3", "선택지4"],
  "correct_index": 0,
  "explanation": "왜 이 답이 맞는지 쉬운 말로 2~3문장. 오답이 왜 틀렸는지도 한 줄 언급."
}}
- correct_index 는 0~3 정수, options 배열에서 정답 위치.
- options 는 정확히 4개.
- 따옴표/JSON 문법 정확히 지킬 것.""",

        "English": f"""{output_lang_instruction}

Create a 4-choice quiz for children about '{principle}' from '{zone_name}'.{glossary_rules}

Style this time: {angle}
Random seed: {variation_seed}

{quality_rules_en}

[Output format — JSON only, no other text]
Output a single JSON object with this schema. No code fences.
{{
  "question": "Single, clear question kids can understand",
  "options": ["option 1", "option 2", "option 3", "option 4"],
  "correct_index": 0,
  "explanation": "2-3 short sentences explaining why the answer is correct, plus one note on why a tempting distractor is wrong."
}}
- correct_index is an integer 0-3 indexing the options array.
- Exactly 4 options.
- Strict JSON syntax."""
    }

    # 日本語 / 中文 prompts (간단하게 영어 베이스에서 파생)
    if language == "日本語":
        language_prompts[language] = f"""'{zone_name}'の'{principle}'をテーマに、子ども向け4択クイズを作ってください。

スタイル: {angle}
ランダムシード: {variation_seed}
{quality_rules_en}

[出力形式 — JSONのみ、他のテキスト禁止]
{{
  "question": "子どもが分かる1文の質問",
  "options": ["選択肢1", "選択肢2", "選択肢3", "選択肢4"],
  "correct_index": 0,
  "explanation": "正解の理由を2-3文で。間違いやすい選択肢の理由も一言。"
}}
"""
    elif language == "中文":
        language_prompts[language] = f"""{output_lang_instruction}

请围绕'{zone_name}'中的'{principle}'，为儿童设计一道四选一测验。

风格: {angle}
随机种子: {variation_seed}
{quality_rules_en}

[输出格式 — 只输出JSON，不要任何其他文本]
{{
  "question": "孩子能理解的一句话提问",
  "options": ["选项1", "选项2", "选项3", "选项4"],
  "correct_index": 0,
  "explanation": "用2-3句话解释为什么正确，并简单点出一个常见错误选项为何不对。"
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


def generate_science_story(zone_name, exhibits, principles, language="한국어"):
    """방문한 놀이터 기반 과학동화 생성 (상상력 강화 버전)"""
    import random

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

    # 핵심 마법 아이템 2개 (제목 + 짧은 설명) — 갈등을 해결하는 키
    core_lines = []
    for ex in exhibits[:2]:
        t = ex.get("metadata", {}).get("title", "") or ""
        d = _short_desc_from_content(ex.get("content", ""))
        if d:
            core_lines.append(f"- {t} (특징: {d})")
        elif t:
            core_lines.append(f"- {t}")
    exhibit_summary = "\n".join(core_lines)

    # 분위기 재료 (다음 5개 전시물 title) — 세계관에 자연스럽게 흩뿌릴 풍경/소품
    atmosphere_titles = []
    for ex in exhibits[2:7]:
        t = ex.get("metadata", {}).get("title", "") or ""
        if t:
            atmosphere_titles.append(t)
    atmosphere_summary = ", ".join(atmosphere_titles) if atmosphere_titles else ""

    # zone 정체성 한 줄 — 모든 전시물 title을 모아 LLM이 분위기를 한눈에 파악하도록
    all_titles = []
    for ex in exhibits[:10]:
        t = ex.get("metadata", {}).get("title", "") or ""
        if t:
            all_titles.append(t)
    zone_identity_line = ", ".join(all_titles[:8]) if all_titles else ""

    principles_text = ", ".join(principles[:1])  # 원리 1개 (갈등 해결용)

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

    language_prompts = {
        "한국어": f"""너는 6~8세 어린이를 위한 감성적이고 신비로운 과학동화 작가야.

[재료 — 모두 CSV 실제 데이터 기반. 반드시 활용할 것]
※ 이 동화는 실재하는 전시관('{zone_name}')을 모티브로 한다. 아래 재료를 무시하고 무관한 설정을 만들지 말 것.

▶ 이 전시관의 정체성 (전시물 목록 — 분위기를 즉시 파악하라):
   {zone_identity_line}
   (예: 새, 공룡, 암석이 있다면 → 자연 관찰관. 회로/로봇이 있다면 → 미래 연구소.)

▶ 배경 분위기(직접 이름은 쓰지 말고 위 정체성을 살린 무대로 변형): {world}

▶ 주인공: 호기심 많은 어린이 '{protagonist}'
▶ 동반자(주인공과 대화하는 단짝): {companion}

▶ ★ 핵심 마법 아이템(아래 전시물 2개를 마법 도구/비밀 장치로 변형해서만 사용. 다른 마법 도구 발명 금지):
{exhibit_summary}

▶ 분위기 재료(이야기 곳곳에 풍경/소품/등장 생물로 자연스럽게 흩뿌려 등장시킬 것 — 최소 2개 이상 본문에 포함):
   {atmosphere_summary}

▶ 이야기의 갈등을 해결하는 단 하나의 과학 현상: {principles_text}

[개연성 규칙 — 매우 중요]
1) **간결한 3막 구조 (총 6~8문단) — 과학 현상이 이야기의 굵직한 축**:
   - 1막(2문단): {protagonist}의 평범한 순간 → **'{principles_text}'와 직접 관련된 이상한 사건** 발생 → "왜 이런 일이?"라는 명확한 하나의 목표.
   - 2막(3~4문단): [핵심 마법 아이템]을 시도 → **현상이 작게 한 번 일어남** (감각 묘사) → 한 번 실패 → 동반자와 함께 같은 현상이 반복되는 걸 관찰하며 **"어? 항상 이렇게 되네?"라는 패턴을 발견**.
   - 3막(1~2문단): **★ 아하 순간**: 주인공이 큰 소리로 깨달음 — "아, 이게 바로 **{principles_text}**(이)구나!" 그 원리를 이용해 위기를 해결 → 1막의 수수께끼도 같은 원리로 설명 → 따뜻한 마무리.
2) **인과 사슬**: 모든 장면은 "~ 때문에 → ~이 일어났다" 순서. 갑자기 새 도구·새 능력 등장 금지.
3) **아이템 제한**: 위에 적힌 [핵심 마법 아이템]만으로 위기를 해결. 새로운 마법/도구를 즉석에서 만들지 말 것.
4) **목표·이름 일관성**: 1막의 목표는 끝까지 유지, 주인공 '{protagonist}'와 동반자 이름은 절대 바뀌지 않음.
5) **★ 과학 표현 규칙 (가장 중요) — "흘려들어도 원리가 박히게"**:
   - **1~2막에서는 용어 사용 금지**. 현상만 감각으로 묘사: "밀자 거꾸로 튕겨 나왔어요", "빛이 둥근 물방울을 지나자 무지개로 흩어졌어요".
   - **3막의 '아하 순간'에서 단 한 번** 원리명('{principles_text}')을 큰따옴표 대사로 명명할 것. 이때 한 줄짜리 쉬운 설명 추가 (예: "물건을 밀면 그 물건도 똑같은 힘으로 나를 밀어내는 거였어!").
   - 결말 부근에서 그 원리명을 **한 번 더 짧게 회상**하면서 위기를 해결 (총 명명 횟수: 2~3회).
   - 강의·백과사전 톤은 절대 금지. 동반자도 같이 깨닫는 친구.
6) **문체 (6~8세 톤)**:
   - 의성어·의태어를 최소 3번 사용 (예: 폴짝폴짝, 윙윙, 반짝반짝, 살랑살랑, 또르르).
   - 짧은 문장 위주, 대사 비중 40% 이상.
   - 감각 묘사(소리/빛/냄새/촉감) 2개 이상 포함.
7) **금지 표현**: "놀이터", "전시물", "체험", "박물관" 같은 단어 절대 금지. 완전한 판타지 모험으로.
8) **결말**: 따뜻하고 희망적, 마지막 한 줄은 잠자리에 어울리는 다정한 인사.

[출력 형식]
- 첫 줄: 제목 (**굵게**)
- 빈 줄
- 본문: 6~8개 문단, 각 문단 2~3문장
- 총 분량: 약 1000~1400자
""",

        "English": f"""You are a tender, imaginative science-fairytale writer for children aged 6–8.{glossary_rules}

[Ingredients — all from REAL CSV data. You MUST use them; do not ignore.]
This story is inspired by a real exhibit zone ('{zone_name}'). Do not invent unrelated settings.

▶ Zone identity (full exhibit list — read the vibe at a glance):
   {zone_identity_line}
   (e.g. birds + dinosaurs + rocks → a nature observation hall. Circuits + robots → a future lab.)

▶ Setting atmosphere (don't name it literally; transform it into a stage that REFLECTS the identity above): {world}

▶ Protagonist: a curious child named '{protagonist}'
▶ Companion (talks with the hero, NOT an encyclopedia): {companion}

▶ ★ Core magical items (use ONLY these two; transform them into magical tools — DO NOT invent other magic items):
{exhibit_summary}

▶ Atmosphere ingredients (sprinkle these as scenery / creatures / props throughout the story — include at least 2 in the body):
   {atmosphere_summary}

▶ The single natural phenomenon that resolves the conflict: {principles_text}

[Coherence Rules — CRITICAL]
1) **Compact 3-act structure (6–8 paragraphs total) — the phenomenon is the BACKBONE of the plot**:
   - Act 1 (2 paragraphs): '{protagonist}'s ordinary moment → a strange event **directly tied to '{principles_text}'** → ONE clear goal ("I must find out why…").
   - Act 2 (3–4 paragraphs): try the magic item → **the phenomenon happens in a small way** (sensory description) → fail once → observe the SAME phenomenon repeating with the companion → "Huh, it always happens this way!" — a clear PATTERN.
   - Act 3 (1–2 paragraphs): **★ Aha moment**: the hero exclaims aloud — "Oh! This is **{principles_text}**!" Use that idea to solve the crisis → the Act-1 mystery is explained by the same idea → warm wrap-up.
2) **Cause-and-effect**: every scene "because of X → Y happened". No sudden new tools or powers.
3) **Item discipline**: only the listed magical items solve the crisis. No improvising new magic.
4) **Goal & name consistency**: Act-1 goal persists; '{protagonist}' and the companion's name NEVER change.
5) **★ Science visibility (most important) — "even a half-listening child must catch it"**:
   - **In Acts 1–2 do NOT use the term**. Show the phenomenon through senses only ("when she pushed it, it bounced back the other way").
   - **At the Act-3 aha moment, name '{principles_text}' EXACTLY ONCE in dialogue**, followed by a one-sentence kid-friendly explanation (e.g., "When you push something, it pushes you back just as hard!").
   - Mention the term ONE more time near the resolution as the hero applies it. (Total namings: 2–3.)
   - Never lecture. The companion discovers WITH the hero, not as a teacher.
6) **Style (ages 6–8)**:
   - Use at least 3 onomatopoeia / mimetic words (whoosh, sparkle-sparkle, plip-plop, thump-thump).
   - Short sentences, dialogue ≥ 40%.
   - At least 2 sensory details (sound, light, smell, texture).
7) **Forbidden words**: "playground", "exhibit", "field trip", "museum" — write it as a true fantasy adventure.
8) **Ending**: warm, hopeful, final line suitable for bedtime.

[Output format]
- Line 1: **Bold title**
- Blank line
- Body: 6–8 paragraphs, each 2–3 sentences
- Length: about 1000–1400 characters total, child-friendly.""",

        "日本語": f"""あなたは6〜8歳の子ども向けに、やさしくて不思議な科学ファンタジーを書く作家です。{glossary_rules}

[素材 — すべて実在のCSVデータ。必ず活用すること]
この物語は実在の展示館（『{zone_name}』）をモチーフにする。下の素材を無視して無関係な設定を作らない。

▶ 展示館の正体（展示物リスト — 雰囲気をひと目で把握）:
   {zone_identity_line}
   （例：鳥・恐竜・岩なら自然観察館。回路・ロボットなら未来の研究所。）

▶ 舞台の雰囲気（言葉自体は使わず、上の正体を活かした舞台に変形）: {world}

▶ 主人公: 好奇心いっぱいの子ども『{protagonist}』
▶ 相棒（主人公と話す友だち。百科事典ではない）: {companion}

▶ ★ 中心となる魔法のアイテム（下の2点だけを魔法の道具に変えて使う。他の魔法は作らない）:
{exhibit_summary}

▶ 雰囲気の素材（物語の風景・生き物・小道具として自然に散りばめる — 本文に最低2つ以上登場させる）:
   {atmosphere_summary}

▶ 物語の事件を解く、たったひとつの自然現象: {principles_text}

[筋の通った物語ルール — 最重要]
1) **コンパクトな3幕構成（全6〜8段落）— 科学現象が物語の太い背骨になる**:
   - 第1幕（2段落）: 『{protagonist}』のふつうの瞬間 → **『{principles_text}』に直接かかわる不思議な出来事** → 「どうして？」というひとつの明確な目的。
   - 第2幕（3〜4段落）: 魔法のアイテムを試す → **現象が小さく一度起きる**（五感で描写） → 一度失敗 → 相棒と一緒に同じ現象が繰り返されるのを観察 → 「あれ？いつもこうなる！」と **パターンに気づく**。
   - 第3幕（1〜2段落）: **★ アハ体験**: 主人公が声をあげて気づく — 「あっ、これって **{principles_text}** だ！」その考えで危機を解決 → 1幕の謎も同じ考えで説明 → あたたかい締めくくり。
2) **因果のつながり**: すべての場面は「〜だから → 〜になった」の順。突然の新しい道具・能力は禁止。
3) **アイテム制限**: 上に挙げた魔法のアイテムだけで危機を解決すること。即興で別の魔法を作らない。
4) **目的と名前の一貫性**: 1幕の目的は最後まで保たれ、『{protagonist}』と相棒の名前は最後まで変えない。
5) **★ 科学の見える化（最重要）— 「聞き流しても原理が頭に残るように」**:
   - **第1〜2幕では用語を使わない**。現象だけを五感で描写（「押すと、ぽいんと逆にはねかえった」など）。
   - **第3幕のアハの瞬間でちょうど一度だけ** 用語『{principles_text}』をセリフで名づける。続けて子ども向けの一文説明（例：「ものを押すと、そのものも同じ強さで自分を押しかえすんだ！」）。
   - 結末近くでもう一度だけ、主人公がその用語を使って危機を解く（合計命名2〜3回）。
   - 講義・百科事典口調は厳禁。相棒は先生ではなく、いっしょに発見する友だち。
6) **文体（6〜8歳向け）**:
   - 擬音語・擬態語を3回以上使う（ぴょんぴょん、ぴかぴか、ふわふわ、ころころ、ぽとんなど）。
   - 短い文中心、会話の割合は40%以上。
   - 五感の描写（音・光・におい・感触）を2つ以上入れる。
7) **禁句**: 「遊び場」「展示」「体験」「博物館」などは禁止。本物のファンタジー冒険として書く。
8) **結末**: あたたかく希望的、最後の一行は寝かしつけにふさわしいやさしい言葉。

[出力形式]
- 1行目: **太字のタイトル**
- 空行
- 本文: 6〜8段落、各段落2〜3文
- 分量: 全体で約1000〜1400字""",

        "中文": f"""你是一位为6〜8岁儿童写作的温柔而充满想象力的科学童话作家。{glossary_rules}

[素材 — 全部来自真实CSV数据，必须使用]
本童话以真实存在的展馆（『{zone_name}』）为蓝本。不要忽略以下素材去编造无关设定。

▶ 展馆身份（展品清单——一眼看清氛围）:
   {zone_identity_line}
   （例：有鸟、恐龙、岩石 → 自然观察馆。有电路、机器人 → 未来研究所。）

▶ 场景氛围（不直接写词，把上述身份活成舞台）: {world}

▶ 主人公: 好奇心旺盛的孩子『{protagonist}』
▶ 伙伴（与主人公对话的朋友，不是百科全书）: {companion}

▶ ★ 核心魔法道具（仅用以下两件展品改写成魔法道具，不要发明其他魔法）:
{exhibit_summary}

▶ 氛围素材（作为风景／生物／道具散布于故事中——正文里至少出现2个以上）:
   {atmosphere_summary}

▶ 推动并解决故事冲突的唯一自然现象: {principles_text}

[开展规则 — 至关重要]
1) **紧凑的三幕结构（共6〜8段）— 科学现象是故事的主干脊梁**:
   - 第一幕（2段）: 『{protagonist}』的平凡时刻 → **直接与『{principles_text}』相关的奇怪事件** → 一个明确目标（"我要弄清楚为什么…"）。
   - 第二幕（3〜4段）: 摆弄魔法道具 → **现象小小地发生一次**（用五感描写）→ 失败一次 → 与伙伴一起观察同一现象反复出现 → "咦？怎么每次都这样！"——发现 **规律**。
   - 第三幕（1〜2段）: **★ 顿悟时刻**: 主人公大声领悟——"啊，原来这就是 **{principles_text}**！"用这个原理化解危机 → 第一幕的谜团也用同一个原理解释 → 温馨收尾。
2) **因果链条**: 所有情节按"因为……所以……"顺序推进。不可突然出现新道具或新能力。
3) **道具限制**: 仅用上面列出的魔法道具来解决危机，不要临时发明新的魔法。
4) **目标与名字一致**: 第一幕设定的目标贯穿到底；『{protagonist}』与伙伴的名字自始至终不变。
5) **★ 让科学"看得见"（最重要）— "就算听漏也能记住原理"**:
   - **第一、二幕中绝不使用术语**，只用五感描写现象（如"她一推，它就反方向弹了回去"）。
   - **第三幕的顿悟瞬间，恰好命名一次** 术语『{principles_text}』（用对话），紧跟一句儿童化解释（例："推一下东西，那东西也会用一样的力气把你推回来！"）。
   - 接近结尾再让主人公简短复述一次该术语来解决危机（合计命名2〜3次）。
   - 严禁讲课口吻或百科全书腔调。伙伴不是老师，是和主人公一起发现的朋友。
6) **文体（6〜8岁口吻）**:
   - 至少使用3个拟声词或叠词（蹦蹦跳跳、闪闪、咕噜咕噜、扑通、轻飘飘）。
   - 以短句为主，对话占比≥40%。
   - 至少加入2处感官描写（声音、光、气味、触感）。
7) **禁用词**: "游乐场""展品""体验""博物馆"等绝对不写。要写成真正的奇幻冒险。
8) **结尾**: 温暖且充满希望，最后一句是适合睡前读的温柔话语。

[输出格式]
- 第1行: **加粗标题**
- 空行
- 正文: 6〜8段，每段2〜3句
- 全文约1000〜1400字""",
    }

    prompt = language_prompts.get(language, language_prompts["한국어"])

    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.8)
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"동화 생성 오류: {e}")
        return None


def text_to_audiobook(story_text, language="한국어", voice_override=None, speed_override=None):
    """텍스트를 오디오북으로 변환 (ElevenLabs > Naver > OpenAI fallback)"""

    eleven_key = os.environ.get("ELEVENLABS_API_KEY")
    if (not eleven_key) and hasattr(st, "secrets"):
        eleven_key = _safe_secret_get("ELEVENLABS_API_KEY", "")

    eleven_voice_id = os.environ.get("ELEVENLABS_VOICE_ID")
    if (not eleven_voice_id) and hasattr(st, "secrets"):
        eleven_voice_id = _safe_secret_get("ELEVENLABS_VOICE_ID", "")
    if not eleven_voice_id:
        eleven_voice_id = "21m00Tcm4TlvDq8ikWAM"

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
            print(f"ElevenLabs TTS 오류: status={resp.status_code}, body={resp.text[:500]}")
            return None
        except Exception as e:
            print(f"ElevenLabs TTS 호출 오류: {e}")
            return None

    ncp_key_id = os.environ.get("X_NCP_APIGW_API_KEY_ID") or os.environ.get("X-NCP-APIGW-API-KEY-ID")
    ncp_key = os.environ.get("X_NCP_APIGW_API_KEY") or os.environ.get("X-NCP-APIGW-API-KEY")
    if (not ncp_key_id) and hasattr(st, "secrets"):
        ncp_key_id = _safe_secret_get("X_NCP_APIGW_API_KEY_ID", "") or _safe_secret_get("X-NCP-APIGW-API-KEY-ID", "")
    if (not ncp_key) and hasattr(st, "secrets"):
        ncp_key = _safe_secret_get("X_NCP_APIGW_API_KEY", "") or _safe_secret_get("X-NCP-APIGW-API-KEY", "")

    if ncp_key_id and ncp_key and language == "한국어":
        url = "https://naveropenapi.apigw.ntruss.com/voice/v1/tts"
        headers = {
            "X-NCP-APIGW-API-KEY-ID": ncp_key_id,
            "X-NCP-APIGW-API-KEY": ncp_key,
        }

        speaker = voice_override or "nara"
        speed = "1" if speed_override is None else str(speed_override)
        data = {
            "speaker": speaker,
            "speed": speed,
            "text": story_text,
        }

        try:
            resp = requests.post(url, headers=headers, data=data, timeout=60)
            if resp.status_code == 200 and resp.content:
                return resp.content
            print(f"네이버 TTS 오류: status={resp.status_code}, body={resp.text[:500]}")
            return None
        except Exception as e:
            print(f"네이버 TTS 호출 오류: {e}")
            return None

    voice_map = {
        "한국어": "alloy",
        "English": "alloy",
        "日本語": "shimmer",
        "中文": "fable"
    }

    voice = voice_override or voice_map.get(language, "nova")
    # 속도: 기본 1.0 (더 느리고 편안하게)
    speed = 1.0 if speed_override is None else speed_override

    try:
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input=story_text,
            speed=speed
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
        st.markdown(f"#### {text['select_zone']}")

        selected = []

        st.markdown(f"##### {text['floor1']}")
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

        st.markdown(f"##### {text['floor2']}")
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

    with tab_quiz:
        selected_zones = _render_zone_selector("quiz")

        if selected_zones:
            st.markdown("---")
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
            for zone in selected_zones:
                zone_rows = all_zone_rows.get(zone, [])
                selected_kw, selected_disp = _render_zone_header(zone, zone_rows, mode="quiz", llm=llm)

                if selected_kw:
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
                                # 영상 row에서 description도 같이 첨부
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

                    if quiz_cache_key not in st.session_state:
                        if st.button(make_quiz_label, key=f"btn_make_quiz_{zone}_{selected_kw}"):
                            with st.spinner(text["quiz_generating"]):
                                quiz = generate_quiz(
                                    zone, selected_kw, llm, language_mode,
                                    variation_seed=st.session_state[seed_key],
                                )
                                st.session_state[quiz_cache_key] = quiz or {}
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
                                st.rerun()
                        else:
                            st.warning("퀴즈 생성에 실패했습니다.")
                            if st.button(f"🔄 {make_quiz_label}", key=f"btn_retry_quiz_{zone}_{selected_kw}"):
                                st.session_state.pop(quiz_cache_key, None)
                                st.rerun()
        else:
            st.info(text["pick_zone_hint"])

    with tab_question:
        selected_zones = _render_zone_selector("question")

        if selected_zones:
            st.markdown("---")
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
            for zone in selected_zones:
                zone_rows = all_zone_rows.get(zone, [])
                _render_zone_header(zone, zone_rows, mode="question", llm=llm)

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
        selected_zones_story = _select_zones_by_group("story", language_mode=language_mode)

        # 선택된 존 표시 (다국어)
        selected_disp_names = [_display_zone_name(z) for z in selected_zones_story]
        selected_label_text = {
            "한국어": "선택된 놀이터",
            "English": "Selected zones",
            "日本語": "選んだゾーン",
            "中文": "已选区域",
        }.get(language_mode, "Selected zones")
        please_select_text = {
            "한국어": "놀이터를 선택해주세요",
            "English": "Please select at least one zone.",
            "日本語": "ゾーンを選んでください。",
            "中文": "请选择区域。",
        }.get(language_mode, "Please select at least one zone.")
        if selected_zones_story:
            st.info(f"{selected_label_text}: {', '.join(selected_disp_names)}")
        else:
            st.warning(please_select_text)

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
            # 동화는 외국어 모드에서 직접 그 언어로 생성되므로 별도의 KO 원문이 없음.
            # → debug_show_korean 또는 debug_backtranslate 가 켜지면 한국어 역번역본을 노출 (사실상 동일한 자료).
            if language_mode != "한국어" and (debug_show_korean or debug_backtranslate):
                bt_story = _backtranslate_to_korean_cached(st.session_state[story_state_key], language_mode)
                if bt_story:
                    label = "🇰🇷 한국어로 보기 (디버그)" if debug_show_korean else "BT (동화 본문 역번역)"
                    with st.expander(label, expanded=False):
                        st.markdown(bt_story)

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
