# voice.py - 음성 입출력 처리 (voice_handler.py에서 이름 변경)
import os
import asyncio
from openai import OpenAI
import tempfile
import streamlit as st
import base64
import re
import requests
import edge_tts

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def _safe_secret_get(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def _get_secret(key: str, default: str = "") -> str:
    """환경변수 → st.secrets 순으로 조회."""
    val = os.environ.get(key)
    if val:
        return val
    if hasattr(st, "secrets"):
        return _safe_secret_get(key, default)
    return default

def speech_to_text(audio_bytes):
    """Convert speech to text using OpenAI Whisper"""
    print("=== Speech-to-Text Debug ===")
    print(f"Audio bytes received: {len(audio_bytes) if audio_bytes else 0}")
    
    try:
        # Check if audio_bytes is valid
        if not audio_bytes or len(audio_bytes) < 100:
            print("Audio bytes too small or empty")
            return None
        
        # Save audio bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
        
        # Transcribe using Whisper with more flexible settings
        with open(temp_audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
                temperature=0.0  # More deterministic
            )
        
        # Clean up temp file
        os.unlink(temp_audio_path)
        
        # Check if transcript is valid
        if transcript and len(transcript.strip()) > 0:
            return transcript.strip()
        else:
            print("Empty transcript received")
            return None
    
    except Exception as e:
        print(f"Speech-to-text error: {e}")
        # Clean up temp file if it exists
        try:
            if 'temp_audio_path' in locals():
                os.unlink(temp_audio_path)
        except:
            pass
        return None

def _tts_elevenlabs(text: str, language: str = "ko") -> bytes | None:
    """ElevenLabs TTS 호출. 키가 없거나 실패하면 None."""
    eleven_key = _get_secret("ELEVENLABS_API_KEY")
    if not eleven_key:
        return None
    voice_map = {
        "ko": "uyVNoMrnUku1dZyVEXwD",
        "en": "8LVfoRdkh4zgjr8v5ObE",
        "ja": "3JDquces8E8bkmvbh6Bc",
        "zh": "vZZLclMx4wouUtKBRfZn",
    }
    voice_id = _get_secret("ELEVENLABS_VOICE_ID") or voice_map.get(language, "uyVNoMrnUku1dZyVEXwD")
    model_id = _get_secret("ELEVENLABS_MODEL_ID") or "eleven_multilingual_v2"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": eleven_key, "accept": "audio/mpeg", "content-type": "application/json"}
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {"stability": 0.45, "similarity_boost": 0.75, "style": 0.0, "use_speaker_boost": True},
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200 and resp.content:
            print(f"[TTS] ElevenLabs voice_id={voice_id} ok ({len(resp.content)} bytes)")
            return resp.content
        print(f"[TTS] ElevenLabs error status={resp.status_code} body={resp.text[:300]}")
        return None
    except Exception as e:
        print(f"[TTS] ElevenLabs exception: {e}")
        return None


EDGE_TTS_VOICES = {
    "ko": "ko-KR-InJoonNeural",
    "en": "en-US-AriaNeural",
    "ja": "ja-JP-NanamiNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
}


async def _tts_edge_async(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data


def _tts_edge(text: str, language: str = "ko") -> bytes | None:
    """edge-tts (Microsoft Neural, 무료, 빠름)."""
    voice = EDGE_TTS_VOICES.get(language, "ko-KR-SunHiNeural")
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        audio_data = loop.run_until_complete(_tts_edge_async(text, voice))
        loop.close()
        print(f"[TTS] edge-tts voice={voice} ok ({len(audio_data)} bytes)")
        return audio_data if audio_data else None
    except Exception as e:
        print(f"[TTS] edge-tts exception: {e}")
        return None


def _tts_openai(text: str, language: str = "ko") -> bytes | None:
    """OpenAI TTS 폴백."""
    voice_map = {
        "ko": "alloy",
        "en": "nova",
        "ja": "shimmer",
        "zh": "fable",
    }
    voice = voice_map.get(language, "alloy")
    try:
        response = client.audio.speech.create(model="tts-1", voice=voice, input=text)
        print(f"[TTS] OpenAI voice={voice} ok")
        return response.content
    except Exception as e:
        print(f"[TTS] OpenAI exception: {e}")
        return None


def text_to_speech(text, language="ko"):
    """TTS 우선순위: ElevenLabs(키 있을 때) → edge-tts → OpenAI 폴백."""
    text = preprocess_tts_text(text, language=language)
    if not text:
        return None

    # 1) ElevenLabs (사용자 설정 음성)
    audio = _tts_elevenlabs(text, language=language)
    if audio:
        return audio

    # 2) edge-tts 폴백
    audio = _tts_edge(text, language=language)
    if audio:
        return audio

    # 3) OpenAI 폴백
    return _tts_openai(text, language=language)


def get_tts_cache_namespace(language: str = "ko") -> str:
    """캐시 키. 음성 변경 시 자동 재생성되도록 voice 이름 포함."""
    eleven_key = _get_secret("ELEVENLABS_API_KEY")
    if eleven_key:
        voice_map = {
            "ko": "uyVNoMrnUku1dZyVEXwD",
            "en": "8LVfoRdkh4zgjr8v5ObE",
            "ja": "3JDquces8E8bkmvbh6Bc",
            "zh": "vZZLclMx4wouUtKBRfZn",
        }
        voice_id = _get_secret("ELEVENLABS_VOICE_ID") or voice_map.get(language, "uyVNoMrnUku1dZyVEXwD")
        model_id = _get_secret("ELEVENLABS_MODEL_ID") or "eleven_multilingual_v2"
        return f"elevenlabs::{model_id}::{voice_id}"
    voice = EDGE_TTS_VOICES.get(language, "ko-KR-InJoonNeural")
    return f"edge-tts::{voice}"

def get_language_code(language_mode):
    """Convert language mode to language code"""
    language_codes = {
        "한국어": "ko",
        "English": "en",
        "日本語": "ja",
        "中文": "zh"
    }
    return language_codes.get(language_mode, "ko")

def autoplay_audio(audio_bytes):
    """Auto-play audio in Streamlit"""
    try:
        b64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        print(f"Autoplay error: {e}")


def preprocess_tts_text(text: str, language: str = "ko") -> str:
    if not text:
        return text
    # 마크다운 기호 제거 (edge-tts가 # ** * 등을 그대로 읽지 않도록)
    text = re.sub(r'#{1,6}\s*', '', text)          # ### 제목
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)  # **bold** / *italic*
    text = re.sub(r'`[^`]*`', '', text)             # `code`
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)  # [link](url) → link䞬식만
    text = re.sub(r'^[-*+]\s+', '', text, flags=re.MULTILINE)  # 목록 기호
    text = re.sub(r'\n{2,}', ' ', text)             # 여러 줄바꿈 → 공백
    text = text.strip()
    if language != "ko":
        return text

    # 번호 목록(줄 시작 "1. " "2. ")을 한자어로 변환 — edge-tts가 "한번/두번"으로 읽는 것 방지
    _sino = {1:'일',2:'이',3:'삼',4:'사',5:'오',6:'육',7:'칠',8:'팔',9:'구',10:'십'}
    def _list_num(m):
        n = int(m.group(1))
        return _sino.get(n, str(n)) + '. '
    text = re.sub(r'(?m)^(\d+)\.\s+', _list_num, text)

    # 층(層)은 한자어로 읽는 게 자연스러움: "1층" → "일층"
    def _floor_sino(m):
        n = int(m.group(1))
        return _sino.get(n, str(n)) + '층'
    text = re.sub(r'(\d+)층', _floor_sino, text)

    def _format_time(hh: str, mm: str) -> str:
        h = int(hh)
        m = int(mm)
        if m == 0:
            return f"{h}시"
        return f"{h}시 {m}분"

    def _repl(match: re.Match) -> str:
        # match.groups() 로 안전하게 추출 (그룹 수가 다른 두 패턴에서 공용 사용)
        groups = match.groups()
        h1 = groups[0] if len(groups) > 0 else None
        m1 = groups[1] if len(groups) > 1 else None
        h2 = groups[2] if len(groups) > 2 else None
        m2 = groups[3] if len(groups) > 3 else None
        if h2 and m2:
            return f"{_format_time(h1, m1)}부터 {_format_time(h2, m2)}까지"
        return _format_time(h1, m1)

    # 범위 패턴(09:00~18:00) 먼저 처리 → 단일 패턴(11:40)
    text = re.sub(r"\b(\d{1,2})\s*:\s*(\d{2})\s*[~∼\-–—]\s*(\d{1,2})\s*:\s*(\d{2})\b", _repl, text)
    text = re.sub(r"\b(\d{1,2})\s*:\s*(\d{2})\b", _repl, text)
    return text
