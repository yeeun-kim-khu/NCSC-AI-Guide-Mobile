# voice.py - 음성 입출력 처리 (voice_handler.py에서 이름 변경)
import os
from openai import OpenAI
import tempfile
import streamlit as st
import base64
import re
import requests

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
                language="ko",  # Korean language
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

def _tts_elevenlabs(text: str) -> bytes | None:
    """ElevenLabs TTS 호출. 키가 없거나 실패하면 None."""
    eleven_key = _get_secret("ELEVENLABS_API_KEY")
    if not eleven_key:
        return None

    voice_id = _get_secret("ELEVENLABS_VOICE_ID") or "21m00Tcm4TlvDq8ikWAM"
    model_id = _get_secret("ELEVENLABS_MODEL_ID") or "eleven_multilingual_v2"

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": eleven_key,
        "accept": "audio/mpeg",
        "content-type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.45,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True,
        },
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200 and resp.content:
            print(f"[TTS] ElevenLabs voice_id={voice_id} model={model_id} ok ({len(resp.content)} bytes)")
            return resp.content
        print(f"[TTS] ElevenLabs error status={resp.status_code} body={resp.text[:300]}")
        return None
    except Exception as e:
        print(f"[TTS] ElevenLabs exception: {e}")
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
    """TTS 우선순위: ElevenLabs(키 있을 때) → OpenAI 폴백."""
    text = preprocess_tts_text(text, language=language)
    if not text:
        return None

    # 1) ElevenLabs (사용자 설정 음성)
    audio = _tts_elevenlabs(text)
    if audio:
        return audio

    # 2) OpenAI 폴백
    return _tts_openai(text, language=language)


def get_tts_cache_namespace(language: str = "ko") -> str:
    """캐시 키. ElevenLabs 사용 시 voice_id를 키에 포함해야 음성 변경 시 재생성됨."""
    eleven_key = _get_secret("ELEVENLABS_API_KEY")
    if eleven_key:
        voice_id = _get_secret("ELEVENLABS_VOICE_ID") or "21m00Tcm4TlvDq8ikWAM"
        model_id = _get_secret("ELEVENLABS_MODEL_ID") or "eleven_multilingual_v2"
        return f"elevenlabs::{model_id}::{voice_id}"

    voice_map = {"ko": "alloy", "en": "nova", "ja": "shimmer", "zh": "fable"}
    voice = voice_map.get(language, "alloy")
    return f"openai::tts-1::{voice}"

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
    if language != "ko":
        return text

    def _format_time(hh: str, mm: str) -> str:
        h = int(hh)
        m = int(mm)
        if m == 0:
            return f"{h}시"
        return f"{h}시 {m}분"

    def _repl(match: re.Match) -> str:
        # 두 정규식이 이 함수를 공유: 범위(HH:MM~HH:MM, 4그룹) / 단일(HH:MM, 2그룹)
        # 단일 매치에서는 group(3)/group(4) 가 없어 IndexError 발생 → groups() 기반으로 안전 접근.
        groups = match.groups()
        h1, m1 = groups[0], groups[1]
        h2 = groups[2] if len(groups) >= 4 else None
        m2 = groups[3] if len(groups) >= 4 else None
        if h2 and m2:
            return f"{_format_time(h1, m1)}부터 {_format_time(h2, m2)}까지"
        return _format_time(h1, m1)

    text = re.sub(r"\b(\d{1,2})\s*:\s*(\d{2})\s*[~∼-]\s*(\d{1,2})\s*:\s*(\d{2})\b", _repl, text)
    text = re.sub(r"\b(\d{1,2})\s*:\s*(\d{2})\b", _repl, text)
    return text
