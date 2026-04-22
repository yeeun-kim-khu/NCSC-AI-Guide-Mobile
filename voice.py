# voice.py - 음성 입출력 처리 (voice_handler.py에서 이름 변경)
import os
from openai import OpenAI
import tempfile
import streamlit as st
import base64
import re

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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

def text_to_speech(text, language="ko"):
    """Convert text to speech using OpenAI TTS"""
    text = preprocess_tts_text(text, language=language)
    try:
        # Select voice based on language
        voice_map = {
            "ko": "alloy",  # Korean - natural voice
            "en": "nova",   # English
            "ja": "shimmer", # Japanese
            "zh": "fable"   # Chinese
        }
        
        voice = voice_map.get(language, "alloy")
        
        # Generate speech
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        
        # Return audio bytes
        return response.content
    
    except Exception as e:
        print(f"Text-to-speech error: {e}")
        return None


def get_tts_cache_namespace(language: str = "ko") -> str:
    voice_map = {
        "ko": "alloy",
        "en": "nova",
        "ja": "shimmer",
        "zh": "fable",
    }
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
        h1, m1 = match.group(1), match.group(2)
        h2, m2 = match.group(3), match.group(4)
        if h2 and m2:
            return f"{_format_time(h1, m1)}부터 {_format_time(h2, m2)}까지"
        return _format_time(h1, m1)

    text = re.sub(r"\b(\d{1,2})\s*:\s*(\d{2})\s*[~∼-]\s*(\d{1,2})\s*:\s*(\d{2})\b", _repl, text)
    text = re.sub(r"\b(\d{1,2})\s*:\s*(\d{2})\b", _repl, text)
    return text
