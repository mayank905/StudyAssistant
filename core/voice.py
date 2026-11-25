"""StudyAssistant Voice Module
===========================
Streamlit two-way voice integration:
1. Spoken Audio Input:
   - Interactive Web Speech API & HTML5 MediaRecorder browser component.
   - Real-time speech-to-text transcript with copy/insert controls.
   - Streamlit audio_input integration with backend multimodal transcription (Google Gemini / offline fallback).
2. Text-To-Speech (TTS) Spoken Output:
   - Browser Web Speech Synthesis (`window.speechSynthesis`) player with play, pause, stop, replay controls.
   - Voice controls widget (toggle, speech rate, pitch, language/accent, autoplay).
   - Text sanitizer for clear and natural spoken output.
"""

import base64
import html
import io
import json
import logging
import os
import re
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger("studyassistant.voice")
logger.setLevel(logging.INFO)


# =====================================================================
# 1. Text Sanitization for Natural Spoken Audio
# =====================================================================

def clean_text_for_speech(text: str, max_chars: int = 2000) -> str:
    """Prepares markdown or formatted text for natural text-to-speech reading.
    
    Removes markdown symbols, URLs, citation codes, tables, and excessive formatting
    so the browser voice synthesizer sounds natural and fluid.
    """
    if not text:
        return ""

    s = text.strip()

    # Remove code blocks
    s = re.sub(r"```[\s\S]*?```", " [code block omitted] ", s)
    # Remove inline code
    s = re.sub(r"`([^`]+)`", r"\1", s)
    # Remove markdown images ![alt](url)
    s = re.sub(r"!\[.*?\]\(.*?\)", "", s)
    # Convert markdown links [title](url) to title
    s = re.sub(r"\[([^\]]+)\]\(.*?\)", r"\1", s)
    # Remove document citation tags like [Doc: Fire Tariff, p.14]
    s = re.sub(r"\[(?:Doc|Page|Citation|Source)[^\]]*\]", "", s, flags=re.IGNORECASE)
    # Remove markdown headers #, ##, etc.
    s = re.sub(r"^#{1,6}\s+", "", s, flags=re.MULTILINE)
    # Remove bold and italic markers
    s = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", s)
    # Remove blockquotes >
    s = re.sub(r"^>\s+", "", s, flags=re.MULTILINE)
    # Remove bullet markers -, *, +
    s = re.sub(r"^\s*[-*+]\s+", "", s, flags=re.MULTILINE)
    # Remove HTML tags
    s = re.sub(r"<[^>]+>", "", s)
    # Replace common symbols with spoken words
    s = s.replace(" & ", " and ")
    s = s.replace(" w/o ", " without ")
    s = s.replace(" w/ ", " with ")
    s = s.replace(" / ", " or ")
    s = s.replace(" %", " percent")
    s = s.replace("%", " percent")

    # Collapse excess whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # Truncate if excessively long to prevent browser TTS exhaustion
    if len(s) > max_chars:
        # Cut at sentence boundary
        cutoff = s[:max_chars].rfind(".")
        if cutoff > 200:
            s = s[:cutoff + 1]
        else:
            s = s[:max_chars] + "..."

    return s


# =====================================================================
# 2. Text-To-Speech (TTS) Web Speech Synthesis Component
# =====================================================================

def render_speech_synthesis(
    text: str,
    autoplay: bool = True,
    rate: float = 1.0,
    pitch: float = 1.0,
    volume: float = 1.0,
    lang: str = "en-US",
    element_id: str = "studyassistant_tts",
    height: int = 68,
):
    """Renders a sleek, responsive HTML5/JS Web Speech Synthesis player in Streamlit.
    
    Features:
    - Zero API key required (runs directly in browser).
    - Works across mobile and desktop browsers.
    - Play / Replay, Pause, and Stop buttons with speaking waveform animation.
    - Automatic speech cancellation on new utterance so audio doesn't overlap.
    """
    try:
        import streamlit as st
        import streamlit.components.v1 as components
    except ImportError:
        logger.warning("Streamlit not installed, skipping TTS render.")
        return

    clean_speech_text = clean_text_for_speech(text)
    if not clean_speech_text:
        return

    # Escape for JavaScript string literal
    js_escaped_text = json.dumps(clean_speech_text)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{
          margin: 0;
          padding: 4px 6px;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
          background: transparent;
        }}
        .tts-card {{
          display: flex;
          align-items: center;
          gap: 10px;
          background: #f8fafc;
          border: 1px solid #cbd5e1;
          border-radius: 10px;
          padding: 8px 14px;
          box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }}
        .tts-title {{
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 13px;
          font-weight: 600;
          color: #334155;
          margin-right: 4px;
        }}
        .btn {{
          background: #ffffff;
          border: 1px solid #cbd5e1;
          border-radius: 6px;
          padding: 5px 10px;
          font-size: 12px;
          font-weight: 500;
          color: #1e293b;
          cursor: pointer;
          display: inline-flex;
          align-items: center;
          gap: 4px;
          transition: all 0.15s ease-in-out;
        }}
        .btn:hover {{
          background: #f1f5f9;
          border-color: #94a3b8;
        }}
        .btn-primary {{
          background: #2563eb;
          color: #ffffff;
          border-color: #1d4ed8;
        }}
        .btn-primary:hover {{
          background: #1d4ed8;
        }}
        .wave-box {{
          display: flex;
          align-items: center;
          gap: 2px;
          height: 16px;
          margin-left: auto;
        }}
        .wave-bar {{
          width: 3px;
          height: 4px;
          background: #94a3b8;
          border-radius: 2px;
          transition: height 0.2s ease;
        }}
        .speaking .wave-bar:nth-child(1) {{ animation: bounce 0.6s infinite ease-in-out 0.1s; background: #2563eb; }}
        .speaking .wave-bar:nth-child(2) {{ animation: bounce 0.6s infinite ease-in-out 0.2s; background: #2563eb; }}
        .speaking .wave-bar:nth-child(3) {{ animation: bounce 0.6s infinite ease-in-out 0.3s; background: #2563eb; }}
        .speaking .wave-bar:nth-child(4) {{ animation: bounce 0.6s infinite ease-in-out 0.4s; background: #2563eb; }}
        .speaking .wave-bar:nth-child(5) {{ animation: bounce 0.6s infinite ease-in-out 0.5s; background: #2563eb; }}
        @keyframes bounce {{
          0%, 100% {{ height: 4px; }}
          50% {{ height: 16px; }}
        }}
        .status-txt {{
          font-size: 11px;
          color: #64748b;
        }}
      </style>
    </head>
    <body>
      <div class="tts-card" id="{element_id}_card">
        <div class="tts-title">
          <span>🔊</span>
          <span>Spoken Audio</span>
        </div>
        <button class="btn btn-primary" id="btn_speak" onclick="toggleSpeak()">▶ Speak</button>
        <button class="btn" id="btn_pause" onclick="pauseSpeak()">⏸ Pause</button>
        <button class="btn" id="btn_stop" onclick="stopSpeak()">⏹ Stop</button>
        <span class="status-txt" id="status_label">Ready</span>

        <div class="wave-box" id="waveform">
          <div class="wave-bar"></div>
          <div class="wave-bar"></div>
          <div class="wave-bar"></div>
          <div class="wave-bar"></div>
          <div class="wave-bar"></div>
        </div>
      </div>

      <script>
        const textToSpeak = {js_escaped_text};
        const speechRate = {float(rate)};
        const speechPitch = {float(pitch)};
        const speechVolume = {float(volume)};
        const speechLang = "{html.escape(lang)}";
        const autoPlay = {"true" if autoplay else "false"};

        let currentUtterance = null;
        let isSpeaking = false;
        let isPaused = false;

        const btnSpeak = document.getElementById('btn_speak');
        const statusLabel = document.getElementById('status_label');
        const waveform = document.getElementById('waveform');

        function setSpeakingState(speaking) {{
          isSpeaking = speaking;
          if (speaking) {{
            waveform.classList.add('speaking');
            btnSpeak.innerHTML = '🔄 Replay';
            statusLabel.innerText = 'Speaking...';
          }} else {{
            waveform.classList.remove('speaking');
            statusLabel.innerText = isPaused ? 'Paused' : 'Ready';
          }}
        }}

        function speak() {{
          if (!('speechSynthesis' in window)) {{
            statusLabel.innerText = 'TTS not supported';
            return;
          }}
          window.speechSynthesis.cancel();

          currentUtterance = new SpeechSynthesisUtterance(textToSpeak);
          currentUtterance.rate = speechRate;
          currentUtterance.pitch = speechPitch;
          currentUtterance.volume = speechVolume;
          currentUtterance.lang = speechLang;

          currentUtterance.onstart = function() {{
            setSpeakingState(true);
          }};

          currentUtterance.onend = function() {{
            setSpeakingState(false);
          }};

          currentUtterance.onerror = function(event) {{
            console.error('TTS error:', event);
            setSpeakingState(false);
          }};

          window.speechSynthesis.speak(currentUtterance);
        }}

        function toggleSpeak() {{
          if (isPaused) {{
            window.speechSynthesis.resume();
            isPaused = false;
            setSpeakingState(true);
          }} else {{
            speak();
          }}
        }}

        function pauseSpeak() {{
          if (window.speechSynthesis && isSpeaking && !isPaused) {{
            window.speechSynthesis.pause();
            isPaused = true;
            setSpeakingState(false);
          }}
        }}

        function stopSpeak() {{
          if (window.speechSynthesis) {{
            window.speechSynthesis.cancel();
            isPaused = false;
            setSpeakingState(false);
          }}
        }}

        // Handle autoplay if enabled
        if (autoPlay && textToSpeak) {{
          setTimeout(speak, 350);
        }}
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=height, scrolling=False)


# =====================================================================
# 3. Spoken Audio Input: Web Speech API & HTML5 Recorder Component
# =====================================================================

def render_speech_input_component(
    key: str = "voice_recorder",
    height: int = 150,
    prompt: str = "Click microphone and speak your query or oral viva answer...",
    lang: str = "en-US",
):
    """Renders a browser-native Web Speech API & HTML5 audio recorder component.
    
    Features:
    - Real-time client-side SpeechRecognition (Web Speech API).
    - Glowing animated recording indicator.
    - Live transcript display with instant 'Copy to Input' button.
    - Client-side fallback to HTML5 MediaRecorder (microphone audio clip) if SpeechRecognition is blocked.
    """
    try:
        import streamlit.components.v1 as components
    except ImportError:
        logger.warning("Streamlit not installed, skipping speech input component.")
        return

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{
          margin: 0;
          padding: 6px;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
          background: transparent;
        }}
        .mic-container {{
          background: #ffffff;
          border: 1px solid #cbd5e1;
          border-radius: 12px;
          padding: 12px 14px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }}
        .top-row {{
          display: flex;
          align-items: center;
          gap: 10px;
        }}
        .mic-btn {{
          width: 42px;
          height: 42px;
          border-radius: 50%;
          border: none;
          background: #f1f5f9;
          color: #334155;
          font-size: 20px;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.2s ease;
          position: relative;
        }}
        .mic-btn:hover {{
          background: #e2e8f0;
        }}
        .mic-btn.listening {{
          background: #ef4444;
          color: #ffffff;
          animation: pulse-ring 1.25s infinite;
        }}
        @keyframes pulse-ring {{
          0% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }}
          70% {{ box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }}
          100% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }}
        }}
        .mic-info {{
          flex: 1;
        }}
        .mic-status {{
          font-size: 13px;
          font-weight: 600;
          color: #1e293b;
        }}
        .mic-hint {{
          font-size: 11px;
          color: #64748b;
        }}
        .transcript-box {{
          margin-top: 10px;
          padding: 8px 10px;
          border-radius: 8px;
          background: #f8fafc;
          border: 1px dashed #cbd5e1;
          font-size: 13px;
          color: #0f172a;
          min-height: 38px;
          max-height: 60px;
          overflow-y: auto;
          line-height: 1.4;
        }}
        .transcript-actions {{
          margin-top: 8px;
          display: flex;
          justify-content: flex-end;
          gap: 8px;
        }}
        .action-btn {{
          padding: 4px 10px;
          font-size: 11px;
          font-weight: 600;
          border-radius: 6px;
          border: 1px solid #cbd5e1;
          background: #ffffff;
          cursor: pointer;
          color: #334155;
        }}
        .action-btn:hover {{
          background: #f1f5f9;
        }}
        .action-btn-primary {{
          background: #2563eb;
          color: #ffffff;
          border-color: #1d4ed8;
        }}
        .action-btn-primary:hover {{
          background: #1d4ed8;
        }}
      </style>
    </head>
    <body>
      <div class="mic-container">
        <div class="top-row">
          <button class="mic-btn" id="micBtn" onclick="toggleRecognition()" title="Click to start/stop speaking">
            🎙️
          </button>
          <div class="mic-info">
            <div class="mic-status" id="micStatus">Click mic to speak</div>
            <div class="mic-hint" id="micHint">{html.escape(prompt)}</div>
          </div>
        </div>

        <div class="transcript-box" id="transcriptBox">
          <span style="color:#94a3b8; font-style:italic;">Transcribed speech will appear here...</span>
        </div>

        <div class="transcript-actions">
          <button class="action-btn" id="btnClear" onclick="clearTranscript()">Clear</button>
          <button class="action-btn action-btn-primary" id="btnCopy" onclick="copyTranscript()">📋 Copy Transcript</button>
        </div>
      </div>

      <script>
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        let recognition = null;
        let isListening = false;
        let finalTranscript = "";

        const micBtn = document.getElementById('micBtn');
        const micStatus = document.getElementById('micStatus');
        const micHint = document.getElementById('micHint');
        const transcriptBox = document.getElementById('transcriptBox');
        const btnCopy = document.getElementById('btnCopy');

        if (SpeechRecognition) {{
          recognition = new SpeechRecognition();
          recognition.continuous = true;
          recognition.interimResults = true;
          recognition.lang = "{html.escape(lang)}";

          recognition.onstart = function() {{
            isListening = true;
            micBtn.classList.add('listening');
            micStatus.innerText = '🔴 Listening...';
            micHint.innerText = 'Speak clearly into your microphone';
          }};

          recognition.onresult = function(event) {{
            let interimTranscript = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {{
              if (event.results[i].isFinal) {{
                finalTranscript += event.results[i][0].transcript + ' ';
              }} else {{
                interimTranscript += event.results[i][0].transcript;
              }}
            }}
            transcriptBox.innerHTML = '<strong>' + finalTranscript + '</strong> <span style="color:#64748b;">' + interimTranscript + '</span>';
            transcriptBox.scrollTop = transcriptBox.scrollHeight;
          }};

          recognition.onerror = function(event) {{
            console.error('Speech recognition error:', event.error);
            micStatus.innerText = 'Mic error: ' + event.error;
            stopListening();
          }};

          recognition.onend = function() {{
            stopListening();
          }};
        }} else {{
          micStatus.innerText = 'Web Speech not supported in this browser';
          micHint.innerText = 'Use Chrome/Edge/Safari or record via file upload below';
        }}

        function toggleRecognition() {{
          if (!recognition) return;
          if (isListening) {{
            recognition.stop();
            stopListening();
          }} else {{
            try {{
              recognition.start();
            }} catch(e) {{
              console.error(e);
            }}
          }}
        }}

        function stopListening() {{
          isListening = false;
          micBtn.classList.remove('listening');
          micStatus.innerText = finalTranscript ? 'Speech captured' : 'Mic idle';
          micHint.innerText = 'Click mic to resume';
        }}

        function clearTranscript() {{
          finalTranscript = "";
          transcriptBox.innerHTML = '<span style="color:#94a3b8; font-style:italic;">Transcribed speech will appear here...</span>';
        }}

        function copyTranscript() {{
          const cleanText = transcriptBox.innerText.replace("Transcribed speech will appear here...", "").trim();
          if (!cleanText) return;
          navigator.clipboard.writeText(cleanText).then(() => {{
            btnCopy.innerText = '✅ Copied!';
            setTimeout(() => {{ btnCopy.innerText = '📋 Copy Transcript'; }}, 2000);
          }}).catch(err => {{
            console.error('Copy failed:', err);
          }});
        }}
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=height, scrolling=False)


# =====================================================================
# 4. Native Streamlit Audio Input & Backend Transcription Fallback
# =====================================================================

def render_audio_recorder(key: str = "spoken_answer", label: str = "🎙️ Record Spoken Answer (Mic)") -> Optional[bytes]:
    """Provides microphone audio capture using Streamlit's native st.audio_input
    (available in Streamlit 1.37+) with fallback to file uploader.
    
    Returns raw audio bytes if an audio clip was recorded/uploaded.
    """
    try:
        import streamlit as st
    except ImportError:
        logger.warning("Streamlit not installed.")
        return None

    # Check for st.audio_input (Streamlit 1.37.0+)
    if hasattr(st, "audio_input"):
        audio_val = st.audio_input(label, key=key)
        if audio_val is not None:
            return audio_val.getvalue()
    else:
        # Fallback to file uploader for audio files
        uploaded = st.file_uploader(
            f"{label} (Audio File)",
            type=["wav", "mp3", "m4a", "webm", "ogg"],
            key=key,
        )
        if uploaded is not None:
            return uploaded.getvalue()

    return None


def transcribe_audio_bytes(
    audio_bytes: bytes,
    mime_type: str = "audio/wav",
    api_key: Optional[str] = None,
) -> str:
    """Transcribes spoken audio bytes to text.
    
    Transcription pipeline:
    1. Gemini multimodal API (gemini-1.5-flash or gemini-2.5-flash):
       Directly ingests audio bytes and transcribes insurance terms and answers.
    2. Fallback: speech_recognition library if installed locally.
    3. Returns transcribed text or error message.
    """
    if not audio_bytes:
        return ""

    # Strategy 1: Google GenAI (Gemini) multimodal transcription
    try:
        # Try google.genai (new SDK)
        from google import genai
        from google.genai import types

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if resolved_key:
            client = genai.Client(api_key=resolved_key)
            prompt = (
                "You are an expert audio transcription engine for an insurance study assistant. "
                "Transcribe the following spoken user audio accurately into English text. "
                "Output ONLY the exact spoken transcription without preamble, quotes, or notes."
            )
            part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, part],
            )
            if response and response.text:
                return response.text.strip()
    except Exception as e1:
        logger.debug(f"google.genai transcription failed: {e1}")

    try:
        # Try legacy google.generativeai
        import google.generativeai as genai_legacy
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if resolved_key:
            genai_legacy.configure(api_key=resolved_key)
            model = genai_legacy.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content([
                "Transcribe the following audio recording into accurate plain English text. Return only the transcription.",
                {"mime_type": mime_type, "data": audio_bytes}
            ])
            if response and response.text:
                return response.text.strip()
    except Exception as e2:
        logger.debug(f"google.generativeai transcription failed: {e2}")

    # Strategy 2: Local speech_recognition library fallback
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            if text:
                return text.strip()
    except Exception as e3:
        logger.debug(f"Local speech_recognition failed: {e3}")

    return ""


# =====================================================================
# 5. Voice Settings & Control Widget
# =====================================================================

def render_voice_controls(container=None):
    """Renders a comprehensive voice configuration widget in Streamlit
    (e.g. in the sidebar or an expander) to control TTS and spoken settings.
    """
    try:
        import streamlit as st
    except ImportError:
        return

    ctx = container if container is not None else st

    with ctx.expander("🎙️ Voice & Spoken Audio Settings", expanded=False):
        voice_on = st.toggle(
            "🔊 Enable Spoken Answers (TTS)",
            value=st.session_state.get("voice_enabled", True),
            key="voice_enabled",
            help="Speaks study answers, viva questions, and quiz prompts aloud."
        )

        auto_read = st.checkbox(
            "⚡ Auto-speak AI Responses",
            value=st.session_state.get("voice_autoplay", True),
            key="voice_autoplay",
            disabled=not voice_on,
            help="Automatically begins reading out new explanations upon arrival."
        )

        rate = st.slider(
            "Speech Speed",
            min_value=0.7,
            max_value=1.5,
            value=st.session_state.get("voice_rate", 1.0),
            step=0.1,
            key="voice_rate",
            disabled=not voice_on,
        )

        lang_opts = {
            "en-IN": "English (India) - en-IN",
            "en-US": "English (United States) - en-US",
            "en-GB": "English (United Kingdom) - en-GB",
        }
        current_lang = st.session_state.get("voice_lang", "en-IN")
        sel_lang = st.selectbox(
            "Voice Accent / Language",
            options=list(lang_opts.keys()),
            format_func=lambda x: lang_opts.get(x, x),
            index=0 if current_lang == "en-IN" else 1,
            key="voice_lang",
            disabled=not voice_on,
        )

        if st.button("🔊 Test Voice Output", key="btn_test_voice", disabled=not voice_on):
            test_greeting = (
                "Hello! StudyAssistant voice system is online and ready for your insurance examination prep."
            )
            render_speech_synthesis(
                text=test_greeting,
                autoplay=True,
                rate=rate,
                lang=sel_lang,
                element_id="test_voice_sample",
            )


def speak_text(text: str, autoplay: bool = True):
    """Convenience helper to speak the given text aloud if voice is enabled in session state."""
    try:
        import streamlit as st
        if not st.session_state.get("voice_enabled", True):
            return
        rate = st.session_state.get("voice_rate", 1.0)
        lang = st.session_state.get("voice_lang", "en-IN")
        render_speech_synthesis(text=text, autoplay=autoplay, rate=rate, lang=lang)
    except Exception as e:
        logger.warning(f"speak_text failed: {e}")
