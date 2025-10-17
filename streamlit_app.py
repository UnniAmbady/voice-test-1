# streamlit_app.py
# Title: Voice test
"""
Single-panel Streamlit app (mobile-friendly) that:
- Records from the microphone
- Transcribes to text using Faster-Whisper (fixed params: model="base", compute_type="int8")
- Clears the editor **every time you press Start new recording**
- Shows transcript in an editable text area
- Has a small **Submit** button (dummy action)

Dependencies (requirements.txt):
    streamlit==1.39.0
    streamlit-mic-recorder==0.0.8
    faster-whisper==1.0.3
    typing-extensions>=4.10.0

Run locally:
    streamlit run streamlit_app.py
"""

import io
import os
import tempfile
from typing import Optional

import streamlit as st

# --- Page setup (single column, mobile-friendly) ---
st.set_page_config(page_title="Voice test", page_icon="🎙️", layout="centered")

# Minimal CSS tweaks for smaller buttons & mobile spacing
st.markdown(
    """
    <style>
    .small-btn button {padding: 0.35rem 0.6rem; font-size: 0.875rem;}
    .tight {margin-top: 0.25rem; margin-bottom: 0.25rem;}
    .stTextArea textarea {font-size: 1rem;}
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Voice test")
st.caption("Microphone → Speech‑to‑Text → Editable text. Submit is a dummy.")

# --- Import libraries (fixed engine: Faster-Whisper) ---
try:
    from faster_whisper import WhisperModel  # type: ignore
except Exception:
    WhisperModel = None  # type: ignore

try:
    from streamlit_mic_recorder import mic_recorder  # type: ignore
except Exception:
    mic_recorder = None  # type: ignore

# --- Session state ---
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""
if "recorder_key" not in st.session_state:
    st.session_state.recorder_key = 0

# --- Controls ---
st.subheader("Record your voice")
col_start, col_spacer = st.columns([1, 5])
with col_start:
    if st.button("Start new recording", key="btn_start", help="Clears the editor, then record", use_container_width=False):
        # Clear editor and reset recorder component (force a fresh instance via key)
        st.session_state.transcribed_text = ""
        st.session_state.recorder_key += 1
        st.rerun()

# Mic widget
if mic_recorder is None:
    st.warning("streamlit-mic-recorder is not installed. Run: pip install streamlit-mic-recorder")
else:
    st.write("Click **Start** to record and **Stop** when done. Transcription runs after stopping.")
    audio = mic_recorder(
        start_prompt="Start",
        stop_prompt="Stop",
        just_once=False,
        use_container_width=True,
        key=f"mic_{st.session_state.recorder_key}",
    )

    # audio is typically a dict with keys: 'bytes', 'sample_rate'
    if audio:
        wav_bytes: Optional[bytes] = None
        if isinstance(audio, dict) and "bytes" in audio:
            wav_bytes = audio["bytes"]
        elif isinstance(audio, (bytes, bytearray)):
            wav_bytes = bytes(audio)

        if wav_bytes:
            st.audio(wav_bytes, format="audio/wav", autoplay=False)

            if WhisperModel is None:
                st.error("faster-whisper not installed. Run: pip install faster-whisper")
            else:
                with st.spinner("Transcribing..."):
                    # Save to a temp wav and run transcription (fixed params)
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                            tmp.write(wav_bytes)
                            tmp.flush()
                            tmp_path = tmp.name

                        model = WhisperModel("base", compute_type="int8")
                        segments, info = model.transcribe(
                            tmp_path,
                            language="en",
                            vad_filter=True,
                        )
                        parts = [seg.text for seg in segments]
                        text = " ".join(parts).strip()
                    finally:
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass

                    if text:
                        # Append to existing text with a separator (editor was cleared when Start new recording was pressed)
                        if st.session_state.transcribed_text:
                            st.session_state.transcribed_text += "

" + text
                        else:
                            st.session_state.transcribed_text = text
                        st.success("Transcription added to editor below.")
                    else:
                        st.info("No speech detected, or empty result.")

# --- Editor & Actions ---
st.subheader("Edit transcript")
st.session_state.transcribed_text = st.text_area(
    "Transcript",
    value=st.session_state.transcribed_text,
    height=220,
)

c1, c2 = st.columns([1, 3])
with c1:
    st.button("Submit", key="btn_submit", type="primary", help="Dummy submit", on_click=lambda: st.success("Submitted (dummy). No action performed."), args=None)
with c2:
    if st.button("Clear", key="btn_clear", help="Clear editor"):
        st.session_state.transcribed_text = ""
        st.rerun()

st.caption("Fixed STT params: Faster‑Whisper base / int8. Designed for single‑panel mobile view.")
