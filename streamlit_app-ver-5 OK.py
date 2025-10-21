# streamlit_app.py
# Title: Voice test (Compact mobile view)
"""
Streamlit app that:
- Records voice and converts to text (Faster-Whisper base/int8)
- Shows only the latest transcription (flushes old text when Start is pressed)
- Single-column, mobile-friendly layout
- Start/Stop buttons auto-sized to label
- Submit button is a dummy

Dependencies (requirements.txt):
    streamlit==1.39.0
    streamlit-mic-recorder==0.0.8
    faster-whisper==1.0.3
    typing-extensions>=4.10.0

Run:
    streamlit run streamlit_app.py
"""

import io
import os
import tempfile
from typing import Optional

import streamlit as st

st.set_page_config(page_title="Voice test", page_icon="🎙️", layout="centered")

# --- Styling for compact UI ---
st.markdown(
    """
    <style>
    .small-btn button {padding: 0.25rem 0.6rem; font-size: 0.85rem; min-width: auto;}
    .stTextArea textarea {font-size: 1rem;}
    .block-container {padding-top: 1rem; padding-bottom: 1.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Voice test")
st.caption("Microphone → Speech‑to‑Text → Editable text. Submit is a dummy.")

# --- Import libraries ---
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

# --- Record control ---
st.subheader("Record your voice")

if st.button("Start new recording", key="btn_start", help="Flush text and record", use_container_width=False):
    # Flush text and reset recorder key
    st.session_state.transcribed_text = ""
    st.session_state.recorder_key += 1
    st.rerun()

if mic_recorder is None:
    st.warning("streamlit-mic-recorder not installed. Run: pip install streamlit-mic-recorder")
else:
    st.write("Click **Start** to record and **Stop** when done. Transcription runs after stopping.")
    audio = mic_recorder(
        start_prompt="Start",
        stop_prompt="Stop",
        just_once=False,
        use_container_width=False,
        key=f"mic_{st.session_state.recorder_key}",
    )

    # Process audio
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
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                            tmp.write(wav_bytes)
                            tmp.flush()
                            tmp_path = tmp.name

                        model = WhisperModel("base", compute_type="int8")
                        segments, info = model.transcribe(tmp_path, language="en", vad_filter=True)
                        parts = [seg.text for seg in segments]
                        text = " ".join(parts).strip()
                    finally:
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass

                    if text:
                        # Always replace text (flush old ones)
                        st.session_state.transcribed_text = text
                        st.success("Transcription added to editor below.")
                    else:
                        st.info("No speech detected or empty result.")

# --- Transcript editor ---
st.subheader("Edit transcript")
st.session_state.transcribed_text = st.text_area(
    "Transcript",
    value=st.session_state.transcribed_text,
    height=200,
)

c1, c2 = st.columns([1, 3])
with c1:
    if st.button("Submit", type="primary", key="btn_submit"):
        st.success("Submitted (dummy). No action performed.")
with c2:
    if st.button("Clear", key="btn_clear"):
        st.session_state.transcribed_text = ""
        st.rerun()

st.caption("Fixed STT params: Faster‑Whisper base / int8. Compact single‑panel mobile UI.")

