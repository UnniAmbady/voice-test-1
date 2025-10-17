# streamlit_app.py
# Title: Voice test
"""
A simple Streamlit app that captures microphone audio, converts speech to text,
shows the transcript in a text editor, and includes a dummy Submit button.

Free, well-known STT backends supported:
1) Faster-Whisper (default) — robust across accents (incl. Singapore English).
2) Vosk (optional) — lightweight offline option (try en-in model for SEA accents).

Install (recommended minimal):
    pip install streamlit faster-whisper streamlit-mic-recorder

Optional (for offline/CPU-light STT):
    pip install vosk
    # Download a small English-accent model (e.g., en-in) from Vosk and set its path in the sidebar.

Run locally:
    streamlit run streamlit_app.py
"""

import io
import os
import tempfile
from typing import Optional
import wave
import json
import audioop  # for simple PCM/channel conversions on Vosk path

import streamlit as st

# Try optional imports gracefully
try:
    from faster_whisper import WhisperModel  # type: ignore
except Exception:
    WhisperModel = None  # type: ignore

try:
    # pip: streamlit-mic-recorder
    from streamlit_mic_recorder import mic_recorder  # type: ignore
except Exception:
    mic_recorder = None  # type: ignore

# Optional Vosk
try:
    import vosk  # type: ignore
    _HAS_VOSK = True
except Exception:
    vosk = None  # type: ignore
    _HAS_VOSK = False

st.set_page_config(page_title="Voice test", page_icon="🎙️")
st.title("Voice test")
st.caption("Microphone -> Speech-to-Text -> Editable text. Submit is a dummy.")

# --- Sidebar settings
st.sidebar.header("Settings")
engine = st.sidebar.selectbox(
    "STT Engine",
    options=["Faster-Whisper (recommended)", "Vosk (offline)"],
    index=0,
)

if engine.startswith("Faster-Whisper"):
    fw_model_size = st.sidebar.selectbox(
        "Whisper model size",
        ["tiny", "base", "small"],
        index=1,
        help=(
            "Smaller = faster but less accurate. 'base' is a good default."
        ),
    )
    compute_type = st.sidebar.selectbox(
        "Compute type",
        ["int8", "int8_float16", "float16", "int8_float32", "float32"],
        index=0,
        help="If you have no GPU, 'int8' is fine.",
    )
else:
    vosk_model_path = st.sidebar.text_input(
        "Vosk model directory",
        value=os.environ.get("VOSK_MODEL_PATH", ""),
        help=(
            "Path to an extracted Vosk English model. For SEA accents, try the "
            "small Indian English model as a starting point."
        ),
    )

# --- Session state
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""

# --- Microphone recorder
st.subheader("Record your voice")
if mic_recorder is None:
    st.warning(
        "streamlit-mic-recorder is not installed. Run: pip install streamlit-mic-recorder"
    )
else:
    st.write("Click to start/stop. After stopping, transcription will run.")
    audio = mic_recorder(
        start_prompt="Start recording",
        stop_prompt="Stop",
        just_once=False,
        use_container_width=True,
        key="mic",
    )

    # audio is typically a dict with keys: 'bytes', 'sample_rate'
    if audio:
        wav_bytes: Optional[bytes] = None
        if isinstance(audio, dict) and "bytes" in audio:
            wav_bytes = audio["bytes"]
        elif isinstance(audio, (bytes, bytearray)):
            wav_bytes = bytes(audio)

        if wav_bytes:
            st.audio(wav_bytes, format="audio/wav")

            with st.spinner("Transcribing..."):
                text = ""
                try:
                    if engine.startswith("Faster-Whisper"):
                        if WhisperModel is None:
                            st.error(
                                "faster-whisper not installed. Run: pip install faster-whisper"
                            )
                        else:
                            # Save to a temp wav and run transcription
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                                tmp.write(wav_bytes)
                                tmp.flush()
                                tmp_path = tmp.name

                            model = WhisperModel(fw_model_size, compute_type=compute_type)
                            segments, info = model.transcribe(
                                tmp_path,
                                language="en",
                                vad_filter=True,
                            )
                            parts = []
                            for seg in segments:
                                parts.append(seg.text)
                            text = " ".join(parts).strip()
                            # Cleanup temp
                            try:
                                os.remove(tmp_path)
                            except Exception:
                                pass

                    else:  # Vosk path with proper WAV -> PCM handling
                        if not _HAS_VOSK:
                            st.error("vosk not installed. Run: pip install vosk")
                        elif not vosk_model_path or not os.path.isdir(vosk_model_path):
                            st.error("Please set a valid Vosk model directory in the sidebar.")
                        else:
                            # Read WAV header, extract PCM frames
                            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                                n_channels = wf.getnchannels()
                                sampwidth = wf.getsampwidth()
                                framerate = wf.getframerate()
                                frames = wf.readframes(wf.getnframes())

                            # Ensure 16-bit mono PCM (Vosk friendly)
                            if n_channels == 2:
                                frames = audioop.tomono(frames, sampwidth, 1, 1)
                                n_channels = 1
                            if sampwidth != 2:
                                frames = audioop.lin2lin(frames, sampwidth, 2)
                                sampwidth = 2

                            model = vosk.Model(vosk_model_path)
                            rec = vosk.KaldiRecognizer(model, framerate)

                            # Feed raw PCM frames in manageable chunks
                            bio = io.BytesIO(frames)
                            chunk = bio.read(4000)
                            while chunk:
                                rec.AcceptWaveform(chunk)
                                chunk = bio.read(4000)

                            res = json.loads(rec.FinalResult())
                            text = (res.get("text") or "").strip()

                    if text:
                        if st.session_state.transcribed_text:
                            st.session_state.transcribed_text += "\n\n" + text
                        else:
                            st.session_state.transcribed_text = text
                        st.success("Transcription added to editor below.")
                    else:
                        st.info("No speech detected, or empty result.")
                                    except Exception as e:
                                        st.exception(e)

# --- Text editor & Submit
st.subheader("Edit transcript")
st.session_state.transcribed_text = st.text_area(
    "Transcript",
    value=st.session_state.transcribed_text,
    height=220,
)

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Submit", type="primary"):
        # Dummy function: does nothing meaningful
        st.success("Submitted (dummy). No action performed.")
with col2:
    if st.button("Clear editor"):
        st.session_state.transcribed_text = ""
        st.rerun()

st.caption(
    "Tip: For clearer recognition, speak close to the mic in a quiet room. Whisper is generally strong across accents, including Singapore English; Vosk is a lighter offline option."
)
