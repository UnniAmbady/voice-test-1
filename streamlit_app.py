# streamlit_app.py
# Title: Voice test (Compact mobile + logging + GitHub sync via Streamlit Secret)
"""
Streamlit app that:
- Records voice and converts to text (Faster-Whisper base/int8)
- Shows only the latest transcription in the editor
- Logs **all** transcriptions to a session-scoped file: `msg/chat-ddmmyy-hhmmss.txt`
- Pushes the log file to your GitHub repo via the REST API using the SSH deploy key stored in Streamlit Secrets.

Secrets format (`.streamlit/secrets.toml`):

    [github]
    token = "ssh-deploy-key"

Dependencies (requirements.txt):
    streamlit==1.39.0
    streamlit-mic-recorder==0.0.8
    faster-whisper==1.0.3
    typing-extensions>=4.10.0
    requests>=2.31.0

Run:
    streamlit run streamlit_app.py
"""

import base64
import io
import os
import tempfile
from typing import Optional
from datetime import datetime

import requests
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
st.caption("Microphone → Speech-to-Text → Editable text. Submit is a dummy.")

# --- Import libraries ---
try:
    from faster_whisper import WhisperModel  # type: ignore
except Exception:
    WhisperModel = None  # type: ignore

try:
    from streamlit_mic_recorder import mic_recorder  # type: ignore
except Exception:
    mic_recorder = None  # type: ignore

# --- GitHub constants ---
GH_REPO = "UnniAmbady/voice-test-1"
GH_BRANCH = "main"
GH_COMMIT_NAME = "Streamlit Sync Bot"
GH_COMMIT_EMAIL = "streamlit-sync@users.noreply.github.com"

# --- GitHub helpers ---
@st.cache_resource(show_spinner=False)
def _gh_cfg():
    token = st.secrets["github"]["token"]
    return token, GH_REPO, GH_BRANCH, GH_COMMIT_NAME, GH_COMMIT_EMAIL


def _gh_get_sha(token: str, repo: str, branch: str, path: str) -> Optional[str]:
    url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"
    r = requests.get(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}, timeout=15)
    if r.status_code == 200:
        return r.json().get("sha")
    return None


def _gh_put_file(token: str, repo: str, branch: str, path: str, content_bytes: bytes, message: str) -> bool:
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    sha = _gh_get_sha(token, repo, branch, path)
    payload = {
        "message": message,
        "branch": branch,
        "content": base64.b64encode(content_bytes).decode("utf-8"),
    }
    if sha:
        payload["sha"] = sha
    r = requests.put(url, json=payload, headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}, timeout=20)
    return r.status_code in (200, 201)


# --- Helpers for logging ---
def _ensure_log_path() -> str:
    if "log_file_path" not in st.session_state:
        session_stamp = datetime.now().strftime("%d%m%y-%H%M%S")
        base_dir = os.getcwd()
        msg_dir = os.path.join(base_dir, "msg")
        os.makedirs(msg_dir, exist_ok=True)
        st.session_state.log_file_path = os.path.join(msg_dir, f"chat-{session_stamp}.txt")
        with open(st.session_state.log_file_path, "a", encoding="utf-8") as f:
            f.write(f"=== Session start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    return st.session_state.log_file_path


def _log_text(text: str) -> None:
    path = _ensure_log_path()
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {text}\n")


def _sync_log_to_github() -> bool:
    token, repo, branch, name, email = _gh_cfg()
    if not token:
        return False
    local_path = _ensure_log_path()
    try:
        with open(local_path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        return False
    remote_path = os.path.relpath(local_path, start=os.getcwd()).replace("\\", "/")
    message = f"Update {remote_path} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ok = _gh_put_file(token, repo, branch, remote_path, data, message)
    return ok


# --- Session state ---
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""
if "recorder_key" not in st.session_state:
    st.session_state.recorder_key = 0

_ensure_log_path()

# --- Record control ---
st.subheader("Record your voice")

if st.button("Start new recording", key="btn_start", help="Flush text and record", use_container_width=False):
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
                        st.session_state.transcribed_text = text
                        _log_text(text)
                        synced = _sync_log_to_github()
                        if synced:
                            st.success("Transcription added, saved to log, and synced to GitHub.")
                        else:
                            st.success("Transcription added and saved to local log (GitHub sync not configured).")
                    else:
                        st.info("No speech detected or empty result.")

# --- Transcript editor ---
st.subheader("Edit transcript")
st.session_state.transcribed_text = st.text_area(
    "Transcript",
    value=st.session_state.transcribed_text,
    height=200,
)

c1, c2, c3 = st.columns([1, 2, 2])
with c1:
    if st.button("Submit", type="primary", key="btn_submit"):
        st.success("Submitted (dummy). No action performed.")
with c2:
    if st.button("Clear", key="btn_clear"):
        st.session_state.transcribed_text = ""
        st.rerun()
with c3:
    if st.button("Sync log to GitHub", key="btn_sync"):
        if _sync_log_to_github():
            st.success("Log synced to GitHub.")
        else:
            st.warning("GitHub sync failed or not configured.")

st.caption("Logs are written to ./msg/chat-<ddmmyy-hhmmss>.txt (session-scoped). Uses Streamlit Secret [github.token] for sync.")
