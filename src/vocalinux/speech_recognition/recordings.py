"""Per-dictation session state and saved recordings.

A ``DictationSession`` collects what one dictation did: the captured audio,
the segments sent to the engine, their timings and the final text. It is used
for the session-prefixed log lines, for the recovery WAV and for its JSON
sidecar. Recordings are debug and recovery data, so they live in the cache
directory and are deleted by age.
"""

import json
import logging
import os
import queue
import threading
import time
import uuid
import wave
from datetime import datetime
from typing import Optional

from ..utils.paths import recordings_dir

logger = logging.getLogger(__name__)

#: Sample rate of the audio the capture sources hand to the recognizer.
SAMPLE_RATE = 16000

#: Bytes per int16 sample.
SAMPLE_WIDTH = 2

_RECORDING_SUFFIXES = (".wav", ".json", ".part", ".tmp")


class DictationSession:
    """State of one dictation, from start to the final emitted text."""

    def __init__(
        self,
        mode: str = "toggle",
        engine: str = "",
        model: str = "",
        language: str = "",
        device_name: Optional[str] = None,
        keep_audio: bool = True,
    ) -> None:
        self.id = uuid.uuid4().hex[:8]
        self.started_wall = time.time()
        self.started_mono = time.monotonic()
        self.stopped_wall: Optional[float] = None
        self.mode = mode
        self.engine = engine
        self.model = model
        self.language = language
        self.device_name = device_name
        self.capture_sample_rate: Optional[int] = None
        self.capture_channels: Optional[int] = None
        # Audio of the whole session. Holds references to the chunk bytes the
        # buffers already hold, so keeping it adds no copy.
        self.keep_audio = keep_audio
        self.chunks: list[bytes] = []
        self.samples = 0
        self.limit_hit = False
        self.errors: list[str] = []
        self.segments: list[dict] = []
        self.final_text_chars: Optional[int] = None
        self.wav_path: Optional[str] = None
        # Texts of intermediate chunks, waiting for the final segment.
        self.text_parts: list[str] = []
        self.first_started_at: Optional[float] = None
        self._lock = threading.Lock()

    def log(self, level: int, message: str, *args, **kwargs) -> None:
        """Log ``message`` with the session id as a greppable prefix."""
        logger.log(level, "[%s] " + message, self.id, *args, **kwargs)

    def add_chunk(self, data: bytes) -> None:
        """Count one captured chunk and keep it for the recording."""
        self.samples += len(data) // SAMPLE_WIDTH
        if self.keep_audio:
            self.chunks.append(data)

    @property
    def audio_seconds(self) -> float:
        return self.samples / SAMPLE_RATE

    def add_segment(self, audio_seconds: float, partial: bool) -> dict:
        """Record a segment sent to the engine and return its stat entry."""
        with self._lock:
            stat = {
                "index": len(self.segments) + 1,
                "partial": partial,
                "audio_seconds": round(audio_seconds, 2),
                "transcribe_seconds": None,
                "chars": None,
            }
            self.segments.append(stat)
        return stat

    def add_error(self, message: str) -> None:
        with self._lock:
            self.errors.append(message)

    def add_text_part(self, text: str, started_at: Optional[float]) -> None:
        """Hold the text of an intermediate chunk until the session ends."""
        with self._lock:
            if self.first_started_at is None:
                self.first_started_at = started_at
            if text.strip():
                self.text_parts.append(text)

    def take_text(self, final_text: str = "") -> tuple[str, Optional[float]]:
        """Join the held parts and ``final_text`` in order, and reset the parts.

        Returns the joined text and the capture start of its first part, or
        None when the parts hold no start time.
        """
        with self._lock:
            parts = self.text_parts
            started_at = self.first_started_at
            self.text_parts = []
            self.first_started_at = None
        texts = [text for text in [*parts, final_text] if text.strip()]
        if not texts:
            return "", started_at
        # Strip inner joints; the last text keeps its trailing newline, which
        # can be an explicit submit signal.
        joined = " ".join([text.strip() for text in texts[:-1]] + [texts[-1].lstrip()])
        return joined, started_at

    def has_text_parts(self) -> bool:
        with self._lock:
            return bool(self.text_parts)

    def to_dict(self) -> dict:
        """Return the JSON sidecar content."""
        stopped = self.stopped_wall
        return {
            "session_id": self.id,
            "start_time": datetime.fromtimestamp(self.started_wall).isoformat(timespec="seconds"),
            "stop_time": (
                datetime.fromtimestamp(stopped).isoformat(timespec="seconds") if stopped else None
            ),
            "mode": self.mode,
            "engine": self.engine,
            "model": self.model,
            "language": self.language,
            "device_name": self.device_name,
            "capture_sample_rate": self.capture_sample_rate,
            "capture_channels": self.capture_channels,
            "wav_sample_rate": SAMPLE_RATE,
            "duration_seconds": round(self.audio_seconds, 2),
            "audio_chunks": len(self.chunks),
            "segment_count": len(self.segments),
            "segments": list(self.segments),
            "final_text_chars": self.final_text_chars,
            "limit_hit": self.limit_hit,
            "errors": list(self.errors),
            "wav_path": self.wav_path,
        }


class SegmentQueue(queue.Queue):
    """Unbounded segment queue that knows the session it belongs to.

    A worker reads only the queue it was started with, so a late worker of an
    old session never takes segments or text of a newer one.
    """

    def __init__(self, session: Optional[DictationSession] = None) -> None:
        super().__init__()
        self.session = session


def queue_session(segment_queue: queue.Queue) -> DictationSession:
    """Return the session of ``segment_queue``, attaching one to a plain queue."""
    session = getattr(segment_queue, "session", None)
    if session is None:
        session = DictationSession()
        setattr(segment_queue, "session", session)
    return session


def _stem(session: DictationSession) -> str:
    started = datetime.fromtimestamp(session.started_wall).strftime("%Y%m%d-%H%M%S")
    return f"{started}_{int(round(session.audio_seconds))}s"


def save_session_audio(session: DictationSession, directory: Optional[str] = None) -> Optional[str]:
    """Write the session audio as 16 kHz mono int16 WAV and return its path."""
    chunks = list(session.chunks)
    if not chunks:
        return None
    directory = directory or recordings_dir()
    os.makedirs(directory, exist_ok=True)
    stem = _stem(session)
    path = os.path.join(directory, stem + ".wav")
    counter = 1
    while os.path.exists(path):
        path = os.path.join(directory, f"{stem}-{counter}.wav")
        counter += 1
    partial_path = path + ".part"
    with wave.open(partial_path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)
        for chunk in chunks:
            wav.writeframes(chunk)
    os.replace(partial_path, path)
    session.wav_path = path
    return path


def sidecar_path(wav_path: str) -> str:
    return os.path.splitext(wav_path)[0] + ".json"


def write_sidecar(session: DictationSession) -> Optional[str]:
    """Write the session JSON next to its WAV and return its path."""
    if not session.wav_path:
        return None
    path = sidecar_path(session.wav_path)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(session.to_dict(), handle, indent=2)
    os.replace(tmp_path, path)
    return path


def cleanup_old_recordings(
    retention_hours: float, directory: Optional[str] = None, now: Optional[float] = None
) -> int:
    """Delete recordings and sidecars older than ``retention_hours``.

    A retention of 0 or less keeps everything. Returns the number of removed
    files.
    """
    if retention_hours is None or retention_hours <= 0:
        return 0
    directory = directory or recordings_dir()
    if not os.path.isdir(directory):
        return 0
    cutoff = (time.time() if now is None else now) - retention_hours * 3600
    removed = 0
    for entry in os.scandir(directory):
        if not entry.is_file() or not entry.name.endswith(_RECORDING_SUFFIXES):
            continue
        try:
            if entry.stat().st_mtime < cutoff:
                os.remove(entry.path)
                removed += 1
        except OSError as error:
            logger.warning("Could not remove old recording %s: %s", entry.path, error)
    if removed:
        logger.info("Removed %d recording files older than %sh", removed, retention_hours)
    return removed
