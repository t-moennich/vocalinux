"""Long push-to-talk dictations: no trimming, duration limit, chunks, recordings."""

import json
import logging
import os
import queue
import re
import sys
import threading
import time
import wave
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from vocalinux.common_types import RecognitionState
from vocalinux.speech_recognition import recognition_manager as rm
from vocalinux.speech_recognition import recordings
from vocalinux.speech_recognition.chunking import split_like_dictation

CHUNK_SAMPLES = 1024
CHUNK_SECONDS = CHUNK_SAMPLES / 16000
SPEECH = np.full(CHUNK_SAMPLES, 5000, dtype=np.int16).tobytes()
SILENCE = np.zeros(CHUNK_SAMPLES, dtype=np.int16).tobytes()


def chunks_for(seconds: float) -> int:
    return int(round(seconds / CHUNK_SECONDS))


class FakeSource:
    """Capture source that plays a script of chunks, then optionally ends."""

    requires_pyaudio = False
    stream = None
    audio = None
    downmix_channel = None
    sample_rate = 16000
    channels = 1

    def __init__(self, manager, script, endless_chunk=None, delay=0.0):
        self.manager = manager
        self.script = list(script)
        self.endless_chunk = endless_chunk
        self.delay = delay

    def open(self, audio):
        return None

    def close(self):
        return None

    def read_chunk(self):
        if self.delay:
            time.sleep(self.delay)
        if self.script:
            return self.script.pop(0)
        if self.endless_chunk is not None:
            return self.endless_chunk
        self.manager.should_record = False
        return SILENCE


@pytest.fixture(autouse=True)
def real_numpy(monkeypatch):
    """Other test modules leave a MagicMock numpy in sys.modules."""
    monkeypatch.setitem(sys.modules, "numpy", getattr(sys, "_vocalinux_real_numpy", np))


@pytest.fixture
def manager(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(rm.SpeechRecognitionManager, "_init_vosk", lambda self: None)
    monkeypatch.setattr(rm, "load_silero_vad", lambda: None)
    # conftest patches _show_notification for every test.
    notify = rm._show_notification
    notify.reset_mock()
    for name in ("play_start_sound", "play_stop_sound", "play_error_sound"):
        monkeypatch.setattr(rm, name, Mock())
    mgr = rm.SpeechRecognitionManager(engine="vosk", playback_duck=Mock())
    mgr.model = object()
    mgr._model_initialized = True
    mgr.stop_sound_guard_ms = 0
    mgr.silence_timeout = 0.5
    mgr.vad_sensitivity = 3
    mgr.notify = notify
    yield mgr
    mgr.should_record = False


def use_script(mgr, script, **kwargs):
    source = FakeSource(mgr, script, **kwargs)
    mgr._new_capture_source = lambda: source
    return source


def collect_texts(mgr):
    texts = []
    mgr.register_text_callback(texts.append)
    segments = []
    mgr.register_segment_callback(lambda text, started_at: segments.append(text))
    return texts, segments


def script_transcripts(mgr, *texts):
    """Make the engine return ``texts`` in order, one per transcribed segment."""
    remaining = list(texts)
    calls = []

    def transcribe(audio_buffer, language=None):
        calls.append(sum(len(chunk) for chunk in audio_buffer) / 2 / 16000)
        return remaining.pop(0) if remaining else ""

    mgr._transcribe_segment = transcribe
    return calls


def run_session(mgr, script, mode, **kwargs):
    use_script(mgr, script, **kwargs)
    assert mgr.start_recognition(mode=mode)
    mgr.audio_thread.join(10)
    assert not mgr.audio_thread.is_alive()
    worker = mgr.recognition_thread
    mgr.stop_recognition()
    worker.join(10)
    assert not worker.is_alive()


class TestNoTrimming:
    def test_buffer_beyond_old_chunk_limit_keeps_all_audio(self, manager):
        manager.audio_buffer = []
        manager.should_record = True
        manager._recognition_mode = "toggle"
        total = manager._max_buffer_size + 1000
        use_script(manager, [SPEECH] * total)
        manager._record_audio()
        assert len(manager.audio_buffer) == total + 1
        assert manager._get_session().samples == (total + 1) * CHUNK_SAMPLES

    def test_stats_still_report(self, manager):
        manager.audio_buffer = [SPEECH] * 3
        stats = manager.get_buffer_stats()
        assert stats["buffer_size"] == 3
        assert stats["buffer_limit"] == manager._max_buffer_size


class TestDurationLimit:
    def test_limit_stops_from_another_thread_and_keeps_audio(self, manager):
        manager.max_recording_minutes = 0.05
        manager.chunk_min_seconds = 0
        texts, _segments = collect_texts(manager)
        script_transcripts(manager, "all of it")
        stop_threads = []
        real_stop = manager.stop_recognition

        def tracking_stop():
            stop_threads.append(threading.current_thread())
            real_stop()

        manager.stop_recognition = tracking_stop
        use_script(manager, [], endless_chunk=SPEECH, delay=0.0005)
        assert manager.start_recognition(mode="push_to_talk")
        audio_thread = manager.audio_thread
        deadline = time.time() + 10
        while manager.state != RecognitionState.IDLE and time.time() < deadline:
            time.sleep(0.01)

        assert manager.state == RecognitionState.IDLE
        assert len(stop_threads) == 1
        assert stop_threads[0] is not audio_thread
        session = manager._get_session()
        assert session.limit_hit
        assert session.audio_seconds >= 3.0
        manager.notify.assert_called_once()
        assert texts == ["all of it"]
        assert sum(stat["audio_seconds"] for stat in session.segments) >= 3.0

    def test_no_limit_when_zero(self, manager):
        manager.max_recording_minutes = 0
        session = manager._get_session()
        session.samples = 10**9
        assert not manager._recording_limit_reached(session)


class TestChunkedPushToTalk:
    def test_pause_splits_chunk_and_text_is_emitted_once(self, manager):
        texts, segments = collect_texts(manager)
        calls = script_transcripts(manager, "eins", "zwei")
        script = (
            [SPEECH] * chunks_for(25)
            + [SILENCE] * chunks_for(1)
            + [SPEECH] * chunks_for(10)
            + [SILENCE] * 2
        )
        run_session(manager, script, "push_to_talk")
        assert len(calls) == 2
        assert calls[0] >= 25
        assert texts == ["eins zwei"]
        assert segments == ["eins zwei"]

    def test_short_pause_splits_with_long_silence_timeout(self, manager):
        """The split rule does not wait for silence_timeout."""
        manager.silence_timeout = 3.0
        texts, _ = collect_texts(manager)
        calls = script_transcripts(manager, "eins", "zwei")
        script = (
            [SPEECH] * chunks_for(22)
            + [SILENCE] * chunks_for(0.8)
            + [SPEECH] * chunks_for(10)
            + [SILENCE] * 2
        )
        run_session(manager, script, "push_to_talk")
        assert len(calls) == 2
        assert calls[0] == pytest.approx(22.8, abs=0.3)
        assert texts == ["eins zwei"]

    def test_pause_shorter_than_chunk_pause_does_not_split(self, manager):
        manager.silence_timeout = 3.0
        texts, _ = collect_texts(manager)
        calls = script_transcripts(manager, "alles")
        script = (
            [SPEECH] * chunks_for(22)
            + [SILENCE] * chunks_for(0.3)
            + [SPEECH] * chunks_for(10)
            + [SILENCE] * 2
        )
        run_session(manager, script, "push_to_talk")
        assert len(calls) == 1
        assert texts == ["alles"]

    def test_chunk_pause_seconds_is_configurable(self, manager):
        manager.chunk_pause_seconds = 1.5
        calls = script_transcripts(manager, "a", "b")
        script = (
            [SPEECH] * chunks_for(22)
            + [SILENCE] * chunks_for(1.0)
            + [SPEECH] * chunks_for(3)
            + [SILENCE] * chunks_for(1.6)
            + [SPEECH] * chunks_for(3)
        )
        run_session(manager, script, "push_to_talk")
        assert len(calls) == 2
        assert calls[0] > 27

    def test_state_stays_listening_while_chunks_are_transcribed(self, manager):
        states = []
        manager.register_state_callback(states.append)
        release = threading.Event()
        started = threading.Event()

        def slow_transcribe(audio_buffer, language=None):
            started.set()
            release.wait(5)
            return "text"

        manager._transcribe_segment = slow_transcribe
        use_script(manager, [SPEECH] * chunks_for(25) + [SILENCE] * chunks_for(1) + [SPEECH] * 5)
        assert manager.start_recognition(mode="push_to_talk")
        manager.audio_thread.join(10)
        assert started.wait(5)
        # The chunk is being transcribed and the key is still held.
        assert manager.state == RecognitionState.LISTENING
        assert RecognitionState.PROCESSING not in states
        release.set()
        manager.stop_recognition()
        manager.recognition_thread.join(10)
        assert manager.state == RecognitionState.IDLE

    def test_release_while_processing_still_stops(self, manager):
        texts, _ = collect_texts(manager)
        script_transcripts(manager, "text")
        use_script(manager, [], endless_chunk=SPEECH, delay=0.0005)
        assert manager.start_recognition(mode="push_to_talk")
        manager._update_state(RecognitionState.PROCESSING)
        manager.stop_recognition()
        manager.recognition_thread.join(10)
        assert manager.state == RecognitionState.IDLE
        assert not manager.should_record
        assert texts == ["text"]

    def test_pause_before_minimum_does_not_split(self, manager):
        texts, _ = collect_texts(manager)
        calls = script_transcripts(manager, "alles")
        script = (
            [SPEECH] * chunks_for(5)
            + [SILENCE] * chunks_for(1)
            + [SPEECH] * chunks_for(5)
            + [SILENCE] * 2
        )
        run_session(manager, script, "push_to_talk")
        assert len(calls) == 1
        assert texts == ["alles"]

    def test_no_pause_splits_at_maximum(self, manager):
        manager.chunk_max_seconds = 50
        texts, _ = collect_texts(manager)
        calls = script_transcripts(manager, "a", "b", "c")
        run_session(manager, [SPEECH] * chunks_for(130), "push_to_talk")
        assert len(calls) == 3
        assert texts == ["a b c"]

    def test_silent_tail_still_delivers_held_text(self, manager):
        texts, _ = collect_texts(manager)
        calls = script_transcripts(manager, "eins")
        script = [SPEECH] * chunks_for(25) + [SILENCE] * chunks_for(3)
        run_session(manager, script, "push_to_talk")
        assert len(calls) == 1
        assert texts == ["eins"]

    def test_toggle_mode_keeps_per_silence_segments(self, manager):
        texts, _ = collect_texts(manager)
        calls = script_transcripts(manager, "eins", "zwei")
        script = (
            [SPEECH] * chunks_for(25)
            + [SILENCE] * chunks_for(1)
            + [SPEECH] * chunks_for(10)
            + [SILENCE] * chunks_for(1)
        )
        run_session(manager, script, "toggle")
        assert len(calls) == 2
        assert texts == ["eins", "zwei"]

    def test_voice_commands_run_once_on_joined_text(self, manager):
        manager._voice_commands_enabled = True
        manager.command_processor = Mock()
        manager.command_processor.process_text.return_value = ("done", [])
        texts, _ = collect_texts(manager)
        script_transcripts(manager, "eins", "zwei")
        script = (
            [SPEECH] * chunks_for(25)
            + [SILENCE] * chunks_for(1)
            + [SPEECH] * chunks_for(5)
            + [SILENCE] * 2
        )
        run_session(manager, script, "push_to_talk")
        manager.command_processor.process_text.assert_called_once_with("eins zwei")
        assert texts == ["done"]

    def test_reload_pending_does_not_split(self, manager):
        manager._reload_pending = True
        manager._recognition_mode = "push_to_talk"
        assert not manager._can_split_segment()


class TestSessionQueues:
    def test_queue_never_drops_segments(self, manager):
        for index in range(50):
            manager._enqueue_audio_segment([bytes([index]) * 4], partial=index % 2 == 0)
        assert manager._segment_queue.qsize() == 50
        assert len(manager._get_session().segments) == 50

    def test_late_worker_reads_only_its_own_queue(self, manager):
        old_queue = recordings.SegmentQueue(recordings.DictationSession())
        new_queue = recordings.SegmentQueue(recordings.DictationSession())
        manager._segment_queue = new_queue
        new_queue.put((rm._AudioSegment([b"new"], None), "de"))
        old_queue.put(None)
        manager._process_audio_buffer = Mock()

        manager._perform_recognition(old_queue)

        manager._process_audio_buffer.assert_not_called()
        assert new_queue.qsize() == 1

    def test_failed_segment_does_not_stop_the_worker(self, manager):
        texts, _ = collect_texts(manager)
        results = iter([ValueError("boom"), "zwei"])

        def transcribe(audio_buffer, language=None):
            result = next(results)
            if isinstance(result, Exception):
                raise result
            return result

        manager._transcribe_segment = transcribe
        manager._enqueue_audio_segment([SPEECH])
        manager._enqueue_audio_segment([SPEECH])
        manager._signal_recognition_stop()

        manager._perform_recognition(manager._segment_queue)

        assert texts == ["zwei"]
        assert manager._get_session().errors == ["ValueError: boom"]

    def test_start_replaces_queue_and_binds_worker(self, manager):
        use_script(manager, [SILENCE] * 3)
        old_queue = manager._segment_queue
        assert manager.start_recognition(mode="toggle")
        assert manager._segment_queue is not old_queue
        assert isinstance(manager._segment_queue, recordings.SegmentQueue)
        manager.audio_thread.join(5)
        manager.stop_recognition()
        manager.recognition_thread.join(5)


class TestRecordings:
    def make_session(self, seconds=2, **kwargs):
        session = recordings.DictationSession(
            mode="push_to_talk", engine="whisper_cpp", model="small", language="de", **kwargs
        )
        for _ in range(chunks_for(seconds)):
            session.add_chunk(SPEECH)
        return session

    def test_saves_16k_mono_wav_with_all_chunks(self, tmp_path):
        session = self.make_session(2)
        path = recordings.save_session_audio(session, str(tmp_path))
        assert re.fullmatch(r"\d{8}-\d{6}_\ds\.wav", os.path.basename(path))
        with wave.open(path, "rb") as wav:
            assert wav.getframerate() == 16000
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            assert wav.getnframes() == session.samples
        assert not [name for name in os.listdir(tmp_path) if name.endswith(".part")]

    def test_session_keeps_references_not_copies(self):
        session = recordings.DictationSession()
        data = b"\x01\x02" * 100
        session.add_chunk(data)
        assert session.chunks[0] is data

    def test_audio_not_kept_when_disabled(self):
        session = recordings.DictationSession(keep_audio=False)
        session.add_chunk(SPEECH)
        assert session.chunks == []
        assert session.samples == CHUNK_SAMPLES

    def test_sidecar_content(self, tmp_path):
        session = self.make_session(2, device_name="Mic")
        session.capture_sample_rate = 48000
        session.capture_channels = 2
        stat = session.add_segment(2.0, False)
        stat["transcribe_seconds"] = 0.5
        stat["chars"] = 12
        session.final_text_chars = 12
        session.limit_hit = True
        session.add_error("ValueError: boom")
        session.stopped_wall = session.started_wall + 3
        wav_path = recordings.save_session_audio(session, str(tmp_path))
        sidecar = recordings.write_sidecar(session)

        assert sidecar == os.path.splitext(wav_path)[0] + ".json"
        with open(sidecar, encoding="utf-8") as handle:
            data = json.load(handle)
        assert data["mode"] == "push_to_talk"
        assert data["engine"] == "whisper_cpp"
        assert data["model"] == "small"
        assert data["language"] == "de"
        assert data["device_name"] == "Mic"
        assert data["capture_sample_rate"] == 48000
        assert data["capture_channels"] == 2
        assert data["start_time"] and data["stop_time"]
        assert data["duration_seconds"] == pytest.approx(2.0, abs=0.1)
        assert data["audio_chunks"] == chunks_for(2)
        assert data["segment_count"] == 1
        assert data["segments"][0]["transcribe_seconds"] == 0.5
        assert data["final_text_chars"] == 12
        assert data["limit_hit"] is True
        assert data["errors"] == ["ValueError: boom"]

    def test_cleanup_deletes_old_wav_and_sidecar_only(self, tmp_path):
        now = time.time()
        old = now - 30 * 3600
        for name in ("old.wav", "old.json", "new.wav", "new.json", "notes.txt"):
            path = tmp_path / name
            path.write_bytes(b"x")
            if name.startswith("old") or name == "notes.txt":
                os.utime(path, (old, old))

        removed = recordings.cleanup_old_recordings(24, str(tmp_path), now=now)

        assert removed == 2
        assert sorted(os.listdir(tmp_path)) == ["new.json", "new.wav", "notes.txt"]

    def test_cleanup_with_zero_retention_keeps_everything(self, tmp_path):
        path = tmp_path / "old.wav"
        path.write_bytes(b"x")
        os.utime(path, (1, 1))
        assert recordings.cleanup_old_recordings(0, str(tmp_path)) == 0
        assert path.exists()

    def test_cleanup_missing_directory(self, tmp_path):
        assert recordings.cleanup_old_recordings(24, str(tmp_path / "missing")) == 0

    def test_manager_cleanup_runs_once_a_day(self, manager):
        with patch.object(rm, "cleanup_old_recordings", return_value=1) as cleanup:
            assert manager.cleanup_recordings(force=False) == 1
            assert manager.cleanup_recordings(force=False) == 0
            assert manager.cleanup_recordings(force=True) == 1
        assert cleanup.call_count == 2

    def test_session_saves_wav_and_final_sidecar(self, manager, tmp_path):
        collect_texts(manager)
        script_transcripts(manager, "eins", "zwei")
        script = (
            [SPEECH] * chunks_for(25)
            + [SILENCE] * chunks_for(1)
            + [SPEECH] * chunks_for(5)
            + [SILENCE] * 2
        )
        run_session(manager, script, "push_to_talk")
        session = manager._get_session()
        directory = tmp_path / "cache" / "vocalinux" / "recordings"

        deadline = time.time() + 10
        data = None
        while time.time() < deadline:
            sidecars = list(directory.glob("*.json")) if directory.exists() else []
            if sidecars:
                data = json.loads(sidecars[0].read_text())
                if data["final_text_chars"] is not None:
                    break
            time.sleep(0.05)

        assert data is not None and data["final_text_chars"] == len("eins zwei")
        assert data["segment_count"] == 2
        assert [s["partial"] for s in data["segments"]] == [True, False]
        wavs = list(directory.glob("*.wav"))
        assert len(wavs) == 1
        with wave.open(str(wavs[0]), "rb") as wav:
            assert wav.getnframes() == session.samples
        assert session.chunks == []

    def test_no_recording_when_disabled(self, manager, tmp_path):
        manager.save_recordings = False
        collect_texts(manager)
        script_transcripts(manager, "eins")
        run_session(manager, [SPEECH] * chunks_for(3) + [SILENCE] * 2, "push_to_talk")
        time.sleep(0.2)
        assert not (tmp_path / "cache" / "vocalinux" / "recordings").exists()


class TestSplitLikeDictation:
    @staticmethod
    def pcm(*parts):
        samples = [np.full(int(sec * 16000), level, dtype=np.int16) for sec, level in parts]
        return np.concatenate(samples)

    @staticmethod
    def loud(step):
        return bool(np.abs(step).mean() > 100)

    def split(self, pcm, **kwargs):
        options = {"silence_timeout": 0.5, "chunk_min_seconds": 20, "chunk_max_seconds": 120}
        options.update(kwargs)
        return split_like_dictation(pcm, self.loud, **options)

    def test_splits_at_pause_after_minimum(self):
        chunks = self.split(self.pcm((25, 5000), (2, 0), (10, 5000)))
        assert len(chunks) == 2
        assert len(chunks[0]) / 16000 == pytest.approx(25.5, abs=0.2)

    def test_splits_at_chunk_pause_despite_long_silence_timeout(self):
        chunks = self.split(self.pcm((22, 5000), (0.8, 0), (10, 5000)), silence_timeout=3.0)
        assert len(chunks) == 2
        assert len(chunks[0]) / 16000 == pytest.approx(22.6, abs=0.2)

    def test_pause_shorter_than_chunk_pause_does_not_split(self):
        pcm = self.pcm((22, 5000), (0.3, 0), (10, 5000))
        assert len(self.split(pcm, silence_timeout=3.0)) == 1

    def test_short_pause_inside_minimum_does_not_split(self):
        assert len(self.split(self.pcm((5, 5000), (2, 0), (5, 5000)))) == 1

    def test_forces_split_without_pause(self):
        chunks = self.split(self.pcm((130, 5000)), chunk_max_seconds=50)
        assert len(chunks) == 3

    def test_silence_only_is_dropped(self):
        assert self.split(self.pcm((10, 0))) == []

    def test_all_speech_samples_are_kept(self):
        pcm = self.pcm((25, 5000), (2, 0), (10, 5000))
        chunks = self.split(pcm)
        assert sum(int(np.count_nonzero(chunk)) for chunk in chunks) == np.count_nonzero(pcm)

    def test_transcribe_file_joins_chunks_once(self, manager, tmp_path):
        manager.chunk_min_seconds = 20
        manager.chunk_max_seconds = 120
        pcm = self.pcm((25, 5000), (2, 0), (10, 5000)).astype(np.float32) / 32768.0
        calls = script_transcripts(manager, "eins", "zwei")
        with patch("vocalinux.speech_recognition.diarization.load_audio", return_value=pcm):
            text = manager.transcribe_file("x.wav")
        assert len(calls) == 2
        assert text == "eins zwei"


class TestFileLogging:
    @pytest.fixture(autouse=True)
    def restore_logging(self, monkeypatch):
        from vocalinux.utils import file_logging

        root = logging.getLogger()
        vocalinux_logger = logging.getLogger("vocalinux")
        old_level = vocalinux_logger.level
        old_handlers = list(root.handlers)
        old_hooks = (sys.excepthook, threading.excepthook)
        monkeypatch.setattr(file_logging, "_file_handler", None)
        yield
        for handler in list(root.handlers):
            if handler not in old_handlers:
                root.removeHandler(handler)
                handler.close()
        vocalinux_logger.setLevel(old_level)
        sys.excepthook, threading.excepthook = old_hooks

    def test_file_gets_debug_but_not_trace(self, tmp_path):
        from vocalinux.utils.file_logging import setup_file_logging

        path = setup_file_logging(str(tmp_path))
        log = logging.getLogger("vocalinux.test_long_recordings")
        log.debug("debug line")
        log.log(rm.LOG_LEVEL_TRACE, "trace line")
        try:
            raise ValueError("boom")
        except ValueError:
            log.exception("with traceback")
        for handler in logging.getLogger().handlers:
            handler.flush()

        content = open(path, encoding="utf-8").read()
        assert "debug line" in content
        assert "trace line" not in content
        assert "Traceback" in content and "ValueError: boom" in content

    def test_setup_twice_adds_one_handler(self, tmp_path):
        from vocalinux.utils.file_logging import setup_file_logging

        before = len(logging.getLogger().handlers)
        setup_file_logging(str(tmp_path))
        setup_file_logging(str(tmp_path))
        assert len(logging.getLogger().handlers) == before + 1


class TestTranscribeCli:
    def test_transcribe_argument_is_parsed(self):
        from vocalinux import main

        with patch.object(sys, "argv", ["vocalinux", "--transcribe", "rec.wav"]):
            args = main.parse_arguments()
        assert args.transcribe == "rec.wav"
        assert args.transcribe_file is None

    def test_transcribe_file_argument_is_unchanged(self):
        from vocalinux import main

        with patch.object(sys, "argv", ["vocalinux", "--transcribe-file", "rec.wav"]):
            args = main.parse_arguments()
        assert args.transcribe is None
        assert args.transcribe_file == "rec.wav"

    def test_main_dispatches_to_transcription(self):
        from vocalinux import main

        with (
            patch.object(sys, "argv", ["vocalinux", "--transcribe", "rec.wav"]),
            patch.object(main, "_run_recording_transcription", return_value=0) as run,
            pytest.raises(SystemExit) as exit_info,
        ):
            main.main()
        assert exit_info.value.code == 0
        assert run.call_args.args[0] == "rec.wav"

    def test_transcription_prints_text(self, capsys):
        from vocalinux import main

        fake_manager = MagicMock()
        fake_manager.model_ready = True
        fake_manager.transcribe_file.return_value = "hallo welt"
        args = MagicMock(engine=None, language=None, model=None)
        with patch(
            "vocalinux.speech_recognition.recognition_manager.SpeechRecognitionManager",
            return_value=fake_manager,
        ):
            code = main._run_recording_transcription("rec.wav", args)
        assert code == 0
        assert capsys.readouterr().out.strip() == "hallo welt"
        fake_manager.transcribe_file.assert_called_once_with("rec.wav")


def test_queue_module_import_is_plain():
    assert issubclass(recordings.SegmentQueue, queue.Queue)
