"""Split a finished recording into chunks the way a live dictation does."""

from typing import Callable, Optional

import numpy as np

from .recordings import SAMPLE_RATE

#: Samples per step, the size of one live capture read.
STEP_SAMPLES = 1024


def split_like_dictation(
    pcm: np.ndarray,
    is_speech: Callable[[np.ndarray], Optional[bool]],
    silence_timeout: float,
    chunk_min_seconds: float,
    chunk_max_seconds: float,
) -> list[np.ndarray]:
    """Return the chunks a push-to-talk dictation of ``pcm`` would transcribe.

    Follows the rules of ``_record_audio``: a chunk ends at the first pause
    longer than ``silence_timeout`` once it holds ``chunk_min_seconds`` of
    audio, or at ``chunk_max_seconds`` without a pause. Audio without speech
    between two pauses is dropped. ``is_speech`` gets each step of int16
    samples; None counts as silence, like a VAD that has no decision yet.
    """
    chunks: list[np.ndarray] = []
    start = 0
    has_speech = False
    silence_seconds = 0.0
    min_samples = chunk_min_seconds * SAMPLE_RATE
    max_samples = chunk_max_seconds * SAMPLE_RATE

    for position in range(0, len(pcm), STEP_SAMPLES):
        end = min(position + STEP_SAMPLES, len(pcm))
        if is_speech(pcm[position:end]):
            has_speech = True
            silence_seconds = 0.0
        else:
            silence_seconds += STEP_SAMPLES / SAMPLE_RATE
            if silence_seconds > silence_timeout:
                silence_seconds = 0.0
                if not has_speech:
                    start = end
                elif chunk_min_seconds > 0 and end - start >= min_samples:
                    chunks.append(pcm[start:end])
                    start = end
                    has_speech = False
        if has_speech and chunk_min_seconds > 0 and end - start >= max_samples:
            chunks.append(pcm[start:end])
            start = end
            has_speech = False

    if has_speech and start < len(pcm):
        chunks.append(pcm[start:])
    return chunks
