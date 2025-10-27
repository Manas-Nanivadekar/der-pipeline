import librosa
from pathlib import Path
from pyannote.core import Annotation


def analyze_diarization(
    audio_path: Path, reference: Annotation, hypothesis: Annotation
) -> dict:
    """Analyse diarization quality."""

    audio_duration = librosa.get_duration(path=str(audio_path))

    # calculate speech duration
    ref_duration = sum(seg.duration for seg in reference.get_timeline())
    hyp_duration = sum(seg.duration for seg in hypothesis.get_timeline())

    # calculate metrics
    missing_speech = ref_duration - hyp_duration
    missing_pct = (missing_speech / ref_duration * 100) if ref_duration > 0 else 0

    return {
        "audio_duration": audio_duration,
        "ref_speech_duration": ref_duration,
        "hyp_speech_duration": hyp_duration,
        "missing_speech_seconds": missing_speech,
        "missing_speech_pct": missing_pct,
        "speakers_detected": len(set(hypothesis.labels())),
        "speakers_expected": len(set(reference.labels())),
    }


def print_diagnostics(rec_id: str, diag: dict):
    """Print diagnostic information."""
    print(f"\n=== {rec_id} ===")
    print(f"Audio: {diag['audio_duration']:.1f}s")
    print(f"Reference speech: {diag['ref_speech_duration']:.1f}s")
    print(f"Detected speech: {diag['hyp_speech_duration']:.1f}s")
    print(
        f"Missing: {diag['missing_speech_seconds']:.1f}s ({diag['missing_speech_pct']:.1f}%)"
    )
    print(
        f"Speakers: {diag['speakers_detected']} (expected {diag['speakers_expected']})"
    )
