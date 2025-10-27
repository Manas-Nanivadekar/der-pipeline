import librosa
import numpy as np
from pathlib import Path
from pyannote.core import Annotation


def analyze_diarization(
    audio_path: Path, reference: Annotation, hypothesis: Annotation
) -> dict:
    """Analyze diarization quality."""
    audio_duration = librosa.get_duration(path=str(audio_path))

    # Calculate speech durations
    ref_duration = sum(seg.duration for seg in reference.get_timeline())
    hyp_duration = sum(seg.duration for seg in hypothesis.get_timeline())

    # Calculate metrics
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


def analyze_audio_quality(audio_path: Path) -> dict:
    """Analyze audio characteristics that might affect diarization."""
    y, sr = librosa.load(audio_path, sr=None)

    # RMS energy
    rms = librosa.feature.rms(y=y)[0]

    # Zero crossing rate (indicator of noise)
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    # Spectral rolloff (frequency content)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

    return {
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "rolloff_mean": float(np.mean(rolloff)),
        "silence_ratio": float(np.sum(rms < 0.01) / len(rms)),  # % of very quiet frames
    }


def identify_problem_type(result: dict) -> str:
    """Identify the primary problem with this recording."""
    fa = result.get("false_alarm", 0)
    md = result.get("missed_detection", 0)
    conf = result.get("confusion", 0)

    if result["DER"] < 0.2:
        return "GOOD"
    elif fa > 0.3:
        return "HIGH_FALSE_ALARM"
    elif conf > 0.4:
        return "HIGH_CONFUSION"
    elif md > 0.3:
        return "HIGH_MISSED_DETECTION"
    elif fa > md and fa > conf:
        return "FALSE_ALARM_DOMINANT"
    elif conf > fa and conf > md:
        return "CONFUSION_DOMINANT"
    else:
        return "MIXED_ISSUES"
