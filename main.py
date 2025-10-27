import json
import os
import warnings
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime

from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

from utils import setup_logging, ensure_directories
from download import download_file
from diagnostics import analyze_diarization, print_diagnostics

log = setup_logging("pipeline")


def configure_torch():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        log.info("TF32 enabled for GPU")


def suppress_known_warnings():
    """Suppress known non-critical warnings."""
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="pyannote.audio.utils.reproducibility"
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="pyannote.audio.pipelines.speaker_diarization",
    )
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="pyannote.audio.models.blocks.pooling"
    )
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="pyannote.metrics.utils"
    )


def load_ground_truth(json_path: str) -> Annotation:
    """Load ground truth annotations from JSON."""
    with open(json_path) as f:
        data = json.load(f)

    annotation = Annotation()
    for seg in data.get("transcriptions", []):
        start, end = float(seg["start_time"]), float(seg["end_time"])
        if end > start:
            annotation[Segment(start, end)] = seg["speaker_id"]

    return annotation


def process_recording(pipeline, rec_id: str, rec_url: str, transcript_url: str) -> dict:
    """Process a single audio recording"""

    audio_path = Path(f"data/audio/{rec_id}.wav")
    transcript_path = Path(f"data/transcripts/{rec_id}.json")

    # Download files
    if not download_file(rec_url, audio_path):
        return {"rec_id": rec_id, "status": "download_failed"}

    if not download_file(transcript_url, transcript_path):
        return {"rec_id": rec_id, "status": "download_failed"}

    try:

        # Perform diarization
        log.info(f"Processing {rec_id}")
        output = pipeline(str(audio_path), min_speaker=2, max_speakers=2)
        hypothesis = output.speaker_diarization

        # Save RTTM
        rttm_path = Path(f"data/diarization/{rec_id}.rttm")
        with open(rttm_path, "w") as f:
            hypothesis.write_rttm(f)

        # Load reference and calculate DER
        reference = load_ground_truth(str(transcript_path))

        der_metric = DiarizationErrorRate()
        metrics = der_metric(reference, hypothesis, detailed=True)

        # Get diagnostics
        diag = analyze_diarization(audio_path, reference, hypothesis)
        print_diagnostics(rec_id, diag)

        result = {
            "rec_id": rec_id,
            "DER": metrics["diarization error rate"],
            "false_alarm": metrics.get("false alarm", 0),
            "missed_detection": metrics.get("missed detection", 0),
            "confusion": metrics.get("confusion", 0),
            "status": "success",
            **diag,
        }

        log.info(f"DER: {result['DER']:.4f}")
        return result

    except Exception as e:
        log.error(f"Failed: {e}")
        return {"rec_id": rec_id, "status": "failed", "error": str(e)}


def main():
    configure_torch()
    suppress_known_warnings()

    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("Set HF_TOKEN environment variable")

    ensure_directories()

    log.info("Loading Pyannote pipeline")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", token=HF_TOKEN
    )

    if torch.cuda.is_available():
        pipeline = pipeline.to(torch.device("cuda"))
        log.info("Using GPU")

    df = pd.read_csv("data.csv")
    if "rec_id" not in df.columns:
        df = pd.read_csv(
            "data.csv", header=None, names=["rec_id", "rec_url", "transcript_url"]
        )

    # Process recordings
    results = []
    for _, row in df.iterrows():
        result = process_recording(
            pipeline, row["rec_id"], row["rec_url"], row["transcript_url"]
        )
        results.append(result)

    # Save report
    report_df = pd.DataFrame(results)
    report_path = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    report_df.to_csv(report_path, index=False)

    # Summary
    success = report_df[report_df["status"] == "success"]
    log.info(f"\n{'='*50}")
    log.info(f"Total: {len(df)} | Success: {len(success)}")
    if len(success) > 0:
        log.info(f"Avg DER: {success['DER'].mean():.4f}")
        log.info(f"Avg Missing Speech: {success['missing_speech_pct'].mean():.1f}%")
    log.info(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
