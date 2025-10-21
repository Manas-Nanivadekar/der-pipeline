import os
import json
import logging
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
import torch

from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def download_file(url: str, output_path: Path) -> bool:
    try:
        log.info(f"Downloading {output_path.name}")
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        return True
    except Exception as e:
        log.error(f"Download failed: {e}")
        return False


def load_ground_truth(json_path: str) -> Annotation:
    with open(json_path, "r") as f:
        data = json.load(f)

    annotation = Annotation()
    for seg in data.get("transcriptions", []):
        start, end = float(seg["start_time"]), float(seg["end_time"])
        if end > start:
            annotation[Segment(start, end)] = seg["speaker_id"]

    log.info(f"Loaded {len(annotation)} segments")
    return annotation


def main():
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("Set HF_TOKEN environment variable")

    log.info("Loading Pyannote pipeline")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", token=HF_TOKEN
    )
    if torch.cuda.is_available():
        pipeline = pipeline.to(torch.device("cuda"))
        log.info("Using GPU")

    der_metric = DiarizationErrorRate()

    log.info("Loading data.csv")
    try:
        df = pd.read_csv("data.csv")
        if "rec_id" not in df.columns:
            raise ValueError()
    except:
        df = pd.read_csv(
            "data.csv", header=None, names=["rec_id", "rec_url", "transcript_url"]
        )

    Path("data/audio").mkdir(parents=True, exist_ok=True)
    Path("data/transcripts").mkdir(parents=True, exist_ok=True)
    Path("data/diarization").mkdir(parents=True, exist_ok=True)
    log.info("Created output directories")

    results = []

    for _, row in df.iterrows():
        rec_id = row["rec_id"]
        log.info(f"\n{'='*50}\nProcessing {rec_id}\n{'='*50}")

        audio_path = Path(f"data/audio/{rec_id}.wav")
        transcript_path = Path(f"data/transcripts/{rec_id}.json")

        if not download_file(row["rec_url"], audio_path):
            results.append({"rec_id": rec_id, "DER": None, "status": "download_failed"})
            continue

        if not download_file(row["transcript_url"], transcript_path):
            results.append({"rec_id": rec_id, "DER": None, "status": "download_failed"})
            continue

        try:

            log.info("Running diarization")
            output = pipeline(str(audio_path))
            hypothesis = output.speaker_diarization

            diar_path = Path(f"data/diarization/{rec_id}.rttm")
            try:
                with open(diar_path, "w") as f:
                    hypothesis.write_rttm(f)
                log.info(f"Saved diarization to {diar_path}")
            except Exception as e:
                log.warning(f"Could not save RTTM: {e}")
                diar_path = None

            reference = load_ground_truth(str(transcript_path))

            log.info("Calculating DER")
            try:
                metrics = der_metric(reference, hypothesis, detailed=True)
                der = metrics["diarization error rate"]
                fa = metrics.get("false alarm", 0)
                md = metrics.get("missed detection", 0)
                conf = metrics.get("confusion", 0)
            except:
                der = der_metric(reference, hypothesis)
                fa = md = conf = 0

            log.info(f"DER: {der:.4f}")
            results.append(
                {
                    "rec_id": rec_id,
                    "DER": der,
                    "false_alarm": fa,
                    "missed_detection": md,
                    "confusion": conf,
                    "diarization_file": str(diar_path) if diar_path else None,
                    "status": "success",
                }
            )

        except Exception as e:
            log.error(f"Processing failed: {e}")
            results.append(
                {"rec_id": rec_id, "DER": None, "status": "failed", "error": str(e)}
            )

    report_df = pd.DataFrame(results)
    report_path = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    report_df.to_csv(report_path, index=False)

    success = report_df[report_df["status"] == "success"]
    log.info(f"\n{'='*50}\nSummary\n{'='*50}")
    log.info(f"Total: {len(df)}")
    log.info(f"Success: {len(success)}")
    if len(success) > 0:
        log.info(f"Average DER: {success['DER'].mean():.4f}")
    log.info(f"Report: {report_path}")


if __name__ == "__main__":
    main()
