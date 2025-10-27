import logging
from pathlib import Path
from datetime import datetime


def setup_logging(name: str) -> logging.Logger:
    """Setup basic logging"""
    log_file = f"{name}_{datetime.now().strftime("%Y%m%d_%H%H%S")}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    return logging.getLogger(name)


def ensure_directories():
    """Create required directories."""
    dirs = ["data/audio", "data/transcripts", "data/diarization"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
