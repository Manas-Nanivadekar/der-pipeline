import requests
from pathlib import Path


def download_file(url: str, output_path: Path) -> bool:
    """Download file from URL."""
    try:
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Download failed for {output_path.name}: {e}")
        return False
