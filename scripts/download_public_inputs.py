"""Download the public FHWA/Census inputs required by the revised workflow.

This helper is optional. It downloads the three single-file, comma-delimited
legacy NBI archives and the 2025 TIGER/Line state boundary archive, then extracts
each NBI archive to the exact filename expected by ``run_all.py``.

Run from the repository root:
    python scripts/download_public_inputs.py

The script never overwrites an existing raw input unless ``--force`` is used.
"""
from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

NBI_URLS = {
    2023: "https://www.fhwa.dot.gov/bridge/nbi/2023hwybronefiledel.zip",
    2024: "https://www.fhwa.dot.gov/bridge/nbi/2024hwybronefiledel.zip",
    2025: "https://www.fhwa.dot.gov/bridge/nbi/2025hwybronefiledel.zip",
}
TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2025/STATE/tl_2025_us_state.zip"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, destination: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "NBI-border-discontinuity-research/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def extract_single_nbi(zip_path: Path, target: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        candidates = [i for i in zf.infolist() if not i.is_dir()]
        if not candidates:
            raise RuntimeError(f"No file found inside {zip_path.name}")
        member = max(candidates, key=lambda i: i.file_size)
        with zf.open(member) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    with target.open("r", encoding="utf-8-sig", errors="replace") as f:
        header = f.readline()
    if "STATE_CODE_001" not in header or "STRUCTURE_NUMBER_008" not in header:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"Extracted {member.filename!r}, but it does not look like the FHWA delimited NBI file")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite existing required raw inputs")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)

    for year, url in NBI_URLS.items():
        target = RAW / f"{year}HwyBridgesDelimitedAllStates.txt"
        if target.exists() and not args.force:
            print(f"skip {target.relative_to(ROOT)} (already exists)")
            continue
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            print(f"download FHWA {year} NBI ...")
            download(url, tmp_path)
            if not zipfile.is_zipfile(tmp_path):
                raise RuntimeError(f"FHWA {year} response was not a ZIP archive")
            extract_single_nbi(tmp_path, target)
            print(f"  {target.relative_to(ROOT)}  sha256={sha256(target)}")
        finally:
            tmp_path.unlink(missing_ok=True)

    tiger = RAW / "tl_2025_us_state.zip"
    if tiger.exists() and not args.force:
        print(f"skip {tiger.relative_to(ROOT)} (already exists)")
    else:
        print("download Census 2025 TIGER/Line state boundaries ...")
        download(TIGER_URL, tiger)
        if not zipfile.is_zipfile(tiger):
            tiger.unlink(missing_ok=True)
            raise RuntimeError("Census response was not a ZIP archive")
        print(f"  {tiger.relative_to(ROOT)}  sha256={sha256(tiger)}")

    print("Public inputs are ready. Run: python run_all.py")


if __name__ == "__main__":
    main()
