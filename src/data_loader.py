"""
data_loader.py

Loads polysomnography (PSG) recordings from the PhysioNet CAP Sleep Database
and pairs each recording with its expert-scored sleep-stage annotations.

Dataset: https://physionet.org/content/capslpdb/1.0.0/
Download instructions are in the project README.

Expected local layout (after downloading):
    data/
        n1.edf          <- narcolepsy patient recordings
        n1.txt          <- corresponding hypnogram / stage annotations
        nfle1.edf       <- (other CAP groups you may add later)
        ...

CAP file naming convention: subject codes starting with 'n' are narcolepsy
patients; healthy controls are typically labeled with a 'sc' prefix in the
CAP database README on PhysioNet. Confirm exact prefixes against the
dataset's own SUBJECTS file when you download it, since labeling
conventions can vary by release.
"""

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# CAP Sleep Database stage labels -> standard AASM-style integer codes
STAGE_MAP = {
    "W": 0,
    "S1": 1,
    "S2": 2,
    "S3": 3,
    "S4": 3,  # S3/S4 are often merged into "N3" in modern scoring
    "R": 4,   # REM
    "MT": -1,  # movement time / artifact, excluded from analysis
}


@dataclass
class Recording:
    subject_id: str
    group: str          # "narcolepsy" or "control"
    raw: mne.io.Raw
    stages: pd.DataFrame  # columns: onset_sec, duration_sec, stage_label, stage_code


def list_available_recordings(data_dir: Path = DATA_DIR) -> list[str]:
    """Return subject IDs for which both .edf and annotation files exist."""
    edf_files = {f.stem for f in data_dir.glob("*.edf")}
    ann_files = {f.stem for f in data_dir.glob("*.txt")}
    return sorted(edf_files & ann_files)


def infer_group(subject_id: str) -> str:
    """
    Rough heuristic based on CAP naming convention. Double-check this
    against the dataset's SUBJECTS documentation before trusting it for
    real analysis -- naming conventions in CAP are not perfectly uniform.
    """
    sid = subject_id.lower()
    if sid.startswith("n") and not sid.startswith("nfle"):
        return "narcolepsy"
    if sid.startswith("sc"):
        return "control"
    return "unknown"


def load_stage_annotations(txt_path: Path) -> pd.DataFrame:
    """
    Parse CAP-format hypnogram text files into a tidy DataFrame.

    NOTE: CAP annotation text files vary somewhat in format between
    recordings. This parser assumes a simple whitespace-delimited format
    with columns like: onset_time, duration, stage_label. You will likely
    need to adjust this parser after inspecting a real downloaded file --
    treat this as a starting point, not a finished parser.
    """
    rows = []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            onset, duration, label = parts[0], parts[1], parts[2]
            try:
                onset_sec = float(onset)
                duration_sec = float(duration)
            except ValueError:
                continue  # skip header / malformed lines
            stage_code = STAGE_MAP.get(label.upper(), -1)
            rows.append((onset_sec, duration_sec, label.upper(), stage_code))

    df = pd.DataFrame(
        rows, columns=["onset_sec", "duration_sec", "stage_label", "stage_code"]
    )
    return df


def load_recording(subject_id: str, data_dir: Path = DATA_DIR) -> Recording:
    edf_path = data_dir / f"{subject_id}.edf"
    txt_path = data_dir / f"{subject_id}.txt"

    if not edf_path.exists() or not txt_path.exists():
        raise FileNotFoundError(
            f"Missing files for subject '{subject_id}'. Expected {edf_path} "
            f"and {txt_path}. Did you download the CAP Sleep Database into "
            f"the data/ folder? See README.md."
        )

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    stages = load_stage_annotations(txt_path)
    group = infer_group(subject_id)

    return Recording(subject_id=subject_id, group=group, raw=raw, stages=stages)


if __name__ == "__main__":
    available = list_available_recordings()
    if not available:
        print(
            "No recordings found in data/. Download CAP Sleep Database "
            "files (.edf + .txt) from PhysioNet into the data/ folder, "
            "then rerun this script."
        )
    else:
        print(f"Found {len(available)} recordings: {available}")
        rec = load_recording(available[0])
        print(f"Loaded {rec.subject_id} (group={rec.group})")
        print(rec.raw.info)
        print(rec.stages.head())
