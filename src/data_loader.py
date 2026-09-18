"""
data_loader.py

Loads polysomnography (PSG) recordings from the PhysioNet CAP Sleep Database
and pairs each recording with its expert-scored sleep-stage annotations.

Dataset: https://physionet.org/content/capslpdb/1.0.0/
Download instructions are in the project README.

Expected local layout (after downloading):
    data/
        n1.edf          <- healthy control recordings (n1-n16)
        n1.txt          <- corresponding hypnogram / stage annotations
        narco1.edf      <- narcolepsy patient recordings (narco1-narco5)
        narco1.txt
        ...

CAP naming convention (confirmed from physionet.org/content/capslpdb):
    n1-n16        -> healthy controls (16 subjects)
    narco1-narco5 -> narcolepsy patients (5 subjects)
    brux*, ins*, nfle*, plm*, rbd*, sdb* -> other sleep-disorder groups,
        not used in this project.
"""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
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
    Based on the official CAP Sleep Database naming convention
    (physionet.org/content/capslpdb):
        n1-n16       -> healthy controls (no pathology)
        narco1-narco5 -> narcolepsy patients
        brux*, ins*, nfle*, plm*, rbd*, sdb* -> other sleep-disorder groups,
            not used by this project
    """
    sid = subject_id.lower()
    if sid.startswith("narco"):
        return "narcolepsy"
    if re.fullmatch(r"n\d+", sid):
        return "control"
    return "unknown"


# Matches a REMlogic data row, e.g.:
#   "W Unknown Position 22:35:17 SLEEP-S0 30 EOG"
#   "S2 Left 23:14:02 SLEEP-S2 30 C4-A1"
# Position can be one or more words, so it's matched lazily between the
# sleep-stage code and the hh:mm:ss timestamp.
# Matches a standalone hh:mm:ss (or hh.mm.ss) timestamp token, used to
# locate the time field regardless of how many other columns precede it.
# CAP .txt files are NOT perfectly consistent between subjects: some
# (e.g. n16) include a "Position" column and colon-separated times, while
# others (e.g. narco2) omit Position entirely and use dot-separated times.
# Rather than assume a fixed column layout, we scan each line's
# whitespace-separated tokens for the one that looks like a timestamp,
# and treat everything after it as Event/Duration/Location -- this works
# for both formats without needing per-subject special-casing.
TIME_TOKEN_PATTERN = re.compile(r"^\d{1,2}[.:]\d{2}[.:]\d{2}$")


def _parse_row(line: str):
    """
    Parse one data row using token-scanning (see TIME_TOKEN_PATTERN
    comment above). Returns (stage_label, time_str, event, duration_sec)
    or None if the line doesn't look like a data row (headers, blank
    lines, metadata lines like "Patient:\tNARCO 2" all fail to match and
    are skipped).
    """
    tokens = line.split()
    if len(tokens) < 4:
        return None

    stage_label = tokens[0].upper()
    if stage_label not in STAGE_MAP:
        return None  # not a stage-labeled row (e.g. a header/metadata line)

    time_idx = None
    for i in range(1, len(tokens)):
        if TIME_TOKEN_PATTERN.match(tokens[i]):
            time_idx = i
            break
    if time_idx is None:
        return None  # no timestamp found on this line

    time_str = tokens[time_idx].replace(".", ":")  # normalize to hh:mm:ss
    remainder = tokens[time_idx + 1:]
    if len(remainder) < 2:
        return None  # need at least Event and Duration after the time

    event = remainder[0]
    try:
        duration_sec = float(remainder[1])
    except ValueError:
        return None

    return stage_label, time_str, event, duration_sec


def load_stage_annotations(txt_path: Path) -> pd.DataFrame:
    """
    Parse a CAP Sleep Database .txt score file (REMlogic export format)
    into a tidy DataFrame of 30-second sleep-stage epochs.

    Handles the two known column layouts seen across CAP subjects:
        n16-style:    W  Unknown Position  22:35:17  SLEEP-S0  30  EOG
        narco2-style: W  21.50.53  SLEEP-S0  30  ECG1-ECG2
    (Position present vs. absent; colon vs. dot time separators.)
    See _parse_row() / TIME_TOKEN_PATTERN for how both are handled.

    Only rows whose Event starts with "SLEEP-" are kept -- these are the
    30-second macrostructure epochs. Other event types (MCAP-A1/A2/A3 =
    CAP microstructure, BUTTON = patient marker) are a different,
    finer-grained annotation layer and are intentionally excluded here.

    Times in the file are wall-clock (hh:mm:ss), not seconds-from-start,
    and overnight recordings cross midnight. This function detects each
    time rollover (time going "backwards") and adds a day, so onset_sec
    is a continuously increasing offset from the first epoch.
    """
    rows = []
    day_offset = 0
    prev_time = None
    base_time = None

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = _parse_row(line.strip())
            if parsed is None:
                continue
            stage_label, time_str, event, duration_sec = parsed

            if not event.upper().startswith("SLEEP-"):
                continue  # skip CAP microstructure / button / other events

            stage_code = STAGE_MAP.get(stage_label, -1)

            t = datetime.strptime(time_str, "%H:%M:%S")
            if base_time is None:
                base_time = t
                prev_time = t
            if t < prev_time:
                day_offset += 1  # crossed midnight
            prev_time = t

            absolute_time = t + timedelta(days=day_offset)
            onset_sec = (absolute_time - base_time).total_seconds()

            rows.append((onset_sec, duration_sec, stage_label, stage_code))

    return pd.DataFrame(
        rows, columns=["onset_sec", "duration_sec", "stage_label", "stage_code"]
    )


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
