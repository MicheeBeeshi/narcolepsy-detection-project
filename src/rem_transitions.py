"""
rem_transitions.py

Identifies REM-onset transitions in a scored PSG recording and extracts
the EEG segment immediately preceding each transition, for later feature
extraction and anomaly detection.

Key clinical concept this encodes:
    Narcolepsy is characterized by abnormally SHORT REM latency and, on
    the Multiple Sleep Latency Test, sleep-onset REM periods (SOREMPs) --
    entering REM very soon after sleep onset, rather than after the usual
    NREM-first progression. We use this as the basis for defining
    "pre-REM-transition" windows to analyze.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_loader import Recording


@dataclass
class RemTransition:
    onset_sec: float          # time REM stage begins
    sleep_onset_sec: float    # time sleep first began (first non-wake epoch)
    rem_latency_sec: float    # onset_sec - sleep_onset_sec
    is_soremp: bool           # True if latency is abnormally short (<15 min is a common threshold)


SOREMP_LATENCY_THRESHOLD_SEC = 15 * 60  # 15 minutes; adjust based on literature/your data


def find_sleep_onset(stages: pd.DataFrame) -> float | None:
    """First epoch scored as anything other than Wake (stage_code 0)."""
    asleep = stages[stages["stage_code"] > 0]
    if asleep.empty:
        return None
    return asleep.iloc[0]["onset_sec"]


def find_rem_transitions(stages: pd.DataFrame) -> list[RemTransition]:
    """
    Find every point where the recording transitions INTO REM (stage_code 4)
    from a non-REM stage. Returns one RemTransition per onset, including
    REM latency relative to overall sleep onset.
    """
    sleep_onset = find_sleep_onset(stages)
    if sleep_onset is None:
        return []

    transitions = []
    prev_code = None
    for _, row in stages.sort_values("onset_sec").iterrows():
        code = row["stage_code"]
        if code == 4 and prev_code is not None and prev_code != 4:
            latency = row["onset_sec"] - sleep_onset
            transitions.append(
                RemTransition(
                    onset_sec=row["onset_sec"],
                    sleep_onset_sec=sleep_onset,
                    rem_latency_sec=latency,
                    is_soremp=latency < SOREMP_LATENCY_THRESHOLD_SEC,
                )
            )
        prev_code = code
    return transitions


def extract_pre_rem_windows(
    recording: Recording,
    window_sec: float = 300.0,  # 5 minutes before REM onset
    channel_picks: list[str] | None = None,
) -> list[np.ndarray]:
    """
    For each REM transition in the recording, pull out the raw EEG segment
    in the `window_sec` seconds immediately preceding it. These segments
    are the input to feature extraction (see features.py, next step).

    Returns a list of arrays, each shape (n_channels, n_samples).
    Skips transitions that don't have a full window available (e.g. REM
    onset too close to the start of the recording).
    """
    raw = recording.raw
    if channel_picks:
        raw = raw.copy().pick(channel_picks)

    sfreq = raw.info["sfreq"]
    transitions = find_rem_transitions(recording.stages)

    windows = []
    for t in transitions:
        start_sec = t.onset_sec - window_sec
        if start_sec < 0:
            continue  # not enough data before this transition
        start_sample = int(start_sec * sfreq)
        stop_sample = int(t.onset_sec * sfreq)
        data, _ = raw[:, start_sample:stop_sample]
        windows.append(data)

    return windows


if __name__ == "__main__":
    from data_loader import list_available_recordings, load_recording

    available = list_available_recordings()
    if not available:
        print("No data found -- see data_loader.py for setup instructions.")
    else:
        rec = load_recording(available[0])
        transitions = find_rem_transitions(rec.stages)
        print(f"Subject {rec.subject_id} ({rec.group}): {len(transitions)} REM transitions")
        for t in transitions:
            flag = "SOREMP" if t.is_soremp else "normal"
            print(f"  onset={t.onset_sec:.0f}s  latency={t.rem_latency_sec/60:.1f}min  [{flag}]")

        windows = extract_pre_rem_windows(rec)
        print(f"Extracted {len(windows)} pre-REM EEG windows for feature extraction.")
