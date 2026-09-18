"""
evaluate.py

Runs the full pipeline end-to-end and produces the comparisons/plots that
go in the README results section:

1. Anomaly score distributions: controls vs. narcolepsy patients
2. REM latency comparison: controls vs. narcolepsy patients (the known
   clinical marker, used here as a sanity check / complementary signal)
3. An example EEG spectrogram with the pre-REM window highlighted

Run this after you have both control ('sc*') and narcolepsy ('n*')
subjects downloaded into data/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal as sp_signal

from data_loader import list_available_recordings, load_recording
from rem_transitions import extract_pre_rem_windows, find_rem_transitions
from features import extract_window_features, features_to_vector
from anomaly_model import NarcolepsyAnomalyDetector

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def build_dataset():
    """
    Load every available recording, extract pre-REM windows and features,
    and split into control vs. narcolepsy groups. Also collects REM
    latency values per subject for the clinical-marker comparison.

    Returns: (control_vectors, test_vectors, test_ids, test_groups, latency_df)
    """
    available = list_available_recordings()
    if not available:
        raise RuntimeError(
            "No data found in data/. Download CAP Sleep Database subjects "
            "(.edf + .txt) first -- see README.md."
        )

    control_vectors = []
    test_vectors, test_ids, test_groups = [], [], []
    latency_rows = []

    for subject_id in available:
        rec = load_recording(subject_id)
        sfreq = rec.raw.info["sfreq"]

        # REM latency, for the clinical-marker sanity check
        transitions = find_rem_transitions(rec.stages)
        if transitions:
            first_latency_min = transitions[0].rem_latency_sec / 60
            latency_rows.append(
                {"subject_id": subject_id, "group": rec.group, "rem_latency_min": first_latency_min}
            )

        # Feature extraction for anomaly detection
        windows = extract_pre_rem_windows(rec)
        for w in windows:
            feats = extract_window_features(w, sfreq, subject_id, rec.group)
            vec = features_to_vector(feats)
            if rec.group == "control":
                control_vectors.append(vec)
            else:
                test_vectors.append(vec)
                test_ids.append(subject_id)
                test_groups.append(rec.group)

    latency_df = pd.DataFrame(latency_rows)
    return (
        np.array(control_vectors) if control_vectors else np.empty((0, 5)),
        np.array(test_vectors) if test_vectors else np.empty((0, 5)),
        test_ids,
        test_groups,
        latency_df,
    )


def plot_score_distributions(control_scores, narcolepsy_scores, save_path):
    """
    Compare anomaly scores between held-out controls and narcolepsy
    patients. Uses a strip plot (individual points, jittered) rather than
    a KDE/histogram -- with only a handful of windows per group and
    heavily quantized Isolation Forest scores (many identical values),
    a smoothed density plot produces a misleading spike rather than a
    meaningful shape. Showing every real point is more honest at this
    sample size.
    """
    import numpy as np

    plt.figure(figsize=(7, 5))
    rng = np.random.default_rng(42)

    groups = (["Controls"] * len(control_scores)) + (["Narcolepsy"] * len(narcolepsy_scores))
    scores = list(control_scores) + list(narcolepsy_scores)
    x_positions = [0] * len(control_scores) + [1] * len(narcolepsy_scores)
    jitter = rng.uniform(-0.08, 0.08, size=len(scores))
    x_jittered = [x + j for x, j in zip(x_positions, jitter)]
    colors = ["steelblue"] * len(control_scores) + ["crimson"] * len(narcolepsy_scores)

    plt.scatter(x_jittered, scores, c=colors, s=80, alpha=0.7, edgecolors="black", linewidths=0.5)

    # Mark each group's mean with a horizontal line
    if control_scores:
        plt.hlines(np.mean(control_scores), -0.25, 0.25, colors="steelblue", linewidth=2, label="Control mean")
    if narcolepsy_scores:
        plt.hlines(np.mean(narcolepsy_scores), 0.75, 1.25, colors="crimson", linewidth=2, label="Narcolepsy mean")

    plt.xticks([0, 1], ["Controls", "Narcolepsy"])
    plt.xlim(-0.5, 1.5)
    plt.ylabel("Anomaly score (higher = more unusual)")
    plt.title("Pre-REM EEG Anomaly Scores: Controls vs. Narcolepsy Patients")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_rem_latency(latency_df, save_path):
    """Compare REM latency (the known clinical marker) across groups."""
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=latency_df, x="group", y="rem_latency_min")
    sns.stripplot(data=latency_df, x="group", y="rem_latency_min", color="black", alpha=0.6, size=6)
    plt.axhline(15, color="gray", linestyle="--", label="SOREMP threshold (15 min)")
    plt.ylabel("REM latency (minutes)")
    plt.title("REM Latency by Group (clinical marker, for comparison)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_example_spectrogram(recording, window_sec, save_path, channel_idx=0):
    """
    Plot a spectrogram of one channel's full recording, with the pre-REM
    window(s) highlighted, as a visual sanity check of what the model
    is looking at.
    """
    raw = recording.raw
    sfreq = raw.info["sfreq"]
    data, times = raw[channel_idx, :]
    data = data.flatten()

    f, t, Sxx = sp_signal.spectrogram(data, fs=sfreq, nperseg=int(sfreq * 4))

    plt.figure(figsize=(12, 5))
    plt.pcolormesh(t / 60, f, 10 * np.log10(Sxx + 1e-12), shading="auto", cmap="viridis")
    plt.ylim(0, 30)
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (minutes)")
    plt.title(f"EEG Spectrogram: {recording.subject_id} (channel {channel_idx})")
    plt.colorbar(label="Power (dB)")

    transitions = find_rem_transitions(recording.stages)
    for t_trans in transitions:
        onset_min = t_trans.onset_sec / 60
        start_min = onset_min - (window_sec / 60)
        plt.axvspan(start_min, onset_min, color="red", alpha=0.2)
        plt.axvline(onset_min, color="red", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def run_full_evaluation():
    print("Building dataset from all available recordings...")
    control_vectors, test_vectors, test_ids, test_groups, latency_df = build_dataset()

    if len(control_vectors) < 2:
        print(
            "Not enough control-group data to train/evaluate meaningfully "
            "(need at least a couple of control subjects). Download more "
            "'sc*' subjects from the CAP Sleep Database."
        )
        return

    # Split controls: train on most, hold out a few to compare against narcolepsy scores
    split_idx = max(1, int(len(control_vectors) * 0.7))
    train_controls = control_vectors[:split_idx]
    heldout_controls = control_vectors[split_idx:]

    detector = NarcolepsyAnomalyDetector()
    detector.fit(train_controls)
    print(f"Trained on {len(train_controls)} control windows.")

    control_results = detector.score(
        heldout_controls,
        subject_ids=["heldout_control"] * len(heldout_controls),
        groups=["control"] * len(heldout_controls),
    ) if len(heldout_controls) else []

    narcolepsy_results = detector.score(test_vectors, test_ids, test_groups) if len(test_vectors) else []

    if control_results and narcolepsy_results:
        control_scores = [r.anomaly_score for r in control_results]
        narcolepsy_scores = [r.anomaly_score for r in narcolepsy_results]
        plot_score_distributions(
            control_scores, narcolepsy_scores, OUTPUT_DIR / "score_distributions.png"
        )
        print(f"Mean anomaly score -- controls: {np.mean(control_scores):.3f}, "
              f"narcolepsy: {np.mean(narcolepsy_scores):.3f}")
    else:
        print("Not enough held-out data on one side to plot score distributions.")

    if not latency_df.empty and latency_df["group"].nunique() > 1:
        plot_rem_latency(latency_df, OUTPUT_DIR / "rem_latency_comparison.png")
    else:
        print("Not enough subjects across groups to plot REM latency comparison.")

    available = list_available_recordings()
    narcolepsy_subject = next(
        (s for s in available if load_recording(s).group == "narcolepsy"), None
    )
    if narcolepsy_subject:
        rec = load_recording(narcolepsy_subject)
        plot_example_spectrogram(rec, window_sec=300.0, save_path=OUTPUT_DIR / "example_spectrogram.png")

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_full_evaluation()
