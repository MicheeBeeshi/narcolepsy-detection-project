"""
features.py

Converts raw pre-REM EEG windows (from rem_transitions.py) into band-power
feature vectors suitable for the anomaly detection model.

Why band power: delta/theta/alpha/beta power ratios are the standard
features used in sleep EEG research to characterize brain state, and are
specifically implicated in the clinical literature distinguishing
narcolepsy patients from healthy sleepers (see README for citations).
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch

# Standard EEG frequency bands (Hz)
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}


@dataclass
class WindowFeatures:
    subject_id: str
    group: str
    band_power: dict[str, float]       # absolute power per band, averaged across channels
    band_power_ratio: dict[str, float]  # each band's power as a fraction of total power
    theta_alpha_ratio: float            # commonly used drowsiness/state indicator


def compute_band_power(window: np.ndarray, sfreq: float) -> dict[str, float]:
    """
    Compute average power in each standard band for a single EEG window.

    window: array of shape (n_channels, n_samples)
    sfreq: sampling frequency in Hz

    Uses Welch's method (scipy.signal.welch) to estimate the power spectral
    density, then integrates power within each band's frequency range.
    Averages across channels to produce one value per band.
    """
    n_channels = window.shape[0]
    band_powers = {band: [] for band in BANDS}

    for ch in range(n_channels):
        freqs, psd = welch(window[ch], fs=sfreq, nperseg=min(len(window[ch]), int(sfreq * 4)))
        for band, (low, high) in BANDS.items():
            mask = (freqs >= low) & (freqs <= high)
            # np.trapz was removed in NumPy 2.0+ in favor of np.trapezoid;
            # fall back for compatibility with either version installed.
            trapz_fn = getattr(np, "trapezoid", None) or np.trapz
            power = trapz_fn(psd[mask], freqs[mask]) if mask.any() else 0.0
            band_powers[band].append(power)

    return {band: float(np.mean(vals)) for band, vals in band_powers.items()}


def extract_window_features(
    window: np.ndarray,
    sfreq: float,
    subject_id: str,
    group: str,
) -> WindowFeatures:
    """Extract the full feature set for one pre-REM EEG window."""
    band_power = compute_band_power(window, sfreq)
    total_power = sum(band_power.values()) or 1e-12  # avoid divide-by-zero
    band_power_ratio = {band: p / total_power for band, p in band_power.items()}

    # Theta/alpha ratio is a commonly used indicator of drowsiness/sleep
    # transition state in EEG research.
    theta_alpha_ratio = band_power["theta"] / (band_power["alpha"] or 1e-12)

    return WindowFeatures(
        subject_id=subject_id,
        group=group,
        band_power=band_power,
        band_power_ratio=band_power_ratio,
        theta_alpha_ratio=theta_alpha_ratio,
    )


def features_to_vector(features: WindowFeatures) -> np.ndarray:
    """
    Flatten a WindowFeatures object into a fixed-order numeric vector for
    use with scikit-learn models. Order: band_power_ratio (4 values),
    theta_alpha_ratio (1 value) = 5 features total.
    """
    band_order = ["delta", "theta", "alpha", "beta"]
    vec = [features.band_power_ratio[b] for b in band_order]
    vec.append(features.theta_alpha_ratio)
    return np.array(vec)


if __name__ == "__main__":
    from data_loader import list_available_recordings, load_recording
    from rem_transitions import extract_pre_rem_windows

    available = list_available_recordings()
    if not available:
        print("No data found -- see data_loader.py for setup instructions.")
    else:
        rec = load_recording(available[0])
        windows = extract_pre_rem_windows(rec)
        sfreq = rec.raw.info["sfreq"]

        print(f"Extracting features for {len(windows)} windows from {rec.subject_id}...")
        for i, w in enumerate(windows):
            feats = extract_window_features(w, sfreq, rec.subject_id, rec.group)
            vec = features_to_vector(feats)
            print(f"  window {i}: band_power_ratio={feats.band_power_ratio}, "
                  f"theta/alpha={feats.theta_alpha_ratio:.3f}")
            print(f"    -> feature vector: {vec}")
