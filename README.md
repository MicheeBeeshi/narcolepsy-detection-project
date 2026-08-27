# REM-Onset Anomaly Detection in Narcolepsy

Unsupervised anomaly detection for identifying abnormal REM-sleep-onset
patterns in EEG, using narcolepsy as the clinical motivating case.

## Problem

Narcolepsy is a chronic neurological sleep disorder marked by abnormally
short REM sleep latency and sleep-onset REM periods (SOREMPs) -- entering
REM sleep much sooner after sleep onset than is typical. These patterns
are part of the clinical diagnostic criteria (via the Multiple Sleep
Latency Test) but are normally only assessed manually, by expert scoring
of overnight polysomnography.

This project asks: **can we automatically flag EEG segments that look
anomalous relative to normal pre-REM-transition brain activity**, using
only unsupervised learning (no narcolepsy-vs-control labels used during
training)?

## Approach

1. Train an anomaly-detection model (Isolation Forest) on EEG band-power
   features extracted from the minutes leading into REM-sleep transitions
   in **healthy control** subjects only -- this defines what a "normal"
   pre-REM EEG pattern looks like.
2. Run the trained model on pre-REM windows from **narcolepsy patients**
   and measure how often/how strongly those windows are flagged as
   anomalous, compared to controls.
3. Compare against the known clinical marker (REM latency) as a sanity
   check and complementary signal.

## Dataset

[CAP Sleep Database](https://physionet.org/content/capslpdb/1.0.0/) on
PhysioNet -- free, no credentialing required. Includes full-night
polysomnography (EEG, EOG, EMG) with expert-scored sleep stages for
multiple subject groups, including narcolepsy patients and healthy
controls.

**Setup:**
```bash
# From PhysioNet, download the .edf and .txt (annotation) files for the
# subjects you want into the data/ folder, e.g.:
#   data/n1.edf   data/n1.txt
#   data/sc1.edf  data/sc1.txt
```
Data files are not committed to this repo (see `.gitignore`) -- download
them yourself from PhysioNet.

## Project structure
```
├── src/
│   ├── data_loader.py       # Load EDF recordings + sleep-stage annotations
│   ├── rem_transitions.py   # Detect REM onsets, extract pre-REM EEG windows
│   ├── features.py          # (next) band-power feature extraction
│   ├── anomaly_model.py     # (next) Isolation Forest training/scoring
│   └── evaluate.py          # (next) score distributions, plots
├── notebooks/                # exploratory analysis
├── tests/                    # unit tests
├── data/                      # (gitignored) raw PhysioNet downloads go here
└── requirements.txt
```

## Status

🚧 Work in progress. Currently implemented:
- [x] EDF + annotation loading (`data_loader.py`)
- [x] REM transition / SOREMP detection (`rem_transitions.py`)
- [ ] Band-power feature extraction
- [ ] Isolation Forest anomaly model
- [ ] Leave-one-subject-out evaluation
- [ ] Results write-up + visualizations

## Important caveat

This is a research/educational project using retrospective, lab-recorded
overnight PSG data. It is **not** a real-time or wearable-device episode
predictor, and is not validated for any clinical use. The goal is to
demonstrate a signal-processing + anomaly-detection pipeline on a
genuine neurological dataset, not to produce a diagnostic tool.

## Setup
```bash
pip install -r requirements.txt
python src/data_loader.py       # sanity-check data loading
python src/rem_transitions.py   # sanity-check REM transition detection
```
