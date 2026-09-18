"""
anomaly_model.py

Trains an Isolation Forest on pre-REM EEG feature vectors from healthy
control subjects only, then scores narcolepsy patients' pre-REM windows
against that learned "normal" baseline.

Isolation Forest, briefly: it works by randomly partitioning the feature
space and measuring how many splits it takes to isolate each point.
Points that are "normal" (in a dense cluster of similar points) take many
splits to isolate. Points that are unusual/rare take very few splits to
isolate, because they're already separated from everything else. That
"how few splits did it take" number becomes the anomaly score.

Nothing about narcolepsy is given to the model -- it only ever learns
what control-subject data looks like, and then measures distance from
that.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class ScoredWindow:
    subject_id: str
    group: str
    anomaly_score: float   # higher = more anomalous (we flip sklearn's sign convention, see below)
    is_anomaly: bool       # True if flagged as an outlier


class NarcolepsyAnomalyDetector:
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        contamination: expected proportion of outliers in training data.
        Since we train ONLY on controls, this should be low (~0.05-0.1) --
        it's not "expected narcolepsy rate," it's "how much noise/atypical
        data do we expect even among healthy controls." Tune this once you
        have real data; it directly affects how sensitive the flagging is.
        """
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=200,
        )
        self._is_fit = False

    def fit(self, control_feature_vectors: np.ndarray) -> None:
        """
        control_feature_vectors: shape (n_windows, n_features), built only
        from healthy control subjects' pre-REM windows.
        """
        scaled = self.scaler.fit_transform(control_feature_vectors)
        self.model.fit(scaled)
        self._is_fit = True

    def score(
        self,
        feature_vectors: np.ndarray,
        subject_ids: list[str],
        groups: list[str],
    ) -> list[ScoredWindow]:
        """
        Score any set of windows (typically narcolepsy patients, but can
        also include held-out controls as a sanity check) against the
        model learned from controls.
        """
        if not self._is_fit:
            raise RuntimeError("Call .fit() on control data before scoring.")

        scaled = self.scaler.transform(feature_vectors)

        # sklearn's decision_function: higher = more normal, lower = more
        # anomalous. We flip the sign so higher = more anomalous, which is
        # more intuitive for reporting ("high anomaly score" should mean
        # "looks unusual").
        raw_scores = -self.model.decision_function(scaled)
        predictions = self.model.predict(scaled)  # -1 = outlier, 1 = inlier

        results = []
        for sid, grp, score, pred in zip(subject_ids, groups, raw_scores, predictions):
            results.append(
                ScoredWindow(
                    subject_id=sid,
                    group=grp,
                    anomaly_score=float(score),
                    is_anomaly=(pred == -1),
                )
            )
        return results


if __name__ == "__main__":
    from data_loader import list_available_recordings, load_recording
    from rem_transitions import extract_pre_rem_windows
    from features import extract_window_features, features_to_vector

    available = list_available_recordings()
    if not available:
        print("No data found -- see data_loader.py for setup instructions.")
        raise SystemExit(0)

    control_vectors = []
    test_vectors, test_ids, test_groups = [], [], []

    for subject_id in available:
        rec = load_recording(subject_id)
        windows = extract_pre_rem_windows(rec)
        sfreq = rec.raw.info["sfreq"]

        for w in windows:
            feats = extract_window_features(w, sfreq, subject_id, rec.group)
            vec = features_to_vector(feats)

            if rec.group == "control":
                control_vectors.append(vec)
            else:
                test_vectors.append(vec)
                test_ids.append(subject_id)
                test_groups.append(rec.group)

    if not control_vectors:
        print(
            "No control-group recordings found -- can't train the model. "
            "Download at least one 'sc*' (healthy control) subject from "
            "the CAP Sleep Database into data/."
        )
        raise SystemExit(0)

    detector = NarcolepsyAnomalyDetector()
    detector.fit(np.array(control_vectors))
    print(f"Trained on {len(control_vectors)} control pre-REM windows.")

    if test_vectors:
        results = detector.score(np.array(test_vectors), test_ids, test_groups)
        n_flagged = sum(r.is_anomaly for r in results)
        print(f"Scored {len(results)} narcolepsy-group windows: "
              f"{n_flagged} flagged as anomalous.")
        for r in results:
            flag = "ANOMALY" if r.is_anomaly else "normal"
            print(f"  {r.subject_id}: score={r.anomaly_score:.3f} [{flag}]")
    else:
        print("No narcolepsy-group windows found to test against.")
