"""
split_protocol.py

WITECH gesture authentication - single Final Test protocol.

Per gesture:
Train
  genuine  : owner's early sessions (all except last 4)
  impostor : P01~P05 except owner

Validation
  genuine  : owner's 2 sessions immediately before the final 2
  impostor : P06, P07
  purpose  : early stopping + authentication threshold selection

Final Test
  genuine  : owner's most recent 2 sessions
  impostor : P08, P09, P10 + X01~X18

The former Internal / External tests are merged into ONE Final Test.
The owner's final genuine samples are included only once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


OWNER_MAP = {
    "G1": "P01",
    "G2": "P02",
    "G3": "P03",
    "G4": "P04",
    "G5": "P05",
}

TRAIN_POOL = tuple(f"P{i:02d}" for i in range(1, 6))
VAL_IMPOSTORS = ("P06", "P07")
FINAL_IMPOSTORS = (
    "P08", "P09", "P10",
    *tuple(f"X{i:02d}" for i in range(1, 19)),
)

N_VAL_OWNER_SESSIONS = 2
N_FINAL_OWNER_SESSIONS = 2


@dataclass(frozen=True)
class GestureSplit:
    gesture: str
    owner: str

    train_idx: np.ndarray
    train_y: np.ndarray

    val_idx: np.ndarray
    val_y: np.ndarray

    final_idx: np.ndarray
    final_y: np.ndarray

    train_owner_sessions: tuple[str, ...]
    val_owner_sessions: tuple[str, ...]
    final_owner_sessions: tuple[str, ...]


def _str_array(x) -> np.ndarray:
    return np.asarray(x).astype(str)


def _pack(pos_idx: np.ndarray, neg_idx: np.ndarray):
    idx = np.concatenate([pos_idx, neg_idx]).astype(np.int64)
    y = np.concatenate([
        np.ones(len(pos_idx), dtype=np.int64),
        np.zeros(len(neg_idx), dtype=np.int64),
    ])
    return idx, y


def build_splits(meta: dict) -> Dict[str, GestureSplit]:
    gesture = _str_array(meta["gesture"])
    performer = _str_array(meta["performer"])
    session = _str_array(meta["session"])
    role = _str_array(meta["role"])

    n = len(gesture)
    if not (len(performer) == len(session) == len(role) == n):
        raise ValueError("metadata arrays have different lengths")

    result: Dict[str, GestureSplit] = {}

    for g, owner in OWNER_MAP.items():
        gmask = gesture == g
        if not np.any(gmask):
            raise ValueError(f"{g}: no samples found")

        own_mask = gmask & (performer == owner) & (role == "own")
        owner_sessions = tuple(sorted(set(session[own_mask])))

        min_sessions = N_VAL_OWNER_SESSIONS + N_FINAL_OWNER_SESSIONS + 1
        if len(owner_sessions) < min_sessions:
            raise ValueError(
                f"{g}/{owner}: need at least {min_sessions} owner sessions, "
                f"found {len(owner_sessions)}: {owner_sessions}"
            )

        final_sessions = owner_sessions[-N_FINAL_OWNER_SESSIONS:]

        val_end = len(owner_sessions) - N_FINAL_OWNER_SESSIONS
        val_start = val_end - N_VAL_OWNER_SESSIONS
        val_sessions = owner_sessions[val_start:val_end]

        train_sessions = owner_sessions[:val_start]

        train_pos = np.where(own_mask & np.isin(session, train_sessions))[0]
        val_pos = np.where(own_mask & np.isin(session, val_sessions))[0]
        final_pos = np.where(own_mask & np.isin(session, final_sessions))[0]

        train_neg_people = tuple(p for p in TRAIN_POOL if p != owner)

        train_neg = np.where(
            gmask
            & np.isin(performer, train_neg_people)
            & (role != "own")
        )[0]

        val_neg = np.where(
            gmask
            & np.isin(performer, VAL_IMPOSTORS)
            & (role != "own")
        )[0]

        final_neg = np.where(
            gmask
            & np.isin(performer, FINAL_IMPOSTORS)
            & (role != "own")
        )[0]

        train_idx, train_y = _pack(train_pos, train_neg)
        val_idx, val_y = _pack(val_pos, val_neg)
        final_idx, final_y = _pack(final_pos, final_neg)

        for name, idx, y in (
            ("Train", train_idx, train_y),
            ("Validation", val_idx, val_y),
            ("Final Test", final_idx, final_y),
        ):
            if len(idx) == 0 or len(np.unique(y)) != 2:
                counts = np.bincount(y, minlength=2).tolist() if len(y) else [0, 0]
                raise ValueError(
                    f"{g}/{owner}: {name} must contain both classes. "
                    f"[impostor, genuine]={counts}"
                )

        # No sample may leak across Train / Validation / Final.
        if np.intersect1d(train_idx, val_idx).size:
            raise AssertionError(f"{g}: Train/Validation overlap")
        if np.intersect1d(train_idx, final_idx).size:
            raise AssertionError(f"{g}: Train/Final overlap")
        if np.intersect1d(val_idx, final_idx).size:
            raise AssertionError(f"{g}: Validation/Final overlap")

        result[g] = GestureSplit(
            gesture=g,
            owner=owner,
            train_idx=train_idx,
            train_y=train_y,
            val_idx=val_idx,
            val_y=val_y,
            final_idx=final_idx,
            final_y=final_y,
            train_owner_sessions=train_sessions,
            val_owner_sessions=val_sessions,
            final_owner_sessions=final_sessions,
        )

    return result


def eer_from_scores(y_true, scores):
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    if len(np.unique(y_true)) != 2:
        raise ValueError("EER requires both classes")

    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1.0 - tpr

    i = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[i] + fnr[i]) / 2.0)
    threshold = float(thresholds[i])

    return eer, threshold


def validation_threshold(y_val, score_val):
    """
    Select the authentication threshold using Validation only.
    Returns: val_eer, threshold
    """
    return eer_from_scores(y_val, score_val)


def evaluate_final(y_true, scores, threshold):
    """
    Final Test metrics.

    AUC / EER:
      diagnostic score-separation metrics on Final Test

    FAR / FRR / Accuracy / Balanced Accuracy:
      computed with the threshold fixed on Validation
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    pred = (scores >= threshold).astype(np.int64)

    pos = y_true == 1
    neg = y_true == 0

    if not np.any(pos) or not np.any(neg):
        raise ValueError("Final Test requires both classes")

    tp = int(np.sum((pred == 1) & pos))
    tn = int(np.sum((pred == 0) & neg))
    fp = int(np.sum((pred == 1) & neg))
    fn = int(np.sum((pred == 0) & pos))

    far = fp / max(int(np.sum(neg)), 1)
    frr = fn / max(int(np.sum(pos)), 1)

    accuracy = (tp + tn) / len(y_true)
    balanced_accuracy = ((1.0 - far) + (1.0 - frr)) / 2.0

    auc = float(roc_auc_score(y_true, scores))
    eer, _ = eer_from_scores(y_true, scores)

    return {
        "auc": auc,
        "eer": float(eer),
        "far": float(far),
        "frr": float(frr),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
    }


def print_split_summary(splits):
    print("=" * 110)
    print("WITECH Single Final Test Protocol")
    print("=" * 110)
    print(
        f"{'Gesture':<9}{'Owner':<8}"
        f"{'Train+':>9}{'Train-':>9}"
        f"{'Val+':>8}{'Val-':>8}"
        f"{'Final+':>9}{'Final-':>9}  Final owner sessions"
    )
    print("-" * 110)

    for g in sorted(splits):
        s = splits[g]
        print(
            f"{g:<9}{s.owner:<8}"
            f"{int(s.train_y.sum()):>9}{int((s.train_y == 0).sum()):>9}"
            f"{int(s.val_y.sum()):>8}{int((s.val_y == 0).sum()):>8}"
            f"{int(s.final_y.sum()):>9}{int((s.final_y == 0).sum()):>9}  "
            f"{', '.join(s.final_owner_sessions)}"
        )

    print("-" * 110)
    print("Validation impostors:", ", ".join(VAL_IMPOSTORS))
    print("Final impostors     :", ", ".join(FINAL_IMPOSTORS))
    print("Final genuine samples are included exactly once.")


if __name__ == "__main__":
    import argparse
    import paths

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(paths.DATASET_NPZ))
    args = ap.parse_args()

    data_path = str(paths.assert_external(args.data, "dataset.npz"))
    z = np.load(data_path, allow_pickle=True)

    meta = {
        "gesture": z["gesture"].astype(str),
        "performer": z["performer"].astype(str),
        "session": z["session"].astype(str),
        "role": z["role"].astype(str),
    }

    print(f"dataset: {data_path}\n")
    print_split_summary(build_splits(meta))