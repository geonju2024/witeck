"""
split_protocol.py

WITECH 제스처 인증 데이터의 고정 Train / Validation / Test split을 정의한다.

목적
----
모델마다 train/test를 다르게 나누면 성능 비교가 의미가 없어진다.

따라서 아래 모델들이 모두 이 파일의 동일한 split을 사용한다.

    03_train_baselines.py
        LogReg / SVM-RBF / RF / GRU / LSTM

    04_train_1dcnn.py

    05_train_transformer.py


Verification split
------------------

1) 등록자 own 데이터
   촬영 session을 날짜순으로 정렬한다.

   마지막 2 session      -> Test
   그 직전 2 session     -> Validation
   나머지 이전 session   -> Train

   현재 12 session이면 자동으로

       Train      8 session
       Validation 2 session
       Test       2 session

   이 된다.


2) Impostor performer

   Train impostor
       P01 ~ P05 중 해당 gesture owner 제외

   Validation impostor
       P06, P07

   Internal Test impostor
       P08, P09, P10

   External Test impostor
       X01 ~ X18


예: G1 owner=P01

    Train
        own       : P01의 초기 촬영 session
        impostor  : P02, P03, P04, P05

    Validation
        own       : P01의 다음 2 session
        impostor  : P06, P07

    Internal Test
        own       : P01의 마지막 2 session
        impostor  : P08, P09, P10

    External Test
        own       : Internal Test와 동일한 마지막 2 session
        impostor  : X01 ~ X18


주의
----
External Test는 공격자 일반화 성능을 보기 위한 별도 평가이다.
genuine 쪽은 Internal Test와 동일하고 impostor 집단만 X01~X18로 바뀐다.

사용 예시
---------
from split_protocol import build_verification_split

split = build_verification_split(meta, "G1")

train_idx = split["train_idx"]
val_idx = split["val_idx"]
test_idx = split["test_idx"]
external_idx = split["external_test_idx"]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


# =========================================================
# Performer groups
# =========================================================

TRAIN_POOL = {
    "P01",
    "P02",
    "P03",
    "P04",
    "P05",
}

VAL_IMPOSTORS = {
    "P06",
    "P07",
}

TEST_IMPOSTORS = {
    "P08",
    "P09",
    "P10",
}

EXTERNAL_IMPOSTORS = {
    f"X{i:02d}"
    for i in range(1, 19)
}


# =========================================================
# Data structure
# =========================================================

@dataclass(frozen=True)
class VerificationSplit:
    gesture: str
    owner: str

    train_sessions: tuple[str, ...]
    val_sessions: tuple[str, ...]
    test_sessions: tuple[str, ...]

    train_impostors: tuple[str, ...]
    val_impostors: tuple[str, ...]
    test_impostors: tuple[str, ...]
    external_impostors: tuple[str, ...]

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    external_test_idx: np.ndarray

    def as_dict(self) -> dict:
        return {
            "gesture": self.gesture,
            "owner": self.owner,

            "train_sessions": self.train_sessions,
            "val_sessions": self.val_sessions,
            "test_sessions": self.test_sessions,

            "train_impostors": self.train_impostors,
            "val_impostors": self.val_impostors,
            "test_impostors": self.test_impostors,
            "external_impostors": self.external_impostors,

            "train_idx": self.train_idx,
            "val_idx": self.val_idx,
            "test_idx": self.test_idx,
            "external_test_idx": self.external_test_idx,
        }


# =========================================================
# Helpers
# =========================================================

def _to_str_array(x) -> np.ndarray:
    return np.asarray(x).astype(str)


def _extract_owner(
    gesture: np.ndarray,
    performer: np.ndarray,
    role: np.ndarray,
    target_gesture: str,
) -> str:
    """
    role == own인 performer를 해당 gesture의 등록자로 사용한다.
    """
    mask = (
        (gesture == target_gesture)
        & (role == "own")
    )

    owners = sorted(
        set(
            performer[mask]
        )
    )

    if len(owners) != 1:
        raise ValueError(
            f"{target_gesture} 등록자가 1명이 아닙니다: {owners}"
        )

    return owners[0]


def _split_owner_sessions(
    sessions: Iterable[str],
    n_val_sessions: int = 2,
    n_test_sessions: int = 2,
):
    """
    날짜 문자열 YYYYMMDD는 문자열 정렬 == 시간순 정렬이다.

    마지막 n_test_sessions개 -> test
    그 직전 n_val_sessions개 -> validation
    나머지 -> train
    """
    sessions = tuple(
        sorted(
            set(
                str(s)
                for s in sessions
            )
        )
    )

    need = (
        1
        + n_val_sessions
        + n_test_sessions
    )

    if len(sessions) < need:
        raise ValueError(
            "등록자 session이 부족합니다. "
            f"최소 {need}개 필요, 현재 {len(sessions)}개: {sessions}"
        )

    test_sessions = sessions[
        -n_test_sessions:
    ]

    val_start = -(
        n_test_sessions
        + n_val_sessions
    )

    val_sessions = sessions[
        val_start:
        -n_test_sessions
    ]

    train_sessions = sessions[
        :val_start
    ]

    if not train_sessions:
        raise ValueError(
            "Train용 등록자 session이 없습니다."
        )

    return (
        tuple(train_sessions),
        tuple(val_sessions),
        tuple(test_sessions),
    )


def _indices_for_split(
    gesture: np.ndarray,
    performer: np.ndarray,
    session: np.ndarray,
    target_gesture: str,
    owner: str,
    owner_sessions: tuple[str, ...],
    impostor_performers: set[str],
) -> np.ndarray:
    """
    하나의 split index 생성.

    positive:
        target gesture + owner + 지정 owner session

    negative:
        target gesture + 지정 impostor performer 전체
    """

    target = (
        gesture == target_gesture
    )

    genuine = (
        target
        & (performer == owner)
        & np.isin(
            session,
            owner_sessions,
        )
    )

    impostor = (
        target
        & np.isin(
            performer,
            list(
                impostor_performers
            ),
        )
    )

    return np.where(
        genuine
        | impostor
    )[0]


def _check_disjoint(
    name_a: str,
    a: np.ndarray,
    name_b: str,
    b: np.ndarray,
):
    overlap = np.intersect1d(
        a,
        b,
    )

    if len(overlap) > 0:
        raise RuntimeError(
            f"{name_a}와 {name_b}에 "
            f"{len(overlap)}개 샘플이 중복됩니다."
        )


def _count_classes(
    idx: np.ndarray,
    performer: np.ndarray,
    owner: str,
):
    y = (
        performer[idx]
        == owner
    ).astype(
        np.int64
    )

    return {
        "total": int(
            len(idx)
        ),
        "own": int(
            y.sum()
        ),
        "impostor": int(
            (y == 0).sum()
        ),
    }


# =========================================================
# Main protocol
# =========================================================

def build_verification_split(
    meta: dict,
    target_gesture: str,
    n_val_sessions: int = 2,
    n_test_sessions: int = 2,
    strict: bool = True,
) -> dict:
    """
    target_gesture 하나에 대한 고정 split 생성.

    Parameters
    ----------
    meta:
        dataset.npz에서 읽은 metadata dict.

        필수:
            gesture
            performer
            session
            role

    target_gesture:
        예: "G1"

    strict:
        True이면 예상하지 못한 performer가 있거나
        필수 split에 샘플이 없을 때 오류를 낸다.

    Returns
    -------
    dict
        train_idx
        val_idx
        test_idx
        external_test_idx
        owner
        session / impostor group 정보
    """

    required = {
        "gesture",
        "performer",
        "session",
        "role",
    }

    missing = (
        required
        - set(
            meta.keys()
        )
    )

    if missing:
        raise ValueError(
            f"meta에 필요한 항목이 없습니다: {sorted(missing)}"
        )

    gesture = _to_str_array(
        meta["gesture"]
    )

    performer = _to_str_array(
        meta["performer"]
    )

    session = _to_str_array(
        meta["session"]
    )

    role = _to_str_array(
        meta["role"]
    )

    if not (
        len(gesture)
        == len(performer)
        == len(session)
        == len(role)
    ):
        raise ValueError(
            "metadata 길이가 서로 다릅니다."
        )

    if target_gesture not in set(
        gesture
    ):
        raise ValueError(
            f"dataset에 {target_gesture}가 없습니다."
        )

    owner = _extract_owner(
        gesture,
        performer,
        role,
        target_gesture,
    )

    # -----------------------------------------------------
    # Owner session split
    # -----------------------------------------------------

    owner_mask = (
        (gesture == target_gesture)
        & (performer == owner)
        & (role == "own")
    )

    owner_sessions = session[
        owner_mask
    ]

    (
        train_sessions,
        val_sessions,
        test_sessions,
    ) = _split_owner_sessions(
        owner_sessions,
        n_val_sessions=n_val_sessions,
        n_test_sessions=n_test_sessions,
    )

    # -----------------------------------------------------
    # Impostor performer split
    # -----------------------------------------------------

    train_impostors = (
        TRAIN_POOL
        - {owner}
    )

    val_impostors = set(
        VAL_IMPOSTORS
    )

    test_impostors = set(
        TEST_IMPOSTORS
    )

    external_impostors = set(
        EXTERNAL_IMPOSTORS
    )

    known_groups = (
        TRAIN_POOL
        | VAL_IMPOSTORS
        | TEST_IMPOSTORS
        | EXTERNAL_IMPOSTORS
    )

    actual_target_performers = set(
        performer[
            gesture
            == target_gesture
        ]
    )

    unknown = (
        actual_target_performers
        - known_groups
    )

    if strict and unknown:
        raise ValueError(
            f"{target_gesture}에 split 규칙에 없는 performer가 있습니다: "
            f"{sorted(unknown)}"
        )

    # -----------------------------------------------------
    # Indices
    # -----------------------------------------------------

    train_idx = _indices_for_split(
        gesture,
        performer,
        session,
        target_gesture,
        owner,
        train_sessions,
        train_impostors,
    )

    val_idx = _indices_for_split(
        gesture,
        performer,
        session,
        target_gesture,
        owner,
        val_sessions,
        val_impostors,
    )

    test_idx = _indices_for_split(
        gesture,
        performer,
        session,
        target_gesture,
        owner,
        test_sessions,
        test_impostors,
    )

    external_test_idx = _indices_for_split(
        gesture,
        performer,
        session,
        target_gesture,
        owner,
        test_sessions,
        external_impostors,
    )

    # -----------------------------------------------------
    # Leakage checks
    # -----------------------------------------------------

    _check_disjoint(
        "train",
        train_idx,
        "validation",
        val_idx,
    )

    _check_disjoint(
        "train",
        train_idx,
        "test",
        test_idx,
    )

    _check_disjoint(
        "train",
        train_idx,
        "external_test",
        external_test_idx,
    )

    _check_disjoint(
        "validation",
        val_idx,
        "test",
        test_idx,
    )

    # test / external_test는 genuine test samples를 의도적으로 공유한다.
    # 따라서 두 집합 자체는 disjoint가 아니다.

    # -----------------------------------------------------
    # Basic validity
    # -----------------------------------------------------

    split_indices = {
        "train": train_idx,
        "validation": val_idx,
        "test": test_idx,
        "external_test": external_test_idx,
    }

    if strict:
        for split_name, idx in split_indices.items():
            counts = _count_classes(
                idx,
                performer,
                owner,
            )

            if counts["own"] == 0:
                raise ValueError(
                    f"{target_gesture} {split_name}: own 샘플이 없습니다."
                )

            if counts["impostor"] == 0:
                raise ValueError(
                    f"{target_gesture} {split_name}: impostor 샘플이 없습니다."
                )

    split = VerificationSplit(
        gesture=target_gesture,
        owner=owner,

        train_sessions=tuple(
            train_sessions
        ),
        val_sessions=tuple(
            val_sessions
        ),
        test_sessions=tuple(
            test_sessions
        ),

        train_impostors=tuple(
            sorted(
                train_impostors
            )
        ),
        val_impostors=tuple(
            sorted(
                val_impostors
            )
        ),
        test_impostors=tuple(
            sorted(
                test_impostors
            )
        ),
        external_impostors=tuple(
            sorted(
                external_impostors
            )
        ),

        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        external_test_idx=external_test_idx,
    )

    return split.as_dict()


# =========================================================
# Inspection helper
# =========================================================

def describe_verification_split(
    meta: dict,
    split: dict,
) -> str:
    """
    사람이 바로 확인할 수 있는 split 요약 문자열.
    """

    performer = _to_str_array(
        meta["performer"]
    )

    owner = split[
        "owner"
    ]

    lines = [
        "=" * 72,
        f"Gesture: {split['gesture']}",
        f"Owner  : {owner}",
        "",
        "Owner sessions",
        f"  Train      : {list(split['train_sessions'])}",
        f"  Validation : {list(split['val_sessions'])}",
        f"  Test       : {list(split['test_sessions'])}",
        "",
        "Impostor performers",
        f"  Train      : {list(split['train_impostors'])}",
        f"  Validation : {list(split['val_impostors'])}",
        f"  Test       : {list(split['test_impostors'])}",
        f"  External   : {list(split['external_impostors'])}",
        "",
        "Sample counts",
    ]

    for label, key in (
        (
            "Train",
            "train_idx",
        ),
        (
            "Validation",
            "val_idx",
        ),
        (
            "Internal Test",
            "test_idx",
        ),
        (
            "External Test",
            "external_test_idx",
        ),
    ):
        counts = _count_classes(
            split[key],
            performer,
            owner,
        )

        lines.append(
            f"  {label:<14} "
            f"total={counts['total']:4d}  "
            f"own={counts['own']:3d}  "
            f"impostor={counts['impostor']:3d}"
        )

    return "\n".join(
        lines
    )


# =========================================================
# Optional standalone test
# =========================================================

def main():
    """
    dataset.npz를 직접 읽어서
    G1~G5 split을 화면에 출력한다.

    사용:
        python split_protocol.py
        python split_protocol.py /path/to/dataset.npz
    """
    import sys

    import paths

    try:
        sys.stdout.reconfigure(
            encoding="utf-8"
        )
    except Exception:
        pass

    dataset_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else str(
            paths.DATASET_NPZ
        )
    )

    z = np.load(
        dataset_path,
        allow_pickle=True,
    )

    meta = {
        "gesture": z[
            "gesture"
        ].astype(str),

        "performer": z[
            "performer"
        ].astype(str),

        "session": z[
            "session"
        ].astype(str),

        "role": z[
            "role"
        ].astype(str),
    }

    gestures = sorted(
        set(
            meta["gesture"]
        )
    )

    print(
        f"dataset: {dataset_path}"
    )

    print(
        f"gestures: {gestures}\n"
    )

    for gesture in gestures:
        split = build_verification_split(
            meta,
            gesture,
        )

        print(
            describe_verification_split(
                meta,
                split,
            )
        )

        print()


if __name__ == "__main__":
    main()