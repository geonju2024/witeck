"""
paths.py

파이프라인이 쓰는 모든 데이터 경로를 한곳에서 정한다.

원칙
  1. 원본 영상도, 생성 데이터도 전부 Google Drive 공유 드라이브(WITECH) 아래에 둔다.
  2. 코드 폴더(C:\\dev\\gesture)에는 파이썬 코드만 남긴다. 이 폴더 안의
     dataset.npz / labels.csv 는 2026-08-08 이전 실험 데이터라 더 이상 쓰지 않는다.
     실수로 지정하면 assert_external() 이 즉시 중단시킨다.

폴더 구조
    WITECH/
      videos/                 원본 영상 (gesture/performer/session/파일)  <- 사람이 관리
      derived/                파이프라인 생성물                          <- 스크립트가 관리
        labels.csv
        landmarks/
        dataset.npz
        runs/

경로를 바꾸려면 환경변수 WITECH_ROOT 를 지정한다.
    set WITECH_ROOT=D:\\WITECH
"""

from __future__ import annotations

import os
from pathlib import Path

# 코드만 두는 폴더. 여기 있는 데이터 파일은 파이프라인이 절대 쓰지 않는다.
CODE_DIR = Path(__file__).resolve().parent

# Google Drive Desktop 이 공유 드라이브를 마운트하는 자리
_ROOT_CANDIDATES = (
    r"G:\공유 드라이브\WITECH",
    r"G:\Shared drives\WITECH",
)

# 공유 드라이브가 위 자리에 안 보이고 '내 드라이브'의 바로가기로만 노출되는 경우,
# 실제 폴더는 이 아래 무작위 ID 폴더 안에 잡힌다.
_SHORTCUT_ROOTS = (
    Path(r"G:\.shortcut-targets-by-id"),
)


def _find_witech_root() -> Path:
    env = os.environ.get("WITECH_ROOT")
    if env:
        p = Path(env)
        if not p.is_dir():
            raise FileNotFoundError(
                f"환경변수 WITECH_ROOT 가 가리키는 폴더가 없습니다: {p}"
            )
        return p.resolve()

    for candidate in _ROOT_CANDIDATES:
        p = Path(candidate)
        if p.is_dir():
            return p.resolve()

    for shortcut_root in _SHORTCUT_ROOTS:
        if shortcut_root.is_dir():
            for hit in sorted(shortcut_root.glob("*/WITECH")):
                if hit.is_dir():
                    return hit.resolve()

    raise FileNotFoundError(
        "WITECH 공유 드라이브 폴더를 찾지 못했습니다.\n"
        "  - Google Drive Desktop 이 실행 중이고 G: 드라이브가 보이는지 확인하세요.\n"
        "  - 다른 위치에 있다면 환경변수로 지정하세요:  set WITECH_ROOT=<폴더>\n"
        f"  - 찾아본 곳: {', '.join(_ROOT_CANDIDATES)}, "
        f"{_SHORTCUT_ROOTS[0]}\\*\\WITECH"
    )


WITECH_ROOT = _find_witech_root()

# 원본 (사람이 관리, 스크립트는 읽기만 한다)
VIDEOS_DIR = WITECH_ROOT / "videos"

# 생성물 (스크립트가 만든다)
DERIVED_DIR = WITECH_ROOT / "derived"
LABELS_CSV = DERIVED_DIR / "labels.csv"
LANDMARKS_DIR = DERIVED_DIR / "landmarks"
DATASET_NPZ = DERIVED_DIR / "dataset.npz"
RUNS_DIR = DERIVED_DIR / "runs"


def assert_external(path, what: str = "데이터") -> Path:
    """
    코드 폴더 안의 옛 데이터를 실수로 쓰는 것을 막는다.

    8/8 실험의 dataset.npz / labels.csv 가 C:\\dev\\gesture 에 남아 있어서,
    경로를 빠뜨리면 조용히 옛 데이터로 학습되고 결과만 그럴듯하게 나온다.
    조용한 오염보다 즉시 중단이 낫다.
    """
    p = Path(path).expanduser().resolve()

    if p == CODE_DIR or CODE_DIR in p.parents:
        raise SystemExit(
            f"[중단] {what} 경로가 코드 폴더 안을 가리킵니다:\n"
            f"    {p}\n"
            f"  {CODE_DIR} 에는 코드만 둡니다. 이 폴더에 남아 있는 "
            f"dataset.npz / labels.csv 는\n"
            f"  2026-08-08 이전 실험 데이터라 사용하지 않습니다.\n"
            f"  공유 드라이브 경로를 쓰세요 (기본값: {DERIVED_DIR})"
        )

    return p


def describe() -> str:
    """현재 잡힌 경로를 사람이 읽을 수 있게 한 줄씩 정리한다."""
    lines = [
        f"WITECH   : {WITECH_ROOT}",
        f"videos   : {VIDEOS_DIR}",
        f"derived  : {DERIVED_DIR}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    print(describe())
    print()
    for label, p in (
        ("videos", VIDEOS_DIR),
        ("labels.csv", LABELS_CSV),
        ("landmarks", LANDMARKS_DIR),
        ("dataset.npz", DATASET_NPZ),
    ):
        print(f"{label:<12} {'있음' if p.exists() else '없음'}  {p}")
