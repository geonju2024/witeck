from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import paths


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

# role 토큰: 등록자 본인은 own, 타인은 impA/impB/... (촬영 회차별로 늘어날 수 있음)
OWNER_ROLE = "own"
ROLE_RE = re.compile(r"^(own|imp[A-Z])$")


def parse_filename(video_path: Path) -> dict[str, str]:
    """
    Expected filename format:

    gesture_performer_session_role_repetition.mp4

    Example:
    G1_P01_20260725_own_001.mp4
    """

    parts = video_path.stem.split("_")

    if len(parts) != 5:
        raise ValueError(
            "Invalid filename format. "
            "Expected: gesture_performer_session_role_repetition.mp4"
        )

    gesture, performer, session, role, repetition = parts

    if not session.isdigit() or len(session) != 8:
        raise ValueError(
            f"Invalid session date: {session}. "
            "Expected YYYYMMDD format."
        )

    if not ROLE_RE.match(role):
        raise ValueError(
            f"Invalid role: {role}. "
            "Use own or impA/impB/..."
        )

    if not repetition.isdigit():
        raise ValueError(
            f"Invalid repetition number: {repetition}"
        )

    return {
        "gesture": gesture,
        "performer": performer,
        "session": session,
        "role": role,
        "repetition": repetition,
    }


def check_folder_agreement(video_path: Path, videos_dir: Path,
                           parsed: dict[str, str]) -> None:
    """
    폴더 구조(gesture/performer/session/파일)와 파일명이 어긋나면 실패시킨다.

    세션은 학습/평가 분할의 기준이라, 둘이 어긋난 채로 통과시키면
    잘못된 세션 라벨이 그대로 평가 결과에 반영된다. 조용히 넘어가면 안 된다.
    """
    parts = video_path.relative_to(videos_dir).parts

    if len(parts) != 4:
        raise ValueError(
            f"Expected videos/<gesture>/<performer>/<session>/<file>, got: "
            f"{'/'.join(parts)}"
        )

    gesture_dir, performer_dir, session_dir, _ = parts
    mismatches = [
        f"{label} folder={folder} filename={name}"
        for label, folder, name in (
            ("gesture", gesture_dir, parsed["gesture"]),
            ("performer", performer_dir, parsed["performer"]),
            ("session", session_dir, parsed["session"]),
        )
        if folder != name
    ]

    if mismatches:
        raise ValueError("Folder/filename mismatch: " + "; ".join(mismatches))


def find_video_files(videos_dir: Path) -> list[Path]:
    videos = [
        path
        for path in videos_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]

    return sorted(videos)


def make_labels(
    videos_dir: Path,
    output_path: Path,
    default_device: str,
    default_location: str,
    default_lighting: str,
) -> None:
    video_files = find_video_files(videos_dir)

    if not video_files:
        raise FileNotFoundError(
            f"No video files found in: {videos_dir}"
        )

    rows: list[dict[str, str]] = []
    errors: list[str] = []

    for video_path in video_files:
        try:
            parsed = parse_filename(video_path)
            check_folder_agreement(video_path, videos_dir, parsed)

            relative_path = video_path.relative_to(videos_dir).as_posix()

            rows.append(
                {
                    "video": video_path.name,
                    "relative_path": relative_path,
                    "gesture": parsed["gesture"],
                    "performer": parsed["performer"],
                    "session": parsed["session"],
                    "role": parsed["role"],
                    "is_owner": "1" if parsed["role"] == OWNER_ROLE else "0",
                    "repetition": parsed["repetition"],
                    "device": default_device,
                    "location": default_location,
                    "lighting": default_lighting,
                }
            )

        except ValueError as error:
            errors.append(f"{video_path}: {error}")

    fieldnames = [
        "video",
        "relative_path",
        "gesture",
        "performer",
        "session",
        "role",
        "is_owner",
        "repetition",
        "device",
        "location",
        "lighting",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Videos found: {len(video_files)}")
    print(f"Rows written: {len(rows)}")
    print(f"Invalid filenames: {len(errors)}")
    print(f"Output: {output_path}")

    if errors:
        print("\nInvalid filename list:")
        for error in errors:
            print(f"- {error}")

    summarize(rows)


def summarize(rows: list[dict[str, str]]) -> None:
    """등록자 매핑과 세션 구성을 요약한다. 평가 설계에 바로 영향을 주는 값들이다."""
    if not rows:
        return

    owners: dict[str, set[str]] = {}
    for row in rows:
        if row["role"] == OWNER_ROLE:
            owners.setdefault(row["gesture"], set()).add(row["performer"])

    print("\nOwner per gesture:")
    for gesture in sorted(owners):
        who = sorted(owners[gesture])
        flag = "" if len(who) == 1 else "   <-- 등록자가 1명이 아닙니다"
        print(f"- {gesture}: {', '.join(who)}{flag}")

    print("\nSessions per performer (own-gesture samples):")
    for performer in sorted({r["performer"] for r in rows}):
        own = [r for r in rows if r["performer"] == performer and r["role"] == OWNER_ROLE]
        sessions = sorted({r["session"] for r in own})
        flag = "   <-- 세션분리 평가 불가" if len(sessions) < 2 else ""
        print(f"- {performer}: {len(sessions)} sessions, {len(own)} own samples{flag}")


def main() -> None:
    # 윈도우 콘솔 기본 코드페이지(cp949)에서 한글 경고문이 깨지는 것을 막는다
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Create labels.csv from video filenames."
    )

    parser.add_argument(
        "--videos",
        type=Path,
        default=paths.VIDEOS_DIR,
        help=f"Video root directory. (기본값: {paths.VIDEOS_DIR})",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=paths.LABELS_CSV,
        help=f"Output CSV path. (기본값: {paths.LABELS_CSV})",
    )

    parser.add_argument(
        "--device",
        default="unknown",
        help="Default recording device.",
    )

    parser.add_argument(
        "--location",
        default="unknown",
        help="Default recording location.",
    )

    parser.add_argument(
        "--lighting",
        default="unknown",
        help="Default lighting condition.",
    )

    args = parser.parse_args()

    videos_dir = paths.assert_external(args.videos, "영상")
    output_path = paths.assert_external(args.out, "labels.csv")
    print(f"영상   : {videos_dir}")
    print(f"출력   : {output_path}\n")

    make_labels(
        videos_dir=videos_dir,
        output_path=output_path,
        default_device=args.device,
        default_location=args.location,
        default_lighting=args.lighting,
    )


if __name__ == "__main__":
    main()