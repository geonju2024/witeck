"""
runlog.py

학습 스크립트의 콘솔 출력을 화면에는 그대로 보여주면서, 동시에 타임스탬프가
붙은 로그 파일로도 남긴다. "이 결과가 언제 나온 거고, 어떤 데이터로 돌린
거지?" 를 나중에 파일만 보고 답할 수 있게 하는 것이 목적이다.

남기는 것
  runs/<script>_<YYYYMMDD-HHMMSS>.log   실행 1건의 전체 출력 + 머리말/꼬리말
  runs/index.csv                        실행 1건 = 1행 요약 (언제/무엇을/어떤 데이터로)

머리말에는 실행 시작 시각, 명령줄, 파이썬/OS, git 커밋, 그리고 입력 데이터
파일의 크기·수정시각·sha256 앞 16자리가 들어간다. 데이터 지문(sha256)이 같으면
서로 다른 두 실행이 정말 같은 데이터셋을 썼다고 단정할 수 있다.

사용법:
    from runlog import start_run_log

    start_run_log(
        "04_train_1dcnn",
        data_files=[args.data],
        extra={"task": args.task, "epochs": args.epochs},
    )
"""

from __future__ import annotations

import atexit
import csv
import hashlib
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

INDEX_COLUMNS = [
    "started",
    "ended",
    "elapsed_sec",
    "script",
    "status",
    "log",
    "data",
    "args",
]


class _Tee:
    """콘솔과 로그 파일에 동시에 쓰는 얇은 래퍼."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, s):
        self._stream.write(s)
        self._fh.write(s)
        # 도중에 죽어도 로그가 남도록 즉시 flush 한다. 출력량이 적어 부담 없다.
        self._fh.flush()
        return len(s)

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def isatty(self):
        return self._stream.isatty()

    def fileno(self):
        return self._stream.fileno()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")


def file_fingerprint(path) -> str:
    """데이터 파일의 크기 / 수정시각 / 내용 해시를 한 줄로 요약한다."""
    p = Path(path)
    if not p.is_file():
        return f"{path}  (파일 없음)"

    st = p.stat()
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"{p.name}  size={st.st_size:,}B  mtime={mtime}  "
        f"sha256={h.hexdigest()[:16]}"
    )


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return "(git 정보 없음)"
        commit = out.stdout.strip()

        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            commit += " (uncommitted 변경 있음)"
        return commit
    except Exception:
        return "(git 정보 없음)"


def _append_index(index_path: Path, row: dict) -> None:
    new_file = not index_path.exists()
    with index_path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def start_run_log(
    script: str,
    out_dir: str = "runs",
    data_files=(),
    extra: dict | None = None,
    enabled: bool = True,
) -> Path | None:
    """
    지금부터의 stdout/stderr 를 타임스탬프 로그 파일로도 복사한다.

    script      로그 파일 이름에 쓰일 스크립트 이름 (예: "04_train_1dcnn")
    out_dir     로그 폴더. 상대경로면 프로젝트 폴더 기준으로 잡는다.
    data_files  지문을 남길 입력 파일 목록 (예: [args.data])
    extra       머리말에 함께 적을 하이퍼파라미터 등
    enabled     False 면 아무것도 하지 않고 None 을 반환한다 (--no-log 용)

    반환값: 로그 파일 경로 (enabled=False 면 None)
    """
    if not enabled:
        return None

    # 윈도우 cp949 콘솔에서 한글이 깨지는 것을 막는다. 이미 utf-8 이면 무해하다.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    started = datetime.now()
    started_mono = time.time()

    log_dir = Path(out_dir)
    if not log_dir.is_absolute():
        log_dir = PROJECT_ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{script}_{started:%Y%m%d-%H%M%S}.log"
    # utf-8-sig(BOM): 메모장이나 PowerShell 5.1 의 Get-Content 로 열어도 한글이 깨지지 않는다.
    fh = log_path.open("w", encoding="utf-8-sig", newline="")

    fingerprints = [file_fingerprint(p) for p in data_files]

    header = [
        "=" * 72,
        f"실행 시작 : {started:%Y-%m-%d %H:%M:%S} ({time.tzname[0]})",
        f"스크립트  : {script}",
        f"명령줄    : {' '.join(sys.argv)}",
        f"작업폴더  : {os.getcwd()}",
        f"파이썬    : {sys.executable} ({platform.python_version()})",
        f"OS        : {platform.platform()}",
        f"git       : {_git_commit()}",
    ]
    for fp in fingerprints:
        header.append(f"데이터    : {fp}")
    if extra:
        header.append(f"설정      : {extra}")
    header.append("=" * 72)
    header.append("")

    fh.write("\n".join(header))
    fh.flush()

    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_stdout, fh)
    sys.stderr = _Tee(orig_stderr, fh)

    state = {"status": "ok"}

    orig_excepthook = sys.excepthook

    def _excepthook(exc_type, exc, tb):
        state["status"] = f"error: {exc_type.__name__}"
        orig_excepthook(exc_type, exc, tb)

    sys.excepthook = _excepthook

    def _finish():
        ended = datetime.now()
        elapsed = time.time() - started_mono
        try:
            fh.write(
                f"\n{'=' * 72}\n"
                f"실행 종료 : {ended:%Y-%m-%d %H:%M:%S}  "
                f"(소요 {elapsed:.1f}초, 결과 {state['status']})\n"
                f"{'=' * 72}\n"
            )
            fh.flush()
        finally:
            sys.stdout, sys.stderr = orig_stdout, orig_stderr
            sys.excepthook = orig_excepthook
            fh.close()

        try:
            _append_index(
                log_dir / "index.csv",
                {
                    "started": f"{started:%Y-%m-%d %H:%M:%S}",
                    "ended": f"{ended:%Y-%m-%d %H:%M:%S}",
                    "elapsed_sec": f"{elapsed:.1f}",
                    "script": script,
                    "status": state["status"],
                    "log": log_path.name,
                    "data": " | ".join(fingerprints),
                    "args": " ".join(sys.argv[1:]),
                },
            )
        except Exception as e:  # 로그 실패가 학습 결과를 덮지 않도록 삼킨다
            print(f"(경고: runs/index.csv 기록 실패: {e})", file=orig_stderr)

        print(f"\n로그 저장: {log_path}", file=orig_stdout)

    atexit.register(_finish)
    return log_path


def add_log_arguments(ap, default_dir="runs") -> None:
    """argparse 파서에 --log-dir / --no-log 를 붙인다."""
    ap.add_argument("--log-dir", default=str(default_dir),
                    help=f"실행 로그를 남길 폴더 (기본값: {default_dir})")
    ap.add_argument("--no-log", action="store_true", help="실행 로그를 남기지 않는다")
