"""
00_make_labels_template.py
영상 폴더를 훑어 labels.csv 템플릿을 만든다.

현재 파일명이 20260723_145335.mp4 처럼 타임스탬프뿐이라 라벨 정보가 없다.
가장 확실한 방법은 촬영 순서/시간대와 대조해 이 CSV를 한 번 채우는 것이다.
폴더를 videos/<제스처>/<수행자>/*.mp4 구조로 정리해두면 자동으로 채워진다.

사용법:
    python 00_make_labels_template.py --videos ./videos --out labels.csv
"""
import argparse
import csv
import glob
import os
import datetime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", default="labels.csv")
    a = ap.parse_args()

    exts = ("mp4", "MP4", "mov", "MOV", "avi", "AVI", "mkv")
    files = sorted({f for e in exts
                    for f in glob.glob(os.path.join(a.videos, "**", f"*.{e}"), recursive=True)})

    with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "gesture", "performer", "mtime", "relpath"])
        for p in files:
            rel = os.path.relpath(p, a.videos)
            parts = rel.replace("\\", "/").split("/")
            # videos/<gesture>/<performer>/xxx.mp4 구조면 자동 추론
            gesture = parts[0] if len(parts) >= 3 else ""
            performer = parts[1] if len(parts) >= 3 else ""
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M:%S")
            w.writerow([os.path.basename(p), gesture, performer, mtime, rel])

    print(f"{len(files)}행 -> {a.out}")
    print("gesture / performer 열을 채운 뒤 02 단계로 넘어가세요.")
    print("mtime(파일 수정시각) 열이 촬영 순서 복원에 도움이 됩니다.")


if __name__ == "__main__":
    main()
