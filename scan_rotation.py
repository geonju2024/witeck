"""동영상 회전 메타데이터 스캔.

ffprobe 없이 MP4/MOV 컨테이너의 박스를 직접 파싱해서
해상도 / fps / 프레임수 / 회전각을 meta/rotation_scan.csv 로 저장한다.

회전각은 trak > tkhd 의 3x3 display matrix 에서 계산한다.
(ffprobe 의 stream_side_data rotation 과 같은 값)

사용법:
    python scan_rotation.py [videos]
"""

import csv
import math
import struct
import sys
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".3gp"}


def iter_boxes(f, end):
    """[start, end) 구간의 박스를 (type, payload_start, box_end) 으로 순회."""
    while f.tell() < end - 8:
        start = f.tell()
        hdr = f.read(8)
        if len(hdr) < 8:
            return
        size = int.from_bytes(hdr[:4], "big")
        typ = hdr[4:8]
        hsize = 8
        if size == 1:
            size = int.from_bytes(f.read(8), "big")
            hsize = 16
        elif size == 0:
            size = end - start
        if size < hsize or start + size > end:
            return
        yield typ, start + hsize, start + size
        f.seek(start + size)


def find_box(f, start, end, path):
    """path(예: [b'moov', b'trak']) 를 따라 내려가 첫 매칭 박스 구간을 반환."""
    if not path:
        return start, end
    f.seek(start)
    for typ, ps, pe in iter_boxes(f, end):
        if typ == path[0]:
            found = find_box(f, ps, pe, path[1:])
            if found:
                return found
            f.seek(pe)
    return None


def parse_tkhd(f, start):
    f.seek(start)
    version = f.read(1)[0]
    f.read(3)  # flags
    f.read(8 * 2 if version == 1 else 4 * 2)  # creation / modification
    f.read(4)  # track_id
    f.read(4)  # reserved
    f.read(8 if version == 1 else 4)  # duration
    f.read(8 + 2 + 2 + 2 + 2)  # reserved, layer, alt_group, volume, reserved
    matrix = struct.unpack(">9i", f.read(36))
    width = struct.unpack(">I", f.read(4))[0] / 65536.0
    height = struct.unpack(">I", f.read(4))[0] / 65536.0

    # matrix = [a b u; c d v; x y w], a/b 는 16.16 고정소수점
    a, b = matrix[0] / 65536.0, matrix[1] / 65536.0
    rotation = round(math.degrees(math.atan2(b, a)))
    if rotation > 180:
        rotation -= 360
    if rotation == -180:
        rotation = 180
    return rotation, width, height


def parse_mdhd(f, start):
    f.seek(start)
    version = f.read(1)[0]
    f.read(3)
    f.read(8 * 2 if version == 1 else 4 * 2)
    timescale = struct.unpack(">I", f.read(4))[0]
    duration = struct.unpack(">Q" if version == 1 else ">I",
                             f.read(8 if version == 1 else 4))[0]
    return timescale, duration


def parse_hdlr(f, start):
    f.seek(start + 8)
    return f.read(4)


def parse_stsz(f, start):
    f.seek(start + 4)
    f.read(4)  # sample_size
    return struct.unpack(">I", f.read(4))[0]


def probe(path):
    """비디오 트랙 정보를 dict 로 반환. 실패 시 error 키를 채운다."""
    info = {"file": str(path), "width": "", "height": "", "fps": "",
            "nb_frames": "", "rotation": "", "display_w": "", "display_h": "",
            "error": ""}
    size = path.stat().st_size
    with path.open("rb") as f:
        moov = find_box(f, 0, size, [b"moov"])
        if not moov:
            info["error"] = "moov not found"
            return info

        # 하위 탐색이 파일 포인터를 옮기므로 trak 목록을 먼저 확정한다
        f.seek(moov[0])
        traks = [(ts, te) for typ, ts, te in iter_boxes(f, moov[1]) if typ == b"trak"]

        for ts, te in traks:
            hdlr = find_box(f, ts, te, [b"mdia", b"hdlr"])
            if not hdlr or parse_hdlr(f, hdlr[0]) != b"vide":
                continue

            tkhd = find_box(f, ts, te, [b"tkhd"])
            if tkhd:
                rot, w, h = parse_tkhd(f, tkhd[0])
                info["rotation"] = rot
                info["width"] = int(w)
                info["height"] = int(h)
                # 회전 적용 후 실제로 보이는 크기
                if abs(rot) == 90:
                    info["display_w"], info["display_h"] = int(h), int(w)
                else:
                    info["display_w"], info["display_h"] = int(w), int(h)

            nb = None
            stsz = find_box(f, ts, te, [b"mdia", b"minf", b"stbl", b"stsz"])
            if stsz:
                nb = parse_stsz(f, stsz[0])
                info["nb_frames"] = nb

            mdhd = find_box(f, ts, te, [b"mdia", b"mdhd"])
            if mdhd and nb:
                timescale, duration = parse_mdhd(f, mdhd[0])
                if timescale and duration:
                    info["fps"] = round(nb / (duration / timescale), 3)
            return info

    info["error"] = "no video track"
    return info


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "videos")
    if not root.is_dir():
        sys.exit(f"디렉터리를 찾을 수 없음: {root}")

    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if not files:
        sys.exit(f"동영상 없음: {root}")

    rows = []
    for p in files:
        try:
            rows.append(probe(p))
        except Exception as e:  # 깨진 파일도 목록에는 남긴다
            rows.append({"file": str(p), "width": "", "height": "", "fps": "",
                         "nb_frames": "", "rotation": "", "display_w": "",
                         "display_h": "", "error": f"{type(e).__name__}: {e}"})

    out = Path("meta/rotation_scan.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 요약
    print(f"{len(rows)}개 스캔 -> {out}\n")
    counts = {}
    for r in rows:
        key = (r["rotation"], r["width"], r["height"])
        counts[key] = counts.get(key, 0) + 1
    print(f"{'rotation':>10} {'w x h':>12} {'개수':>6}")
    for (rot, w_, h_), n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"{str(rot):>10} {f'{w_}x{h_}':>12} {n:>6}")

    bad = [r for r in rows if r["error"]]
    if bad:
        print(f"\n파싱 실패 {len(bad)}개:")
        for r in bad[:10]:
            print(f"  {r['file']}: {r['error']}")


if __name__ == "__main__":
    main()
