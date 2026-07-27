"""
02_build_features.py
랜드마크(.npz) -> 고정 길이 특징 행렬 X (+ 라벨 y).

핵심 설계
  1) 정규화: 손목 원점 이동 + 손 크기 스케일. 카메라 거리/화면 위치가 달라도 같은 값이 나온다.
     (영상 해상도가 제각각이어도 MediaPipe 좌표가 이미 0~1 정규화라 문제 없음)
  2) 결측 처리: 손 미검출 프레임은 선형보간 + '유효 마스크'를 별도 채널로 보존.
     실측상 손 검출률이 영상마다 38~91%로 편차가 크므로 이 처리가 정확도를 좌우한다.
  3) 시간축 리샘플: 프레임 수(55~79, fps도 24/30 혼재)를 T=32로 통일.
  4) Pose를 보조 특징으로 추가. Pose는 100% 가까이 잡히므로 손이 안 잡히는 구간의 보험이다.

사용법:
    python 02_build_features.py --landmarks ./landmarks --labels labels.csv --out dataset.npz
"""
import argparse
import os
import glob
import numpy as np

T_OUT = 32          # 리샘플 후 프레임 수
WRIST, IDX_MCP, MID_MCP, PINKY_MCP = 0, 5, 9, 17
# 상체 Pose 인덱스만 사용 (하체는 제스처와 무관하고 프레임 밖인 경우도 많음)
POSE_UPPER = [11, 12, 13, 14, 15, 16, 0]  # 양 어깨, 양 팔꿈치, 양 손목, 코


def resample_time(seq, t_out=T_OUT):
    """(T, D) -> (t_out, D) 선형보간 리샘플."""
    T = seq.shape[0]
    if T == t_out:
        return seq.astype(np.float32)
    src = np.linspace(0.0, 1.0, T)
    dst = np.linspace(0.0, 1.0, t_out)
    flat = seq.reshape(T, -1)
    out = np.stack([np.interp(dst, src, flat[:, d]) for d in range(flat.shape[1])], axis=1)
    return out.astype(np.float32).reshape(t_out, *seq.shape[1:])


def interp_missing(seq, valid):
    """valid==0 인 프레임을 앞뒤 유효 프레임으로 선형보간. 전부 결측이면 0 유지."""
    T = seq.shape[0]
    idx = np.where(valid > 0)[0]
    if len(idx) == 0:
        return seq, False
    if len(idx) == T:
        return seq, True
    flat = seq.reshape(T, -1).copy()
    all_t = np.arange(T)
    for d in range(flat.shape[1]):
        flat[:, d] = np.interp(all_t, idx, flat[idx, d])
    return flat.reshape(seq.shape), True


def pick_primary_hand(hand, hand_valid):
    """
    (T, 2, 21, 3) 중 '주 손' 하나를 고른다.
    두 손이 잡힌 프레임에서는 더 크게 잡힌(=카메라에 가까운, 제스처를 수행 중인) 손을 택한다.
    수어 제스처가 한 손 위주인 데이터라 이 단순 규칙으로 충분하다.
    양손 제스처가 섞여 있으면 --two-hands 옵션으로 두 손 모두 쓰도록 확장할 것.
    """
    T = hand.shape[0]
    out = np.zeros((T, 21, 3), np.float32)
    valid = np.zeros((T,), np.uint8)
    for t in range(T):
        best, best_size = -1, -1.0
        for k in range(hand.shape[1]):
            if not hand_valid[t, k]:
                continue
            pts = hand[t, k, :, :2]
            size = float(np.ptp(pts[:, 0]) + np.ptp(pts[:, 1]))
            if size > best_size:
                best, best_size = k, size
        if best >= 0:
            out[t] = hand[t, best]
            valid[t] = 1
    return out, valid


def normalize_hand(seq):
    """손목 원점 이동 + 손 크기 정규화. (T,21,3) -> (T,21,3)"""
    seq = seq.copy()
    seq -= seq[:, WRIST:WRIST + 1, :]                     # 화면 내 위치 무효화
    span = np.linalg.norm(seq[:, MID_MCP, :2], axis=1)    # 손목~중지 MCP 거리 = 손 크기
    span = np.maximum(span, 1e-6)[:, None, None]
    return seq / span                                      # 카메라 거리 무효화


def normalize_pose(pose_seq):
    """어깨 중심 원점 + 어깨 너비 스케일. (T,33,4) -> (T,len(POSE_UPPER),3)"""
    xyz = pose_seq[:, :, :3]
    center = (xyz[:, 11, :] + xyz[:, 12, :]) / 2.0
    width = np.linalg.norm(xyz[:, 11, :2] - xyz[:, 12, :2], axis=1)
    width = np.maximum(width, 1e-6)[:, None, None]
    sub = xyz[:, POSE_UPPER, :] - center[:, None, :]
    return (sub / width).astype(np.float32)


def features_from_npz(path):
    """npz 1개 -> 1차원 특징 벡터. 실패하면 None."""
    z = np.load(path)
    hand, hand_valid = z["hand"], z["hand_valid"]
    pose, pose_valid = z["pose"], z["pose_valid"]

    h, hv = pick_primary_hand(hand, hand_valid)
    det_rate = float(hv.mean())
    h, ok = interp_missing(h, hv)
    if not ok:
        return None, det_rate      # 손이 한 프레임도 안 잡힌 영상

    h = normalize_hand(h)
    p, pok = interp_missing(pose, pose_valid)
    p = normalize_pose(p) if pok else np.zeros((hand.shape[0], len(POSE_UPPER), 3), np.float32)

    h32 = resample_time(h)                                  # (32,21,3)
    p32 = resample_time(p)                                  # (32,7,3)
    v32 = resample_time(hv.astype(np.float32)[:, None])     # (32,1) 유효 마스크

    # 속도(1차 차분)는 '동작의 흐름/템포'를 담는다.
    # 제안서가 요구하는 "정지 동작이 아닌 궤적·속도·템포" 가 바로 이 항이다.
    dh = np.diff(h32, axis=0, prepend=h32[:1])
    dp = np.diff(p32, axis=0, prepend=p32[:1])

    feat = np.concatenate([
        h32.reshape(T_OUT, -1),   # 63
        dh.reshape(T_OUT, -1),    # 63
        p32.reshape(T_OUT, -1),   # 21
        dp.reshape(T_OUT, -1),    # 21
        v32,                      # 1
    ], axis=1)                    # (32, 169)
    return feat.reshape(-1).astype(np.float32), det_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks", required=True)
    ap.add_argument("--labels", required=True,
                    help="CSV: filename,gesture,performer  (헤더 포함)")
    ap.add_argument("--out", default="dataset.npz")
    ap.add_argument("--min-det", type=float, default=0.15,
                    help="손 검출률이 이 값 미만인 영상은 제외")
    a = ap.parse_args()

    import csv
    meta = {}
    with open(a.labels, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            stem = os.path.splitext(os.path.basename(row["filename"].strip()))[0]
            meta[stem] = (row["gesture"].strip(), row["performer"].strip())

    X, gestures, performers, names, dropped = [], [], [], [], []
    for path in sorted(glob.glob(os.path.join(a.landmarks, "*.npz"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem not in meta:
            dropped.append((stem, "labels.csv 에 없음"))
            continue
        feat, det = features_from_npz(path)
        if feat is None:
            dropped.append((stem, "손 미검출 100%"))
            continue
        if det < a.min_det:
            dropped.append((stem, f"손 검출률 {det:.0%}"))
            continue
        g, p = meta[stem]
        X.append(feat); gestures.append(g); performers.append(p); names.append(stem)

    if not X:
        # 대부분 labels.csv 를 영상 폴더보다 먼저 만들어 비어 있는 경우다.
        print(f"사용 가능한 샘플이 0개입니다. (npz {len(dropped)}개 전부 제외됨)")
        print(f"labels.csv 행 수: {len(meta)}")
        if not meta:
            print("-> labels.csv 에 데이터 행이 없습니다. "
                  "영상을 videos/ 에 넣은 뒤 00_make_labels_template.py 를 다시 실행하세요.")
        else:
            print("-> 제외 사유:")
            for s, r in dropped[:20]:
                print(f"     {s}: {r}")
            print("   npz 파일명과 labels.csv 의 filename 열이 일치하는지 확인하세요.")
        raise SystemExit(1)

    X = np.stack(X)
    np.savez_compressed(a.out, X=X,
                        gesture=np.array(gestures), performer=np.array(performers),
                        name=np.array(names), T=T_OUT, D=X.shape[1] // T_OUT)
    print(f"X = {X.shape}  (영상 {X.shape[0]}개 x {X.shape[1]}차원)")
    print(f"제스처 종류: {sorted(set(gestures))}")
    print(f"수행자: {sorted(set(performers))}")
    if dropped:
        print(f"\n제외 {len(dropped)}개:")
        for s, r in dropped[:20]:
            print(f"  {s}: {r}")
    print(f"\n저장 -> {a.out}")


if __name__ == "__main__":
    main()
