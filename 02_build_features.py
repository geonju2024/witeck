"""
02_build_features.py
랜드마크(.npz) -> 고정 길이 특징 행렬 X (+ 라벨 y).

핵심 설계
  0) 종횡비 보정: MediaPipe 좌표는 0~1 정규화지만 x 는 '너비'로, y 는 '높이'로 나눈
     값이다. 그래서 16:9 영상과 9:16 영상은 같은 손 모양이라도 x:y 비가 달라진다.
     정규화 전에 등방 좌표로 되돌려야 한다. (자세한 근거는 apply_aspect 참고)
  1) 정규화: 손목 원점 이동 + 손 크기 스케일. 카메라 거리/화면 위치가 달라도 같은 값이 나온다.
  2) 결측 처리: 손 미검출 프레임은 선형보간 + '유효 마스크'를 별도 채널로 보존.
     실측상 손 검출률이 영상마다 38~91%로 편차가 크므로 이 처리가 정확도를 좌우한다.
  3) 시간축 리샘플: 프레임 수(55~79, fps도 24/30 혼재)를 T=32로 통일.
  4) Pose를 보조 특징으로 추가. Pose는 100% 가까이 잡히므로 손이 안 잡히는 구간의 보험이다.

사용법:
    python 02_build_features.py --landmarks ./landmarks --labels labels.csv --out dataset.npz
"""
# labels.csv 의 relative_path 를 그대로 키로 쓴다. 파일명(basename)만으로 맞추면
# 세션 폴더가 다른 동명 파일이 서로를 덮어쓴다.
import argparse
import os
import sys
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
    chosen = np.full((T,), -1, np.int8)      # 프레임별로 고른 손의 인덱스
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
            chosen[t] = best
    return out, valid, chosen


def primary_hand_side(chosen, handedness):
    """주 손의 좌우를 다수결로 판정. 1=Right, 0=Left, -1=판정 불가."""
    sides = [int(handedness[t, k]) for t, k in enumerate(chosen)
             if k >= 0 and handedness[t, k] >= 0]
    if not sides:
        return -1
    return 1 if float(np.mean(sides)) >= 0.5 else 0


def mirror_to_right(h, p):
    """
    왼손 수행분을 오른손 기준으로 좌우 반전한다.

    왜 필요한가:
      같은 제스처를 왼손으로 하면 좌표가 거울상이 되는데, normalize_hand 는 원점 이동과
      크기 나눗셈만 하므로 이 반전이 그대로 남는다. 모델에게는 전혀 다른 동작으로 보인다.
      실측상 P03 은 7월 세션은 오른손, 8월 세션은 왼손으로 수행했다. 정규화하지 않으면
      세션분리 평가에서 본인이 통째로 거부된다.

      인증 관점에서도 '어느 손을 쓰는가'는 1비트짜리 정보이고, 공격자가 영상만 보면
      바로 따라할 수 있다. 이 1비트만으로 등록자/타인이 최대 78%까지 갈리는 제스처가
      있어(G4), 정규화하지 않으면 EER 이 실제 보안 강도보다 낙관적으로 나온다.

    손은 손목 기준 좌표라 x 부호만 뒤집으면 된다.
    Pose 는 어깨 중심 좌표이므로 x 를 뒤집은 뒤 좌우 쌍(어깨/팔꿈치/손목)의 순서도 맞바꾼다.
    """
    h = h.copy()
    h[..., 0] *= -1.0
    p = p.copy()
    p[..., 0] *= -1.0
    # POSE_UPPER = [11,12, 13,14, 15,16, 0] -> 좌우 쌍끼리 교환, 코(0)는 그대로
    p = p[:, [1, 0, 3, 2, 5, 4, 6], :]
    return h, p


def apply_aspect(seq, width, height):
    """
    MediaPipe 정규화 좌표(0~1)를 등방(isotropic) 좌표로 되돌린다.

    landmark.x 는 프레임 '너비'로, y 는 '높이'로 나눈 값이다. 따라서 같은 물리적 손이라도
    프레임 종횡비가 다르면 정규화 좌표에서의 가로:세로 비가 달라진다.
    이 데이터셋 실측값(손 가로/세로 비 중앙값):
        9:16 세로 영상  1.61 ~ 1.72
        16:9 가로 영상  0.59          <- 혼자 2.7배 어긋남
    normalize_hand 의 크기 나눗셈은 등방 스케일만 없애므로 이 눌림은 보정되지 않는다.

    x 에 (width/height) 를 곱하면 두 축이 같은 픽셀 배율을 갖는다. 남는 공통 배율은
    이후 손 크기로 나누는 단계에서 상쇄된다.
    z 는 x 와 같은 스케일이라고 MediaPipe 문서에 명시돼 있어 함께 보정한다.
    """
    seq = seq.astype(np.float32, copy=True)
    ratio = float(width) / float(height)
    seq[..., 0] *= ratio
    if seq.shape[-1] > 2:
        seq[..., 2] *= ratio
    return seq


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


def features_from_npz(path, canonical_hand=True):
    """npz 1개 -> (특징 벡터, 손 검출률, 주 손 좌우). 실패하면 특징이 None."""
    z = np.load(path)
    W, H = float(z["width"]), float(z["height"])
    hand, hand_valid = apply_aspect(z["hand"], W, H), z["hand_valid"]
    pose, pose_valid = z["pose"].astype(np.float32, copy=True), z["pose_valid"]
    pose[:, :, :3] = apply_aspect(pose[:, :, :3], W, H)   # visibility(4번째)는 건드리지 않는다

    h, hv, chosen = pick_primary_hand(hand, hand_valid)
    side = primary_hand_side(chosen, z["handedness"])
    det_rate = float(hv.mean())
    h, ok = interp_missing(h, hv)
    if not ok:
        return None, det_rate, side      # 손이 한 프레임도 안 잡힌 영상

    h = normalize_hand(h)
    p, pok = interp_missing(pose, pose_valid)
    p = normalize_pose(p) if pok else np.zeros((hand.shape[0], len(POSE_UPPER), 3), np.float32)

    if canonical_hand and side == 0:      # 왼손 -> 오른손 기준으로 통일
        h, p = mirror_to_right(h, p)

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
    return feat.reshape(-1).astype(np.float32), det_rate, side


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # 윈도우 cp949 콘솔에서 한글 깨짐 방지
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks", required=True)
    ap.add_argument("--labels", required=True,
                    help="CSV: relative_path,gesture,performer,session,role (헤더 포함)")
    ap.add_argument("--out", default="dataset.npz")
    ap.add_argument("--min-det", type=float, default=0.15,
                    help="손 검출률이 이 값 미만인 영상은 제외")
    ap.add_argument("--keep-hand-side", action="store_true",
                    help="좌우 정규화를 끈다(왼손/오른손을 다른 동작으로 취급). "
                         "기본값은 정규화 켬. 비교 실험용")
    a = ap.parse_args()

    import csv
    meta = {}
    with open(a.labels, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rel = row["relative_path"].strip().replace("\\", "/")
            meta[rel] = row

    # labels.csv 를 기준으로 돈다. 이래야 '랜드마크가 없는 영상'이 조용히 빠지지 않고
    # 제외 목록에 남는다.
    X, gestures, performers, sessions, roles, hands, names, dropped = [], [], [], [], [], [], [], []
    for rel, row in sorted(meta.items()):
        path = os.path.join(a.landmarks, os.path.splitext(rel)[0] + ".npz")
        if not os.path.exists(path):
            dropped.append((rel, "랜드마크 npz 없음 (01번을 먼저 실행)"))
            continue
        feat, det, side = features_from_npz(path, canonical_hand=not a.keep_hand_side)
        if feat is None:
            dropped.append((rel, "손 미검출 100%"))
            continue
        if det < a.min_det:
            dropped.append((rel, f"손 검출률 {det:.0%}"))
            continue
        X.append(feat)
        gestures.append(row["gesture"].strip())
        performers.append(row["performer"].strip())
        sessions.append(row["session"].strip())
        roles.append(row["role"].strip())
        hands.append({1: "R", 0: "L"}.get(side, "?"))
        names.append(rel)

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
            print("   labels.csv 의 relative_path 열과 landmarks/ 하위 경로가 맞는지 확인하세요.")
        raise SystemExit(1)

    X = np.stack(X)
    np.savez_compressed(a.out, X=X,
                        gesture=np.array(gestures), performer=np.array(performers),
                        session=np.array(sessions), role=np.array(roles),
                        hand=np.array(hands), name=np.array(names),
                        canonical_hand=(not a.keep_hand_side),
                        T=T_OUT, D=X.shape[1] // T_OUT)
    print(f"X = {X.shape}  (영상 {X.shape[0]}개 x {X.shape[1]}차원)")
    print(f"제스처 종류: {sorted(set(gestures))}")
    print(f"수행자: {sorted(set(performers))}")
    print(f"세션: {sorted(set(sessions))}")
    hc = {s: hands.count(s) for s in sorted(set(hands))}
    mode = "끔(왼손/오른손 구분 유지)" if a.keep_hand_side else "켬(오른손 기준 통일)"
    print(f"주 손 분포: {hc}   좌우 정규화: {mode}")
    if dropped:
        print(f"\n제외 {len(dropped)}개:")
        for s, r in dropped[:20]:
            print(f"  {s}: {r}")
    print(f"\n저장 -> {a.out}")


if __name__ == "__main__":
    main()
