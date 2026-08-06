"""
03_train.py
dataset.npz -> 두 가지 과제를 각각 학습/평가한다. 전체 실행 시간 수 초.

과제 A) 제스처 분류 (4-class)
    "지금 들어온 동작이 어떤 제스처인가?"
    쉬운 과제다. 여기서 95% 넘게 나와도 프로젝트가 끝난 게 아니다.

과제 B) 본인 인증 / verification  <-- 이게 진짜 과제다
    "이 제스처의 등록자 본인이 한 것인가, 남이 흉내낸 것인가?"
    제스처별로 본인 15개 vs 타인 40개. 제안서의 '정확도 90%' 목표는
    여기서 측정해야 의미가 있다. FAR(타인 통과율)/FRR(본인 거부율)/EER 로 보고한다.

사용법:
    python 03_train.py --data dataset.npz --owners owners.csv
    # owners.csv:  gesture,owner   (제스처별 등록자 1명)
"""
import argparse
import csv
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc


def models():
    """작은 데이터(수백 샘플)에서는 이 셋이 딥러닝보다 빠르고 대개 더 강하다."""
    return {
        "LogReg":  make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=3000, C=1.0)),
        "SVM-RBF": make_pipeline(StandardScaler(),
                                 SVC(kernel="rbf", C=10, gamma="scale")),
        "RF":      RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                          n_jobs=-1, random_state=0),
    }


def eer_from_scores(y_true, score):
    """EER(Equal Error Rate)와 그때의 임계값. 인증 시스템의 표준 지표."""
    fpr, tpr, thr = roc_curve(y_true, score)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return (fpr[i] + fnr[i]) / 2, thr[i], auc(fpr, tpr)


def task_a_gesture(X, gesture, performer):
    classes = sorted(set(gesture))
    print("=" * 62)
    print(f"과제 A) 제스처 분류 ({len(classes)}-class)")
    print("=" * 62)
    y = np.array([classes.index(g) for g in gesture])

    # 수행자 단위로 분할한다. 같은 사람의 영상이 train/test에 동시에 들어가면
    # 모델이 '제스처'가 아니라 '그 사람의 손'을 외워서 정확도가 부풀려진다.
    n_groups = len(set(performer))
    cv = StratifiedGroupKFold(n_splits=min(5, n_groups), shuffle=True, random_state=0)

    for name, mdl in models().items():
        pred = cross_val_predict(mdl, X, y, cv=cv, groups=performer, n_jobs=1)
        acc = accuracy_score(y, pred)
        print(f"\n[{name}]  수행자-분리 교차검증 정확도 = {acc:.3f}")
        if name == "SVM-RBF":
            print(classification_report(y, pred, target_names=classes, digits=3, zero_division=0))
            print("혼동행렬 (행=정답, 열=예측)")
            print("        " + "  ".join(f"{c:>6}" for c in classes))
            for c, row in zip(classes, confusion_matrix(y, pred)):
                print(f"{c:>6}  " + "  ".join(f"{v:>6d}" for v in row))


def impostor_grouped_folds(yg, pg, seed=0):
    """
    타인(impostor)을 '수행자 단위로' 통째로 홀드아웃하는 CV 폴드를 만든다.

    왜 필요한가:
      StratifiedKFold 는 같은 타인의 영상을 학습과 평가에 동시에 넣는다. 그러면 모델이
      "등록자가 아닌 동작"을 배우는 대신 "이 4명의 동작"을 외워도 통과한다.
      실제 인증에서 FAR 이 의미를 갖는 건 '처음 보는 사람'을 거부할 때이므로,
      평가 대상 타인은 학습에 한 번도 등장하지 않아야 한다.

    폴드 k:  test = (타인 수행자 k 의 전체 샘플) + (등록자 샘플의 1/K)
             train = 나머지 타인 수행자 전원 + 나머지 등록자 샘플
    등록자는 한 명뿐이라 그룹 분리가 불가능하므로 등록자 샘플만 K등분한다.
    (=> FRR 은 여전히 같은 세션 샘플로 측정된다는 한계가 남는다)

    각 샘플은 정확히 한 번만 test 에 들어간다 -> cross_val_predict 사용 가능.
    폴드를 만들 수 없으면 None.
    """
    pg = np.asarray(pg)
    pos = np.where(yg == 1)[0]
    imp = sorted(set(pg[yg == 0]))
    K = len(imp)
    if K < 2 or len(pos) < K:
        return None
    rng = np.random.RandomState(seed)
    pos_folds = np.array_split(pos[rng.permutation(len(pos))], K)
    all_idx = np.arange(len(yg))
    folds = []
    for k, who in enumerate(imp):
        te = np.concatenate([pos_folds[k], np.where((yg == 0) & (pg == who))[0]])
        folds.append((np.setdiff1d(all_idx, te), te))
    return folds


def _split_list(items, k):
    """리스트를 k 개 그룹으로 최대한 고르게 나눈다."""
    n = len(items)
    return [items[i * n // k:(i + 1) * n // k] for i in range(k)]


def session_folds(yg, pg, sg, seed=0):
    """
    등록자의 '세션(촬영일)' 을 통째로 홀드아웃한다.

    왜 필요한가:
      impostor_grouped_folds 는 등록자 샘플을 무작위로 K등분하므로, 같은 날 찍은 영상이
      학습과 평가에 함께 들어간다. 같은 조명/의상/카메라 위치가 그대로 재현되므로
      FRR 이 실제보다 낙관적으로 나온다. 5일에 걸쳐 촬영한 이유가 이걸 없애기 위해서다.

    폴드 k: test = (등록자의 세션 k 전체) + (타인 샘플의 1/K)
    타인은 무작위 분할이므로 FAR 은 여전히 낙관적이다. 이 방식이 정직해지는 건 FRR 이다.
    """
    sg, pg = np.asarray(sg), np.asarray(pg)
    pos = np.where(yg == 1)[0]
    sessions = sorted(set(sg[pos]))
    K = len(sessions)
    if K < 2:
        return None
    neg = np.where(yg == 0)[0]
    rng = np.random.RandomState(seed)
    neg_folds = np.array_split(neg[rng.permutation(len(neg))], K)
    all_idx = np.arange(len(yg))
    folds = []
    for k, s in enumerate(sessions):
        te = np.concatenate([pos[sg[pos] == s], neg_folds[k]])
        folds.append((np.setdiff1d(all_idx, te), te))
    return folds


def session_impostor_folds(yg, pg, sg):
    """
    등록자 세션과 타인 수행자를 '동시에' 홀드아웃한다. 가장 엄격한 평가.

    폴드 k: test = (등록자 세션 그룹 k) + (타인 수행자 그룹 k 전원)
            train = 나머지 세션의 등록자 + 나머지 타인 전원

    FRR 은 '학습에 없던 날', FAR 은 '학습에 없던 사람' 에서 측정된다.
    실사용 조건(등록 후 다른 날 인증, 처음 보는 공격자)에 가장 가까우므로
    제안서 수치는 이 값으로 보고해야 방어할 수 있다.

    등록자 세션이 1개뿐이면(=추가 촬영을 하지 않은 수행자) 폴드를 만들 수 없어 None.
    """
    sg, pg = np.asarray(sg), np.asarray(pg)
    pos = np.where(yg == 1)[0]
    sessions = sorted(set(sg[pos]))
    imps = sorted(set(pg[yg == 0]))
    K = min(len(sessions), len(imps))
    if K < 2:
        return None
    sess_groups = _split_list(sessions, K)
    imp_groups = _split_list(imps, K)
    all_idx = np.arange(len(yg))
    folds = []
    for k in range(K):
        te = np.concatenate([
            pos[np.isin(sg[pos], sess_groups[k])],
            np.where((yg == 0) & np.isin(pg, imp_groups[k]))[0],
        ])
        folds.append((np.setdiff1d(all_idx, te), te))
    return folds


def eval_scheme(Xg, yg, cv, n_pos, n_neg):
    """CV 하나에 대해 모델 3종을 평가하고 (모델명, EER, AUC, FAR, FRR) 목록을 돌려준다."""
    out = []
    for name, mdl in models().items():
        # 점수는 확률이 아니어도 된다. ROC/EER 은 순위만 쓰므로
        # SVM/LogReg 는 decision_function, RF 는 predict_proba 를 쓴다.
        method = "predict_proba" if name == "RF" else "decision_function"
        sc = cross_val_predict(mdl, Xg, yg, cv=cv, method=method, n_jobs=1)
        prob = sc[:, 1] if sc.ndim == 2 else sc
        eer, thr, roc = eer_from_scores(yg, prob)
        pred = (prob >= thr).astype(int)          # EER 임계값에서의 FAR/FRR
        far = float(((pred == 1) & (yg == 0)).sum() / max(n_neg, 1))
        frr = float(((pred == 0) & (yg == 1)).sum() / max(n_pos, 1))
        out.append((name, eer, roc, far, frr))
    return out


SCHEME_ORDER = ["혼합", "타인분리", "세션분리", "이중분리"]


def task_b_verification(X, gesture, performer, owners, session=None):
    print("\n" + "=" * 62)
    print("과제 B) 본인 인증 (제스처별 본인 vs 타인)")
    print("=" * 62)
    print("교차검증 네 방식을 나란히 측정한다. 아래로 갈수록 엄격하다.")
    print("  [혼합]     StratifiedKFold. 같은 타인이 학습/평가에 모두 등장 -> FAR 낙관적")
    print("  [타인분리] 타인을 수행자 단위로 홀드아웃 -> FAR 정직, FRR 은 여전히 같은 세션")
    print("  [세션분리] 등록자 세션을 홀드아웃      -> FRR 정직('다른 날'), FAR 낙관적")
    print("  [이중분리] 세션 + 타인 동시 홀드아웃   -> 둘 다 정직. 실사용에 가장 가까움")

    performer = np.array(performer)
    session = np.array(session) if session is not None else None
    rows = []
    for g in sorted(set(gesture)):
        owner = owners.get(g)
        if owner is None:
            print(f"\n[{g}] 등록자 정보 없음 -> 건너뜀")
            continue
        m = np.array([x == g for x in gesture])
        Xg, pg = X[m], performer[m]
        sg = session[m] if session is not None else None
        yg = np.array([1 if p == owner else 0 for p in pg])
        n_pos, n_neg = int(yg.sum()), int((1 - yg).sum())
        n_sess = len(set(sg[yg == 1])) if sg is not None else 0
        print(f"\n[{g}] 등록자={owner}  본인 {n_pos}개 / 타인 {n_neg}개"
              f"  (타인 수행자 {len(set(pg[yg == 0]))}명, 등록자 세션 {n_sess}개)")
        if n_pos < 4 or n_neg < 4:
            print("  샘플이 너무 적어 평가 생략")
            continue

        # 본인 샘플이 15개뿐이므로 fold를 많이 쪼갠다.
        schemes = [("혼합", StratifiedKFold(n_splits=min(5, n_pos), shuffle=True, random_state=0))]
        folds = impostor_grouped_folds(yg, pg)
        if folds is None:
            print("  타인 수행자가 부족해 [타인분리] 생략")
        else:
            schemes.append(("타인분리", folds))
        if sg is not None:
            for label, fn in (("세션분리", session_folds), ("이중분리", session_impostor_folds)):
                f = fn(yg, pg, sg)
                if f is None:
                    print(f"  등록자 세션이 {n_sess}개뿐이라 [{label}] 생략"
                          f" -> 추가 촬영 없이는 이 수치를 낼 수 없습니다")
                else:
                    schemes.append((label, f))

        best = {}
        for sname, cv in schemes:
            for name, eer, roc, far, frr in eval_scheme(Xg, yg, cv, n_pos, n_neg):
                print(f"  [{sname:<5}] {name:8s} AUC={roc:.3f}  EER={eer:.3f}  "
                      f"FAR={far:.3f} FRR={frr:.3f}")
                if sname not in best or eer < best[sname][1]:
                    best[sname] = (name, eer, roc, far, frr)
        rows.append((g, owner, best))

    if not rows:
        return
    used = [s for s in SCHEME_ORDER if any(s in b for _, _, b in rows)]

    print("\n" + "-" * 62)
    print("요약: 제스처별 최적 EER (낮을수록 좋음)")
    print(f"{'제스처':<8}{'등록자':<8}" + "".join(f"{s:>12}" for s in used))
    for g, o, b in rows:
        line = f"{g:<8}{o:<8}"
        for s in used:
            line += f"{b[s][1]:>12.3f}" if s in b else f"{'-':>12}"
        print(line)

    line = f"{'평균':<8}{'':<8}"
    for s in used:
        vals = [b[s][1] for _, _, b in rows if s in b]
        line += f"{np.mean(vals):>12.3f}" if vals else f"{'-':>12}"
    print(line)

    print("\n정확도 환산 (1 - EER):")
    for s in used:
        vals = [b[s][1] for _, _, b in rows if s in b]
        n_missing = len(rows) - len(vals)
        note = f"  (제스처 {n_missing}개 측정 불가)" if n_missing else ""
        if vals:
            print(f"  [{s}] {1 - float(np.mean(vals)):.1%}{note}")

    print("\n제안서의 '인식 정확도 90% 이상' 은 [이중분리] 수치로 보고해야 방어 가능합니다.")
    print("나머지 셋은 각각 FAR 또는 FRR 이 낙관적으로 측정됩니다.")


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # 윈도우 cp949 콘솔에서 한글 깨짐 방지
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset.npz")
    ap.add_argument("--owners", help="CSV: gesture,owner")
    a = ap.parse_args()

    z = np.load(a.data, allow_pickle=True)
    X, gesture, performer = z["X"], list(z["gesture"]), list(z["performer"])
    session = list(z["session"]) if "session" in z.files else None
    role = list(z["role"]) if "role" in z.files else None
    print(f"데이터 {X.shape[0]}개, 특징 {X.shape[1]}차원")
    if "canonical_hand" in z.files:
        print(f"좌우 정규화: {'켬' if bool(z['canonical_hand']) else '끔'}")
    if session is None:
        print("경고: dataset.npz 에 session 이 없습니다 -> 세션분리 평가 생략"
              " (02_build_features.py 를 다시 실행하세요)")
    print()

    task_a_gesture(X, gesture, performer)

    owners = {}
    if role is not None:
        # role 은 데이터와 같은 labels.csv 에서 왔으므로 owners.csv 보다 신뢰할 수 있다
        cand = {}
        for g, p, r in zip(gesture, performer, role):
            if r == "own":
                cand.setdefault(g, set()).add(p)
        for g, who in cand.items():
            if len(who) == 1:
                owners[g] = next(iter(who))
            else:
                print(f"경고: {g} 의 등록자가 {sorted(who)} 로 여러 명입니다 -> 건너뜀")
    elif a.owners:
        with open(a.owners, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                owners[row["gesture"].strip()] = row["owner"].strip()
    else:
        # 라벨이 없으면 '해당 제스처를 가장 많이 수행한 사람'을 등록자로 추정
        for g in set(gesture):
            cnt = {}
            for gg, pp in zip(gesture, performer):
                if gg == g:
                    cnt[pp] = cnt.get(pp, 0) + 1
            owners[g] = max(cnt, key=cnt.get)
        print("\n(등록자 정보 없음 -> 최다 수행자를 등록자로 자동 추정)")

    task_b_verification(X, gesture, performer, owners, session)


if __name__ == "__main__":
    main()
