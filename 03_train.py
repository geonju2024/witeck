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


def task_b_verification(X, gesture, performer, owners):
    print("\n" + "=" * 62)
    print("과제 B) 본인 인증 (제스처별 본인 vs 타인)")
    print("=" * 62)
    print("교차검증 두 방식을 나란히 측정한다.")
    print("  [혼합]     StratifiedKFold. 같은 타인이 학습/평가에 모두 등장 -> FAR 낙관적")
    print("  [타인분리] 타인을 수행자 단위로 홀드아웃 -> '처음 보는 사람'을 거부하는지 측정")

    performer = np.array(performer)
    rows = []
    for g in sorted(set(gesture)):
        owner = owners.get(g)
        if owner is None:
            print(f"\n[{g}] owners.csv 에 등록자 정보 없음 -> 건너뜀")
            continue
        m = np.array([x == g for x in gesture])
        Xg, pg = X[m], performer[m]
        yg = np.array([1 if p == owner else 0 for p in pg])
        n_pos, n_neg = int(yg.sum()), int((1 - yg).sum())
        print(f"\n[{g}] 등록자={owner}  본인 {n_pos}개 / 타인 {n_neg}개"
              f"  (타인 수행자 {len(set(pg[yg == 0]))}명)")
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

        best = {}
        for sname, cv in schemes:
            for name, eer, roc, far, frr in eval_scheme(Xg, yg, cv, n_pos, n_neg):
                print(f"  [{sname:<5}] {name:8s} AUC={roc:.3f}  EER={eer:.3f}  "
                      f"FAR={far:.3f} FRR={frr:.3f}")
                if sname not in best or eer < best[sname][1]:
                    best[sname] = (name, eer, roc)
        rows.append((g, owner, best))

    if not rows:
        return
    print("\n" + "-" * 62)
    print("요약 (제스처별, 각 CV 방식의 최적 모델)")
    print(f"{'제스처':<11}{'등록자':<9}{'혼합: 모델':<13}{'EER':>7}"
          f"   {'타인분리: 모델':<15}{'EER':>7}")
    for g, o, b in rows:
        mix = b.get("혼합")
        sep = b.get("타인분리")
        line = f"{g:<11}{o:<9}{mix[0]:<13}{mix[1]:>7.3f}   "
        line += f"{sep[0]:<15}{sep[1]:>7.3f}" if sep else f"{'-':<15}{'-':>7}"
        print(line)

    for sname in ("혼합", "타인분리"):
        vals = [b[sname][1] for _, _, b in rows if sname in b]
        if vals:
            mean_eer = float(np.mean(vals))
            print(f"\n[{sname}] 평균 EER = {mean_eer:.3f}"
                  f"  ->  '정확도' 환산 약 {1 - mean_eer:.1%}")
    print("\n제안서의 '인식 정확도 90% 이상' 은 [타인분리] 수치로 보고해야 방어 가능합니다.")
    print("([혼합] 은 같은 타인을 외울 수 있어 FAR 이 실제보다 낮게 나옵니다)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset.npz")
    ap.add_argument("--owners", help="CSV: gesture,owner")
    a = ap.parse_args()

    z = np.load(a.data, allow_pickle=True)
    X, gesture, performer = z["X"], list(z["gesture"]), list(z["performer"])
    print(f"데이터 {X.shape[0]}개, 특징 {X.shape[1]}차원\n")

    task_a_gesture(X, gesture, performer)

    owners = {}
    if a.owners:
        with open(a.owners, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                owners[row["gesture"].strip()] = row["owner"].strip()
    else:
        # owners.csv 가 없으면 '해당 제스처를 가장 많이 수행한 사람'을 등록자로 추정
        for g in set(gesture):
            cnt = {}
            for gg, pp in zip(gesture, performer):
                if gg == g:
                    cnt[pp] = cnt.get(pp, 0) + 1
            owners[g] = max(cnt, key=cnt.get)
        print("\n(owners.csv 미지정 -> 최다 수행자를 등록자로 자동 추정)")

    task_b_verification(X, gesture, performer, owners)


if __name__ == "__main__":
    main()
