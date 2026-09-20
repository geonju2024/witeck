"""Evaluate WITECK Shared-Backbone Dual-Head authentication.

This evaluator DOES NOT tune thresholds on test data.
It loads Tg / Tu from the trained checkpoint and evaluates:

1) Genuine
   same target user + registered gesture, later sessions
2) Wrong gesture
   same target user + different gesture, later sessions
3) Same-gesture impostor
   different user + registered gesture
4) Random impostor
   different user + different gesture

Final accept rule:
    gesture_score >= Tg AND user_score >= Tu

For the current legacy dataset, the target users default to the checkpoint's
unseen_users (P08/P09/P10) and the attack pool is every performer not used for
training, excluding the current target. This naturally includes P08~P10 peers
and X01~X18 when they are present in the dataset.
"""

from __future__ import annotations

import argparse
import csv
import importlib
from pathlib import Path

import numpy as np
import torch


train_mod = importlib.import_module("45_train_shared_dual_head")
legacy = importlib.import_module("16_train_supcon_embedding")
base = importlib.import_module("08_train_embedding")

SharedDualHead1DCNN = train_mod.SharedDualHead1DCNN


def parse_id_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def normalize_template(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x) + 1e-12)


def normalized_scores(
    embeddings: np.ndarray,
    template: np.ndarray,
) -> np.ndarray:
    if len(embeddings) == 0:
        return np.asarray([], dtype=np.float64)

    template = normalize_template(template)
    embeddings = embeddings / (
        np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )
        + 1e-12
    )
    return embeddings @ template


@torch.no_grad()
def extract_dual(
    model,
    X,
    duration,
    device,
    batch_size=128,
):
    if len(X) == 0:
        empty = np.empty(
            (0, model.gesture_head.out_features),
            dtype=np.float32,
        )
        return empty.copy(), empty.copy()

    return train_mod.extract_dual_embeddings(
        model,
        X,
        duration,
        device,
        batch_size=batch_size,
    )


def category_summary(rows: list[dict], category: str) -> dict:
    subset = [r for r in rows if r["category"] == category]

    if not subset:
        return {
            "trials": 0,
            "gesture_pass_rate": float("nan"),
            "user_pass_rate": float("nan"),
            "combined_accept_rate": float("nan"),
            "mean_gesture_score": float("nan"),
            "mean_user_score": float("nan"),
        }

    gesture_pass = np.asarray(
        [int(r["gesture_pass"]) for r in subset],
        dtype=np.float64,
    )
    user_pass = np.asarray(
        [int(r["user_pass"]) for r in subset],
        dtype=np.float64,
    )
    combined = np.asarray(
        [int(r["accepted"]) for r in subset],
        dtype=np.float64,
    )
    gesture_scores = np.asarray(
        [float(r["gesture_score"]) for r in subset],
        dtype=np.float64,
    )
    user_scores = np.asarray(
        [float(r["user_score"]) for r in subset],
        dtype=np.float64,
    )

    return {
        "trials": len(subset),
        "gesture_pass_rate": float(gesture_pass.mean()),
        "user_pass_rate": float(user_pass.mean()),
        "combined_accept_rate": float(combined.mean()),
        "mean_gesture_score": float(gesture_scores.mean()),
        "mean_user_score": float(user_scores.mean()),
    }


def main() -> None:
    project_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default=str(
            project_dir
            / "dataset"
            / "dataset_1955_recent8_updated_20260905_hand_only.npz"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(
            project_dir
            / "output"
            / "shared_dual_head"
            / "seed_40"
            / "shared_dual_head.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            project_dir
            / "output"
            / "shared_dual_head_eval"
            / "seed_40"
        ),
    )
    parser.add_argument("--enroll", type=int, default=None)
    parser.add_argument(
        "--target-users",
        default=None,
        help="Comma-separated override, e.g. P08,P09,P10",
    )
    parser.add_argument("--batch-size", type=int, default=128)

    args = parser.parse_args()

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    _, X_seq, duration, meta, T, D = legacy.load_dataset(args.data)

    seq_mean = np.asarray(checkpoint["sequence_mean"])
    seq_std = np.asarray(checkpoint["sequence_std"])
    dur_mean = checkpoint["duration_mean"]
    dur_std = checkpoint["duration_std"]

    X_norm = legacy.apply_sequence_stats(
        X_seq,
        seq_mean,
        seq_std,
    )
    duration_norm = legacy.apply_duration_stats(
        duration,
        dur_mean,
        dur_std,
    )

    model = SharedDualHead1DCNN(
        checkpoint["input_dim"],
        len(checkpoint["train_users"]),
        checkpoint["embedding_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    gesture_threshold = float(checkpoint["gesture_threshold"])
    user_threshold = float(checkpoint["user_threshold"])

    enrollment_per_gesture = (
        int(args.enroll)
        if args.enroll is not None
        else int(checkpoint.get("enrollment_per_gesture", 3))
    )

    target_users = parse_id_list(args.target_users)
    if target_users is None:
        target_users = list(checkpoint["unseen_users"])

    train_users = set(checkpoint["train_users"])

    performer = np.asarray(meta["performer"])
    gesture = np.asarray(meta["gesture"])
    session = np.asarray(meta["session"])

    all_performers = sorted(np.unique(performer).tolist())

    # Current final-test attack pool:
    # every identity excluded from training.
    nontrain_performers = [
        p
        for p in all_performers
        if p not in train_users
    ]

    rows: list[dict] = []

    print("=" * 78)
    print("WITECK Shared Dual-Head Evaluation")
    print("=" * 78)
    print(f"device={device}")
    print(f"N={len(X_seq)} T={T} D={D}")
    print(f"targets={target_users}")
    print(f"nontrain_attack_pool={nontrain_performers}")
    print(f"enrollment_per_gesture={enrollment_per_gesture}")
    print(f"gesture_threshold={gesture_threshold:.9f}")
    print(f"user_threshold={user_threshold:.9f}")
    print(
        "final_accept = "
        "gesture_score >= Tg AND user_score >= Tu"
    )

    for target_user in target_users:
        target_idx = np.where(
            performer == target_user
        )[0]

        if len(target_idx) == 0:
            print(f"[SKIP] target {target_user}: no samples")
            continue

        target_sessions = sorted(
            np.unique(session[target_idx]).tolist()
        )
        if not target_sessions:
            continue

        target_gestures = ["G5"]

        print()
        print(f"[TARGET] {target_user}")

        for registered_gesture in target_gestures:

            # 이 사용자에게 해당 제스처가 존재하는 세션만 찾기
            gesture_sessions = sorted(
                np.unique(
                    session[
                    (performer == target_user)
                    & (gesture == registered_gesture)
                    ]
                ).tolist()
            )

            enroll_session = None

            # G5 샘플이 enrollment 개수 이상 있는 첫 세션 선택
            for candidate_session in gesture_sessions:
                candidate_idx = np.where(
                    (performer == target_user)
                    & (gesture == registered_gesture)
                    & (session == candidate_session)
                )[0]

                if len(candidate_idx) >= enrollment_per_gesture:
                    enroll_session = candidate_session
                    break

            # 조건을 만족하는 등록 세션이 없으면 skip
            if enroll_session is None:
                print(
                    f"  {registered_gesture}: SKIP "
                    f"(no session has >= "
                    f"{enrollment_per_gesture} enrollment samples)"
                )
                continue

            enrollment_candidates = np.where(
                (performer == target_user)
                & (gesture == registered_gesture)
                & (session == enroll_session)
            )[0]

            print(
                f"  {registered_gesture}: "
                f"enroll_session={enroll_session}, "
                f"enrollment_candidates={len(enrollment_candidates)}"
            )

            enroll_idx = enrollment_candidates[
                :enrollment_per_gesture
            ]

            enroll_gesture_emb, enroll_user_emb = extract_dual(
                model,
                X_norm[enroll_idx],
                duration_norm[enroll_idx],
                device,
                batch_size=args.batch_size,
            )

            gesture_template = enroll_gesture_emb.mean(axis=0)
            user_template = enroll_user_emb.mean(axis=0)

            # 1) Genuine:
            # same user + same registered gesture + later session.
            genuine_idx = np.where(
                (performer == target_user)
                & (gesture == registered_gesture)
                & (session != enroll_session)
            )[0]

            # 2) Wrong gesture:
            # same user + any other gesture + later session.
            wrong_idx = np.where(
                (performer == target_user)
                & (gesture != registered_gesture)
                & (session != enroll_session)
            )[0]

            other_nontrain = [
                p
                for p in train_users
                if p != target_user
            ]

            # 3) Same-gesture impostor:
            # another non-training identity performs the same gesture.
            same_imp_idx = np.where(
                np.isin(performer, other_nontrain)
                & (gesture == registered_gesture)
            )[0]

            # 4) Random impostor:
            # another non-training identity performs a different gesture.
            random_imp_idx = np.where(
                np.isin(performer, other_nontrain)
                & (gesture != registered_gesture)
            )[0]

            categories = [
                ("genuine", genuine_idx),
                ("wrong_gesture", wrong_idx),
                ("same_gesture_impostor", same_imp_idx),
                ("random_impostor", random_imp_idx),
            ]

            for category, idx in categories:
                if len(idx) == 0:
                    continue

                g_emb, u_emb = extract_dual(
                    model,
                    X_norm[idx],
                    duration_norm[idx],
                    device,
                    batch_size=args.batch_size,
                )

                gesture_scores = normalized_scores(
                    g_emb,
                    gesture_template,
                )
                user_scores = normalized_scores(
                    u_emb,
                    user_template,
                )

                gesture_pass = (
                    gesture_scores >= gesture_threshold
                )
                user_pass = (
                    user_scores >= user_threshold
                )
                accepted = gesture_pass & user_pass

                for local_pos, sample_idx in enumerate(idx):
                    rows.append(
                        {
                            "target_user": target_user,
                            "registered_gesture": registered_gesture,
                            "category": category,
                            "sample_index": int(sample_idx),
                            "performer": str(
                                performer[sample_idx]
                            ),
                            "performed_gesture": str(
                                gesture[sample_idx]
                            ),
                            "session": str(
                                session[sample_idx]
                            ),
                            "gesture_score": float(
                                gesture_scores[local_pos]
                            ),
                            "user_score": float(
                                user_scores[local_pos]
                            ),
                            "gesture_pass": int(
                                gesture_pass[local_pos]
                            ),
                            "user_pass": int(
                                user_pass[local_pos]
                            ),
                            "accepted": int(
                                accepted[local_pos]
                            ),
                        }
                    )

    if not rows:
        raise RuntimeError("No evaluation trials were created")

    summaries = {
        category: category_summary(rows, category)
        for category in (
            "genuine",
            "wrong_gesture",
            "same_gesture_impostor",
            "random_impostor",
        )
    }

    genuine_accept = summaries["genuine"][
        "combined_accept_rate"
    ]
    genuine_frr = 1.0 - genuine_accept

    wrong_far = summaries["wrong_gesture"][
        "combined_accept_rate"
    ]
    same_imp_far = summaries["same_gesture_impostor"][
        "combined_accept_rate"
    ]
    random_imp_far = summaries["random_impostor"][
        "combined_accept_rate"
    ]

    attack_rows = [
        r
        for r in rows
        if r["category"] != "genuine"
    ]
    combined_attack_far = float(
        np.mean(
            [int(r["accepted"]) for r in attack_rows]
        )
    )

    # Balanced accuracy over all current attack categories combined.
    balanced_accuracy = (
        genuine_accept
        + (1.0 - combined_attack_far)
    ) / 2.0

    total_correct = sum(
        int(r["accepted"] == 1)
        if r["category"] == "genuine"
        else int(r["accepted"] == 0)
        for r in rows
    )
    accuracy = total_correct / len(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "trials.csv"
    fieldnames = list(rows[0].keys())

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "summary.txt"

    lines = [
        "WITECK Shared Dual-Head Final-Test Evaluation",
        f"checkpoint={Path(args.checkpoint).resolve()}",
        f"data={Path(args.data).resolve()}",
        f"device={device}",
        f"target_users={target_users}",
        f"enrollment_per_gesture={enrollment_per_gesture}",
        f"gesture_threshold={gesture_threshold:.9f}",
        f"user_threshold={user_threshold:.9f}",
        f"registrations_evaluated={len(set((r['target_user'], r['registered_gesture']) for r in rows))}",
        f"trials={len(rows)}",
        "",
        "[Overall]",
        f"accuracy={accuracy:.6f}",
        f"balanced_accuracy={balanced_accuracy:.6f}",
        f"genuine_frr={genuine_frr:.6f}",
        f"combined_attack_far={combined_attack_far:.6f}",
        f"wrong_gesture_far={wrong_far:.6f}",
        f"same_gesture_impostor_far={same_imp_far:.6f}",
        f"random_impostor_far={random_imp_far:.6f}",
        "",
    ]

    for category, info in summaries.items():
        lines.extend(
            [
                f"[{category}]",
                f"trials={info['trials']}",
                f"gesture_pass_rate={info['gesture_pass_rate']:.6f}",
                f"user_pass_rate={info['user_pass_rate']:.6f}",
                f"combined_accept_rate={info['combined_accept_rate']:.6f}",
                f"mean_gesture_score={info['mean_gesture_score']:.6f}",
                f"mean_user_score={info['mean_user_score']:.6f}",
                "",
            ]
        )

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("Evaluation complete")
    print("=" * 78)
    print(f"trials={len(rows)}")
    print(f"accuracy={accuracy:.6f}")
    print(f"balanced_accuracy={balanced_accuracy:.6f}")
    print(f"genuine_FRR={genuine_frr:.6f}")
    print(f"combined_attack_FAR={combined_attack_far:.6f}")
    print(f"wrong_gesture_FAR={wrong_far:.6f}")
    print(f"same_gesture_impostor_FAR={same_imp_far:.6f}")
    print(f"random_impostor_FAR={random_imp_far:.6f}")
    print()
    print(
        "Genuine head pass rates: "
        f"gesture={summaries['genuine']['gesture_pass_rate']:.6f}, "
        f"user={summaries['genuine']['user_pass_rate']:.6f}"
    )
    print(
        "Wrong-gesture head pass rates: "
        f"gesture={summaries['wrong_gesture']['gesture_pass_rate']:.6f}, "
        f"user={summaries['wrong_gesture']['user_pass_rate']:.6f}"
    )
    print(
        "Same-gesture-impostor head pass rates: "
        f"gesture={summaries['same_gesture_impostor']['gesture_pass_rate']:.6f}, "
        f"user={summaries['same_gesture_impostor']['user_pass_rate']:.6f}"
    )
    print(f"Saved trials: {csv_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
