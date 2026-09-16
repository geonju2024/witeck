# WITECK 파일 안내 — 여기부터 확인

## 1. 백엔드팀에 전달할 것

다른 파일을 찾을 필요 없이 다음 ZIP 하나만 전달합니다.

```text
FINAL_DELIVERY/WITECK_AI_BACKEND_FINAL_v1.0.0.zip
```

세부 전달 안내는 `FINAL_DELIVERY/README_FIRST.md`에 있습니다.

## 2. 현재 최종 학습·평가 코드

| 파일 | 역할 |
|---|---|
| `02_build_features.py` | raw landmark 전처리; `--hand-only` 사용 시 D=127 |
| `13_train_gesture_embedding_1dcnn.py` | 최종 제스처 1D-CNN embedding 학습 |
| `14_evaluate_embedding_end_to_end.py` | 제스처→인증 End-to-End 평가 |
| `39_make_hand_only_dataset.py` | 기존 D=169 NPZ에서 pose를 제거해 D=127 생성 |
| `41_train_hand_only_supcon.py` | 최종 손 전용 Two-stream SupCon 인증 모델 |
| `42_run_hand_only_tailored_five_seed.py` | 최종 인증 모델 5-seed 평가 |
| `43_prepare_hand_only_release_metrics.py` | threshold·등록 1/3/4회·배포 설정 생성 |

## 3. 현재 최종 데이터와 결과

| 경로 | 의미 |
|---|---|
| `dataset/dataset_1955_recent8_updated_20260905_hand_only.npz` | 최종 D=127 hand-only 데이터셋 |
| `output/hand_only_five_seed/` | hand-only 제스처 및 일반 SupCon 비교 결과 |
| `output/hand_only_tailored_five_seed/` | 최종 손 전용 인증 모델 5-seed 결과 |
| `output/hand_only_release_candidate/` | 선택 checkpoint End-to-End 결과 |

## 4. 공통 코드 — 삭제하거나 이동하지 말 것

- `08_train_embedding.py`
- `16_train_supcon_embedding.py`
- `two_stage_common.py`
- `split_protocol.py`
- `paths.py`

최종 학습 코드가 위 파일의 평가·학습 함수를 import합니다.

## 5. 이전 실험 파일

이전 모델 실험은 `archive/experiments/`, 이전 결과는 `archive/outputs/`, 보조
도구는 `archive/tools/`, 이전 배포본과 개발용 배포 폴더는
`archive/releases/`로 이동했습니다. 최종 백엔드 배포에는 사용하지 않습니다.
과거 실험을 다시 실행해야 한다면 프로젝트 루트에서 다음처럼 실행합니다.

```bash
PYTHONPATH=. python archive/experiments/<실험파일>.py
```

## 6. 이름이 비슷한 배포 파일 구분

| 경로 | 용도 | 백엔드 전달 여부 |
|---|---|---:|
| `FINAL_DELIVERY/WITECK_AI_BACKEND_FINAL_v1.0.0.zip` | 실제 최종 전달본 | **전달** |
| `archive/releases/backend_delivery_source/ai_release/` | 최종 ZIP의 원본 폴더 | 전달 불필요 |
| `archive/releases/backend_delivery_source/witeck_ai_release_runtime_v1.0.0.zip` | 같은 실행본의 이전 이름 | 전달 불필요 |
| `archive/releases/ai_release_full/` | 보고서까지 포함된 개발용 폴더 | 전달 불필요 |
| `archive/releases/ai_release_handonly_supcon_v1.0.0.zip` | 개발·검증 보관본 | 전달 불필요 |
