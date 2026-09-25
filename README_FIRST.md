# WITECK AI팀 재현용 — Mobile Shared Dual Head G1~G24 v1.0.0

이 묶음은 모바일 방식 Landmark 추출, D=127 feature 생성, G1~G24 학습 및
Same-Gesture 보안 평가를 AI팀이 확인·재실행하기 위한 것이다.

## 최종 모델

- 구조: Shared two-stream 1D-CNN Backbone + Gesture/User Dual Head
- 학습: `49_train_shared_dual_head_g1_g24.py`
- 평가: `53_evaluate_shared_dual_head_same_attack_pool.py`
- 입력: `[N,32,127]`
- 최종 후보 가중치: `final_result/shared_dual_head.pt`
- 모델 버전: `witeck-mobile-shared-dual-head-g1g24-v1.0.0`

## 실행

```bash
python 49_train_shared_dual_head_g1_g24.py \
  --data dataset_G1_G24_hand_only_2902_stabilized_p10_v100_20260923.npz \
  --output-dir output/retrain_seed42 \
  --epochs 50 \
  --seed 42

python 53_evaluate_shared_dual_head_same_attack_pool.py \
  --data dataset_G1_G24_hand_only_2902_stabilized_p10_v100_20260923.npz \
  --checkpoint output/retrain_seed42/shared_dual_head.pt \
  --output-dir output/retrain_seed42/evaluation \
  --target-users P08,P09,P10 \
  --registered-gesture G5 \
  --attack-pool nontrain
```

CPU/MPS 및 라이브러리 버전에 따라 재학습 결과는 달라질 수 있다. 포함된
가중치는 CPU seed 40~44 중 final test를 보지 않고 User validation EER이 가장
낮은 seed 42를 선택한 것이다.

## 포함 파일 구분

- `09`: 저장 영상에서 모바일 방식 MediaPipe landmark JSON 추출
- `10`: AH650 JSON을 hand-only D=127 NPZ로 변환
- `11`: 기본 병합본 생성
- `12`: collapse-aware 안정화 병합본 생성
- `45`: 모델 구조 및 공통 학습 함수
- `49`: G1~G5 identity + G1~G24 gesture 학습
- `53`: 동일 attack pool 최종 평가
- `08`, `16`, `two_stage_common`: import 의존 코드

## 포함 결과

| 지표 | 결과 |
|---|---:|
| Accuracy | 95.46% |
| Balanced Accuracy | 92.05% |
| Genuine FRR | 11.67% |
| Combined FAR | 4.23% |
| Same-Gesture FAR | 23.98% |
| Wrong-Gesture FAR | 0.00% |
| Random-Impostor FAR | 0.00% |

9/24 별도 실행 보고서의 Accuracy 96.56%, Same-Gesture FAR 15.04% 가중치는
현재 폴더에 전달되지 않았다. 해당 성능을 배포하려면 그 실행에서 생성된
`shared_dual_head.pt` 원본을 별도로 확보해야 한다.
