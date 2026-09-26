# Experiments

이 폴더에는 최종 모델을 선택하기 전에 사용한 비교 실험 코드가 있습니다.

- `evaluate_baseline_model.py`: 초기 Shared Dual Head 평가
- `evaluate_unseen_gesture.py`: 학습에서 제외한 제스처 평가
- `train_hard_negative_model.py`: Hard Negative 학습 실험
- `run_loss_ablation.py`: 손실 함수 조합 비교

최종 학습과 평가는 저장소 루트의 `train_final_model.py`와 `evaluate_final_model.py`를 사용합니다.
