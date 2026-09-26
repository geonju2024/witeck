# WITECK Mobile Gesture Authentication

카메라로 손동작을 입력받아 등록한 제스처와 사용자가 모두 일치하는지 확인하는 비접촉 인증 프로젝트입니다.

최종 모델은 손 랜드마크 시계열을 처리하는 Shared Backbone + Dual Head 1D-CNN입니다. Gesture Head는 등록 제스처와의 일치 여부를 확인하고, User Head는 등록 사용자와의 일치 여부를 확인합니다. 두 점수가 모두 기준값 이상일 때만 인증을 허용합니다.

## 최종 모델

- 입력: 32 프레임, 프레임당 127개 특징
- 특징: 손 위치 63개, 속도 63개, 유효 프레임 정보 1개
- 출력: 128차원 Gesture Embedding과 128차원 User Embedding
- 등록: 개인 제스처 3회 촬영 후 두 임베딩의 평균 템플릿 저장
- 인증: Gesture Score와 User Score를 각각 계산한 뒤 두 조건을 모두 만족하면 통과
- 모델 버전: `witeck-mobile-shared-dual-head-g1g24-v1.0.0`

## 최종 성능

선택된 CPU seed 42 모델의 오프라인 최종 평가 결과입니다.

| 지표 | 결과 |
| --- | ---: |
| Accuracy | 95.46% |
| Balanced Accuracy | 92.05% |
| Genuine FRR | 11.67% |
| Combined FAR | 4.23% |
| Same-Gesture Impostor FAR | 23.98% |

이 수치는 구축 데이터셋을 이용한 오프라인 평가 결과입니다. 실제 모바일 환경에서 신규 사용자와 자유 제스처를 등록한 성능은 별도의 현장 검증이 필요합니다.

## 폴더 구성

```text
witeck-main/
├── data/
│   └── processed/                   최종 학습 데이터
├── experiments/                     이전 비교 실험
├── deployment/                       백엔드 연동 패키지
├── release_packages/                 팀 전달용 압축파일
├── output/                           학습 가중치와 평가 결과
├── extract_mobile_landmarks.py       모바일 방식 랜드마크 추출
├── build_mobile_hand_features.py     32 x 127 특징 생성
├── merge_gesture_datasets.py         G1-G5와 G6-G24 데이터 병합
├── stabilize_gesture_dataset.py      이상값 안정화 및 최종 데이터 생성
├── train_final_model.py              최종 모델 학습
├── evaluate_final_model.py           최종 보안 성능 평가
├── shared_dual_head_model.py         모델 구조와 공통 학습 함수
├── train_embedding_baseline.py       기본 임베딩 학습 의존 코드
├── train_supcon_embedding.py         SupCon 학습 의존 코드
├── model_utils.py                    데이터와 평가 공통 함수
└── requirements.txt
```

## 실행 방법

저장소 루트에서 다음 명령을 실행합니다.

```bash
python -m pip install -r requirements.txt
python train_final_model.py
python evaluate_final_model.py
```

최종 학습 데이터의 기본 경로는 다음과 같습니다.

```text
data/processed/witeck_g1_g24_mobile_v1.npz
```

학습 결과는 `output/final_model`에 저장되고, 평가는 해당 가중치와 학습 과정에서 정한 threshold를 그대로 사용합니다. 평가 데이터로 threshold를 다시 조정하면 안 됩니다.

## 백엔드 연동

백엔드에서 실제로 사용하는 패키지는 다음 폴더입니다.

```text
deployment/backend/
```

이 폴더에는 모델 로딩, 모바일 랜드마크 전처리, 임베딩 생성, 등록, 인증 함수와 최종 가중치가 포함되어 있습니다. 백엔드는 사용자마다 Gesture Template과 User Template을 각각 저장해야 합니다.

## 주의사항

- 원본 영상은 인물 식별 가능성과 용량 문제 때문에 저장소에 올리지 않습니다.
- 새로운 개인 제스처의 ID는 사용자별로 고유하게 지정해야 합니다.
- 모델 가중치가 바뀌면 모델 버전을 올리고 기존 등록 랜드마크에서 임베딩을 다시 생성해야 합니다.
- Same-Gesture Impostor FAR은 아직 개선이 필요한 보안 지표입니다.
