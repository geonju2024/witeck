# WITECK Gesture Authentication

수어 제스처 기반 비접촉 본인 인증 시스템

## 데이터 구조

```text
G1/
├── P01/ → own
├── P02~P10/ → impostor
└── X01~X18/ → impostor

G2/
├── P02/ → own
└── 나머지 → impostor

G3 → P03 own
G4 → P04 own
G5 → P05 own
```

각 수행자 폴더 아래에는 `session_01`, `session_02` 형식으로 촬영 세션을 구분한다.

## 전체 파이프라인

```text
Video
↓
Label 생성
↓
MediaPipe Hands + Pose
↓
Landmark 추출
↓
좌표 및 시간축 정규화
↓
dataset.npz
↓
ML / 1D CNN
↓
본인 인증
```

## 00_make_labels_template.py

폴더 구조를 기준으로 다음 정보를 자동 생성한다.

```text
gesture
performer
session
role
```

등록자 매핑:

```text
G1 → P01
G2 → P02
G3 → P03
G4 → P04
G5 → P05
```

등록자가 자신의 제스처를 수행하면 `own`, 다른 수행자가 모방하면 `impostor`로 지정한다.

## 01_extract_landmarks.py

원본 영상에서 MediaPipe landmark를 추출한다.

```text
Hands
21 points × x, y, z

Pose
33 points × x, y, z, visibility
```

영상별 landmark를 `.npz`로 저장한다.

## 02_build_features.py

```text
종횡비 보정
↓
손목 기준 위치 정규화
↓
손 크기 정규화
↓
미검출 프레임 보간
↓
좌우 손 정규화
↓
T=32 시간축 리샘플링
↓
속도 특징 계산
```

최종 feature:

```text
Hand xyz        63
Hand velocity   63
Pose            21
Pose velocity   21
Valid mask       1

총 169 features/frame
```

영상 하나의 최종 입력:

```text
32 frames × 169 features
```

## 03_train.py

Machine Learning baseline

```text
Logistic Regression
SVM-RBF
Random Forest
```

주요 과제:

```text
own
vs
impostor
```

평가 지표:

```text
AUC
EER
FAR
FRR
```

## 04_train_1dcnn.py

동일한 시계열 feature를 1D CNN으로 학습한다.

```text
[N, 32, 169]
↓
[N, 169, 32]
↓
Conv1D
↓
own / impostor
```

## 09_train_siamese_embedding.py

사용자 인증 전용 Dilated Siamese 1D CNN이다. 기존 입력 `[N, 32, 169]`를
그대로 받아 128차원 움직임 임베딩을 만들고, 등록 템플릿과 cosine similarity로
본인 여부를 판정한다.

```text
1x1 Conv
↓
Residual Dilated Conv1D (1, 2, 4, 8)
↓
Masked Mean + Std + Max Pooling
↓
128D L2-normalized embedding
↓
Cosine similarity + validation threshold
```

```bash
python 09_train_siamese_embedding.py --data /path/to/dataset.npz
```

등록 영상 개수 1/3/5개를 비교하려면 다음을 실행한다.

```bash
python 10_siamese_enrollment_sweep.py --data /path/to/dataset.npz
```

## 실행 순서

```bash
python paths.py

python 00_make_labels_template.py

python scan_rotation.py

python 01_extract_landmarks.py --workers 4

python 02_build_features.py

python 03_train.py

python 04_train_1dcnn.py
```

## Next

```text
1D CNN
↓
GRU
↓
LSTM
↓
Transformer
```
