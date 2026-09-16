# WITECK Hand-only Authentication
## Shared Backbone + Dual Embedding Head

## 1. 프로젝트 목표

WITECK의 최종 목표는 다음과 같다.

> 신규 사용자가 자신만의 자유 제스처를 소수 횟수 등록하면, 해당 사용자의 데이터로 AI 모델을 다시 학습하지 않고도 이후 동일 사용자를 인증하는 시스템을 구현한다.

최종 목표는 단순한 G1~G5 제스처 분류가 아니라 다음 조건을 만족하는 인증이다.

- Unseen User
- Unseen Gesture
- Few-shot Enrollment
- No Retraining

즉 신규 사용자가 학습에 없던 개인 제스처를 등록하더라도 모델의 weight를 변경하지 않고, 등록 시 생성한 embedding template만 저장하여 인증할 수 있어야 한다.

---

# 2. 기존 시스템

기존 파이프라인은 다음과 같았다.

```text
입력 영상
   ↓
MediaPipe Hands
   ↓
Hand-only Feature (D=127)
   ↓
Gesture 1D-CNN
   ↓
G1~G5 분류
   ↓
Two-stream SupCon 1D-CNN
   ↓
사용자 인증
   ↓
Accept / Reject
```

기존 제스처 분류는 높은 routing accuracy를 보였으나, 구조적으로 G1~G5의 closed-set classification을 전제로 한다.

따라서 학습에 존재하지 않는 새로운 제스처 G6가 입력되어도 G6라고 출력할 수 없고, G1~G5 중 하나로 강제 분류된다.

이 구조는 최종 목표인 자유 제스처 등록과 직접적으로 맞지 않는다.

---

# 3. 기존 사용자 인증 Baseline

기존 Hand-only 사용자 인증 모델은 다음 구조를 사용한다.

- Feature dimension: D=127
- Position: 63
- Velocity: 63
- Valid mask: 1
- Architecture: Two-stream 1D-CNN
- Pooling: per-stream mean + std
- Embedding dimension: 128
- Loss: Cross Entropy + Gesture-conditioned SupCon

관련 파일:

```text
41_train_hand_only_supcon.py
44_evaluate_direct_template_auth.py
```

기존 SupCon의 핵심 pair rule은 다음과 같다.

```text
Positive:
same_user AND same_gesture

Eligible comparison:
same_gesture
```

즉 다음 관계는 학습된다.

```text
P01-G1 ↔ P01-G1
→ Positive

P01-G1 ↔ P02-G1
→ Negative
```

그러나 다음 관계는 직접적인 비교 대상이 아니다.

```text
P01-G1 ↔ P01-G2
```

따라서 동일 사용자가 다른 제스처를 수행했을 때 이를 충분히 거부하지 못하는 문제가 발생할 수 있다.

---

# 4. Direct Template 평가에서 확인된 문제

`44_evaluate_direct_template_auth.py`는 제스처 분류 모델을 사용하지 않고, `userId + gestureId`가 이미 주어진 상태에서 해당 template을 선택하여 사용자 인증을 수행한다.

구조:

```text
userId + gestureId
        ↓
등록 Template 선택
        ↓
사용자 인증 Encoder
        ↓
Embedding
        ↓
Cosine Similarity
        ↓
Threshold
        ↓
Accept / Reject
```

## Threshold 0.327836

```text
Accuracy                    63.24%
Balanced Accuracy           69.86%
Combined FAR                38.67%
FRR                         21.62%
Wrong-Gesture FAR           58.70%
Same-Gesture Impostor FAR   18.21%
```

## Threshold 0.627516

```text
Accuracy                    74.76%
Balanced Accuracy           69.11%
Combined FAR                23.60%
FRR                         38.18%
Wrong-Gesture FAR           37.84%
Same-Gesture Impostor FAR    9.06%
```

Threshold를 높이면 FAR은 감소하지만 FRR이 크게 증가했다.

따라서 threshold 조절만으로는 본인이 다른 제스처를 수행했을 때 통과하는 문제를 충분히 해결하기 어렵다고 판단했다.

---

# 5. 새 구조: Shared Backbone + Dual Embedding Head

현재 새로 실험 중인 구조는 다음과 같다.

```text
영상
 ↓
MediaPipe Hands
 ↓
Hand-only D=127
 ↓
Two-stream 1D-CNN
Shared Backbone
       ↓
 ┌─────┴─────┐
 ↓           ↓
Gesture     User
Head        Head
 ↓           ↓
Gesture     User
Embedding   Embedding
```

하나의 backbone이 공통 특징을 추출하고 마지막 embedding head만 두 개로 분리한다.

## Gesture Head

목표:

> 현재 수행한 동작이 등록된 제스처와 같은가?

학습:

```text
같은 gesture id
→ embedding을 가깝게

다른 gesture id
→ embedding을 멀게
```

## User Head

목표:

> 현재 수행자가 등록된 사용자와 같은가?

학습:

```text
같은 user
→ embedding을 가깝게

다른 user
→ embedding을 멀게
```

보조적으로 closed-set User Cross Entropy를 사용하지만, 이는 학습 안정화를 위한 보조 loss이며 신규 사용자 등록이나 실제 인증 시 사용하지 않는다.

최종 loss:

```text
CE(User)
+ 0.2 × User SupCon
+ 0.2 × Gesture SupCon
```

---

# 6. 최종 인증 방식

등록 시 사용자는 개인 제스처를 약 3회 수행한다.

```text
Take 1
Take 2
Take 3
   ↓
Shared Backbone
   ↓
Gesture Embedding 3개
User Embedding 3개
   ↓
각각 평균
   ↓
Gesture Template
User Template
```

신규 사용자 등록 시 모델 재학습은 수행하지 않는다.

인증 시:

```text
입력
 ↓
Shared Backbone
 ↓
Gesture Embedding
User Embedding
 ↓
등록 Template과 각각 비교
 ↓
Gesture score
User score
```

최종 판정:

```text
Gesture score >= Tg
AND
User score >= Tu

→ Accept
```

둘 중 하나라도 threshold를 통과하지 못하면 Reject한다.

---

# 7. 새 코드

관련 파일:

```text
45_train_shared_dual_head.py
46_evaluate_shared_dual_head.py
```

기존 코드는 비교를 위한 Baseline으로 보존한다.

```text
41_train_hand_only_supcon.py
44_evaluate_direct_template_auth.py
```

백업:

```text
16_train_supcon_embedding_BASELINE.py
41_train_hand_only_supcon_BASELINE.py
44_evaluate_direct_template_auth_BASELINE.py
```

---

# 8. Shared Dual-Head 학습 실행

## Smoke Test

먼저 짧은 실행으로 코드 동작을 확인한다.

```bat
python 45_train_shared_dual_head.py --epochs 3 --seed 40 --output-dir output\shared_dual_head_smoke\seed_40
```

확인 항목:

- UserSupCon이 계산되는가
- GestureSupCon이 계산되는가
- 두 embedding head가 정상적으로 생성되는가
- Gesture threshold Tg가 생성되는가
- User threshold Tu가 생성되는가

---

# 9. Shared Dual-Head 본 학습

```bat
python 45_train_shared_dual_head.py --epochs 50 --seed 40 --output-dir output\shared_dual_head\seed_40
```

실제 seed 40 결과:

```text
Early stopping epoch = 32

Gesture threshold = 0.902031779
Gesture validation EER = 0.00%
Gesture validation FAR = 0.00%
Gesture validation FRR = 0.00%

User threshold = 0.342350125
User validation EER = 4.17%
User validation FAR = 4.05%
User validation FRR = 4.29%
```

주의:

Gesture validation EER 0%는 현재 G1~G5라는 고정된 기존 제스처에서 나온 결과이다.

이 값은 아직 학습에 없던 자유 제스처에서도 0%라는 의미가 아니다.

---

# 10. Shared Dual-Head 평가 실행

```bat
python 46_evaluate_shared_dual_head.py
```

기본 설정:

```text
Dataset:
dataset\dataset_1955_recent8_updated_20260905_hand_only.npz

Checkpoint:
output\shared_dual_head\seed_40\shared_dual_head.pt

Target users:
P08, P09, P10

Enrollment:
각 제스처 3개

Gesture threshold:
0.902031779

User threshold:
0.342350125
```

---

# 11. 평가 항목

`46_evaluate_shared_dual_head.py`는 다음 네 가지를 분리해서 평가한다.

## Genuine

```text
같은 사용자
+
등록한 제스처
```

기대:

```text
Accept
```

## Wrong Gesture

```text
같은 사용자
+
다른 제스처
```

기대:

```text
Gesture Head → Reject
```

## Same-Gesture Impostor

```text
다른 사용자
+
등록 제스처와 같은 제스처
```

기대:

```text
User Head → Reject
```

## Random Impostor

```text
다른 사용자
+
다른 제스처
```

기대:

```text
Reject
```

추후 자유 제스처 데이터에서는 Imitation Attack도 별도 평가할 예정이다.

---

# 12. Shared Dual-Head 현재 평가 결과

seed 40 / P08~P10 기준:

```text
Trials                      7,275
Accuracy                    95.77%
Balanced Accuracy           84.37%

Genuine FRR                 28.04%
Combined Attack FAR          3.22%

Wrong-Gesture FAR            1.52%
Same-Gesture Impostor FAR   17.34%
Random Impostor FAR          0.13%
```

Head별 pass rate:

## Genuine

```text
Gesture Head pass rate = 88.85%
User Head pass rate    = 77.70%
```

## Wrong Gesture

```text
Gesture Head pass rate = 1.69%
User Head pass rate    = 58.70%
```

즉 동일 사용자가 다른 제스처를 수행하더라도 User Head는 통과할 수 있지만 Gesture Head가 대부분 거부한다.

## Same-Gesture Impostor

```text
Gesture Head pass rate = 84.47%
User Head pass rate    = 20.79%
```

즉 타인이 같은 제스처를 수행하면 Gesture Head는 통과할 수 있지만 User Head가 대부분 거부한다.

---

# 13. 기존 Direct Template과 비교

| Metric | Direct Template | Shared Dual-Head |
|---|---:|---:|
| Wrong-Gesture FAR | 37.84~58.70% | 1.52% |
| Same-Gesture Impostor FAR | 9.06~18.21% | 17.34% |
| Genuine FRR | 21.62~38.18% | 28.04% |
| Combined Attack FAR | 23.60~38.67% | 3.22% |
| Balanced Accuracy | 약 69% | 84.37% |
| Accuracy | 63~75% | 95.77% |

가장 큰 개선은 Wrong-Gesture FAR이다.

기존 Direct Template에서는 본인이 다른 제스처를 수행해도 많은 경우 통과했지만, Shared Dual-Head에서는 약 1.52%까지 감소했다.

다만 Same-Gesture Impostor FAR 17.34%와 Genuine FRR 28.04%는 추가 개선이 필요하다.

현재 주요 병목은 Gesture Head보다 User Head에 가깝다.

---

# 14. Accuracy 해석 주의

현재 Accuracy 95.77%만을 대표 성능으로 사용하면 안 된다.

전체 trial에는 Random Impostor가 많이 포함되어 있고, Random Impostor FAR이 매우 낮기 때문에 Accuracy가 높아질 수 있다.

따라서 현재 인증 성능은 다음 지표를 함께 확인해야 한다.

- Balanced Accuracy
- Genuine FRR
- Wrong-Gesture FAR
- Same-Gesture Impostor FAR
- Random Impostor FAR
- Combined Attack FAR
- 향후 EER
- 향후 Imitation FAR

---

# 15. 현재 데이터셋의 한계

현재 최종 데이터셋은 1,955개이며 총 28명이 참여했다.

현재 문제는 사람 수 부족이 아니라 제스처 다양성이다.

기존 데이터는 기본적으로 G1~G5라는 정해진 gesture vocabulary를 사용한다.

따라서 현재 결과로 검증할 수 있는 것은:

```text
Unseen User
+
Known Gesture(G1~G5)
```

이다.

하지만 최종 프로젝트 목표는:

```text
Unseen User
+
Unseen Free Gesture
+
Few-shot Enrollment
+
No Retraining
```

이다.

따라서 자유 제스처 데이터를 추가로 수집해야 한다.

---

# 16. 자유 제스처 데이터 수집 계획

현재 10명이 3일 동안 촬영 가능하다고 가정한다.

## Development 7명

구성 예:

```text
P01~P05
+
신규 Development 2명
```

각 사용자:

```text
개인 자유 제스처 3개
× 하루 4회
× 3일
```

총:

```text
7 × 3 × 4 × 3
= 252개
```

기존 P01~P05가 자유 제스처 데이터 촬영에 다시 참여해도 된다.

단, 이들은 Development 용도로만 사용한다.

## Final Test 3명

기존 모델 학습에 전혀 참여하지 않은 완전 신규 사용자.

각 사용자:

```text
개인 자유 제스처 1개
× 하루 5회
× 3일
```

총:

```text
3 × 1 × 5 × 3
= 45개
```

Final Test 사용자의 데이터는 Training, Validation, threshold 결정에 사용하지 않는다.

---

# 17. 자유 제스처 ID 규칙

새 자유 제스처 데이터의 gesture ID는 전역적으로 고유하게 저장해야 한다.

좋은 예:

```text
D01_PG1
D01_PG2
D01_PG3

D02_PG1
D02_PG2
D02_PG3
```

나쁜 예:

```text
D01 → PG1
D02 → PG1
D03 → PG1
```

서로 다른 사용자가 만든 서로 다른 자유 제스처를 모두 동일한 `PG1`로 저장하면 Gesture Head가 서로 다른 동작을 같은 gesture class로 학습하게 된다.

---

# 18. Final Test 등록 방식

예: T01

Day 1:

```text
take1
take2
take3
→ Enrollment Template 생성
```

모델 재학습 없음.

나머지:

```text
Day1 take4~5
Day2 take1~5
Day3 take1~5
```

→ Genuine Test

즉 최종 평가 조건은 다음과 같다.

```text
Unseen User
+
Unseen Gesture
+
3-shot Enrollment
+
No Retraining
```

---

# 19. Imitation Attack

추후 Final Test에서는 타인이 등록 사용자의 자유 제스처를 보고 따라하는 공격도 수집한다.

예:

```text
T01 개인 제스처
      ↓
T02가 따라함
T03가 따라함
```

이때 기대 결과:

```text
Gesture Head
→ 통과할 수 있음

User Head
→ 거부되어야 함

Final
→ Reject
```

최종 평가에서는 Imitation FAR을 별도로 측정한다.

---

# 20. 앞으로의 작업 순서

```text
1. 기존 코드 보존
   ✅ 완료

2. Shared Dual-Head 코드 작성
   ✅ 완료

3. 기존 G1~G5 데이터 Smoke Test
   ✅ 완료

4. seed 40 본 학습
   ✅ 완료

5. P08~P10 Dual-Head 평가
   ✅ 완료

6. 기존 Direct Template과 비교
   ✅ 완료

7. 자유 제스처 데이터 3일간 수집
   ⬅ 다음 단계

8. 자유 제스처 Development 데이터 전처리

9. 기존 1,955개
   +
   자유 제스처 Development 데이터
   ↓
   Shared Dual-Head 재학습

10. Development / Validation에서
    Gesture threshold Tg
    User threshold Tu
    재결정

11. 모델 고정

12. 신규 T01~T03
    자유 제스처 3회 등록

13. 모델 재학습 없이 Final Test

14. 평가
    - Genuine FRR
    - Wrong-Gesture FAR
    - Same-Gesture Impostor FAR
    - Random Impostor FAR
    - Imitation FAR
    - FAR
    - FRR
    - EER
    - Balanced Accuracy

15. 필요 시 User Head 개선
    - Hard Negative Mining
    - same-gesture impostor 강화
    - imitation negative 강화
    - SupCon weight 조정
    - 다른 metric-learning loss 비교
```

---

# 21. 최종 시스템 목표

최종적으로 사용자가 경험하는 시스템은 다음과 같다.

## 신규 등록

```text
회원가입 / userId 생성
 ↓
자신만의 자유 제스처 3회 수행
 ↓
Gesture Template 저장
User Template 저장
 ↓
등록 완료

모델 재학습 없음
```

## 인증

```text
사용자가 자신의 userId 선택 / 로그인
 ↓
개인 제스처 수행
 ↓
Shared Dual-Head Encoder
 ↓
Gesture score
+
User score
 ↓
둘 다 threshold 통과
 ↓
인증 성공
```

즉 최종적으로 WITECK이 증명해야 할 핵심은 다음이다.

> AI 모델 학습에 참여하지 않은 신규 사용자가 AI가 학습한 적 없는 자신만의 자유 제스처를 소수 횟수 등록해도, 모델을 다시 학습하지 않고 이후 동일 사용자를 인증할 수 있다.

현재 Shared Dual-Head 실험은 기존 G1~G5 환경에서 이 구조의 가능성을 확인한 단계이며, 다음 핵심 단계는 자유 제스처 데이터셋을 이용한 Unseen User + Unseen Gesture 검증이다.

Backend Release v1.1.1

현재 백엔드 연동용 최종 배포 버전은 다음과 같다.

shared-dual-head-v1.1.1

최종 전달 파일:

WITECK_AI_RELEASE_SHARED_DUAL_HEAD_v1.1.1.zip

이 배포본은 기존 v1.0.0의 백엔드 인터페이스를 최대한 유지하면서 내부 인증 모델을 Shared Dual-Head 구조로 교체한 버전이다.

포함 파일
ai_release/
├── __init__.py
├── encoder.py
├── features.py
├── model_defs.py
├── preprocess.json
├── thresholds.json
├── calibration_report.md
├── manifest.json
├── requirements.txt
├── smoke_test.py
└── weights/
    └── shared_dual_head_v1.pt
Backend 공개 API

encoder.py에서 다음 API를 제공한다.

load_model()

embed_user()
embed_gesture()

embed_user_batch()
embed_gesture_batch()

embed_both()
embed_both_batch()

MODEL_VERSION

Embedding dimension:

Gesture embedding = 128-D
User embedding    = 128-D

두 embedding 모두 L2 normalization된 상태로 반환된다.

따라서 백엔드에서는 등록 template과 cosine similarity를 계산하여 사용할 수 있다.

입력 전처리

새 릴리스에는 기존 v1.0.0에서 사용하던 raw MediaPipe landmark 전처리 front-end를 다시 포함했다.

입력:

raw MediaPipe hand landmarks
+
timestamp
+
camera width / height
+
handedness

전처리 과정:

Raw landmark payload
↓
입력 검증
↓
21개 hand landmark parsing
↓
누락 frame 보간
↓
aspect ratio 보정
↓
wrist-relative coordinate 변환
↓
hand-size normalization
↓
velocity 계산
↓
32-frame resampling
↓
[32,127]

최종 feature layout:

hand_xyz       63
hand_velocity  63
valid_mask      1
-----------------
total          127
입력 검증

잘못된 입력이 모델에 그대로 들어가지 않도록 InvalidSequenceError 검증을 포함한다.

주요 검증 항목:

최소 입력 frame 수
최소 valid frame 수
최소 촬영 시간 750 ms
camera width / height 존재 여부
timestamp 유효성
timestamp 증가 순서
landmark shape 및 finite 값
Right hand 여부

따라서 0으로 채워진 임의 tensor를 직접 모델에 넣는 구조가 아니라, raw payload 단계에서 유효성 검사를 수행한다.

Threading

기존 백엔드 계약을 유지하기 위해 AI 모듈 내부에:

threading.RLock

을 사용한다.

모델 load 및 forward 과정이 내부 lock으로 보호되므로, 백엔드에서 별도의 model-forward lock을 추가할 필요는 없다.

Operating Points
Default

기본 운영점:

Gesture threshold Tg = 0.902032
User threshold Tu    = 0.342350

최종 판정:

gesture_score >= Tg
AND
user_score >= Tu

→ Accept

Validation 기준:

Gesture FAR = 0.00%
Gesture FRR = 0.00%

User FAR = 4.05%
User FRR = 4.29%
Demo Relaxed

시연 중 genuine rejection을 줄이기 위한 완화 운영점도 추가했다.

Gesture threshold Tg = 0.902032
User threshold Tu    = 0.293309

Validation 기준:

Gesture FAR = 0.00%
Gesture FRR = 0.00%

User FAR = 4.76%
User FRR = 2.86%

Gesture Head는 기본 threshold에서도 validation FAR/FRR이 모두 0%였기 때문에 threshold를 변경하지 않았다.

User Head만:

0.342350
↓
0.293309

로 완화했다.

이 운영점은:

validation FAR <= 5%

조건을 유지하면서 validation FRR을 최대한 낮추는 기준으로 산출했다.

중요:

P08~P10 final-test 결과는 threshold 산출에 사용하지 않았다.

따라서 demo_relaxed는 known-user validation만을 이용해 calibration한 운영점이다.

Current Evaluation Result

P08~P10 unseen user 기준 평가:

Enrollment = 3 takes
Trials     = 7,275

결과:

Accuracy                    95.77%
Balanced Accuracy           84.37%

Genuine FRR                 28.04%
Combined Attack FAR          3.22%

Wrong-Gesture FAR            1.52%
Same-Gesture Impostor FAR   17.34%
Random Impostor FAR          0.13%

현재 결과에서 가장 큰 개선은 Wrong-Gesture FAR이다.

기존 Direct Template:

37.84% ~ 58.70%

Shared Dual-Head:

1.52%

로 크게 감소했다.

다만 현재 주요 개선 대상은:

Genuine FRR                 28.04%
Same-Gesture Impostor FAR   17.34%

이다.

Backend Authentication Flow

현재 1인 1개 개인 제스처 정책에서는 앱에서 gestureId를 보낼 필요가 없다.

인증 흐름:

App
userId + raw landmark payload
↓
Backend
userId 기준 template 조회
↓
AI Release
features.py
↓
[32,127]
↓
Shared Dual-Head
↓
Gesture embedding [128]
+
User embedding [128]
↓
Backend
cosine similarity 계산
↓
Gesture threshold
+
User threshold
↓
최종 인증

사용자별 저장 데이터 예:

userId
├── gesture_template [128]
└── user_template    [128]
Enrollment

신규 사용자 등록 시 약 3회의 제스처를 촬영한다.

Take 1
Take 2
Take 3
↓
Gesture embedding 3개
User embedding 3개
↓
각각 평균
↓
Gesture Template
User Template 저장

신규 사용자 등록 과정에서는:

모델 재학습 X
Backpropagation X
Weight update X

이며 template만 생성한다.

Release Integrity

최종 릴리스는 finalize_release_v1_1_1.py로 생성한다.

빌드 순서:

checkpoint 복사
↓
preprocess 통계 반영
↓
threshold 반영
↓
smoke test
↓
__pycache__ / *.pyc 삭제
↓
manifest hash 생성
↓
ZIP 생성

__pycache__ 및 .pyc 파일은 최종 ZIP과 manifest에서 제외된다.

이를 통해 smoke test 이후 생성된 Python bytecode 때문에 manifest hash 검증이 실패하는 문제를 방지한다.

Current Limitation

현재 shared_dual_head_v1.pt checkpoint는 아직 기존:

G1
G2
G3
G4
G5

제스처 데이터를 기반으로 학습되어 있다.

따라서 현재 결과는:

Unseen User
+
Known Gesture

조건을 평가한 결과이다.

아직 최종 목표인:

Unseen User
+
Unseen Free Gesture
+
Few-shot Enrollment
+
No Retraining

을 완전히 검증한 상태는 아니다.

Next Retraining Stage

다음 단계에서는 자유 제스처 데이터를 추가 수집한다.

예정 흐름:

자유 제스처 데이터 수집
↓
전처리 및 dataset 구성
↓
기존 데이터 + Development 자유 제스처 데이터
↓
Shared Dual-Head 재학습
↓
Validation에서 Tg / Tu 재calibration
↓
새 ai_release 생성
↓
완전 신규 사용자 Final Test

자유 제스처 모델로 재학습한 뒤에도 백엔드 공개 API 구조는 최대한 동일하게 유지할 예정이다.

즉:

embed_user()
embed_gesture()
embed_user_batch()
embed_gesture_batch()

와:

128-D embedding
cosine similarity
Tg / Tu

구조는 유지하고, 이후 릴리스에서는 주로 다음 항목이 교체된다.

weights
preprocess normalization statistics
thresholds
calibration report

따라서 백엔드는 현재 v1.1.1 기준으로 연동을 진행하고, 이후 자유 제스처 모델이 확정되면 새로운 release 파일로 교체하는 방식으로 진행할 수 있다.
