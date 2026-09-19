# WITECK Generalization & Security Evaluation

## 1. 실험 목적

본 브랜치는 WITECK 손 제스처 인증 시스템의 **실험 환경 성능과 실제 서비스 환경 성능 사이의 차이**를 분석하기 위한 학습 및 평가 코드를 정리한 브랜치이다.

기존 WITECK AI 모델은 주로 G1~G5의 5가지 제스처 데이터를 이용하여 학습 및 평가하였다.

그러나 실제 서비스에서는 신규 사용자가 기존 G1~G5 중 하나를 사용하는 것이 아니라, **자신만의 새로운 제스처를 등록하여 인증에 사용하는 환경**을 목표로 한다.

따라서 기존 데이터셋에서 높은 성능을 얻었다고 하더라도 다음 상황에서도 동일한 성능이 유지된다고 볼 수 없다.

* 학습에 참여하지 않은 신규 사용자
* 학습 데이터에 존재하지 않는 신규 제스처
* 타인이 등록 사용자의 제스처를 따라 하는 공격

본 브랜치에서는 이러한 실제 서비스 환경에 가까운 조건에서 모델의 일반화 및 보안 성능을 확인하는 것을 목적으로 한다.

---

## 2. 현재 모델 구조

현재 모델은 Hand Landmark 기반 입력을 사용하는 **Shared Backbone + Dual Head 구조**이다.

```text
Hand Landmark Sequence
        │
        ▼
 Shared Backbone
        │
 ┌──────┴──────┐
 ▼             ▼
Gesture Head   User Head
 │             │
 ▼             ▼
Gesture        User
Embedding      Embedding
```

하나의 Backbone에서 특징을 추출한 뒤,

* Gesture Head는 수행한 제스처가 등록된 제스처와 같은지 확인하고
* User Head는 해당 동작을 수행한 사람이 등록된 사용자와 같은지 확인한다.

최종 인증은 다음 조건을 모두 만족하는 경우에만 PASS한다.

```text
Gesture Score >= Gesture Threshold
AND
User Score >= User Threshold
```

즉, 사람만 동일하거나 제스처만 동일한 경우에는 인증을 통과시키지 않는 것을 목표로 한다.

---

## 3. 왜 추가 평가가 필요한가

### 3.1 제한된 제스처 데이터

현재 학습 데이터는 주로 G1~G5의 5가지 제스처로 구성되어 있다.

따라서 모델이 해당 제스처에서는 높은 성능을 보이더라도, 실제 서비스에서 사용자가 자유롭게 만든 새로운 제스처에서도 동일한 성능을 보장할 수 없다.

---

### 3.2 신규 사용자 + 신규 제스처

실제 서비스에서는 다음 두 조건이 동시에 발생한다.

```text
Unseen User
+
Unseen Gesture
```

기존 Train/Validation/Test가 동일한 제스처 종류를 공유하는 환경이라면 이러한 실제 서비스 상황에 대한 일반화 성능을 충분히 확인하기 어렵다.

따라서 신규 사용자 및 신규 제스처 환경을 별도로 평가할 필요가 있다.

---

### 3.3 보안 성능

전체 Accuracy뿐만 아니라 잘못된 사용자가 인증되는 False Acceptance를 별도로 확인해야 한다.

특히 다음 네 상황을 구분하여 평가한다.

| 평가 상황               | 기대 결과 |
| ------------------- | ----- |
| 본인 + 정상 제스처         | PASS  |
| 본인 + 다른 Known 제스처   | FAIL  |
| 본인 + 학습되지 않은 다른 제스처 | FAIL  |
| 타인 + 동일 제스처         | FAIL  |

이 중 **타인 + 동일 제스처(Same-Gesture Impostor)** 상황의 FAR은 실제 인증 시스템의 보안 성능을 확인하기 위한 중요한 지표이다.

---

## 4. 파일 설명

### `45_train_shared_dual_head.py`

Shared Backbone + Dual Head 모델의 기본 학습 코드이다.

Gesture Embedding과 User Embedding을 하나의 Backbone에서 동시에 학습한다.

---

### `45_train_shared_dual_head_pseudoval.py`

Pseudo-Unseen Validation 실험을 위한 학습 코드이다.

기존 단순 Random Split보다 실제 신규 사용자 환경에 가까운 조건에서 일반화 성능을 확인하기 위해 사용한다.

Seed를 변경하여 여러 번 학습한 뒤 결과 변동도 함께 확인할 수 있다.

---

### `46_evaluate_shared_dual_head.py`

Shared Dual Head 모델의 기본 인증 성능 평가 코드이다.

Gesture Score와 User Score를 각각 계산하고 두 Threshold를 모두 만족하는지 확인한다.

주요 평가 대상은 다음과 같다.

```text
Genuine
Wrong Gesture
Same-Gesture Impostor
Random Impostor
```

---

### `46_evaluate_shared_dual_head_unseen_g5.py`

실제 서비스 환경에 가까운 일반화 성능을 확인하기 위한 평가 코드이다.

특히 기존 학습 데이터와 다른 조건에서 Gesture/User Head가 얼마나 안정적으로 동작하는지 확인한다.

본 브랜치에서 가장 중요한 평가 코드 중 하나이다.

---

### `baseline/44_evaluate_direct_template_auth.py`

기존 Direct Template 인증 방식과 현재 Dual Head 방식의 차이를 확인하기 위한 비교용 코드이다.

기존 방식에서는 같은 사용자가 다른 제스처를 수행했을 때 User Similarity가 높아 인증될 가능성이 존재하였다.

이 문제를 해결하기 위해 현재 구조에서는 Gesture와 User를 별도의 Head로 평가한다.

---

## 5. 사용 데이터

현재 실험에서는 Hand Landmark 기반 데이터셋을 사용한다.

대표 데이터셋:

```text
dataset_1955_recent8_updated_20260905_hand_only.npz
```

입력 특징 차원:

```text
D = 127
```

본 브랜치에는 데이터셋 자체를 중복 저장하지 않을 수 있으며, 데이터셋이 위치한 경로를 코드 실행 시 지정하여 사용한다.

---

## 6. 주요 평가 지표

본 실험에서는 단순 Accuracy뿐만 아니라 인증 시스템의 특성을 고려하여 다음 지표를 확인한다.

| 지표                | 의미                 |
| ----------------- | ------------------ |
| Accuracy          | 전체 인증 판단 정확도       |
| FAR               | 타인을 잘못 본인으로 인증한 비율 |
| FRR               | 본인을 잘못 거부한 비율      |
| EER               | FAR과 FRR이 같아지는 지점  |
| Gesture PASS Rate | Gesture Head 통과 비율 |
| User PASS Rate    | User Head 통과 비율    |
| Final PASS Rate   | 두 조건을 모두 만족한 비율    |

보안 관점에서는 특히 **Same-Gesture Impostor FAR**을 중요하게 확인한다.

---

## 7. 실험 시 확인해야 할 테스트

최종적으로 다음 네 가지 상황을 별도로 기록한다.

```text
1. Genuine
   본인 + 등록된 정상 제스처

2. Known Wrong Gesture
   본인 + 다른 기존 제스처

3. OOD Wrong Gesture
   본인 + 학습 데이터에 존재하지 않는 다른 제스처

4. Same-Gesture Impostor
   타인 + 등록 사용자와 동일한 제스처
```

목표 동작은 다음과 같다.

```text
Genuine                 → PASS
Known Wrong Gesture     → FAIL
OOD Wrong Gesture       → FAIL
Same-Gesture Impostor   → FAIL
```

---

## 8. 현재 실험의 한계

현재 데이터셋은 제한된 사용자와 G1~G5 제스처를 중심으로 구축되어 있다.

따라서 현재 결과는 기존 데이터 환경에서의 모델 성능을 나타내며, 실제 서비스에서 사용자가 자유롭게 만든 모든 제스처에 대한 성능을 의미하지 않는다.

향후에는 다음 조건을 포함하는 별도의 데이터 수집 및 평가가 필요하다.

```text
Unseen User
+
Unseen Gesture
+
Same-Gesture Impostor Attack
```

특히 사용자가 직접 만든 자유 제스처 데이터를 추가하여 모델 재학습 없이 등록 및 인증이 가능한지를 검증해야 한다.

---

## 9. 최종 목표

WITECK의 최종 목표는 신규 사용자가 자신만의 제스처를 소수 회 등록한 뒤 모델을 다시 학습하지 않고 바로 인증에 사용할 수 있는 시스템을 구축하는 것이다.

```text
신규 사용자
      │
      ▼
자유 제스처 선택
      │
      ▼
3회 등록
      │
      ▼
Embedding Template 생성
      │
      ▼
모델 재학습 없이 인증
```

따라서 앞으로의 성능 평가는 기존 G1~G5 분류 정확도뿐만 아니라 **신규 사용자와 신규 제스처에 대한 일반화 성능 및 타인 공격에 대한 FAR**을 중심으로 진행한다.
