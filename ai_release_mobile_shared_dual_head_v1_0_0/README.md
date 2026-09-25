# WITECK Mobile Shared Dual Head v1.0.0

백엔드 FastAPI 프로세스 안에서 직접 import하는 최종 후보 AI 패키지다.

## 모델 계약

- 입력: 모바일 MediaPipe Hand 21점과 각 프레임의 `tMs`, `frameIndex`
- 전처리 출력: `[32,127]`
- 출력: Gesture embedding 128차원 + User embedding 128차원
- 등록: 동일한 개인 제스처 최소 3회
- 판정: Gesture와 User threshold를 모두 통과해야 성공
- 신규 사용자 등록 시 재학습하지 않는다.

## 서버 시작

```python
from ai_release_mobile_shared_dual_head_v1_0_0.encoder import load_model

load_model("cpu")
```

## 등록

```python
from ai_release_mobile_shared_dual_head_v1_0_0.encoder import enroll

templates = enroll(takes)  # takes: 모바일 캡처 payload 3개 이상
```

DB에는 다음을 저장한다.

- `gesture_template`: float32 128차원
- `user_template`: float32 128차원
- `model_version`
- 재색인을 위한 원본 landmark payload

평균 template은 `enroll()` 내부에서 다시 L2 정규화된다.

## 인증

```python
from ai_release_mobile_shared_dual_head_v1_0_0.encoder import verify

result = verify(
    frames=current_capture,
    gesture_template=stored_gesture_template,
    user_template=stored_user_template,
)
```

반환 예:

```python
{
    "model_version": "witeck-mobile-shared-dual-head-g1g24-v1.0.0",
    "gesture_score": 0.95,
    "user_score": 0.91,
    "gesture_threshold": 0.937420845,
    "user_threshold": 0.824398994,
    "gesture_passed": True,
    "user_passed": True,
    "passed": True,
}
```

## 입력 payload 필수 항목

```json
{
  "video": {
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "total_frames": 60
  },
  "majority_handedness": "Right",
  "frames": [
    {
      "frame_index": 0,
      "tMs": 0,
      "handedness": "Right",
      "landmarks": [{"x": 0.1, "y": 0.2, "z": -0.01}]
    }
  ]
}
```

`landmarks`에는 21개 점이 필요하다. 손을 검출하지 못한 프레임은 생략할 수
있지만 검출 프레임의 원래 `frame_index`와 전체 `total_frames`가 필요하다.
최소 8개 검출 프레임이 필요하다.

## FastAPI 동시 요청

모델은 서버 시작 시 한 번 로드한다. 추론은 내부 `RLock`으로 보호되므로 여러
스레드에서 호출해도 안전하지만 한 프로세스 안에서는 순차 forward가 된다.

## 기존 백엔드에서 바뀌는 점

기존 단일 `embed() -> [128]` 계약을 그대로 사용하면 안 된다. 이 버전의
`embed()`는 `(gesture_embedding, user_embedding)` 두 벡터를 반환한다. DB의
template과 similarity/threshold도 두 개가 필요하다. 기존 G1~G5 closed-set
`classify_gesture()`는 개인 자유 제스처 인증 경로에서 사용하지 않는다.

## 현재 성능과 한계

포함 가중치의 재현 평가 결과:

| 지표 | 결과 |
|---|---:|
| Accuracy | 95.46% |
| Balanced Accuracy | 92.05% |
| Genuine FRR | 11.67% |
| Combined FAR | 4.23% |
| Wrong-Gesture FAR | 0.00% |
| Same-Gesture Impostor FAR | 23.98% |
| Random-Impostor FAR | 0.00% |

Same-Gesture FAR은 아직 배포 수준으로 충분히 낮지 않다. 또한 실제 평가 조건은
P08~P10의 G5 등록 proxy이며, 완전히 새로운 사용자와 자유 제스처 조합을 앱에서
직접 평가한 결과는 아니다. 따라서 본 패키지는 캡스톤 통합 및 추가 실시간
검증용 최종 후보이며, 보안 성능 보장을 의미하지 않는다.

## 테스트

```bash
python -m pip install -r requirements.txt
python smoke_test.py
```
