# WITECK 1D-CNN Authentication Models

Ba mô hình PyTorch cho user authentication bằng hand gesture, thiết kế cho input
`[N, 32, 169]`:

1. **InceptionTime Siamese** — nhiều temporal kernel song song.
2. **TCN Siamese** — residual depthwise temporal convolution với dilation 1/2/4/8.
3. **Prototypical 1D CNN** — enrollment template trung bình và episodic training.

Tất cả embedding có 128 chiều và được L2-normalize. Hai Siamese model trả về cosine
similarity. Prototypical model tính prototype từ các enrollment sample.

## Chuẩn dữ liệu

File NPZ cần chứa ba mảng:

```python
X.shape == (N, 32, 169)
user_ids.shape == (N,)
gesture_ids.shape == (N,)
```

Loader cũng chấp nhận các alias phổ biến:

- `X`, `features`, `data` hoặc `sequences`;
- `user_ids`, `users`, `subjects`, `subject_ids` hoặc `y_user`;
- `gesture_ids`, `gestures` hoặc `y_gesture`.

Pair và episode luôn được tạo **trong cùng một gesture**, để mạng học đặc trưng user
thay vì phân biệt G1–G5.

## Cài đặt

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

## Huấn luyện InceptionTime Siamese

```bash
python train_siamese.py --data train.npz --model inception \
  --validation-data validation.npz --output checkpoints/inception_siamese.pt
```

## Huấn luyện TCN Siamese

```bash
python train_siamese.py --data train.npz --model tcn \
  --validation-data validation.npz --output checkpoints/tcn_siamese.pt
```

## Huấn luyện Prototypical 1D CNN

```bash
python train_prototypical.py --data train.npz \
  --support 3 --query 2 --output checkpoints/prototypical_cnn.pt
```

Mỗi Prototypical episode chọn một gesture rồi tạo support/query từ những user có
đủ sample cho gesture đó.

## Đánh giá Siamese

```bash
python evaluate_pairs.py --data final_test.npz \
  --checkpoint checkpoints/inception_siamese.pt
```

Script báo cáo AUC, EER, FAR, FRR và balanced accuracy. Threshold trong checkpoint
chỉ được lưu khi `--validation-data` được cung cấp. Khi đánh giá Final Test, có thể
truyền threshold đã chọn trên Validation bằng `--threshold` hoặc dùng threshold đó.

## Enrollment bằng Prototypical model

```python
import torch
from witeck_auth.models import Prototypical1DCNN

model = Prototypical1DCNN()
enrollment = torch.randn(5, 32, 169)
owner_labels = torch.zeros(5, dtype=torch.long)
prototype, _ = model.compute_prototypes(enrollment, owner_labels)

query = torch.randn(1, 32, 169)
score = model.authenticate(query, prototype[0])
accepted = score.item() >= threshold
```

Trong triển khai thật, enrollment và query phải được chuẩn hóa bằng mean/std lưu
trong checkpoint.

## Kiểm tra

```bash
pip install pytest
pytest -q
```

## Cấu trúc

```text
witeck_auth/
  data.py
  losses.py
  metrics.py
  models/
    common.py
    inception_siamese.py
    tcn_siamese.py
    prototypical_cnn.py
train_siamese.py
train_prototypical.py
evaluate_pairs.py
tests/test_model_shapes.py
```

## Multi-Stream Dilated Siamese

The feature-aware multi-stream model is available through
`--model multistream`. See [README_MULTISTREAM.md](README_MULTISTREAM.md).
