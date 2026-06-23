# Yêu Cầu Seminar - Implementation, Evaluation, Ablation

Tài liệu này tóm tắt những nội dung cần chứng minh trong buổi seminar cho 3 phần chính: **Implementation**, **Evaluation**, và **Ablation**. Mục tiêu là giúp phần trình bày đi thẳng vào bằng chứng thực nghiệm, tránh kể lan man về paper hoặc code chi tiết.

## 1. Implementation

### Mục tiêu cần chứng minh

Phần Implementation phải chứng minh rằng nhóm đã xây được một pipeline thực nghiệm thật theo hướng **Strong Official-Mini GR00T**, không chỉ là notebook demo hoặc toy model.

### Nội dung bắt buộc nên có

- Dữ liệu dùng là dataset gốc `NVIDIA PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`.
- Pipeline có đủ các bước:
  - download subset;
  - prepare state/action sequence;
  - encode video-language feature bằng frozen VLM;
  - train DiT/action head;
  - evaluate;
  - ablation.
- Cần nói rõ đây là **official-mini**, không phải full official GR00T reproduction.
- Cần nói rõ phần được train from scratch là **DiT/action head**, còn VLM/Eagle2 được frozen.

### Số liệu quan trọng cần trình bày

```text
Subsets used: 20
Downloaded size: ~119.36 GB
Video setting: partial ego-view video
Video files: 61,999
Parquet files: 192,226

Train samples: 3,022,069
Test samples: 755,702
```

```text
State shape:  [N, 1, 44]
Action shape: [N, 16, 44]
Mask shape:   [N, 16, 44]
Action horizon H = 16
State horizon S = 1
```

```text
Frozen VLM: NVIDIA Eagle2-2B
Fallback used: false
VL feature dim: 1536
VL tokens: 2

Trainable module: DiT/action head
Params: ~35.27M
Pretrain samples: 3,022,069
Pretrain steps: 11,805
Posttrain subset: gr1_arms_only.CanSort
Posttrain samples: 50,000
Posttrain steps: 196
Best checkpoint: posttrain_checkpoint.pt
```

### Câu nên nói khi trình bày

```text
Do giới hạn tài nguyên, nhóm không train full VLM như paper gốc.
Nhóm dùng Eagle2-2B làm frozen VLM để extract visual-language features,
sau đó train DiT/action head from scratch trên dữ liệu GR00T-X.
```

### Những điều không nên overclaim

- Không nói đây là full GR00T reproduction.
- Không nói đã train VLM/backbone.
- Không nói đã dùng full dataset hoặc full video multi-camera.

## 2. Evaluation

### Mục tiêu cần chứng minh

Phần Evaluation phải chứng minh model được đánh giá thật trên full test split, bằng metric có ý nghĩa trong raw action space, và kết quả tốt hơn baseline đơn giản.

### Nội dung bắt buộc nên có

- Evaluation chạy trên full test samples, không phải smoke test.
- Metric chính tính trên **denormalized raw action space**.
- Có baseline để so sánh.
- Có so sánh pretrain và posttrain.
- Có nói rõ zero-shot NVIDIA baseline được attempted nhưng skipped, không dùng làm kết luận chính.

### Metric cần trình bày

```text
raw_mse
raw_mae
raw_rmse
masked_raw_mse
```

Metric chính nên nhắc nhiều nhất là `raw_mse`.

### Số liệu quan trọng cần trình bày

```text
Eval samples: 755,702
Metric space: denormalized raw action space
Action mask used: true
K-step Euler inference: K = 4
```

| Model | raw_mse | raw_mae | raw_rmse |
|---|---:|---:|---:|
| Train action mean baseline | 0.6378 | 0.4209 | 0.7986 |
| Last-state repeat baseline | 0.6160 | 0.3315 | 0.7849 |
| OfficialMiniVLDiT pretrain | 0.5953 | 0.2112 | 0.7715 |
| OfficialMiniVLDiT posttrain | **0.4085** | 0.2358 | **0.6391** |

### Kết luận số liệu nên nói

```text
Posttrain giảm raw_mse khoảng 35.9% so với train_action_mean baseline.
Posttrain giảm raw_mse khoảng 33.7% so với last_state_repeat baseline.
Posttrain giảm raw_mse khoảng 31.4% so với pretrain-only.
```

### Câu nên nói khi trình bày

```text
Kết quả chính là posttrain checkpoint đạt raw_mse 0.4085 trên 755,702 test samples,
thấp hơn rõ rệt so với baseline 0.6378 và pretrain-only 0.5953.
```

### Những điều không nên overclaim

- Không dùng zero-shot NVIDIA làm kết luận vì trạng thái là `skipped`.
- Không chỉ nói loss train giảm; phải nói metric eval trên test split.
- Không chỉ nói normalized loss; phải nhấn mạnh denormalized raw action metrics.

## 3. Ablation

### Mục tiêu cần chứng minh

Phần Ablation phải chứng minh các lựa chọn thiết kế trong pipeline có tác động thật đến kết quả, không phải model tốt lên do ngẫu nhiên.

### Nội dung bắt buộc nên có

- So sánh state-only với state+VL.
- So sánh pretrain-only với posttrain task-specific.
- So sánh learning rate.
- So sánh action horizon.
- Có nhắc K-step inference tradeoff nếu còn thời gian.

### Số liệu quan trọng cần trình bày

```text
State-only vs State+VL:
state_only raw_mse = 0.6852
state+VL   raw_mse = 0.5808
=> VL feature giúp giảm lỗi.
```

```text
Pretrain vs Posttrain:
pretrain_only raw_mse = 0.5952
posttrain     raw_mse = 0.4085
=> task-specific posttrain là cải thiện quan trọng nhất.
```

```text
Learning rate:
lr=1e-4 raw_mse = 0.5808
lr=5e-5 raw_mse = 0.7788
=> lr=1e-4 phù hợp hơn trong setup này.
```

```text
Action horizon:
H=16 raw_mse = 0.5808
H=8  raw_mse = 0.3256
=> horizon ngắn dễ dự đoán hơn.
```

### Insight chính cần nói

- `state+VL` tốt hơn `state-only`, chứng minh visual-language features có ích.
- `posttrain` tốt hơn `pretrain-only`, chứng minh task-specific adaptation quan trọng.
- `lr=1e-4` tốt hơn `lr=5e-5` trong setup hiện tại.
- `H=8` có raw_mse thấp hơn `H=16`, nhưng cần nói rõ H=8 là derived từ H16 prefix.

### Câu nên nói khi trình bày

```text
Ablation cho thấy phần quan trọng nhất là task-specific posttraining:
raw_mse giảm từ 0.5952 xuống 0.4085.
Ngoài ra, state+VL tốt hơn state-only, chứng minh video-language features có tác dụng.
```

### Những điều không nên overclaim

- Không nói H=8 là một prepare pipeline độc lập; hiện tại H=8 là prefix từ H=16.
- Không nói layer 4 tốt hơn layer 8 một cách tuyệt đối; chỉ nói trong setup 1 epoch hiện tại, layer 4 cho raw_mse thấp hơn.
- Không biến ablation thành danh sách dài; chỉ trình bày insight.

## 4. Checklist Trước Khi Trình Bày

- Có nói rõ pipeline là **official-mini**.
- Có nhắc data scale: **20 subsets**, **3.02M train samples**, **755k test samples**.
- Có nói rõ VLM frozen, DiT/action head train from scratch.
- Có trình bày raw_mse posttrain: **0.4085**.
- Có so sánh posttrain với baseline và pretrain-only.
- Có ít nhất 3 ablation insights:
  - state+VL tốt hơn state-only;
  - posttrain tốt hơn pretrain;
  - horizon H=8 dễ hơn H=16.
- Có limitation trung thực:
  - không full GR00T;
  - không train/fine-tune VLM;
  - partial ego-view video;
  - zero-shot NVIDIA skipped;
  - H=8 derived từ H16 prefix.

## 5. Một Câu Kết Luận Gọn

```text
Với tài nguyên sinh viên/Kaggle, nhóm đã xây được một Strong Official-Mini GR00T pipeline
có train thật trên GR00T-X, evaluation thật trên full test split, và ablation đủ để chứng minh
các lựa chọn thiết kế chính.
```
