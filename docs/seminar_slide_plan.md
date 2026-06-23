# Kế Hoạch Slide Seminar - Implementation, Evaluation, Ablation

Tổng thời lượng: **7 phút 30 giây**.

Mục tiêu: trình bày gọn các bằng chứng quan trọng nhất cho thấy pipeline Strong Official-Mini GR00T đã được implement, train, evaluate và ablate thật. Không dành thời gian cho paper background dài; chỉ nhắc paper khi cần giải thích vì sao gọi là **official-mini**.

## Tổng Quan Thời Lượng Tối Ưu

| Phần | Slide | Thời lượng |
|---|---:|---:|
| Implementation | 1-2 | 2m15s |
| Evaluation | 3-4 | 2m00s |
| Ablation | 5-6 | 2m25s |
| Takeaway + limitations | 7 | 50s |
| **Tổng** | **7** | **7m30s** |

Lý do chỉnh so với plan cũ: Implementation không nên chiếm quá nhiều thời gian vì giảng viên thường quan tâm nhiều hơn đến **kết quả thực nghiệm** và **ablation insight**. Vì vậy plan mới gộp pipeline + scale vào một slide, rồi tách Ablation thành hai slide để nói sâu hơn.

## Slide 1 - Pipeline Overview + Data Scale

**Thời lượng:** 65s

**Thông điệp chính:**  
Nhóm đã chuyển từ toy MiniDiT sang một pipeline **Official-Mini GR00T** có dữ liệu thật, video-language feature, action chunk và masked flow-matching loss.

**Nội dung cần đưa lên slide:**

```text
video frame + language + state history
-> frozen Eagle2-2B
-> cached VL features
-> DiT/action head train from scratch
-> action chunk H=16
-> masked flow-matching loss
-> denormalized evaluation
```

**4 số lớn nên đặt cạnh pipeline diagram:**

```text
20 subsets
~119.36 GB downloaded
3,022,069 train samples
755,702 test samples
```

**Prepared tensor contract chỉ nên để rất gọn:**

```text
State [N, 1, 44]
Action [N, 16, 44]
Mask [N, 16, 44]
```

**Note trình bày:**  
Không cần đưa "Old mini" lên slide; chỉ nói miệng một câu: "So với bản mini ban đầu chỉ dùng state vector, bản này đã có video-language feature và action chunk H=16." Nhấn mạnh đây là **official-mini**, không phải full GR00T reproduction.

**Gợi ý hình ảnh:**  
Một pipeline diagram bên trái, 4 number cards bên phải.

## Slide 2 - Training: Frozen VLM + DiT Head

**Thời lượng:** 70s

**Thông điệp chính:**  
Training được chia thành **pretrain + posttrain**, trong đó VLM được frozen, còn DiT/action head được train from scratch.

**Số liệu cần đưa lên slide:**

```text
Frozen VLM: NVIDIA Eagle2-2B
Trainable module: DiT/action head
Params: ~35.27M

Pretrain samples: 3,022,069
Pretrain steps: 11,805

Posttrain subset: gr1_arms_only.CanSort
Posttrain samples: 50,000
Posttrain steps: 196

Best checkpoint: posttrain_checkpoint.pt
```

**Chi tiết chỉ nói miệng nếu cần:**

```text
Fallback used: false
VL feature dim: 1536
VL tokens: 2
```

**Note trình bày:**  
Nói: "Eagle2-2B không bị fallback, nên visual-language features là thật. DiT/action head được train from scratch trên 3.02M samples, sau đó posttrain task-specific trên CanSort."

**Gợi ý hình ảnh:**  
Flow nhỏ: `Eagle2 frozen -> DiT pretrain -> DiT posttrain`. Nếu có hình `loss_curve.png` từ Notebook 03b, đặt nhỏ ở góc slide để chứng minh training thật sự chạy.

## Slide 3 - Evaluation Setup: Vì Sao Metric Này Công Bằng?

**Thời lượng:** 45s

**Thông điệp chính:**  
Evaluation được tính trên **denormalized raw action space**, nên kết quả phản ánh lỗi trong không gian action thật hơn là normalized tensor.

**Nội dung cần đưa lên slide:**

```text
Eval samples: 755,702
Metric space: denormalized raw action space
Main metrics: raw_mse, raw_mae, raw_rmse
Action mask used: true
K-step Euler inference: K = 4
```

**Các model/baseline được so sánh:**

```text
baseline_train_action_mean
baseline_last_state_repeat
OfficialMiniVLDiT_pretrain
OfficialMiniVLDiT_posttrain
NVIDIA_GR00T_zero_shot: attempted, skipped
```

**Note trình bày:**  
Nói ngắn: "Zero-shot NVIDIA có attempt nhưng skipped do Kaggle/offline/dependency limit, nên không dùng làm kết luận chính." Đừng dành quá 10 giây cho zero-shot.

**Gợi ý hình ảnh:**  
Một metric setup box nhỏ. Không cần bảng kết quả ở slide này.

## Slide 4 - Evaluation Results: Posttrain Wins Clearly

**Thời lượng:** 75s

**Thông điệp chính:**  
Posttrain checkpoint giảm lỗi rõ so với simple baselines và pretrain-only trên full test split.

**Bảng cần đưa lên slide:**

| Model | raw_mse | raw_mae | raw_rmse |
|---|---:|---:|---:|
| Train action mean baseline | 0.6378 | 0.4209 | 0.7986 |
| Last-state repeat baseline | 0.6160 | 0.3315 | 0.7849 |
| OfficialMiniVLDiT pretrain | 0.5953 | 0.2112 | 0.7715 |
| OfficialMiniVLDiT posttrain | **0.4085** | 0.2358 | **0.6391** |

**Số liệu cần nhấn mạnh:**

```text
Posttrain vs train_action_mean:
raw MSE reduction ~35.9%

Posttrain vs last_state_repeat:
raw MSE reduction ~33.7%

Posttrain vs pretrain-only:
raw MSE reduction ~31.4%
```

**Note trình bày:**  
Không đọc cả bảng. Chỉ nói: "Dòng quan trọng nhất là posttrain: raw_mse 0.4085, thấp hơn baseline 0.6378 và pretrain 0.5953."

**Câu chuẩn bị nếu bị hỏi vì sao raw_mae posttrain cao hơn pretrain:**  
Posttrain tập trung vào task-specific CanSort nên giảm mạnh squared error trên các lỗi lớn, làm raw_mse giảm rõ. Tuy nhiên raw_mae trên toàn bộ 20-subset test có thể hơi cao hơn pretrain do distribution mismatch giữa task-specific posttrain và toàn bộ test distribution.

**Gợi ý hình ảnh:**  
Raw MSE bar chart gồm 4 cột: mean baseline, repeat baseline, pretrain, posttrain. Nếu có prediction-vs-ground-truth plot thì để rất nhỏ, không thay thế bảng chính.

## Slide 5 - Ablation: Components & Training Stage

**Thời lượng:** 70s

**Thông điệp chính:**  
Hai yếu tố quan trọng nhất là **visual-language feature** và **task-specific posttraining**.

**Comparison 1 - State-only vs State+VL:**

```text
state_only raw_mse = 0.6852
state+VL   raw_mse = 0.5808
=> VL feature giúp giảm lỗi
```

**Comparison 2 - Pretrain vs Posttrain:**

```text
pretrain_only raw_mse = 0.5952
posttrain     raw_mse = 0.4085
=> task-specific posttrain là cải thiện quan trọng nhất
```

**Note trình bày:**  
Chốt insight thay vì kể từng run: "VL giúp model khai thác visual-language context; posttrain giúp model thích nghi task-specific, và đây là cải thiện mạnh nhất."

**Gợi ý hình ảnh:**  
Hai paired bar charts đặt cạnh nhau:

```text
State-only | State+VL
Pretrain   | Posttrain
```

## Slide 6 - Ablation: Architecture, Horizon, Inference

**Thời lượng:** 75s

**Thông điệp chính:**  
Các lựa chọn về learning rate, action horizon và inference step đều ảnh hưởng đến raw_mse; tuy nhiên cần diễn giải cẩn thận để không overclaim.

**Learning rate:**

```text
lr=1e-4 raw_mse = 0.5808
lr=5e-5 raw_mse = 0.7788
=> lr=1e-4 tốt hơn trong setup 1 epoch hiện tại
```

**Action horizon:**

```text
H=16 raw_mse = 0.5808
H=8  raw_mse = 0.3256
=> horizon ngắn dễ dự đoán hơn
```

**Layers:**

```text
8 layers raw_mse = 0.5808
4 layers raw_mse = 0.4915
=> model nhỏ hơn tối ưu tốt hơn trong setup 1 epoch hiện tại
```

**K-step inference:**  
Nếu còn thời gian, nói 1 câu: Notebook 05 đã chạy K = 1, 4, 8, 16 để kiểm tra tradeoff giữa inference cost và quality. Không cần đọc hết bảng K-step.

**Note trình bày:**  
Nói rõ H=8 hiện là **derived từ H=16 prefix**, không phải một pipeline prepare H8 độc lập. Với layers, không kết luận 4 layers luôn tốt hơn 8 layers; chỉ nói trong setup 1 epoch này, 4 layers cho raw_mse thấp hơn.

**Gợi ý hình ảnh:**  
3 mini blocks: `LR`, `Horizon`, `Layers`. Nếu dùng chart, ưu tiên `horizon_ablation.png` đã sửa thành 2 cột H=16/H=8.

## Slide 7 - Final Takeaway, Limitations, Q&A Hooks

**Thời lượng:** 50s

**Thông điệp chính:**  
Pipeline đủ mạnh như một **official-mini seminar reproduction**, nhưng chưa phải full-scale GR00T training.

**What we achieved:**

```text
- Real GR00T-X data: 20 subsets, 3.02M train samples
- Frozen Eagle2-2B + DiT/action head train from scratch
- Pretrain + posttrain
- Full evaluation trên raw action space
- Multi-axis ablation
```

**Limitations:**

```text
- Not full GR00T training
- VLM frozen, not fine-tuned
- Partial ego-view video, not full multi-camera
- Zero-shot NVIDIA baseline skipped
- H=8 derived from H=16 prefix
```

**Q&A hooks cần chuẩn bị:**

```text
1. Vì sao raw_mae posttrain cao hơn pretrain?
2. Flow matching khác DDPM/DDIM như thế nào?
3. Zero-shot NVIDIA skipped vì lý do gì?
4. H=8 derived từ H=16 prefix có bias không?
5. Nếu có thêm GPU/data, bước tiếp theo là gì?
```

**Note trình bày:**  
Kết bằng câu: "Với tài nguyên sinh viên/Kaggle, nhóm đã tái hiện được một Official-Mini GR00T pipeline có training thật, evaluation thật và ablation thật. Phần nằm ngoài scope là full VLM/backbone training và paper-scale data/model size."

**Gợi ý hình ảnh:**  
Hai cột `Achieved` vs `Limitations`, thêm một dòng nhỏ `Prepared Q&A`.

## Delivery Checklist

- Không dành nhiều thời gian cho background paper.
- Mở đầu bằng định vị Official-Mini để tránh overclaim.
- Không đưa quá nhiều số phụ như số video files/parquet files nếu slide đã chật.
- Dùng `raw_mse` làm metric chính được nhắc lại nhiều lần.
- Zero-shot chỉ nói là attempted/skipped, không dùng làm bằng chứng chính.
- Với ablation, nói insight thay vì kể từng run.
- Luôn nói rõ limitation của H=8 và frozen VLM.
