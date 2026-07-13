# Báo Cáo Cuối Kỳ: Official-Mini Isaac GR00T Pipeline

## 1. Thông Tin Đề Tài

**Tên đề tài:** Isaac GR00T: N1 through N1.6  
**Paper gốc:** *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*  
**Mục tiêu thực nghiệm:** xây dựng một pipeline **official-mini reproduction** trên dữ liệu GR00T-X thật, gồm data preparation, frozen VLM encoding, DiT/action head training, evaluation và ablation.

Báo cáo này không claim reproduce full GR00T ở paper-scale. Do giới hạn tài nguyên sinh viên/Kaggle, nhóm tập trung tái hiện các ý tưởng chính của pipeline: dùng visual-language context, state/action sequence, action chunk, flow matching, pretrain và posttrain. Phần được train from scratch là **DiT/action head**, còn VLM/Eagle2-2B được dùng ở trạng thái frozen.

## 2. Tóm Tắt Paper Gốc

GR00T là một Vision-Language-Action (VLA) model cho humanoid robot. Model nhận quan sát thị giác, instruction ngôn ngữ và trạng thái robot để sinh ra action cho robot thực hiện nhiệm vụ.

Các ý chính của paper:

- Dùng VLM backbone để mã hóa visual-language context.
- Dùng DiT/action head để mô hình hóa và sinh action.
- Dùng flow matching objective để học quá trình biến noise/action trung gian thành action thật.
- Training gồm pretraining trên dữ liệu đa dạng và posttraining/fine-tuning cho task cụ thể.
- Paper gốc dùng quy mô tài nguyên rất lớn, gồm model billion-scale, dữ liệu đa dạng, nhiều embodiment và cụm GPU lớn.

Trong phạm vi môn học, nhóm không thể train full backbone hay reproduce toàn bộ benchmark của paper. Vì vậy nhóm xây một bản **Official-Mini GR00T Pipeline**: giữ cấu trúc và tinh thần chính, nhưng scale xuống để chạy được trên Kaggle/Nemotron RTX6000.

## 3. Mục Tiêu Thực Nghiệm Của Nhóm

Mục tiêu của nhóm là chứng minh pipeline không còn là toy demo state-only, mà đã có đầy đủ các thành phần thực nghiệm quan trọng:

- Dùng dataset gốc `NVIDIA PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`.
- Prepare state/action thành sequence/chunk format.
- Encode video frame và language instruction bằng frozen Eagle2-2B.
- Train DiT/action head from scratch.
- Chạy pretrain trên nhiều subset và posttrain trên task-specific subset.
- Evaluate trên test split bằng metric denormalized về raw action space.
- Chạy ablation để kiểm tra vai trò của VL features, posttraining, learning rate, model depth và inference steps.

Điểm quan trọng: nhóm gọi đây là **official-mini reproduction**, không phải full reproduction. Cách gọi này phản ánh đúng phạm vi: pipeline bám sát paper về ý tưởng, nhưng không match scale dữ liệu, model, GPU và robot rollout evaluation như paper gốc.

## 4. Dataset Và Preprocessing

Dataset sử dụng là `NVIDIA PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`. Do video raw rất lớn, nhóm không tải full toàn bộ dataset. Thay vào đó, nhóm tải 20 subsets với partial ego-view video để đủ chạy frozen VLM encoding mà không tràn disk Kaggle.

| Hạng mục | Giá trị |
|---|---:|
| Subsets used | 20 |
| Downloaded size | ~119.36 GB |
| Video setting | Partial ego-view video |
| Video files | 61,999 |
| Parquet files | 192,226 |
| Train samples | 3,022,069 |
| Test samples | 755,702 |

Sau bước prepare, dữ liệu được chuẩn hóa thành contract chính:

| Tensor | Shape | Ý nghĩa |
|---|---|---|
| State | `[N, 1, 44]` | State history với `S=1` |
| Action | `[N, 16, 44]` | Action chunk với `H=16` |
| Mask | `[N, 16, 44]` | Mask cho action dimensions hợp lệ |

Action horizon `H=16` nghĩa là mỗi sample dự đoán 16 bước action tương lai. State horizon hiện tại là `S=1`, nhưng vẫn giữ shape `[N, 1, 44]` để DiT xử lý state như sequence token và dễ mở rộng lên state history dài hơn.

Notebook 02 cũng lưu normalization statistics để Notebook 04 có thể denormalize prediction và target về raw action space trước khi tính metric:

```python
pred_raw = pred_norm * action_std + action_mean
target_raw = target_norm * action_std + action_mean
```

## 5. Pipeline Implementation

Pipeline được chia thành 5 nhóm notebook:

| Phase | Notebook | Vai trò |
|---|---|---|
| 01 | Download subset notebooks | Tải meta/data và partial ego-view video cho từng subset |
| 02 | Prepare notebooks | Tạo state/action/mask arrays, split train/test và video frame manifest |
| 03a | Encode notebooks | Dùng frozen Eagle2-2B để encode video-language features |
| 03b | Train notebook | Train DiT/action head bằng pretrain + posttrain |
| 04 | Evaluation notebook | Evaluate checkpoints trên raw denormalized action metrics |
| 05 | Ablation notebook | Chạy ablation theo component, model, hyperparameter và inference |

Luồng dữ liệu tổng quát:

```text
GR00T-X subset data + partial video
-> prepare state/action/mask + video manifest
-> frozen Eagle2-2B encode video-language features
-> train DiT/action head from scratch
-> evaluate pretrain/posttrain checkpoints
-> ablation study
```

Frozen VLM encoding được tách khỏi training. Cách này giúp giảm VRAM và thời gian training, vì Notebook 03b không cần decode video hay load Eagle2-2B trong lúc train DiT/action head.

## 6. Model Training

Trong pipeline của nhóm, VLM là **NVIDIA Eagle2-2B** ở trạng thái frozen. Model này nhận video frame và instruction để tạo visual-language features.

| Thành phần | Giá trị |
|---|---:|
| Frozen VLM | NVIDIA Eagle2-2B |
| Fallback used | False |
| VL feature dim | 1536 |
| VL tokens | 2 |
| Trainable module | DiT/action head |
| Trainable params | ~35.27M |

DiT/action head nhận noisy action, state history, timestep và VL features. Training dùng masked flow-matching objective. Thay vì MSE đơn giản trên toàn bộ tensor, loss được mask theo action dimensions hợp lệ:

```python
loss = (mse_per_element * action_mask).sum() / (action_mask.sum() + 1e-6)
```

Training gồm hai stage:

| Stage | Data | Samples | Steps | Mục tiêu |
|---|---|---:|---:|---|
| Pretrain | Full train split đã encode | 3,022,069 | 11,805 | Học hành vi tổng quát từ nhiều task |
| Posttrain | `gr1_arms_only.CanSort` | 50,000 | 196 | Fine-tune task-specific |

Posttrain chỉ có 196 steps vì đây là fine-tuning task-specific trên 50,000 samples sau pretrain, không phải train lại từ đầu. Checkpoint tốt nhất dùng cho evaluation chính là `posttrain_checkpoint.pt`.

## 7. Evaluation

Evaluation được chạy trên full test split của 20 subsets, gồm 755,702 samples. Metric chính được tính trên **denormalized raw action space**, không chỉ trên normalized tensor. Điều này giúp kết quả phản ánh lỗi action thực tế hơn.

Thiết lập evaluation:

| Hạng mục | Giá trị |
|---|---:|
| Eval samples | 755,702 |
| Metric space | Denormalized raw action space |
| Action mask used | True |
| Main inference | K-step Euler |
| K chính | 4 |
| Action horizon | H=16 |

Các baseline/model được so sánh:

- `baseline_train_action_mean`: luôn dự đoán action mean từ train set.
- `baseline_last_state_repeat`: dùng last state làm action repeat baseline.
- `OfficialMiniVLDiT_pretrain`: checkpoint sau pretrain.
- `OfficialMiniVLDiT_posttrain`: checkpoint sau posttrain.
- `NVIDIA_GR00T_zero_shot`: attempted nhưng skipped do giới hạn môi trường/dependency/API, không dùng làm kết luận chính.

Kết quả chính:

| Model | Status | raw_mse | raw_mae | raw_rmse |
|---|---|---:|---:|---:|
| Train action mean baseline | completed | 0.6378 | 0.4209 | 0.7986 |
| Last-state repeat baseline | completed | 0.6160 | 0.3315 | 0.7849 |
| OfficialMiniVLDiT pretrain | completed | 0.5953 | 0.2112 | 0.7715 |
| OfficialMiniVLDiT posttrain | completed | **0.4085** | 0.2358 | **0.6391** |
| NVIDIA GR00T zero-shot | skipped | N/A | N/A | N/A |

Posttrain checkpoint là kết quả chính tốt nhất theo raw MSE:

- Giảm raw MSE khoảng **35.9%** so với train action mean baseline.
- Giảm raw MSE khoảng **33.7%** so với last-state repeat baseline.
- Giảm raw MSE khoảng **31.4%** so với pretrain-only.

Kết quả này cho thấy posttraining task-specific có tác dụng rõ rệt, đặc biệt khi so với baseline đơn giản và checkpoint pretrain-only.

## 8. Ablation Study

Ablation nhằm kiểm tra thành phần nào trong pipeline thật sự ảnh hưởng đến kết quả. Metric chính vẫn là raw MSE trên denormalized raw action space.

### 8.1 State-only vs State+VL

| Run | Input | raw_mse | raw_mae | raw_rmse |
|---|---|---:|---:|---:|
| state_only_layers8_lr1e4 | State only | 0.6852 | 0.2361 | 0.8278 |
| state_vl_layers8_lr1e4 | State + VL | **0.5808** | **0.2056** | **0.7621** |

Kết quả cho thấy visual-language features giúp giảm lỗi so với chỉ dùng state. Đây là bằng chứng quan trọng rằng frozen Eagle2-2B features không chỉ là phần trang trí, mà thật sự đóng góp vào action prediction.

### 8.2 Pretrain-only vs Posttrain

| Run | Stage | raw_mse | raw_mae | raw_rmse |
|---|---|---:|---:|---:|
| pretrain_only | Pretrain | 0.5952 | **0.2112** | 0.7715 |
| posttrain_task_specific | Posttrain | **0.4085** | 0.2358 | **0.6391** |

Đây là ablation quan trọng nhất. Posttraining task-specific giảm raw MSE mạnh so với pretrain-only, chứng minh fine-tuning trên task cụ thể giúp model thích nghi tốt hơn.

### 8.3 Learning Rate

| Run | Learning rate | raw_mse | raw_mae | raw_rmse |
|---|---:|---:|---:|---:|
| state_vl_layers8_lr1e4 | 1e-4 | **0.5808** | **0.2056** | **0.7621** |
| state_vl_layers8_lr5e5 | 5e-5 | 0.7788 | 0.2567 | 0.8825 |

Trong setup 1 epoch hiện tại, learning rate `1e-4` hội tụ tốt hơn `5e-5`. Không nên kết luận `1e-4` luôn tốt hơn trong mọi setup, vì kết quả còn phụ thuộc số epoch, batch size và scheduler.

### 8.4 Model Depth

| Run | Layers | Params | raw_mse | raw_mae | raw_rmse |
|---|---:|---:|---:|---:|---:|
| state_vl_layers8_lr1e4 | 8 | ~35.27M | 0.5808 | 0.2056 | 0.7621 |
| state_vl_layers4_lr1e4 | 4 | ~18.46M | **0.4915** | **0.1652** | **0.7011** |

Trong run hiện tại, 4 layers cho raw MSE thấp hơn 8 layers. Tuy nhiên, không nên overclaim rằng model 4 layers luôn tốt hơn. Với setup chỉ 1 epoch, model nhỏ hơn có thể tối ưu dễ hơn và ít undertrained hơn.

### 8.5 Horizon và K-step Inference

Action horizon analysis cho thấy horizon ngắn hơn thường dễ dự đoán hơn. Trong slide/seminar, H=8 được dùng như **derived prefix từ H=16**, không phải một independent H8 prepare/training pipeline. Vì vậy, kết luận đúng là: horizon ngắn hơn có thể cho lỗi thấp hơn, nhưng cần chạy pipeline H8 độc lập nếu muốn kết luận chắc chắn.

K-step Euler inference được dùng để refine action prediction theo velocity field học được từ flow matching. Evaluation chính dùng `K=4` vì đây là điểm cân bằng giữa chất lượng và chi phí inference. Khi tăng K lên 8 hoặc 16, kết quả không nhất thiết cải thiện nhiều, trong khi chi phí inference tăng.

## 9. So Sánh Với Paper Gốc

| Khía cạnh | Paper gốc GR00T | Pipeline của nhóm |
|---|---|---|
| Mục tiêu | Generalist humanoid robot foundation model | Official-mini reproduction cho seminar |
| VLM/backbone | Model lớn, paper-scale | Eagle2-2B frozen feature extractor |
| Train from scratch | Quy mô lớn, nhiều thành phần | DiT/action head train from scratch |
| Action modeling | DiT/action head, flow matching | DiT/action head mini, masked flow matching |
| Data | Dữ liệu rất lớn, nhiều embodiment/camera | 20 GR00T-X subsets, partial ego-view video |
| Training | Distributed GPU cluster | Kaggle/Nemotron RTX6000 workflow |
| Evaluation | Simulation/real robot benchmark, success rate | Offline open-loop action prediction |
| Posttraining | Task-specific adaptation | CanSort task-specific posttrain |

Các điểm giống chính:

- Có vision-language context.
- Có state/action input.
- Có action chunk prediction.
- Có DiT/action head.
- Có flow matching objective.
- Có pretrain và posttrain.

Các điểm khác chính:

- Không train/fine-tune full VLM.
- Không dùng full dataset/multi-camera như paper-scale.
- Không chạy robot rollout hoặc simulator success-rate evaluation.
- Model và tài nguyên nhỏ hơn rất nhiều.

Vì vậy, kết luận đúng là nhóm đã xây được một **Strong Official-Mini GR00T Pipeline**, không phải full GR00T reproduction.

## 10. Limitations Và Hướng Phát Triển

### 10.1 Limitations

- **Không full GR00T training:** VLM/Eagle2-2B được frozen, không train từ đầu.
- **Không full dataset:** chỉ dùng 20 subsets và partial ego-view video.
- **Không full multi-camera:** pipeline hiện tại ưu tiên ego-view để tránh tràn disk.
- **Không robot rollout:** evaluation hiện tại là offline open-loop action prediction.
- **Zero-shot NVIDIA baseline skipped:** có attempt nhưng không dùng làm kết luận chính do giới hạn môi trường/API.
- **H=8 chưa phải independent run:** hiện chỉ nên xem như horizon prefix analysis từ H=16.

### 10.2 Hướng Phát Triển

- Tạo `06_single_video_demo.ipynb` để visualize predicted action vs ground-truth action trên một video/episode cụ thể.
- Chạy H=8 prepare/train/eval độc lập để có horizon ablation sạch hơn.
- Thêm simulator hoặc replay environment để đánh giá rollout/task success.
- Tải thêm camera view hoặc nhiều chunks hơn nếu có đủ disk/quota.
- Thử fine-tune một phần VLM hoặc adapter nhỏ nếu có tài nguyên GPU tốt hơn.

## 11. Q&A / Risk Notes

### Flow matching khác DDPM thế nào?

DDPM học quá trình denoise nhiều bước từ noise về data. Flow matching học trực tiếp velocity field, tức hướng di chuyển từ noise/action trung gian về action thật. Trong pipeline này, target velocity được hiểu đơn giản là:

```text
target_velocity = action - noise
```

### Vì sao frozen VLM?

Full VLM/backbone training cần tài nguyên paper-scale. Với giới hạn sinh viên/Kaggle, nhóm dùng Eagle2-2B frozen để extract visual-language features, rồi train DiT/action head from scratch. Đây là lựa chọn thực tế và trung thực cho official-mini reproduction.

### Vì sao posttrain chỉ 196 steps?

Posttrain dùng 50,000 samples từ `gr1_arms_only.CanSort`. Với batch size khoảng 256, số step xấp xỉ 196. Đây là fine-tuning task-specific sau pretrain trên hơn 3 triệu samples, nên ngắn hơn pretrain là hợp lý.

### Vì sao zero-shot NVIDIA baseline bị skipped?

Notebook 04 có attempt zero-shot baseline, nhưng môi trường Kaggle/offline/dependency/API không đảm bảo load được model NVIDIA GR00T pretrained với action prediction API tương thích. Vì vậy baseline này được ghi `skipped` và không dùng làm kết luận chính.

### Vì sao H=8 không nên overclaim?

H=8 hiện được dùng như derived prefix từ H=16 trong phần phân tích horizon, không phải một pipeline H8 độc lập được prepare/train/evaluate riêng. Do đó chỉ nên kết luận rằng horizon ngắn hơn có xu hướng dễ hơn, chưa nên claim H=8 setup chính thức tốt hơn H=16.

## 12. Kết Luận

Với tài nguyên sinh viên và môi trường Kaggle/Nemotron RTX6000, nhóm đã xây dựng được một pipeline **Strong Official-Mini GR00T** có dữ liệu thật, training thật, evaluation thật và ablation thật.

Kết quả chính là `OfficialMiniVLDiT_posttrain` đạt raw MSE **0.4085** trên **755,702 test samples**, thấp hơn rõ rệt so với baseline và pretrain-only. Ablation cũng cho thấy visual-language features và task-specific posttraining đều đóng góp tích cực.

Pipeline hiện tại chưa phải full GR00T paper-scale, nhưng đủ để chứng minh nhóm đã hiểu và tái hiện được các thành phần quan trọng của GR00T ở mức official-mini: frozen VLM feature extraction, DiT/action head training, action chunk prediction, masked flow matching, denormalized evaluation và ablation có ý nghĩa.
