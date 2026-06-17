# Strong Official-Mini GR00T Pipeline Plan

Tài liệu này là README cho pipeline mới sau khi nâng cấp từ bản MiniDiT state-only sang bản **Strong Official-Mini**. Mục tiêu là để lần sau mở lại repo hoặc trao đổi tiếp, có thể hiểu ngay pipeline mới làm gì, notebook nào phụ trách phần nào, output cần có là gì, và tiêu chí nào chứng minh pipeline đã đủ mạnh cho seminar cuối kì.

## 1. Mục Tiêu

Pipeline cũ chỉ là bản mini kiểm thử:

```text
state [N, 44]
-> MiniDiT nhỏ
-> dự đoán 1 action step
-> MSE đơn giản
```

Pipeline mới nâng lên hướng gần official GR00T hơn:

```text
video frame + language + state history
-> frozen NVIDIA/Eagle/GR00T encoder
-> cached VL features
-> DiT/action head train from scratch
-> action chunk prediction H=16
-> masked flow-matching loss
-> denormalized open-loop evaluation
-> strong ablation
```

Điểm quan trọng: pipeline này **không train full VLM từ scratch**. VLM/Eagle/GR00T chỉ dùng làm frozen feature extractor vì tài nguyên Kaggle/T4 không đủ để train full backbone như paper gốc. Phần train from scratch nằm ở **DiT/action head**, đúng với hướng Mini Implementation của đồ án.

Nếu NVIDIA/Eagle/GR00T không chạy được do gated model, dependency, OOM hoặc quota GPU, pipeline cho phép fallback sang SigLIP/OpenCLIP, nhưng vẫn giữ chung interface:

```text
vl_features: [N, T, D]
cross_attention_dim = D
```

## 2. Mức Độ So Với Yêu Cầu Cuối Kì

Theo ghi chú yêu cầu của thầy, mức Implementation Mini cần:

- train thật trên dataset gốc của paper;
- train tối thiểu 1 epoch;
- có evaluation thật trên test split;
- có ablation study thật;
- giải thích rõ metric, setup, limitation.

Pipeline Strong Official-Mini đáp ứng hướng này bằng cách:

- dùng dataset gốc `nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`;
- prepare dữ liệu thành state/action sequence đúng hơn;
- encode video/language bằng frozen VLM;
- train DiT/action head từ đầu;
- đánh giá trên raw denormalized action space;
- có ablation ở component, model, inference, horizon và hyperparameter.

Đây vẫn là **official-mini**, không phải full official. Các phần không làm full do giới hạn tài nguyên:

- không train VLM/Eagle backbone;
- không train action head cỡ 1.7B params;
- không dùng full 1.5TB/113 subsets;
- không chạy distributed multi-GPU/DeepSpeed như môi trường nghiên cứu lớn.

Trong seminar nên nói rõ:

```text
Do giới hạn tài nguyên, nhóm tái hiện mini pipeline của GR00T:
VLM dùng pretrained/frozen backbone để trích xuất visual-language features,
còn DiT/action head được khởi tạo và train từ đầu trên GR00T-X-Embodiment-Sim.
```

## 3. Notebook Map

Pipeline mới được chia theo phase/notebook:

| Phase | Notebook | Vai trò | Output chính |
|---|---|---|---|
| 01 | `new_notebook/01_download/` | tải raw subset, mặc định meta/data + partial ego-view video | raw subset folders, download reports |
| 02 | `new_notebook/02_prepare/02_prepare_official_sequence_data.ipynb` | chuyển raw data sang official sequence/chunk format | `gr00t_prepared_official*` |
| 03 | `new_notebook/03_train/03_encode_vlm_and_train_official_mini_vldit.ipynb` | frozen VLM encode + train DiT/action head from scratch | VL features, checkpoints, train report |
| 04 | `new_notebook/04_eval/04_evaluate_official_mini_denormalized.ipynb` | evaluate bằng K-step Euler ODE, metric trên raw action space | `gr00t_official_eval/` |
| 05 | `new_notebook/05_ablation/05_strong_ablation_official_mini.ipynb` | ablation mạnh cho seminar | `gr00t_official_ablation/` |

Notebook 01 được tách thành nhiều notebook nhỏ theo từng subset để tránh tràn disk Kaggle. Mỗi subset có thể chạy riêng, xuất output riêng, sau đó Notebook 02 merge những subset đã tải thành một prepared dataset. Mặc định hiện tại của các notebook download subset là `data_meta_ego_chunks`, tức là không tải full video mà chỉ tải `meta/`, `data/`, và video `observation.images.ego_view` cho các chunk được chọn.

## 4. Data Strategy

### 4.1 Lý do tách Notebook 01 theo subset

Raw video của GR00T-X-Embodiment-Sim lớn, Kaggle disk dễ đầy nếu một notebook tải quá nhiều subset. Vì vậy phase download dùng nhiều notebook nhỏ:

```text
01_download_full_subset_INDEX.ipynb
01_download_full_subset_001_*.ipynb
01_download_full_subset_002_*.ipynb
...
```

Mỗi notebook phụ trách một subset. Khi subset nào chạy xong, output đó có thể được add làm input cho Notebook 02.

### 4.2 Full video hay chỉ meta/data?

Với pipeline mới, nên tải video nếu còn disk/quota, vì Notebook 03 cần frame để tạo VL features. Nếu chỉ tải meta/data thì:

- Notebook 02 vẫn prepare được state/action;
- Notebook 03 phải fallback sang state-only hoặc dummy/no-video feature;
- chất lượng và độ giống official giảm mạnh;
- ablation `state_only vs state_vl` mất ý nghĩa hoặc yếu hơn.

Vì vậy chiến lược hiện tại là: **ưu tiên meta/data đầy đủ và partial video đại diện**, thay vì cố tải full video.

Cập nhật thực tế cho Kaggle disk: các notebook 01 hiện mặc định dùng `DOWNLOAD_MODE = "data_meta_ego_chunks"` với `VIDEO_CHUNKS = [0, 1, 2]`. Nếu vẫn đầy disk thì giảm `VIDEO_CHUNKS = [0]`; nếu vẫn đầy nữa thì chuyển riêng subset đó sang `DOWNLOAD_MODE = "data_meta_only"`.

### 4.3 Nhóm subset ưu tiên

Ưu tiên theo mức độ an toàn cho seminar:

| Priority | Nhóm subset | Lý do |
|---|---|---|
| P1 | GR-1 arms/arms-waist/full-upper-body | cùng embodiment, dễ dùng action dim 44 |
| P2 | GR-1 unified | tăng đa dạng task nhưng vẫn cùng họ GR-1 |
| P3 | cross-embodiment Panda/R1/Unitree | tăng độ đa dạng, nhưng xử lý mask/embodiment phức tạp hơn |

Cho seminar, nên hoàn thành chắc P1/P2 trước. P3 là phần mở rộng nếu còn thời gian.

## 5. Notebook 02: Prepare Official Sequence Data

Notebook 02 biến dữ liệu raw subset thành dạng sequence/chunk để Notebook 03 train chính thức.

### 5.1 Output contract

Output chính:

```text
gr00t_prepared_official/
  states_train.npy
  states_test.npy
  actions_train_chunk.npy
  actions_test_chunk.npy
  action_mask_train.npy
  action_mask_test.npy
  train_samples.parquet
  test_samples.parquet
  video_frame_manifest_train.parquet
  video_frame_manifest_test.parquet
  normalization_stats.json
  prepare_report.json
```

Shape bắt buộc:

```text
states_train.npy          [N, S, 44]
states_test.npy           [N, S, 44]
actions_train_chunk.npy   [N, H, 44]
actions_test_chunk.npy    [N, H, 44]
action_mask_train.npy     [N, H, 44]
action_mask_test.npy      [N, H, 44]
```

Default config:

```text
STATE_HORIZON = 1
ACTION_HORIZON = 16
STATE_DIM = 44
ACTION_DIM = 44
```

Ngay cả khi `STATE_HORIZON = 1`, state vẫn phải có shape `[N, 1, 44]`, không dùng `[N, 44]`. Lý do là DiT xử lý state như sequence token, gần official hơn và dễ mở rộng lên `S=2..4`.

### 5.2 Action chunk

Với sample tại frame `t`, target action chunk là:

```text
[t, t+1, ..., t+H-1]
```

Rule:

- không chunk nào được vượt episode boundary;
- nếu episode không đủ dài cho H step thì bỏ sample đó;
- default `H=16`;
- H ablation cần re-prepare riêng `H=8` và `H=16`.

### 5.3 Action mask

`action_mask` bắt buộc có shape `[N, H, 44]`.

Với GR-1 44-dim, mask mặc định toàn 1:

```text
action_mask = ones([N, H, 44])
```

Khi mở rộng cross-embodiment, mask sẽ dùng để loại joint/action dim không thuộc embodiment hiện tại. Training loss không được dùng MSE đơn giản, mà phải dùng masked loss ở Notebook 03.

### 5.4 Normalization stats

Notebook 02 phải lưu mean/std của action để Notebook 04 denormalize:

```text
normalization_stats.json
  action_mean
  action_std
  state_mean
  state_std
```

Metric chính trong Notebook 04 phải tính trên raw action space:

```python
pred_raw = pred_norm * action_std + action_mean
target_raw = target_norm * action_std + action_mean
```

### 5.5 Video frame manifest

Notebook 02 không cần lưu raw frame đã decode. Thay vào đó tạo manifest để Notebook 03 biết cần đọc frame nào từ video.

Schema:

```text
sample_id
subset
episode_index
frame_index
timestamp
video_path
camera_key
task_text
has_video
split
```

Default camera:

```text
observation.images.ego_view
```

Decoder:

- primary: PyAV;
- fallback: OpenCV `cv2.VideoCapture`.

Nên ghi thêm report:

```text
video_decode_report.json
```

để biết bao nhiêu sample có video thật, bao nhiêu sample bị missing/fallback.

## 6. Notebook 03: Frozen VLM Encoding + Official Mini DiT Training

Notebook 03 là notebook quan trọng nhất vì nó chứng minh pipeline đã vượt khỏi state-only test.

Notebook gồm 3 bước chính:

```text
Phase 1: frozen VLM encode
Phase 2: pretrain DiT/action head from scratch on all encoded train samples
Phase 3: posttrain the same DiT/action head on one task-specific subset
```

### 6.1 Phase 1: Frozen VLM encode

Primary encoder:

```text
NVIDIA/GR00T/Eagle frozen encoder
```

Fallback encoder:

```text
SigLIP/OpenCLIP frozen encoder
```

Output:

```text
vl_features_train.npy
vl_features_test.npy
vl_feature_index.parquet
encode_report.json
```

Feature interface:

```text
vl_features_train.npy  [N_train, T, D]
vl_features_test.npy   [N_test, T, D]
```

`D` là dynamic theo encoder. Vì SigLIP/OpenCLIP và Eagle có feature dimension khác nhau, model không được hard-code cross attention dim. Bắt buộc:

```text
cross_attention_dim = feature_dim = D
```

`encode_report.json` nên có:

```text
encoder_name
feature_dim
num_tokens
fallback_used
num_video_samples
num_missing_video_samples
```

### 6.2 Phase 2/3: Pretrain + Posttrain DiT/action head

Input model:

```text
state_history:      [B, S, 44]
noisy_action_chunk: [B, H, 44]
vl_features:        [B, T, D]
timestep:           [B]
action_mask:        [B, H, 44]
```

Flow matching:

```python
t = Beta(1.5, 1.0).sample([B])
t = t[:, None, None]

noise = torch.randn_like(action_chunk)
noisy = (1.0 - t) * noise + t * action_chunk
target_velocity = action_chunk - noise
```

Masked loss:

```python
mse_per_element = (pred_velocity - target_velocity).pow(2)
loss = (mse_per_element * action_mask).sum() / (action_mask.sum() + 1e-6)
```

Điểm cần nhấn mạnh: `target_velocity = action_chunk - noise`. Nếu ngược dấu, model sẽ học sai hướng vector field.

Notebook 03 phải xuất rõ hai checkpoint:

```text
official_mini_vldit_H16/
  pretrain_checkpoint.pt
  posttrain_checkpoint.pt
  checkpoint_last.pt
  pretrain_log.csv
  posttrain_log.csv
  train_summary.json
```

Luồng mặc định:

- `pretrain`: random-init DiT/action head, train trên toàn bộ encoded train samples.
- `posttrain`: load `pretrain_checkpoint.pt`, fine-tune tiếp trên task-specific subset.
- `POSTTRAIN_SUBSETS = ["gr1_arms_only.CanSort"]`.
- Nếu `CanSort` không tồn tại trong prepared data, notebook tự fallback sang subset có nhiều sample nhất và ghi `posttrain_subset_fallback_used = true`.
- `checkpoint_last.pt` trỏ tới checkpoint tốt nhất: ưu tiên posttrain nếu posttrain completed, nếu không thì dùng pretrain.

### 6.3 DiT architecture minimum

Model không cần to như official, nhưng cần có các thành phần tối thiểu:

- state projection cho `[S, 44]`;
- action token projection cho `[H, 44]`;
- learned action position embedding;
- timestep bucket embedding với `NUM_TIMESTEP_BUCKETS = 1000`;
- cross-attention từ state/action tokens sang `vl_features`;
- hidden dim default `512`, fallback `256`;
- layers default `8`, fallback `4`;
- AMP/mixed precision nếu GPU hỗ trợ.

Acceptance cho Notebook 03:

```text
full_epoch_completed = true
pretrain_completed = true
posttrain_completed = true
teacher_requirement_alignment = pretrain_plus_posttrain_dit_head
target_velocity_sign = action_minus_noise
timestep_sampling = beta_1.5_1.0
action_horizon = 16
state_horizon >= 1
uses_vl_features = true
cross_attention_dim = feature_dim
masked_loss = true
```

## 7. Notebook 04: Denormalized Open-Loop Evaluation

Notebook 04 đánh giá checkpoint official-mini bằng K-step Euler ODE.

### 7.1 Inference

```python
x = noise
for k in range(K):
    t = k / K
    v = model(x, state_history, vl_features, t)
    x = x + (1 / K) * v

pred_action_chunk_norm = x
```

Default:

```text
K = 4
H = 16
```

K sẽ được ablation trong Notebook 05.

### 7.2 Metric trên raw denormalized action space

Notebook 04 phải tính metric chính trên action đã denormalize:

```python
pred_raw = pred_action_chunk_norm * action_std + action_mean
target_raw = target_action_chunk_norm * action_std + action_mean
diff = pred_raw - target_raw
```

Required metrics:

```text
raw_action_mse
raw_action_mae
raw_action_rmse
masked_raw_action_mse
masked_raw_action_mae
masked_raw_action_rmse
flow_matching_test_loss on normalized space
per-subset raw metrics
per-horizon-step raw metrics h=0..H-1
per-action-dim raw metrics
inference latency
video/VL coverage
```

Output:

```text
gr00t_official_eval/
  eval_summary.json
  eval_metrics.csv
  per_subset_metrics.csv
  per_horizon_metrics.csv
  per_action_dim_metrics.csv
  prediction_vs_ground_truth_raw.png
  horizon_error_curve_raw.png
```

### 7.3 Bảng so sánh cần có

| Model | Input | S | H | K | Raw MSE | Raw MAE | Raw RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline train action mean | none | 1 | 16 | 0 | new | new | new |
| Baseline last-state repeat | state | 1 | 16 | 0 | new | new | new |
| NVIDIA GR00T zero-shot | state + video/lang | 1 | 16 | official/API | optional/skipped | optional/skipped | optional/skipped |
| OfficialMiniVLDiT pretrain | state + video/lang | 1 | 16 | 4 | new | new | new |
| OfficialMiniVLDiT posttrain | state + video/lang | 1 | 16 | 4 | new | new | new |

Zero-shot NVIDIA baseline là optional-safe:

- Notebook 04 cố load `nvidia/GR00T-N1.6-3B`.
- Nếu model bị gated, OOM, thiếu dependency, hoặc không expose action prediction API tương thích, notebook ghi row `NVIDIA_GR00T_zero_shot` với `status = "skipped"` và `skip_reason`.
- Zero-shot fail không được làm crash eval pipeline.

Acceptance:

```text
eval uses held-out episodes
eval metrics are computed on denormalized raw action space
eval still reports normalized flow-matching test loss separately
eval uses action_mask
eval reports number of video-backed samples
eval reports pretrain_eval_completed and posttrain_eval_completed
eval reports zeroshot_attempted, zeroshot_status, and zeroshot_model_id
```

## 8. Notebook 05: Strong Ablation

Notebook 05 là phần giúp pipeline đủ sức bảo vệ seminar, vì thầy yêu cầu ablation không chỉ là chạy metric.

Required ablations:

| Type | Run | Setting |
|---|---|---|
| Component | `state_only` | không dùng VL features |
| Component | `state_vl` | dùng frozen VL features + cross-attention |
| Model | `layers_4` vs `layers_8` | model-level ablation |
| Inference | `K=1,4,8,16` | số bước ODE denoising |
| Horizon | `H=8` vs `H=16` | action horizon ablation |
| Hyperparameter | `lr=1e-4` vs `5e-5` | learning rate |
| Training stage | `pretrain_only` vs `posttrain_task_specific` | before/after task-specific posttrain |

### 8.1 H ablation

H ablation là required trong plan mới.

Rule:

- chạy `H=8` và `H=16`;
- phải re-run Notebook 02 với `ACTION_HORIZON` khác nhau;
- không cần đổi model code, cùng model class phải nhận dynamic H;
- train/eval mỗi H với cùng config, chỉ đổi horizon.

Prepared outputs:

```text
gr00t_prepared_official_H8/
gr00t_prepared_official_H16/
```

Report cần nói rõ trade-off:

- H lớn hơn giúp dự đoán kế hoạch dài hơn;
- H lớn hơn cũng khó hơn, thường lỗi cuối horizon tăng;
- biểu đồ `horizon_error_curve_raw.png` dùng để minh họa lỗi theo bước tương lai.

Optional nếu còn GPU:

```text
hidden_dim 256 vs 512
STATE_HORIZON 1 vs 4
data scale 1 subset vs 3 subsets
optimizer AdamW vs Apollo
```

Output:

```text
gr00t_official_ablation/
  ablation_summary.csv
  ablation_summary.json
  ablation_bar_chart.png
  k_step_tradeoff.png
  horizon_ablation.png
  component_ablation.png
  pretrain_posttrain_comparison.csv
```

Acceptance:

```text
>= 6 ablation runs completed
includes component ablation
includes model-level ablation
includes inference-time K ablation
includes action horizon H ablation
includes pretrain vs posttrain comparison if Notebook 03 checkpoints exist
all reported MSE/MAE/RMSE are denormalized raw action metrics
all losses/metrics use action_mask where applicable
```

## 9. Smoke Test Config

Trước khi chạy full trên Kaggle, dùng config nhỏ để test pipeline end-to-end:

```text
MAX_EPISODES_PER_SUBSET = 5
MAX_ENCODE_SAMPLES = 256
MAX_TRAIN_SAMPLES = 2048
POSTTRAIN_MAX_SAMPLES = 512
MAX_EVAL_SAMPLES = 1024
EPOCHS_PRETRAIN = 1
EPOCHS_POSTTRAIN = 1
```

Smoke test pass khi:

```text
states_train.shape == [N, S, 44]
actions_train_chunk.shape == [N, H, 44]
action_mask.shape == [N, H, 44]
no chunk crosses episode boundary
vl_features.shape == [N_vl, T, D]
cross_attention_dim == D
masked loss runs without NaN
checkpoint reloads successfully
pretrain checkpoint exists
posttrain checkpoint exists or is explicitly skipped with reason
eval computes denormalized raw metrics
zero-shot NVIDIA fail path writes skipped row instead of crashing
H=8 and H=16 prepare outputs both run successfully
```

## 10. Full Acceptance Checklist

Pipeline mới được xem là hoàn tất cho seminar khi có đủ:

- Notebook 01 tải được raw subset có video/meta/data;
- Notebook 02 tạo prepared dataset official format cho `H=16`;
- Notebook 02 tạo thêm prepared dataset cho `H=8` nếu chạy H ablation;
- Notebook 03 encode được VL features hoặc fallback có ghi report rõ ràng;
- Notebook 03 pretrain DiT/action head from scratch ít nhất 1 epoch;
- Notebook 03 posttrain DiT/action head trên task-specific subset;
- Notebook 03 dùng Beta timestep sampling, correct flow target sign và masked loss;
- Notebook 04 evaluate held-out test split;
- Notebook 04 có bảng so sánh baseline mean, baseline last-state, NVIDIA zero-shot optional, pretrain, posttrain;
- Notebook 04 report MSE/MAE/RMSE trên raw denormalized action space;
- Notebook 05 có component ablation;
- Notebook 05 có model-level ablation;
- Notebook 05 có K ablation;
- Notebook 05 có H ablation;
- toàn bộ notebook có output report/json/csv/figure đủ để đưa vào slide.

## 11. Cách Chạy Lại Pipeline

Thứ tự chạy đề xuất:

```text
1. Run các notebook 01_download cho subset cần dùng.
2. Add output của các notebook 01 làm input cho Notebook 02.
3. Run Notebook 02 với ACTION_HORIZON=16.
4. Add output Notebook 02 làm input cho Notebook 03.
5. Run Notebook 03 để encode VL features, pretrain DiT, rồi posttrain DiT trên task-specific subset.
6. Add checkpoint/output Notebook 03 làm input cho Notebook 04.
7. Run Notebook 04 để lấy denormalized eval metrics cho pretrain, posttrain, và zero-shot optional.
8. Run Notebook 05 để tạo ablation tables/plots và bảng pretrain-vs-posttrain.
9. Nếu cần H ablation, quay lại Notebook 02 với ACTION_HORIZON=8 rồi train/eval lại nhánh H=8.
```

Trong Kaggle, mỗi notebook mới cần add input từ notebook trước:

```text
Notebook 02 input: output của các Notebook 01
Notebook 03 input: output của Notebook 02
Notebook 04 input: output của Notebook 03 + prepared data từ Notebook 02
Notebook 05 input: output của Notebook 02/03/04 hoặc tự chạy lại config nhỏ
```

## 12. Những Gì Có Và Chưa Có So Với Plan Cũ

Pipeline mới mạnh hơn bản mini cũ ở các điểm:

- có video/language feature thay vì chỉ state;
- có action chunk H=16 thay vì single-step;
- có state shape `[N, S, 44]`;
- có action mask;
- có Beta timestep sampling;
- có cross-attention dynamic theo feature dimension;
- có K-step ODE evaluation;
- có metric denormalized;
- có H ablation và component ablation.

Pipeline mới hiện đã đưa hai ý quan trọng trong conversation với thầy vào luồng chính:

- pretrain DiT/action head trên encoded train samples;
- posttrain cùng DiT/action head trên task-specific subset.

Zero-shot pretrained NVIDIA baseline cũng đã được thêm vào Notebook 04 theo dạng optional-safe. Đây là mốc so sánh tốt cho seminar, nhưng không phải điều kiện bắt buộc để pipeline pass vì có thể fail do gated model, VRAM, dependency hoặc API mismatch.

## 13. Cách Trình Bày Trong Seminar

Nên trình bày pipeline theo logic:

1. Paper gốc dùng VLM + DiT/action expert + flow matching cho robot action.
2. Tài nguyên sinh viên không đủ để train full VLM/action head lớn.
3. Nhóm xây dựng bản Strong Official-Mini:
   - frozen VLM để encode video/language;
   - train DiT/action head từ scratch;
   - dự đoán action chunk H=16;
   - dùng masked flow matching loss;
   - evaluate trên raw action space;
   - ablation nhiều trục.
4. Nhấn mạnh đây không còn là toy state-only pipeline.
5. Thành thật nêu limitation: không train VLM, không full dataset, không multi-GPU.

Một câu chốt phù hợp:

```text
Pipeline của nhóm không tái hiện full-scale GR00T do giới hạn GPU,
nhưng tái hiện các thành phần cốt lõi ở mức official-mini:
visual-language conditioning, action chunking, flow matching, masked loss,
open-loop denoising evaluation và ablation có ý nghĩa.
```
