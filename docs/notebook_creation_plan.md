# Kế Hoạch Tạo Notebook Cho CK Isaac GR00T N1 / N1.6

File này liệt kê các notebook cần có để thực hiện phần thực nghiệm CK theo `ck_action_plan.md`, sau khi đã điều chỉnh scope cho tài nguyên sinh viên.

## Mục Tiêu Pipeline

Mục tiêu thực nghiệm là chạy được pipeline end-to-end trên khoảng **3 subset GR-1** của dataset gốc `nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`:

```text
download data -> prepare split/features -> train DiT/action head from scratch -> evaluate test split -> ablation
```

Cách hiểu "from scratch" trong đồ án:

- VLM / vision-language backbone dùng pretrained và freeze.
- Phần DiT/action head khởi tạo mới và train từ đầu.
- Đây là phiên bản mini/reproduce thực tế của pipeline GR00T, không phải train full VLA 2B/3B từ đầu.

## Danh Sách Notebook

| Thứ tự | Notebook | Trạng thái | GPU | Vai trò trong plan CK |
|---:|---|---|---|---|
| 01A | `01_download_subsets_1_5.ipynb` | Đã có | Không cần | Download raw dataset gốc, hiện dùng cho subset 1-5 |
| 01B | `01_download_subsets_6_10.ipynb` | Đã có | Không cần | Download raw dataset gốc, hiện dùng cho subset 6-10 |
| 02 | `02_prepare_splits_and_features.ipynb` | Cần tạo | CPU trước, GPU optional | Chuẩn bị train/test split và cache dữ liệu train-ready |
| 03 | `03_train_dit_from_scratch.ipynb` | Đã có | Cần GPU | Train DiT/action head từ scratch ít nhất 1 epoch |
| 04 | `04_evaluate_test_split.ipynb` | Cần tạo | GPU khuyến nghị | Evaluation trên test split |
| 05 | `05_ablation_study.ipynb` | Cần tạo | Cần GPU | Ablation hyperparameters / cấu trúc DiT |
| 06 | `06_collect_results_for_report.ipynb` | Optional | Không cần | Tổng hợp bảng, biểu đồ, file kết quả cho slide/báo cáo |

## Notebook 01 - Download Subsets

### Mục tiêu

Tải subset từ HuggingFace về Kaggle để có dữ liệu gốc phục vụ các bước sau.

### Notebook hiện có

- `01_download_subsets_1_5.ipynb`
- `01_download_subsets_6_10.ipynb`

### Công đoạn trong plan CK

Notebook này thuộc phần **Implementation / Data Pipeline**.

Nó chứng minh nhóm sử dụng dataset gốc `GR00T-X-Embodiment-Sim`, không dùng dữ liệu tự chế.

### Input

- HuggingFace token trong Kaggle Secret: `HF_TOKEN`
- Dataset: `nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`

### Output

```text
/kaggle/working/gr00t_x_embodiment_sim/
/kaggle/working/download_report_*.json
```

### Ghi chú quan trọng

Không nên tải quá nhiều subset raw cùng lúc. Với disk Kaggle khoảng 57GB, chiến lược an toàn là tải khoảng **3 subset GR-1** trước để chạy end-to-end.

Ba subset ưu tiên:

```text
gr1_arms_only.CanSort
gr1_arms_waist.CanToDrawer
gr1_arms_waist.CupToDrawer
```

## Notebook 02 - Prepare Splits And Features

### File cần tạo

```text
02_prepare_splits_and_features.ipynb
```

### Mục tiêu

Biến raw dataset đã tải thành dữ liệu sạch để train/evaluate.

Notebook này là cầu nối giữa download và train. Không nên train ngay nếu chưa có notebook 02, vì dễ bị lỗi split sai, thiếu action label, hoặc leakage giữa train/test.

### Công đoạn trong plan CK

Notebook này thuộc phần:

- **Implementation**: chuẩn bị dữ liệu train từ dataset gốc.
- **Evaluation**: tạo test split đúng.
- **Ablation**: tạo cùng một split cố định để các cấu hình so sánh công bằng.

### Việc notebook làm

- Tìm thư mục raw dataset trong:

```text
/kaggle/input/.../gr00t_x_embodiment_sim/
/kaggle/working/gr00t_x_embodiment_sim/
```

- Kiểm tra từng subset có:
  - `meta/`
  - `data/`
  - `videos/`
- Đọc các file parquet trong `data/`.
- Tự detect các cột:
  - episode id
  - frame id
  - state
  - action
  - task/language nếu có
- Chia train/test theo episode, mặc định 80/20.
- Lưu cache `.npy` và `.parquet` cho notebook 03.

### Output bắt buộc

```text
/kaggle/working/gr00t_prepared_3subsets/
  dataset_scan_report.json
  prepare_report.json
  splits_train_episodes.json
  splits_test_episodes.json
  train_samples.parquet
  test_samples.parquet
  actions_train.npy
  actions_test.npy
  sample_index.parquet
```

Nếu đọc được state thì có thêm:

```text
states_train.npy
states_test.npy
```

### Tiêu chí chạy xong

- Có ít nhất 3 subset được scan.
- `action_dim > 0`.
- `num_train_episodes > 0`.
- `num_test_episodes > 0`.
- Không có episode nào xuất hiện đồng thời ở train và test.

## Notebook 03 - Train DiT From Scratch

### File cần tạo

```text
03_train_dit_from_scratch.ipynb
```

### Mục tiêu

Train DiT/action head khởi tạo mới trên dữ liệu đã chuẩn bị từ notebook 02.

Đây là notebook quan trọng nhất cho yêu cầu:

```text
Implement Mini: train from scratch ít nhất 1 epoch trên dataset gốc.
```

### Công đoạn trong plan CK

Notebook này thuộc phần **Implementation**.

### Input

```text
/kaggle/input/gr00t_prepared_3subsets/
```

Các file chính:

```text
actions_train.npy
states_train.npy
train_samples.parquet
prepare_report.json
```

### Việc notebook làm

- Load cache từ notebook 02.
- Khởi tạo model action head/DiT mini từ đầu.
- Train ít nhất 1 epoch.
- Log train loss theo step/epoch.
- Lưu checkpoint và config.

### Output bắt buộc

```text
/kaggle/working/gr00t_dit_runs/
  config.json
  train_log.csv
  loss_curve.png
  checkpoint_last.pt
  train_summary.json
```

### Config baseline đề xuất

```text
epochs = 1
batch_size = 64
learning_rate = 1e-4
optimizer = AdamW
hidden_dim = 256 hoặc 512
num_layers = 4
seed = 42
```

Nếu GPU yếu hoặc hết RAM, giảm theo thứ tự:

```text
batch_size -> hidden_dim -> num_layers
```

## Notebook 04 - Evaluate Test Split

### File cần tạo

```text
04_evaluate_test_split.ipynb
```

### Mục tiêu

Đánh giá checkpoint từ notebook 03 trên test split đã tạo ở notebook 02.

### Công đoạn trong plan CK

Notebook này thuộc phần **Evaluation Yes**.

### Input

```text
/kaggle/input/gr00t_prepared_3subsets/
/kaggle/input/gr00t_dit_runs/
```

### Việc notebook làm

- Load model checkpoint.
- Load `actions_test.npy`, `states_test.npy`, `test_samples.parquet`.
- Chạy inference offline/open-loop.
- Tính metric:
  - MSE
  - MAE
  - RMSE nếu cần
  - per-subset MSE nếu đủ thông tin subset trong `test_samples.parquet`
- Vẽ biểu đồ loss/eval error.

### Output bắt buộc

```text
/kaggle/working/gr00t_eval_results/
  eval_summary.json
  eval_metrics.csv
  per_subset_metrics.csv
  prediction_vs_ground_truth.png
```

### Bảng cần đưa vào báo cáo

| Model | Train data | Test data | MSE | MAE |
|---|---|---|---:|---:|
| DiT from scratch baseline | 3 GR-1 subsets | held-out episodes | ... | ... |

## Notebook 05 - Ablation Study

### File cần tạo

```text
05_ablation_study.ipynb
```

### Mục tiêu

Chạy so sánh khi thay đổi hyperparameter hoặc thành phần cấu trúc, đáp ứng yêu cầu **Ablation Study Yes**.

### Công đoạn trong plan CK

Notebook này thuộc phần **Ablation Study**.

### Input

```text
/kaggle/input/gr00t_prepared_3subsets/
```

Có thể dùng lại code train/eval từ notebook 03 và 04.

### Ablation tối thiểu nên chạy

| Run | Learning rate | Layers | Hidden dim | Mục đích |
|---|---:|---:|---:|---|
| baseline | `1e-4` | 4 | 256/512 | cấu hình chính |
| lr_low | `5e-5` | 4 | 256/512 | kiểm tra ảnh hưởng learning rate |
| shallow | `1e-4` | 2 | 256/512 | kiểm tra ảnh hưởng số layer |

Nếu còn thời gian/GPU, thêm:

| Run | Thay đổi | Mục đích |
|---|---|---|
| hidden_small | hidden dim nhỏ hơn | model capacity |
| batch_small | batch nhỏ hơn | stability |
| optimizer_sgd_or_adamw_variant | optimizer | training dynamics |

### Output bắt buộc

```text
/kaggle/working/gr00t_ablation_results/
  ablation_summary.csv
  ablation_summary.json
  ablation_bar_chart.png
  run_baseline/
  run_lr_low/
  run_shallow/
```

### Bảng cần đưa vào báo cáo

| Run | LR | Layers | Hidden dim | Test MSE | Test MAE | Nhận xét |
|---|---:|---:|---:|---:|---:|---|
| baseline | ... | ... | ... | ... | ... | ... |
| lr_low | ... | ... | ... | ... | ... | ... |
| shallow | ... | ... | ... | ... | ... | ... |

## Notebook 06 - Collect Results For Report

### File optional

```text
06_collect_results_for_report.ipynb
```

### Mục tiêu

Tổng hợp tất cả output từ các notebook trước thành bảng/ảnh dùng trực tiếp cho slide CK.

### Công đoạn trong plan CK

Notebook này hỗ trợ phần trình bày:

- Method
- Implementation
- Evaluation
- Ablation

### Input

```text
/kaggle/input/gr00t_prepared_3subsets/
/kaggle/input/gr00t_dit_runs/
/kaggle/input/gr00t_eval_results/
/kaggle/input/gr00t_ablation_results/
```

### Output

```text
/kaggle/working/final_report_assets/
  dataset_summary_table.csv
  training_summary_table.csv
  evaluation_table.csv
  ablation_table.csv
  loss_curve.png
  ablation_bar_chart.png
  final_ck_summary.md
```

## Thứ Tự Chạy Khuyến Nghị

Không chạy lan man nhiều subset trước. Chạy theo thứ tự này để có kết quả CK nhanh nhất:

1. Chạy notebook 01 để có 3 subset GR-1.
2. Save output notebook 01 thành Kaggle Dataset.
3. Chạy `02_prepare_splits_and_features.ipynb`.
4. Save output notebook 02 thành Kaggle Dataset.
5. Chạy `03_train_dit_from_scratch.ipynb` ít nhất 1 epoch.
6. Chạy `04_evaluate_test_split.ipynb`.
7. Chạy `05_ablation_study.ipynb` với ít nhất 3 run.
8. Nếu còn thời gian, chạy `06_collect_results_for_report.ipynb`.

## Mapping Với Yêu Cầu Giảng Viên

| Yêu cầu | Notebook chứng minh |
|---|---|
| Train from scratch trên dataset gốc ít nhất 1 epoch | 01 + 02 + 03 |
| Evaluation trên test split | 02 + 04 |
| Ablation Study | 05 |
| Dùng dataset gốc GR00T-X-Embodiment-Sim | 01 + report download |
| Giải thích hạn chế tài nguyên và cách tiếp cận mini | 03 + slide/report |

## Deliverables Tối Thiểu Để Đủ CK

Nếu thời gian gấp, tối thiểu phải có:

```text
01_download_subsets_*.ipynb
02_prepare_splits_and_features.ipynb
03_train_dit_from_scratch.ipynb
04_evaluate_test_split.ipynb
05_ablation_study.ipynb
```

Kết quả tối thiểu phải có:

```text
download_report.json
prepare_report.json
train_summary.json
eval_summary.json
ablation_summary.csv
loss_curve.png
ablation_bar_chart.png
```

Đây là bộ bằng chứng đủ để trình bày rằng nhóm đã đi qua đầy đủ pipeline thực nghiệm của plan CK.
