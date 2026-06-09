# Tóm Tắt Bối Cảnh Và Tiến Độ CK - Isaac GR00T N1 / N1.6

Ngày cập nhật: 09/06/2026.

File này dùng để chuyển ngữ cảnh sang repo/cuộc trò chuyện khác. Nội dung ghi lại yêu cầu đồ án, hướng tiếp cận đã thống nhất, các notebook đã tạo/sửa, dữ liệu đã tải, lỗi đã gặp, và trạng thái hiện tại.

## 1. Đề Tài Và Yêu Cầu CK

Đề tài nhóm: **Isaac GR00T: N1 through N1.6**.

Nhóm có 3 thành viên.

Yêu cầu giảng viên cho phần thực nghiệm CK:

1. **Implementation - Mức Mini**
   - Train from scratch mô hình trên dataset gốc **GR00T-X-Embodiment-Sim** ít nhất 1 epoch.
   - Do hạn chế tài nguyên sinh viên, giảng viên cho phép cách tiếp cận:
     - VLM / vision-language backbone dùng pretrained và freeze.
     - Phần **DiT/action head** khởi tạo mới và train từ đầu.
   - Cần trình bày rõ: "from scratch" ở đây là train mới action module/DiT trong pipeline mini, không phải pretrain full VLA 2B/3B từ đầu.

2. **Evaluation - Mức Yes**
   - Đánh giá trên **test split** của dataset.
   - Cần có metric như MSE/MAE/loss trên test split.

3. **Ablation Study - Mức Yes**
   - Chạy benchmark so sánh khi thay đổi hyperparameters hoặc thành phần cấu trúc.
   - Ablation khả thi:
     - learning rate: `1e-4` vs `5e-5`
     - DiT layers: 4 vs 2
     - hidden dim / batch size / optimizer nếu còn thời gian
     - data scale: 3 subsets vs thêm subset nếu pipeline đã ổn

## 2. Cách Hiểu Paper Và Hướng From Scratch

Đã tham khảo paper gốc trong file `2503.14734v2.txt`.

Ý chính:

- GR00T N1 là Vision-Language-Action model gồm:
  - VLM / Eagle-2 hoặc Cosmos-style backbone xử lý image + language.
  - DiT/action module sinh action chunk bằng diffusion/flow matching.
- Paper gốc dùng tài nguyên rất lớn, ví dụ hàng chục nghìn H100 GPU hours cho pretraining.
- Vì vậy đồ án sinh viên không thể train full VLA từ đầu.

Cách trình bày được chốt:

```text
Nhóm reproduce mini pipeline của GR00T:
dùng pretrained/frozen VLM làm feature extractor,
train mới DiT/action head trên GR00T-X-Embodiment-Sim.
```

## 3. Scope Dữ Liệu Đã Chốt

Plan CK ban đầu trong `ck_action_plan.md` khá tham vọng:

- 2-phase pipeline:
  1. VLM encode features từ raw dataset.
  2. Train DiT/action head trên encoded features.
- Plan ban đầu nhắc đến khoảng 36 subsets hoặc 39 subsets tùy file hướng dẫn.

Sau khi kiểm tra giới hạn Kaggle:

- Không nên tải/encode 36-39 subset trong giai đoạn hiện tại.
- Chiến thuật tốt nhất:

```text
chạy end-to-end với 3 subset trước
-> có train/eval/ablation đầy đủ
-> nếu ổn mới tải thêm subset và train lại hoặc fine-tune tiếp
```

Ba subset GR-1 đang dùng:

```text
gr1_arms_only.CanSort
gr1_arms_waist.CupToDrawer
gr1_arms_waist.CanToDrawer
```

Đánh giá hiện tại:

- 3 subset là đủ để hoàn thành CK nếu pipeline đầy đủ.
- Không nên tải thêm trước khi chạy xong notebook 02/03/04/05.
- Nếu pipeline ổn, thêm subset sau có thể biến thành ablation "data scale".

## 4. Trạng Thái Dữ Liệu Đã Tải Trên Kaggle

Hiện đã tải xong các phần sau, theo minh chứng từ ảnh/output Kaggle:

| Kaggle notebook | Nội dung tải | Thời gian | Ghi chú |
|---|---|---:|---|
| `notebook1700561abd` | `gr1_arms_only.CanSort` | 6 phút | Có output `gr00t_x_embodiment_sim` và `download_report_1_5.json` |
| `notebookbcd29698a0` | `gr1_arms_waist.CupToDrawer` | 1 tiếng 40 phút | Full subset, có data/meta/videos |
| `notebookb6c37dbbd0` | `gr1_arms_waist.CanToDrawer` | 24 phút | Chỉ tải `data/` + `meta/` |
| `notebookd3f6a10db4` | Video supplement của `gr1_arms_waist.CanToDrawer` | 20 phút | Camera `observation.images.ego_view`, chunks `0,1,2` |

Kết luận dữ liệu:

- `CanSort`: đã có.
- `CupToDrawer`: đã có full raw subset, gồm video.
- `CanToDrawer`: đã có data/meta và có thêm video supplement giới hạn chunk 0,1,2.
- Cách tải này phù hợp với giới hạn Kaggle hơn so với tải full video toàn bộ `CanToDrawer`.

## 5. Notebook Đã Có / Đã Tạo

### 5.1 `01_download_subsets_1_5.ipynb`

Mục đích:

- Download subset từ HuggingFace.
- Tạo `download_report_1_5.json`.
- CPU-only, không cần GPU.

Lưu ý:

- Pattern full subset `allow_patterns=f"{subset_name}/**"` có thể gây tràn disk với subset lớn.
- Hiện đã dùng thành công cho một số subset, nhưng không nên tiếp tục tải nhiều full raw subset cùng lúc.

### 5.2 `01_download_subsets_6_10.ipynb`

Mục đích:

- Download subset nhóm 6-10.
- Tạo `download_report_6_10.json`.

Lưu ý:

- Notebook cũ full raw từng gây tràn disk với `CanToDrawer`.
- Sau đó chiến lược đã đổi: `CanToDrawer` data/meta tải riêng, video supplement tải riêng.

### 5.3 `01_supplement_can_to_drawer_videos.ipynb`

Mục đích:

- Bổ sung video cho `gr1_arms_waist.CanToDrawer`.
- Không tải full video toàn subset.
- Mặc định tải:

```python
camera_key = "observation.images.ego_view"
CHUNK_IDS = [0, 1, 2]
```

Output:

```text
/kaggle/working/gr00t_x_embodiment_sim_video_supplement/
/kaggle/working/video_supplement_report_CanToDrawer.json
```

Trạng thái:

- Đã chạy trên Kaggle và tải xong video supplement theo ảnh minh chứng.

### 5.4 `02_prepare_splits_and_features.ipynb`

Mục đích:

- Đọc 3 subset đã tải.
- Scan `meta/`, `data/`, `videos/`.
- Đọc parquet.
- Detect episode/frame/action/state columns.
- Split train/test theo episode.
- Lưu cache `.npy`, `.parquet`, `.json` cho notebook train.

Trạng thái hiện tại:

- **Đã tạo file notebook.**
- **Đã sửa để phù hợp với cách tải hiện tại của 4 Kaggle outputs.**
- **Chưa chạy trên Kaggle.**

Notebook 2 hiện đã hỗ trợ:

- Không còn giả định 3 subset nằm chung một `RAW_ROOT`.
- Tự scan nhiều Kaggle input bằng `RAW_ROOTS`.
- Tạo `SUBSET_SOURCES` để mỗi subset có thể lấy:
  - `data/meta` từ một output dataset;
  - `videos` từ output dataset khác.
- Ghi minh chứng tải vào:
  - `dataset_scan_report.json`
  - `prepare_report.json`

Khi chạy notebook 2 trên Kaggle, cần Add Input đủ 4 output datasets:

```text
notebook1700561abd
notebookbcd29698a0
notebookb6c37dbbd0
notebookd3f6a10db4
```

Notebook 2 **không cần GPU**. Để Kaggle:

```text
Accelerator: None
```

Nên chạy thử với:

```python
MAX_FRAMES_PER_EPISODE = 200
```

Nếu chạy ổn, có thể đổi thành:

```python
MAX_FRAMES_PER_EPISODE = None
```

Output mong đợi của notebook 2:

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
  states_train.npy      # nếu đọc được state
  states_test.npy       # nếu đọc được state
```

### 5.5 `notebook_creation_plan.md`

Mục đích:

- Liệt kê toàn bộ notebook cần tạo cho CK.
- Ghi mục tiêu từng notebook.
- Mapping từng notebook với các phần của `ck_action_plan.md`.

Các notebook trong plan:

```text
01_download_subsets_*.ipynb
02_prepare_splits_and_features.ipynb
03_train_dit_from_scratch.ipynb
04_evaluate_test_split.ipynb
05_ablation_study.ipynb
06_collect_results_for_report.ipynb   # optional
```

## 6. Lỗi Đã Gặp Và Cách Xử Lý

### 6.1 HuggingFace 403 Với Gated Model

Lỗi:

```text
403 Forbidden: Please enable access to public gated repositories...
```

Nguyên nhân:

- Token HuggingFace chưa có quyền public gated repos hoặc chưa accept model/dataset gated.

Cách xử lý:

- Tạo HF token mới.
- Bật permission read public gated repos.
- Accept model/dataset cần dùng.
- Lưu token vào Kaggle Secret `HF_TOKEN`.

### 6.2 Kaggle Save Version Lỗi Notebook Source > 1MB

Lỗi:

```text
The kernel source must be less than 1 megabytes in size.
```

Nguyên nhân:

- Notebook cũ chứa output/log/papermill metadata quá lớn.

Cách xử lý:

- Clear outputs.
- Reset metadata.
- Tạo notebook sạch, ít cell.

### 6.3 HuggingFace HTTP 429 Rate Limit

Nguyên nhân:

- `snapshot_download` tạo quá nhiều request.

Cách xử lý:

- Dùng `max_workers=2` hoặc `1`.
- Tránh chạy nhiều notebook download cùng lúc bằng cùng token.

### 6.4 Kaggle Tràn Disk Khi Tải Full Raw Subset

Log cũ cho thấy `CanToDrawer` bị tràn disk khi tải full raw:

```text
OSError: [Errno 28] No space left on device
videos/chunk-008/observation.images.ego_view
```

Kết luận:

- Không phải lỗi token hay cú pháp.
- Nguyên nhân là tải full raw subset gồm nhiều video/chunk/camera.

Cách xử lý đã chọn:

```text
CanToDrawer:
  - tải data/meta riêng
  - tải video supplement riêng, chỉ camera ego_view, chunk 0,1,2
```

## 7. Có Cần Video Không?

Dựa vào `ck_action_plan.md`:

- Nếu làm đúng tinh thần 2-phase pipeline "VLM encode features -> train DiT", thì cần ảnh/video frame cho VLM.
- Nhưng không cần tải full video cho mọi subset.

Chiến lược hiện tại:

```text
3 subset: có data/meta để train/eval action pipeline
CupToDrawer: có full video
CanToDrawer: có video supplement chunks 0,1,2
CanSort: có dữ liệu từ notebook download, cần notebook 2 scan lại xác nhận số video/parquet
```

Điều này đủ hợp lý cho CK:

- Dùng parquet/action/state cho pipeline train/eval.
- Có video representative cho phần visual/VLM/demo.
- Không làm Kaggle tràn disk.

## 8. Tiến Độ Hiện Tại

Đã xong:

- Phân tích yêu cầu CK.
- Chốt cách hiểu from scratch: train DiT/action head từ đầu, VLM pretrained/frozen.
- Chốt chiến lược 3 subset trước.
- Tải xong dữ liệu theo 4 Kaggle notebooks như mục 4.
- Tạo `notebook_creation_plan.md`.
- Tạo và sửa `02_prepare_splits_and_features.ipynb`.
- Tạo `01_supplement_can_to_drawer_videos.ipynb`.
- Tạo `teacher_audio_notes.md`.

Chưa xong:

- **Chưa chạy notebook 2 trên Kaggle.**
- Chưa tạo notebook 3 train DiT.
- Chưa train 1 epoch.
- Chưa evaluate test split.
- Chưa chạy ablation.

Việc cần làm ngay tiếp theo:

1. Mở `02_prepare_splits_and_features.ipynb` trên Kaggle.
2. Add Input đủ 4 output datasets:

```text
notebook1700561abd
notebookbcd29698a0
notebookb6c37dbbd0
notebookd3f6a10db4
```

3. Để Accelerator: `None`.
4. Chạy với:

```python
MAX_FRAMES_PER_EPISODE = 200
```

5. Kiểm tra output:

```text
prepare_report.json
dataset_scan_report.json
actions_train.npy
actions_test.npy
train_samples.parquet
test_samples.parquet
```

6. Nếu notebook 2 chạy ổn, Save Version output thành Kaggle Dataset.
7. Sau đó tạo/chạy notebook 3: `03_train_dit_from_scratch.ipynb`.

## 9. File Quan Trọng Trong Repo

```text
ck_action_plan.md
ck_context_summary.md
notebook_creation_plan.md
teacher_audio_notes.md
01_download_subsets_1_5.ipynb
01_download_subsets_6_10.ipynb
01_supplement_can_to_drawer_videos.ipynb
02_prepare_splits_and_features.ipynb
```

## 10. Prompt Gợi Ý Khi Chuyển Sang Repo/Cuộc Trò Chuyện Khác

```text
Chúng tôi đang làm CK Isaac GR00T N1/N1.6.
Yêu cầu: train from scratch ít nhất 1 epoch trên GR00T-X-Embodiment-Sim, evaluation trên test split, ablation study.
Giảng viên cho phép dùng pretrained/frozen VLM và train DiT/action head từ scratch.

Hiện đã tải xong 3 subset GR-1:
- CanSort từ notebook1700561abd
- CupToDrawer từ notebookbcd29698a0
- CanToDrawer data/meta từ notebookb6c37dbbd0
- CanToDrawer video supplement chunks 0,1,2 từ notebookd3f6a10db4

Đã tạo và sửa notebook 2: 02_prepare_splits_and_features.ipynb.
Notebook 2 hiện chưa chạy.
Cần chạy notebook 2 trên Kaggle với Accelerator None, Add Input đủ 4 datasets, MAX_FRAMES_PER_EPISODE=200.
Sau đó tạo notebook 3 để train DiT/action head from scratch.
```
