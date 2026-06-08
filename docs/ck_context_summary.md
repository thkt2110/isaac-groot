# Tóm Tắt Bối Cảnh CK - Isaac GR00T N1 / N1.6

## 1. Đề Tài Và Yêu Cầu CK

Đề tài nhóm: **Isaac GR00T: N1 through N1.6**.

Nhóm có 3 thành viên.

Yêu cầu giảng viên cho phần thực nghiệm cuối kỳ:

1. **Implementation - Mức Mini**
   - Tự train from scratch mô hình trên dataset gốc **GR00T-X-Embodiment-Sim** ít nhất 1 epoch.
   - Trong bối cảnh tài nguyên sinh viên, giảng viên chấp nhận cách tiếp cận:
     - VLM / vision-language backbone dùng pretrained và freeze/load lại.
     - Phần **DiT/action head** train from scratch.
   - Cần trình bày rõ: "from scratch" ở đây là train mới action module/DiT theo pipeline mini, không phải pretrain full VLA 2B/3B từ đầu.

2. **Evaluation - Mức Yes**
   - Đánh giá trên **test split** của dataset.
   - Cần có bảng kết quả như MSE/MAE/loss trên test split.

3. **Ablation Study - Mức Yes**
   - Chạy benchmark so sánh khi thay đổi hyperparameters hoặc thành phần cấu trúc.
   - Các ablation khả thi:
     - learning rate: `1e-4` vs `5e-5`
     - DiT layers: 4 vs 8
     - optimizer: AdamW vs Apollo nếu kịp
     - inference K / action horizon H nếu có eval pipeline sẵn sàng

## 2. Paper Gốc Và Cách Hiểu From Scratch

Đã tham khảo file paper gốc: `2503.14734v2.txt`.

Ý chính cần nhớ:

- GR00T N1 là Vision-Language-Action model gồm:
  - VLM / Eagle-2 backbone xử lý image + language.
  - DiT/action module sinh action chunk bằng flow matching.
- Paper có nói quy mô train rất lớn, ví dụ GR00T-N1-2B dùng khoảng **50,000 H100 GPU hours** cho pretraining.
- Vì vậy với đồ án sinh viên, không khả thi train full model từ đầu.
- Cách trình bày nên là:
  - "Chúng em reproduce mini version của pipeline: dùng pretrained/frozen VLM làm feature extractor, train mới DiT/action head trên GR00T-X-Embodiment-Sim."

## 3. Kế Hoạch Ban Đầu Và Điều Chỉnh Scope

File plan ban đầu: `ck_action_plan.md`.

Plan ban đầu khá tham:

- 2-phase pipeline:
  1. Phase 1: VLM encode features từ raw dataset.
  2. Phase 2: train DiT/action head trên encoded features.
- Dự kiến xử lý nhiều subset:
  - Bản `ck_action_plan.md` ghi **36 subsets**: 27 GR-1 + 9 cross-embodiment.
  - File hướng dẫn phụ có danh sách gần **39 subsets**, hơi lệch với plan chính.

Kết luận sau khi xem giới hạn Kaggle:

- Không nên cố gắng tải/encode 36 subset trong 7 ngày.
- Nên chốt scope CK an toàn:
  - 1-3 subset GR-1 là tối thiểu khả thi.
  - 3-5 subset nếu mọi thứ chạy tốt.
  - Quan trọng là end-to-end: download -> split -> encode -> train 1 epoch -> eval -> ablation.

## 4. Notebook Cũ Và Kết Luận Về Caption

Notebook cũ ban đầu tên `groot-n1.ipynb` / notebook Kaggle cũ:

- Cài `transformers`, `datasets`, `decord`, `accelerate`.
- Login HuggingFace.
- Load model `nvidia/Cosmos-Reason2-2B`.
- Stream dataset `nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`.
- Tải từng video episode.
- Dùng Cosmos VLM để caption video: "robot đang làm gì".
- Lưu output `.jsonl` gồm `episode_index`, `video_file`, `output`.

Kết luận:

- Phần caption video **không quan trọng cho pipeline train DiT**.
- Load model Cosmos chỉ có ý nghĩa nếu muốn sinh caption demo/qualitative cho slides.
- Caption không thay thế được:
  - VLM backbone embeddings
  - state
  - action
  - attention mask
  - target action cho flow matching
- Để đi đường chính CK, đã quyết định **bỏ caption khỏi notebook download**.

## 5. Notebook Đã Tạo / Đã Sửa

### 5.1 `01_download_subsets_1_5.ipynb`

Mục đích:

- Notebook CPU-only để download + verify subset 1-5.
- Không cần GPU.
- Dùng Kaggle Secret `HF_TOKEN`.
- Download raw data từ HuggingFace.
- Tạo report JSON.

Subsets:

1. `gr1_arms_only.CanSort`
2. `gr1_arms_waist.CanToDrawer`
3. `gr1_arms_waist.CupToDrawer`
4. `gr1_arms_waist.CuttingboardToBasket`
5. `gr1_arms_waist.CuttingboardToCardboardBox`

Output mong đợi trên Kaggle:

```text
/kaggle/working/gr00t_x_embodiment_sim/
/kaggle/working/download_report_1_5.json
```

Hiện notebook đang dùng:

```python
max_workers=3
time.sleep(15)
```

Lưu ý: cách download full raw subset bằng `allow_patterns=f"{subset_name}/**"` đã gây tràn disk với subset lớn. Cần sửa chiến thuật nếu tiếp tục dùng.

### 5.2 `01_download_subsets_6_10.ipynb`

Mục đích:

- Notebook CPU-only để download + verify subset 6-10.
- Tương tự notebook 1-5.

Subsets:

6. `gr1_arms_waist.CuttingboardToPan`
7. `gr1_arms_waist.CuttingboardToPot`
8. `gr1_arms_waist.CuttingboardToTieredBasket`
9. `gr1_arms_waist.PlacematToBasket`
10. `gr1_arms_waist.PlacematToBowl`

Output mong đợi:

```text
/kaggle/working/gr00t_x_embodiment_sim/
/kaggle/working/download_report_6_10.json
```

Hiện notebook đã được sửa:

```python
max_workers=3
time.sleep(15)
```

File sạch, khoảng 9KB, không còn output/papermill metadata cũ.

## 6. Lỗi Đã Gặp

### 6.1 HuggingFace 403 với Cosmos

Lỗi:

```text
403 Forbidden: Please enable access to public gated repositories...
```

Nguyên nhân:

- Token HuggingFace chưa có quyền public gated repo hoặc chưa accept model `nvidia/Cosmos-Reason2-2B`.

Đã hướng dẫn khắc phục:

- Tạo token HuggingFace mới.
- Tick permission: read public gated repos.
- Accept model page `nvidia/Cosmos-Reason2-2B`.
- Lưu token vào Kaggle Secret `HF_TOKEN`.
- Login bằng:

```python
from huggingface_hub import login, whoami
from kaggle_secrets import UserSecretsClient

token = UserSecretsClient().get_secret("HF_TOKEN")
login(token=token)
print(whoami()["name"])
```

### 6.2 Kaggle Save Version lỗi notebook > 1MB

Lỗi:

```text
The kernel source must be less than 1 megabytes in size.
```

Nguyên nhân:

- Notebook cũ có output/log/papermill widget metadata rất lớn do đã caption 1000 episode.

Đã khắc phục:

- Clear all outputs.
- Reset metadata.
- Tạo notebook download sạch chỉ 5 cell, file chỉ khoảng 9KB.

### 6.3 HuggingFace HTTP 429 Rate Limit

Lỗi:

```text
HTTP Error 429
Rate limited. Waiting...
```

Nguyên nhân:

- `snapshot_download` gọi quá nhiều request HEAD/GET, đặc biệt dataset có rất nhiều file nhỏ.

Hướng xử lý:

- Dùng `max_workers=2` hoặc `3`.
- Nếu vẫn bị 429 nhiều thì hạ xuống `max_workers=1`.
- Tránh chạy nhiều Kaggle notebooks cùng lúc bằng cùng token.

### 6.4 Lỗi Chính Hiện Tại: Kaggle Tràn Disk

File log: `bug.txt`.

Dòng quan trọng:

```text
Your notebook tried to use more disk space than is available.
Not enough free disk space to download the file.
OSError: [Errno 28] No space left on device
```

Chuyện gì đã xảy ra:

- Notebook subset 1-5 tải xong subset đầu:

```text
gr1_arms_only.CanSort
Subset size: 1.25 GB
```

- Sau đó tải subset thứ 2:

```text
gr1_arms_waist.CanToDrawer
```

- Dataset đang tải đến:

```text
videos/chunk-008/observation.images.ego_view
```

- Disk còn 0MB và notebook chết.

Kết luận:

- Không phải tải sai cú pháp.
- Vấn đề là `allow_patterns=f"{subset_name}/**"` tải **toàn bộ raw subset**, gồm data/meta/videos và nhiều camera views/chunks.
- Một số subset lớn hơn dự kiến rất nhiều.
- Kaggle 57GB không đủ để giữ 5 full raw subset lớn cùng lúc.
- Trong lúc download, HuggingFace còn tạo cache tạm:

```text
/kaggle/working/gr00t_x_embodiment_sim/.cache/huggingface/download/...
```

nên dung lượng tạm thời còn có thể cao hơn final size.

## 7. Chiến Lược Download Nên Đổi Sang

Sau khi đọc log, chiến lược tốt nhất:

```text
1 notebook = 1 subset + tải partial camera
```

Không nên:

```text
5 subset raw full cùng lúc
```

Nên tải partial:

```text
meta/**
data/**
1 camera view video
```

Mẫu pattern:

```python
CAMERA_KEYS = [
    "observation.images.ego_view",
]

allow_patterns = [
    f"{subset_name}/meta/**",
    f"{subset_name}/data/**",
]

for cam in CAMERA_KEYS:
    allow_patterns.append(f"{subset_name}/videos/**/{cam}/**")

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=allow_patterns,
    local_dir=str(local_dir),
    max_workers=2,
)
```

Nếu subset không có `ego_view`, thử:

```python
CAMERA_KEYS = [
    "observation.images.front_view",
]
```

Nếu cần kiểm tra camera nào tồn tại, có thể mở HuggingFace file tree hoặc dùng partial probe bằng API.

## 8. Trạng Thái Hiện Tại

Đã làm được:

- Hiểu rõ yêu cầu CK và cách giải thích "from scratch" cho DiT/action head.
- Xác định notebook caption cũ không phải đường chính.
- Tạo notebook download sạch:
  - `01_download_subsets_1_5.ipynb`
  - `01_download_subsets_6_10.ipynb`
- Xử lý vấn đề token HuggingFace gated repo.
- Xử lý vấn đề notebook output quá lớn.
- Phân tích log `bug.txt`: lỗi hiện tại là tràn disk, không phải token/code.

Đang ở giai đoạn:

```text
Stage 01 - Download raw dataset / data access
```

Chưa sang:

```text
Stage 02 - train/test split
Stage 03 - VLM feature encoding
Stage 04 - train DiT from scratch
Stage 05 - evaluation
Stage 06 - ablation
```

## 9. Việc Cần Làm Tiếp Theo

### Việc gấp nhất

1. Sửa lại chiến lược download:
   - mỗi notebook 1 subset
   - tải partial camera
   - dùng `max_workers=2`
   - không tải full subset `/**`

2. Nếu dùng session Kaggle cũ bị đầy disk:
   - restart session hoặc xóa folder lỗi:

```python
!rm -rf /kaggle/working/gr00t_x_embodiment_sim/.cache
!rm -rf /kaggle/working/gr00t_x_embodiment_sim/gr1_arms_waist.CanToDrawer
!df -h /kaggle/working
```

3. Chạy lại download cho subset nhỏ trước:
   - ưu tiên `gr1_arms_only.CanSort`
   - sau đó thử 1 subset GR-1 khác với partial camera.

### Notebook cần tạo tiếp

Nên tạo theo thứ tự:

```text
01_download_subset_<name>.ipynb
02_make_train_test_split.ipynb
03_encode_vlm_features.ipynb
04_train_dit_from_scratch.ipynb
05_eval_and_ablation.ipynb
```

### Output của Stage 01

Mỗi notebook download nên tạo:

```text
/kaggle/working/gr00t_x_embodiment_sim/<subset_name>/
/kaggle/working/download_report_<subset_name>.json
```

Sau khi Save Version trên Kaggle, notebook tiếp theo dùng:

```text
Add Input -> output của notebook download
```

dữ liệu sẽ nằm ở:

```text
/kaggle/input/<notebook-or-dataset-name>/gr00t_x_embodiment_sim/
```

### Output của Stage 02

Train/test split:

```text
/kaggle/working/splits/
  <subset_name>_train.json
  <subset_name>_test.json
```

### Output của Stage 03

Encoded features:

```text
/kaggle/working/encoded_features/
  <subset_name>/
    train/
    test/
```

Mỗi sample nên có:

```python
{
    "backbone_features": ...,
    "backbone_attention_mask": ...,
    "image_mask": ...,
    "state": ...,
    "action": ...,
    "episode_index": ...,
    "subset_name": ...
}
```

### Output của Stage 04

Train DiT/action head from scratch:

```text
/kaggle/working/checkpoints/
/kaggle/working/train_loss.csv
```

Cần có bảng:

```text
epoch, train_loss, val_loss
```

### Output của Stage 05

Evaluation:

```text
/kaggle/working/eval_results.csv
/kaggle/working/plots/
```

Metrics:

```text
MSE
MAE
test loss
```

### Output của Stage 06

Ablation:

```text
/kaggle/working/ablation_results.csv
```

Bảng để báo cáo:

```text
Run | LR | DiT layers | Batch size | Epoch | MSE | MAE
```

## 10. Scope 7 Ngày Khả Thi

Có thể hoàn thành CK trong 7 ngày nếu cắt scope:

Không nên:

```text
36 subset raw/full
```

Nên:

```text
1-3 subset GR-1, chạy end-to-end thật chắc
```

Mục tiêu nhỏ nhưng đủ điểm:

1. Train DiT/action head from scratch trên 1 subset ít nhất 1 epoch.
2. Eval trên test split.
3. Ablation 2-3 cấu hình.
4. Báo cáo rõ constraint tài nguyên và lý do freeze VLM.

Timeline đề xuất:

- Ngày 1: download partial 1-2 subset + split.
- Ngày 2: encode VLM features.
- Ngày 3: train DiT/action head 1 epoch.
- Ngày 4: eval test split.
- Ngày 5: ablation.
- Ngày 6: tổng hợp bảng/plot/slides.
- Ngày 7: dry run và sửa lỗi.

## 11. Ghi Chú Khi Mang Sang Repo Khác

Khi chuyển sang repo mới, nên mang theo:

1. File này: `ck_context_summary.md`
2. File plan CK: `ck_action_plan.md` hoặc `plan_ck`
3. Notebook download hiện có:
   - `01_download_subsets_1_5.ipynb`
   - `01_download_subsets_6_10.ipynb`
4. File log lỗi: `bug.txt`
5. Source official repo nếu cần: `Isaac-GR00T_offical/`

Nội dung cần nói với assistant mới:

```text
Chúng tôi đang làm CK Isaac GR00T N1/N1.6. Hiện đã xong phần phân tích yêu cầu và notebook download, nhưng bị tràn disk khi tải full raw subsets. Cần tiếp tục bằng chiến lược 1 notebook = 1 subset + partial camera, sau đó tạo split, encode VLM features, train DiT/action head from scratch, eval và ablation.
```
