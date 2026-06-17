# Tom Tat Boi Canh Va Tien Do CK - Isaac GR00T N1 / N1.6

Ngay cap nhat: 10/06/2026.

File nay dung de chuyen ngu canh sang repo/cuoc tro chuyen khac. Noi dung ghi lai muc tieu do an, scope du lieu, cac notebook da co, ket qua Kaggle da chay, va viec can lam tiep.

## Cap Nhat Moi - Strong Official-Mini Pipeline

Da tao va nang cap pipeline moi trong `new_notebook/` theo huong Strong Official-Mini, manh hon pipeline state-only cu:

- Notebook 02 moi prepare official sequence/chunk data voi `states [N,S,44]`, `actions_chunk [N,H,44]`, `action_mask [N,H,44]`, video frame manifest, normalization stats.
- Notebook 03 moi encode frozen VLM features, sau do **pretrain DiT/action head from scratch** va **posttrain cung DiT/action head tren task-specific subset**.
- Notebook 03 output moi gom `pretrain_checkpoint.pt`, `posttrain_checkpoint.pt`, `checkpoint_last.pt`, `pretrain_log.csv`, `posttrain_log.csv`, `train_summary.json`.
- Notebook 04 moi evaluate denormalized raw metrics cho `OfficialMiniVLDiT_pretrain`, `OfficialMiniVLDiT_posttrain`, va them optional-safe `NVIDIA_GR00T_zero_shot` baseline.
- Notebook 05 moi giu ablation bat buoc va them bang `pretrain_posttrain_comparison.csv` neu co checkpoint tu Notebook 03.
- Tai lieu README/plan cua pipeline moi nam o `docs/strong_official_mini_pipeline_plan.md`.
- Notebook 01 download subsets hien mac dinh khong tai full video nua: `DOWNLOAD_MODE = "data_meta_ego_chunks"`, `CAMERA_KEY = "observation.images.ego_view"`, `VIDEO_CHUNKS = [0, 1, 2]`. Neu van day disk thi giam `VIDEO_CHUNKS = [0]` hoac chuyen subset do sang `data_meta_only`.

Ket luan moi: pipeline Strong Official-Mini da khop hon voi conversation cua thay vi dua `pretrain + posttrain DiT head` vao luong chinh. Zero-shot NVIDIA baseline la optional-safe, neu khong load duoc thi ghi skipped reason thay vi lam crash notebook.

## 1. De Tai Va Yeu Cau CK

De tai nhom: **Isaac GR00T: N1 through N1.6**.

Yeu cau thuc nghiem CK da thong nhat:

1. **Implementation - muc Mini**
   - Su dung dataset goc `nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`.
   - Train from scratch it nhat 1 epoch tren data goc.
   - Do gioi han tai nguyen sinh vien, cach trinh bay hop ly la: frozen/pretrained VLM neu dung visual backbone, con phan DiT/action head khoi tao moi va train tu dau.
   - Pipeline code hien tai la **mini state-action pipeline**: train action/DiT head tu scratch tren state/action features da cache, chua encode VLM image embedding that.

2. **Evaluation - Yes**
   - Danh gia tren held-out test split.
   - Can co MSE/MAE/RMSE hoac loss tren test split.

3. **Ablation Study - Yes**
   - So sanh it nhat 3 cau hinh: baseline, learning rate thap hon, model shallow hon.

## 2. Scope Du Lieu Da Chot

Khong tai/encode 36-39 subset trong giai doan CK vi vuot tai nguyen Kaggle. Scope hop ly hien tai la chay end-to-end voi 3 subset GR-1:

```text
gr1_arms_only.CanSort
gr1_arms_waist.CanToDrawer
gr1_arms_waist.CupToDrawer
```

Ket luan:

- 3 subset la du de hoan thanh plan CK neu co day du train/eval/ablation.
- Khong can tai them subset truoc khi chay xong notebook 04 va 05.
- Neu con thoi gian, co the them subset de lam ablation data scale, nhung khong bat buoc.

## 3. Trang Thai Du Lieu Da Tai Tren Kaggle

| Kaggle notebook | Noi dung tai | Thoi gian | Ghi chu |
|---|---|---:|---|
| `notebook1700561abd` | `gr1_arms_only.CanSort` | 6 phut | Output co `download_report_1_5.json` |
| `notebookbcd29698a0` | `gr1_arms_waist.CupToDrawer` | 1 gio 40 phut | Full subset |
| `notebookb6c37dbbd0` | `gr1_arms_waist.CanToDrawer` data/meta | 24 phut | Khong tai full video |
| `notebookd3f6a10db4` | Video supplement cho `gr1_arms_waist.CanToDrawer` | 20 phut | Camera `observation.images.ego_view`, chunks `0,1,2` |

Ve `partial camera` va `meta/data`:

- `meta/data` la phan quan trong nhat cho pipeline hien tai vi chua parquet/state/action/episode metadata.
- `partial camera` la video/image supplement. No huu ich neu can minh hoa visual/VLM/demo, nhung pipeline Notebook 02-05 hien tai khong bat buoc can full video tat ca camera.
- Cach tai hien tai la hop ly cho CK: co data/meta day du de train/eval, co video dai dien de giai thich visual part, khong lam Kaggle tran disk.

## 4. Notebook Va Ket Qua Hien Tai

### Notebook 01 - Download data

Trang thai:

- Da chay thanh cong cac output download can thiet cho 3 subset.
- CPU-only, khong can GPU.

Output dung lam Kaggle input cho Notebook 02:

```text
notebook1700561abd
notebookbcd29698a0
notebookb6c37dbbd0
notebookd3f6a10db4
```

### Notebook 02 - Prepare splits/features

File chinh:

```text
notebook/02_split/02_prepare_splits_and_features.ipynb
```

Kaggle run moi nhat:

```text
notebook24d25a4217.ipynb
```

Trang thai: **da chay xong va on de lam input cho Notebook 03/04/05**.

Ket qua quan trong:

```text
output_dir: gr00t_prepared_3subsets
MAX_FRAMES_PER_EPISODE: null
rows total: 6,192,174
train samples: 4,952,021
test samples: 1,240,153
action_dim: 44
state_dim: 44
split: episode-level 80/20
```

Rows theo subset:

| Subset | Rows | Episodes |
|---|---:|---:|
| `gr1_arms_only.CanSort` | 316,878 | 1,000 |
| `gr1_arms_waist.CanToDrawer` | 3,210,426 | 10,107 |
| `gr1_arms_waist.CupToDrawer` | 2,664,870 | 10,036 |

Output Notebook 02:

```text
actions_train.npy
actions_test.npy
states_train.npy
states_test.npy
train_samples.parquet
test_samples.parquet
sample_index.parquet
splits_train_episodes.json
splits_test_episodes.json
dataset_scan_report.json
prepare_report.json
```

Ghi chu ky thuat:

- Notebook 02 da duoc sua thanh pipeline that, khong con test-only.
- `MAX_FRAMES_PER_EPISODE = None`, tuc la dung full frame thay vi cat 200 frame/episode.
- Action/state column dung la `action` va `observation.state`.
- Cac cot annotation text bi bo qua dung cach, khong bi nham thanh action numeric.

### Notebook 03 - Train DiT/action head from scratch

File da tao:

```text
notebook/03_train/03_train_dit_from_scratch.ipynb
```

Kaggle run da chay:

```text
notebook91f64cc351.ipynb
```

Trang thai: **da train xong 1 epoch, dat yeu cau Implementation Mini**.

Ket qua chinh:

```text
GPU: Tesla T4
input: notebook24d25a4217/gr00t_prepared_3subsets
model: MiniDiTActionHead
params: 3,325,484
hidden_dim: 256
num_layers: 4
batch_size: 1024
epochs: 1
steps: 4,835
full_epoch_completed: true
elapsed: about 108 sec
first logged loss: 2.02877
last logged loss: 0.11877
epoch mean loss: 0.176141
```

Output Notebook 03:

```text
gr00t_dit_runs/
  config.json
  train_log.csv
  loss_curve.png
  checkpoint_last.pt
  train_summary.json
```

Danh gia:

- Notebook 03 du manh de chung minh train from scratch action/DiT head tren dataset goc.
- Ket qua loss giam ro va full epoch completed, co the tiep tuc sang evaluation.

### Notebook 04 - Evaluate test split

File da tao:

```text
notebook/04_eval/04_evaluate_test_split.ipynb
```

Trang thai: **da tao, chua co thong tin user da chay tren Kaggle**.

Input can add tren Kaggle:

```text
notebook24d25a4217/gr00t_prepared_3subsets
notebook91f64cc351/gr00t_dit_runs
```

Output mong doi:

```text
gr00t_eval_results/
  eval_summary.json
  eval_metrics.csv
  per_subset_metrics.csv
  prediction_vs_ground_truth.png
```

Vai tro:

- Day la notebook chung minh **Evaluation Yes**.
- Sau khi chay xong Notebook 04 moi co bang MSE/MAE/RMSE tren test split de dua vao bao cao.

### Notebook 05 - Ablation study

File da tao:

```text
notebook/05_ablation/05_ablation_study.ipynb
```

Trang thai: **da tao, chua co thong tin user da chay tren Kaggle**.

Input can add tren Kaggle:

```text
notebook24d25a4217/gr00t_prepared_3subsets
```

Mac dinh chay 3 cau hinh:

| Run | Learning rate | Layers | Hidden dim |
|---|---:|---:|---:|
| `baseline` | `1e-4` | 4 | 256 |
| `lr_low` | `5e-5` | 4 | 256 |
| `shallow` | `1e-4` | 2 | 256 |

Output mong doi:

```text
gr00t_ablation_results/
  ablation_summary.csv
  ablation_summary.json
  ablation_bar_chart.png
  ablation_per_subset_metrics.csv
  run_baseline/
  run_lr_low/
  run_shallow/
```

Vai tro:

- Day la notebook chung minh **Ablation Study Yes**.
- Sau khi chay xong Notebook 05, phan thuc nghiem cot loi cua plan CK gan nhu hoan tat.

## 5. Mapping Phase

Theo pipeline 2 phase da dieu chinh cho scope CK:

### Phase 1 - Data preparation / train-ready features

Notebook thuoc Phase 1:

```text
Notebook 01: download 3 subset GR-1
Notebook 02: prepare train/test split, cache state/action features
```

Ghi chu trung thuc khi viet bao cao:

- Phase 1 hien tai la prepare state/action train-ready cache.
- Neu slide noi "VLM encode image features" thi can sua wording thanh "prepare train-ready features" hoac bo sung notebook encode VLM that.

### Phase 2 - Train, evaluate, ablate action model

Notebook thuoc Phase 2:

```text
Notebook 03: train MiniDiT/action head from scratch
Notebook 04: evaluate on held-out test split
Notebook 05: ablation study
```

### Phase 3 optional - Report packaging

```text
Notebook 06: collect results for report/slides
```

Notebook 06 khong bat buoc de dat yeu cau CK, nhung huu ich neu muon gom bang/anh tu dong.

## 6. Phan Tram Hoan Thanh

Trang thai thuc te tai ngay 10/06/2026:

- Notebook 01: da chay xong.
- Notebook 02: da chay xong va output tot.
- Notebook 03: da chay xong 1 epoch va output tot.
- Notebook 04: da tao, can chay tren Kaggle.
- Notebook 05: da tao, can chay tren Kaggle.

Uoc luong hien tai:

```text
Hien tai: khoang 65-70% plan CK thuc nghiem.
```

Neu chay xong thanh cong ca 5 notebook 01-05:

```text
Khoang 90-95% plan CK.
```

Phan con lai sau Notebook 01-05:

- Gom bang/anh vao slide va report.
- Viet nhan xet evaluation/ablation.
- Giai thich gioi han: mini pipeline, train action head tu scratch, khong pretrain full VLA.
- Optional: tao Notebook 06 de collect report assets.

## 7. Viec Can Lam Tiep

Thu tu nen lam tiep:

1. Chay `04_evaluate_test_split.ipynb` tren Kaggle voi input Notebook 02 + Notebook 03.
2. Kiem tra `eval_summary.json`, `eval_metrics.csv`, `per_subset_metrics.csv`.
3. Chay `05_ablation_study.ipynb` tren Kaggle voi input Notebook 02.
4. Kiem tra `ablation_summary.csv`, `ablation_bar_chart.png`.
5. Dua cac bang/anh vao bao cao CK.

Neu can tiet kiem GPU:

- Notebook 04 chay truoc vi nhanh va bat buoc cho Evaluation.
- Notebook 05 co the chay sau; neu thoi gian it, van nen chay du 3 config mac dinh de co ablation hop le.
