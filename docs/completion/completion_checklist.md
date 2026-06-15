# ✅ Checklist hoàn thiện 100% — Implementation / Evaluation / Ablation

---

## 🔧 Implementation (15%) — Hiện ~80%

### Đã xong
- [x] Train MiniDiT from scratch (3.3M params, 1 epoch)
- [x] Loss converge (2.03 → 0.12)
- [x] Checkpoint + config + loss curve saved

### Cần làm thêm

- [ ] **Mở rộng data**: Encode + train thêm subsets (ít nhất 5-10 subsets GR-1 nữa)
  - Ưu tiên: dùng notebook 01 download thêm subsets → notebook 02 prepare → notebook 03 re-train
  - ⏱️ ~2-4 giờ GPU | 🎯 Chứng minh scale data tốt hơn

- [ ] **Thêm cross-embodiment**: Train trên ít nhất 2-3 subsets khác embodiment
  - Ví dụ: `bimanual_panda_gripper.Threading`, `single_panda_gripper.PnPCounterToCab`
  - ⏱️ ~1-2 giờ GPU | 🎯 Đáp ứng yêu cầu "cross-embodiment" của GR00T

- [ ] **Train thêm epochs**: Thử 3-5 epochs (thay vì chỉ 1)
  - ⏱️ ~5-10 phút/epoch (rất nhanh với 3.3M params)
  - 🎯 Cho thấy model tiếp tục cải thiện hay đã converge

- [ ] **Ghi chú kỹ thuật**: Viết mô tả rõ ràng về:
  - Config mini model (layers, heads, hidden_dim) vs model gốc
  - Tại sao scale down (constraint GPU sinh viên)
  - Training setup: optimizer, lr, batch_size, AMP, grad_clip

---

## 📊 Evaluation (20%) — Hiện ~65%

### Đã xong
- [x] Eval full test split (1.24M samples)
- [x] MSE, MAE, RMSE, flow-matching loss
- [x] Prediction vs GT visualization
- [x] Per-subset metrics

### Cần làm thêm

- [ ] **⭐ Eval zero-shot pre-trained NVIDIA** (QUAN TRỌNG NHẤT)
  - Chạy pre-trained `GR00T-N1.6-3B` trên cùng test split → lấy MSE baseline
  - 🎯 Đây là con số so sánh chuẩn nhất: "model mình vs model NVIDIA"
  - ⏱️ ~30 phút GPU

- [ ] **Bảng so sánh chính**:
  | Model | MSE ↓ | MAE ↓ | RMSE ↓ |
  |---|---|---|---|
  | Pre-trained NVIDIA (zero-shot) | ??? | ??? | ??? |
  | **From-scratch (MiniDiT 3.3M)** | 0.022 | 0.023 | 0.148 |

- [ ] **Eval denormalized / raw action space** (BẮT BUỘC cho bản official-mini)
  - Metrics chính phải tính sau khi unnormalize action:
    ```python
    pred_raw = pred_norm * action_std + action_mean
    target_raw = target_norm * action_std + action_mean
    ```
  - Báo cáo chính dùng MSE/MAE/RMSE trên `raw action space`, không chỉ normalized action.
  - Có thể lưu normalized metrics phụ, nhưng phải ghi rõ chúng không so sánh trực tiếp được với official/open-loop GR00T.

- [ ] **Eval trên thêm subsets** (nếu đã train thêm ở phần Implementation)
  - Per-subset breakdown: subset nào tốt, subset nào xấu
  - ⏱️ ~20 phút GPU

- [ ] **Eval cross-embodiment** (nếu đã train thêm)
  - So sánh performance GR-1 vs Panda vs R1
  - 🎯 Insight: model generalize tốt hay chỉ tốt cho 1 embodiment

- [ ] **Thêm visualization**:
  - Loss curve (đã có) + thêm eval metrics theo epoch (nếu train nhiều epoch)
  - Per-joint error heatmap (joint nào predict tốt/xấu nhất)

---

## 🔬 Ablation (10%) — Hiện ~55%

### Đã xong
- [x] 3 training-time runs: baseline, lr_low, shallow
- [x] Bar chart + summary + per-subset metrics

### Cần làm thêm

- [ ] **⭐ Ablation inference-time: K (denoising steps)** (KHÔNG cần re-train)
  - K=2, K=4 (baseline), K=8, K=16
  - Chạy trên model đã train, chỉ đổi số bước denoising khi inference
  - ⏱️ ~10-15 phút/run, tổng ~1 giờ | 🎯 Yêu cầu đề bài: thay đổi components

- [ ] **⭐ Ablation action horizon: H=8 vs H=16** (required)
  - Re-prepare data cho từng setting `ACTION_HORIZON = 8` và `ACTION_HORIZON = 16`.
  - Không cần thay đổi model class; chỉ thay config/output shape `[B, H, action_dim]`.
  - Nên train/eval lại với cùng DiT config, cùng episode split, cùng denormalized metrics.
  - ⏱️ Chủ yếu tốn thời gian prepare + train/eval lại | 🎯 Chứng minh hiểu action chunking/horizon theo paper.

- [ ] **Thêm ablation training-time** (nếu còn GPU time):
  - Hidden_dim: 128 vs 256 (baseline) vs 512
  - Num_layers: 2 vs 4 (baseline) vs 8
  - ⏱️ ~30 phút/run

- [ ] **Bảng ablation tổng hợp đầy đủ**:
  | # | Loại | Setting | MSE ↓ | Δ vs Baseline |
  |---|---|---|---|---|
  | 1 | Training | lr_low | có rồi | có rồi |
  | 2 | Training | shallow | có rồi | có rồi |
  | 3 | Inference | K=2 | **cần chạy** | |
  | 4 | Inference | K=8 | **cần chạy** | |
  | 5 | Action horizon | H=8 | **cần chạy** | |
  | 6 | Action horizon | H=16 | **cần chạy / baseline official-mini** | |

- [ ] **Nhận xét/phân tích** cho từng ablation:
  - K↑ → chất lượng tốt hơn nhưng chậm hơn (trade-off)
  - H↑ → predict action trajectory dài hơn, giống GR00T hơn, nhưng khó học hơn và có thể tăng sai số ở step xa.
  - LR↓ → converge chậm nhưng stable hơn

---

## 📋 Thứ tự ưu tiên (nên làm theo thứ tự này)

| # | Task | Thời gian | GPU? | Tác động |
|---|---|---|---|---|
| 1 | Eval zero-shot pre-trained NVIDIA | ~30 phút | ✅ | 🔴 Rất cao — thiếu cái này thì phần Eval mất nhiều điểm |
| 2 | Ablation inference-time K (2,4,8) | ~45 phút | ✅ | 🔴 Cao — đáp ứng yêu cầu "thay đổi components" |
| 3 | Ablation action horizon H=8 vs H=16 | ~1-2 giờ | ✅ | 🟡 Cao — cần re-prepare data, không cần đổi model class |
| 4 | Train thêm epochs (3-5) | ~30 phút | ✅ | 🟡 Trung bình |
| 5 | Mở rộng thêm subsets | ~2-4 giờ | ✅ | 🟡 Trung bình |
| 6 | Cross-embodiment data | ~2-3 giờ | ✅ | 🟢 Bonus |
| 7 | Ghi chú kỹ thuật + bảng tổng hợp | ~1 giờ | ❌ | 🟡 Cần cho slides |

> **Ước tính**: Làm hết #1-#4 mất ~2.5 giờ GPU → đạt ~90-95%
> Làm thêm #5-#7 → đạt 100%
