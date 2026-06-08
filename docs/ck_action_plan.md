# 🎯 Action Plan CK — Isaac GR00T N1 v7 (7 ngày)

> **Deadline**: 30/05/2026 | **Bắt đầu**: 24/05 | **Nhóm**: 3 người
> **Trọng số CK**: Method 35% · Evaluation 20% · Implementation 15% · Ablation 10%
>
> **Approach (theo hướng dẫn giảng viên):**
> 1. **2-phase pipeline**: Encoding (VLM rút trích features) → Training (DiT train trên features)
> 2. Pretrained Eagle-2 VLM → **frozen**, chạy **1 lần duy nhất** rút features → lưu disk → xóa raw data
> 3. DiT head **random init** → **pre-train** trên features đã lưu → **post-train** trên task-specific data
> 4. Kỹ thuật tối ưu: gradient accumulation, Apollo optimizer, mixed precision (bf16)
> 5. **Xử lý TOÀN BỘ data theo từng chunk**: download chunk → encode → xóa raw → download chunk tiếp

---

## Tại Sao 2-Phase Pipeline?

```
=== TRƯỚC (1-phase): VLM + DiT cùng trên GPU ===
GPU VRAM: [VLM 1.34B ~3GB] + [DiT] + [batch data] = CHẬT → phải scale DiT xuống 3-5M params

=== SAU (2-phase): VLM và DiT KHÔNG cùng lúc ===
Phase 1 (Encoding):  GPU = [VLM 1.34B ~3GB] + [batch images] → save features to disk
Phase 2 (Training):  GPU = [DiT] + [batch features]           → TOÀN BỘ 16GB cho DiT!
```

**Lợi ích**: DiT có thể lớn hơn nhiều (8-16 layers thay vì 4), batch size lớn hơn, train tốt hơn.

---

## 📦 Chiến Lược Data: Chunk-by-Chunk Encoding

> [!IMPORTANT]
> **Theo thầy**: Download từng chunk → VLM encode → lưu features → xóa raw → download chunk tiếp.
> Features (embedding) nhỏ hơn raw data nhiều lần → lưu trữ thoải mái trên Kaggle.

### Subsets được chọn: 27 GR-1 + 9 cross-embodiment = **36 subsets**

| # | Nhóm | Subsets | Lý do chọn |
|---|---|---|---|
| 27 | **GR-1 (all)** | `CanSort`, 21× `arms_waist.*`, 2× `full_upper_body.*`, 3× `unified.*` | Core target — cùng embodiment, eval trên GR1 |
| 2 | **Bimanual Panda Gripper** | `Threading`, `Transport` | Đa dạng: 2 tay + gripper |
| 1 | **Bimanual Panda Hand** | `LiftTray` | Đa dạng: 2 tay + bàn tay |
| 3 | **Single Panda** | `PnPCounterToCab`, `OpenDrawer`, `CoffeeServeMug` | Đa dạng: pick-place + articulated + multi-step |
| 2 | **Behavior R1** | `task-0001_picking_up_trash`, `task-0020_sorting_vegetables` | Đa dạng: mobile manipulation |
| 1 | **Unitree G1** | `LMPnPAppleToPlateDC` | Đa dạng: humanoid khác |

> **Tại sao 36 mà không phải 113?** GPU quota 30h/tuần. Encode 36 subsets ≈ 8-12h, còn 18-22h cho train + ablation. Encode 113 subsets ≈ 25-30h → hết quota, không còn thời gian train.

### Workflow: Xử lý tăng dần trên Kaggle (theo `download_strategy.txt`)

```
Kaggle Notebook (có GPU):
  1. Dùng HuggingFace Hub Python API tải subset trực tiếp vào notebook
  2. Load pretrained VLM (freeze) → forward pass → rút trích features
  3. Lưu features (embeddings + state + action + mask) thành Kaggle Dataset artifact
  4. Xóa raw data → giải phóng disk
  5. Lặp lại cho subset tiếp theo

Có thể chạy NHIỀU Kaggle notebooks SONG SONG để tăng tốc encoding!
```

**Lưu features thành Kaggle Dataset** (thay vì chỉ lưu local):
- Persist giữa các sessions (không mất khi session hết 12h)
- Dùng lại ở notebook khác (notebook train DiT chỉ cần mount dataset features)
- Chia sẻ cho C eval dễ dàng

### Thứ tự ưu tiên encoding

| Priority | Nhóm | Số subsets |
|---|---|---|
| 🔴 P1 | GR-1 arms only + arms+waist + full upper body | 24 |
| 🟡 P2 | GR-1 unified (chọn 3 đại diện) | 3 |
| 🟢 P3 | Cross-embodiment (Panda + R1 + Unitree) | 9 |

> **Mục tiêu Day 1-2**: encode hết P1 (24) + P2 (3) + bắt đầu P3. Có thể chia cho nhiều notebooks song song.

---

## Yêu Cầu & Cách Đáp Ứng

| Phần | Mức | Yêu cầu | Cách nhóm đáp ứng |
|---|---|---|---|
| **Method** | 35% | Trình bày rõ method | A: overview → chi tiết từng module + diagram tự vẽ |
| **Implementation** | Mini | Train from scratch, ≥1 epoch | B: 2-phase pipeline, process toàn bộ data, DiT from scratch, pre-train + post-train |
| **Evaluation** | Yes | Eval trên test split | C: eval pre-trained NVIDIA (zero-shot) + model from scratch |
| **Ablation** | Yes | Thay đổi HP + components | B: ablation LR/optimizer. C: ablation inference-time (K, H) |

---

## Phân Vai

| | **A — Method & Slides** | **B — Train Lead** | **C — Eval & Ablation Inference** |
|---|---|---|---|
| **Chính** | Sửa Method + tổng hợp slides CK | 2-phase: encode toàn bộ data → train DiT, ablation | Eval pipeline, ablation inference-time (K, H) |
| **GPU** | Không cần | **GPU chính** | **GPU inference/eval (nhẹ)** |
| **Output** | Slides Method + full deck | Encoded features + DiT checkpoints + loss curves | Bảng eval + ablation + charts |

### Nguyên tắc song song

```
B: [Setup+Encode chunk 1] → [Encode chunks 2-N] → [Train DiT] → [Post-train] → [Ablation]
C: [Setup] → [Eval zero-shot ngay] → [Ablation K,H] → [Eval B's checkpoints] → [Tổng hợp]
A: [Sửa Method + vẽ diagram] ──────→ [Ghép slides] → [Review]
```

> [!IMPORTANT]
> **C KHÔNG ĐỢI B.** Từ Day 1, C tải pre-trained model NVIDIA về chạy eval + ablation inference ngay.

---

## Day 1 (24/05) — SETUP + PHASE 1 ENCODING (BATCH 1)

### B — Train Lead
- [ ] Setup Kaggle Notebook + cài env
- [ ] Clone repo + checkout `n1.6-release`
- [ ] **Đọc kỹ code**: `setup.py`, `eagle_backbone.py`, `processing_gr00t_n1d6.py`
- [ ] Download pretrained checkpoint `nvidia/GR00T-N1.6-3B`
- [ ] Smoke test inference → confirm env OK
- [ ] **Viết encoding script** (Phase 1 — key deliverable)
- [ ] **Bắt đầu encoding loop**:
```
Chunk 1: Download gr1_arms_only.CanSort → Encode → Lưu features → Xóa raw
Chunk 2: Download gr1_arms_waist.CanToDrawer → Encode → Lưu → Xóa
Chunk 3: Download gr1_arms_waist.CupToDrawer → Encode → Lưu → Xóa
... (tiếp tục đến hết session hoặc hết P1 subsets)
```
- [ ] Ghi log: bao nhiêu subsets đã encode, features size vs raw size

> 📄 **Hướng dẫn chi tiết**: xem [`guides/day1_member_b.md`](./guides/day1_member_b.md)

### C — Eval Lead
- [ ] Clone repo + cài env (Kaggle riêng)
- [ ] Download pre-trained `nvidia/GR00T-N1.6-3B`
- [ ] **Chạy eval zero-shot ngay** → ghi MSE baseline NVIDIA
- [ ] Nếu gặp bug eval → fix ngay

### A — Method Lead
- [ ] Đọc lại feedback GK + paper Section 2.1-2.3
- [ ] Lên outline Method
- [ ] Chọn tool vẽ diagram

> **✅ Day 1 done khi**: B đã encode ≥5 subsets. C có MSE zero-shot. A có outline.

---

## Day 2 (25/05) — ENCODING TIẾP + TRAIN DiT + C ABLATION

### B — Train Lead
- [ ] **Tiếp tục encoding loop** (nếu chưa hết P1+P2):
```
Chunk 4-N: Download subset → Encode → Lưu → Xóa raw
Mục tiêu: encode hết P1 (22 GR-1 subsets) + bắt đầu P2 nếu kịp
```
- [ ] **Viết DiT-only training script** (Phase 2):
  - Load saved features từ disk
  - Init DiT head random → train trên features
  - gradient accumulation + mixed precision bf16
  - Apollo optimizer (nếu kịp, không thì AdamW)
- [ ] **Pre-train DiT head from scratch** trên features đã encode:
```
Config:
  - DiT layers: 8-16
  - hidden_size: 512
  - batch_size: 32-64
  - gradient_accumulation_steps: 4-8
  - mixed precision: bf16
  - LR: 1e-4
  - max_steps: 2000-5000
```
- [ ] Log loss curve, training time, đếm params

### C — Ablation Inference
- [ ] **Ablation inference-time trên pre-trained model** (4 runs):

| # | Setting | K | H | MSE ↓ |
|---|---|---|---|---|
| 0 | **Baseline (pre-trained)** | 4 | 16 | (Day 1) |
| 1 | K↓ | **2** | 16 | — |
| 2 | K↑ | **8** | 16 | — |
| 3 | H↓ | 4 | **8** | — |
| 4 | H↑ | 4 | **32** | — |

### A — Method Lead
- [ ] Vẽ architecture diagram (tự vẽ)
- [ ] Bắt đầu slides Method

> **✅ Day 2 done khi**: B encode ≥15 subsets + DiT đang/đã train. C có bảng ablation. A đang slides.

---

## Day 3 (26/05) — POST-TRAIN + EVAL + ABLATION TRAIN

### B — Train Lead
- [ ] (Nếu còn) Encode thêm P2/P3 subsets
- [ ] Confirm pre-train DiT xong
- [ ] **Post-train DiT** trên task-specific data (CanSort features only, ~500-1000 steps)
- [ ] **Ablation training-time** (1-2 runs):
```
# Ablation 1: LR thấp
... lr=5e-5

# Ablation 2: Apollo optimizer (nếu kịp)
... optimizer=apollo
```
- [ ] Ném checkpoint cho C eval

### C — Eval Lead
- [ ] **Eval from-scratch model** (DiT pre-trained) trên test split
- [ ] **Eval post-trained model** (nếu B có) trên test split
- [ ] Bảng so sánh:

| Model | Params | Data | MSE ↓ |
|---|---|---|---|
| Pre-trained NVIDIA (zero-shot) | 2.2B | — | — |
| DiT from-scratch (pre-train, N subsets) | ~50-100M | all encoded | — |
| DiT from-scratch (post-train, CanSort) | ~50-100M | CanSort only | — |

### A — Method Lead
- [ ] **Hoàn thiện slides Method**
- [ ] Tạo khung slides CK

> **✅ Day 3 done khi**: DiT pre-trained model eval xong. Post-train/ablation đang chạy. A xong Method.

---

## Day 4 (27/05) — THU KẾT QUẢ + SLIDES

### B
- [ ] Ablation xong → ném checkpoint cho C
- [ ] Thu thập training logs, loss curves, GPU info
- [ ] Viết ghi chú: 2-phase pipeline, data pipeline (bao nhiêu subsets, features size), config, params

### C
- [ ] Eval ablation checkpoints
- [ ] **Tổng hợp bảng kết quả**:

**Bảng Evaluation:**
| Model | Params | Data encoded | MSE ↓ |
|---|---|---|---|
| Pre-trained NVIDIA (zero-shot) | 2.2B | — | — |
| DiT from-scratch pre-train (lr=1e-4) | ~50-100M | N subsets | — |
| DiT from-scratch post-train | ~50-100M | CanSort | — |

**Bảng Ablation:**
| # | Loại | Setting | MSE ↓ | Δ vs Baseline |
|---|---|---|---|---|
| 1 | Inference | K=2 | — | — |
| 2 | Inference | K=8 | — | — |
| 3 | Inference | H=8 | — | — |
| 4 | Inference | H=32 | — | — |
| 5 | Training | lr=5e-5 | — | — |
| 6 | Training | Apollo optimizer | — | — |

- [ ] Vẽ charts + viết nhận xét

### A — Slides Lead
- [ ] **Ghép slides CK hoàn chỉnh**:
  1. Formulation (từ GK)
  2. **Method** (overview→detail + diagram)
  3. Impact (từ GK)
  4. **Implementation** (2-phase diagram, data pipeline, config, params, loss curve)
  5. **Evaluation** (bảng + visualization)
  6. **Ablation** (bảng + chart + nhận xét)

> **✅ Day 4 done khi**: Tất cả số liệu đủ. Slides ~90%.

---

## Day 5-6-7: Hoàn thiện + Dry run + Nộp

### Day 5 (28/05)
- A: Điền số liệu thật, embed charts
- B: Review slides Implementation, viết README
- C: Review slides Evaluation

### Day 6 (29/05)
- **Dry run trình bày**
- Chuẩn bị Q&A:
  - *"2-phase pipeline là gì?"* → Tách VLM encoding và DiT training. VLM rút features 1 lần, lưu disk, xóa raw data. DiT train trên features. Tiết kiệm VRAM → DiT lớn hơn.
  - *"Process toàn bộ data thế nào?"* → Download từng chunk, encode, xóa raw, download chunk tiếp. Features nhỏ hơn raw nhiều lần → lưu hết trên disk.
  - *"Tại sao VLM pretrained?"* → VLM = frozen feature extractor. DiT head hoàn toàn from scratch.
  - *"Pre-train vs post-train?"* → Pre-train trên toàn bộ encoded data → post-train trên task cụ thể (đúng paper).

### Day 7 (30/05)
- Fix cuối + nộp + trình bày

---

## 📊 Deliverables

| # | Deliverable | Ai | Khi nào |
|---|---|---|---|
| 1 | Env OK + eval zero-shot | B + C | Day 1 |
| 2 | Encoding script + encode ≥5 subsets | **B** | Day 1 |
| 3 | Encode ≥15 subsets (P1 done) | **B** | Day 2 |
| 4 | Ablation inference (K, H) — 4 runs | **C** | Day 2 |
| 5 | Phase 2: DiT pre-train checkpoint + bảng params | **B** | Day 2 |
| 6 | DiT post-train checkpoint | **B** | Day 3 |
| 7 | Eval from-scratch models | **C** | Day 3 |
| 8 | Ablation training (LR, optimizer) | **B** | Day 3-4 |
| 9 | Bảng tổng hợp + charts | **C** | Day 4 |
| 10 | Slides Method + full deck | **A** | Day 3-5 |
| 11 | Dry run + repo clean | All | Day 6 |
| 12 | Nộp + trình bày | All | Day 7 |

---

## 🔑 Kỹ Thuật Tối Ưu (Theo Thầy)

| Kỹ thuật | Mô tả | Ưu tiên |
|---|---|---|
| **2-phase pipeline** | VLM encode → save features → DiT train trên features | 🔴 BẮT BUỘC |
| **Chunk-based encoding** | Download từng phần data → encode → xóa raw → tiếp | 🔴 BẮT BUỘC |
| **Gradient accumulation** | Tăng effective batch size mà không tăng VRAM | 🔴 BẮT BUỘC |
| **Mixed precision (bf16)** | Train DiT ở bfloat16 → giảm VRAM 50% | 🟡 NÊN CÓ |
| **Apollo optimizer** | Thay AdamW, memory-efficient hơn | 🟢 NẾU KỊP |

---

## 🔑 Phân Loại Ablation

| Loại | Ablation | Ai chạy | Cần train? | Thời gian |
|---|---|---|---|---|
| **Inference-time** | K=2, K=8 | **C** | ❌ | ~30 phút/run |
| **Inference-time** | H=8, H=32 | **C** | ❌ | ~30 phút/run |
| **Training-time** | lr=5e-5 | **B** | ✅ | ~1-2h/run |
| **Training-time** | Apollo optimizer | **B** | ✅ | ~1-2h/run |

---

## ⚠️ Risk & Mitigation

| Risk | Mitigation |
|---|---|
| Encoding script phức tạp | Dựa trên `standalone_inference_script.py` — đã có code VLM forward |
| Encode quá chậm (1 subset ~30min) | Chạy liên tục, tự động hóa loop. Ưu tiên GR-1 trước |
| Disk hết khi download raw + encode | Xóa raw **ngay sau khi encode xong** mỗi chunk. Luôn kiểm tra `df -h` |
| Features quá lớn trên disk | ~20-50MB/subset → 113 subsets ≈ 2-5GB max. Kaggle 70GB → thừa |
| DiT-only training loop khó viết | Modify `experiment.py` — skip VLM, load features từ disk |
| T4 OOM khi encode VLM | Encode batch_size=1, bf16, torch.no_grad() |
| Apollo optimizer chưa biết dùng | Fallback: AdamW + gradient accumulation |
| Thầy hỏi "sao không encode hết?" | Ghi rõ bao nhiêu subsets đã encode, show bảng data pipeline |
