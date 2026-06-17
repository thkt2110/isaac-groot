Hiện tại giống official bao nhiêu %?
~25-30% — chỉ giống ở concept flow matching loss + train/test split. Phần còn lại (architecture, data pipeline, input modalities) khác gần như hoàn toàn.

Thành phần	Trọng số	Giống?	Chi tiết
Flow matching loss concept	15%	✅ ~80%	Đúng idea, nhưng target ngược dấu
DiT architecture	25%	❌ ~15%	Vanilla Transformer vs custom DiT + cross-attention
VLM backbone	25%	❌ 0%	Hoàn toàn không có
Data pipeline	15%	❌ ~20%	Single-step, no video, no action chunking
Encoders/Decoders	10%	❌ ~10%	Linear vs CategorySpecificMLP
Train/test split	5%	✅ 90%	Giống
Eval pipeline	5%	⚠️ ~40%	Có metrics nhưng thiếu ODE inference
Cần làm gì để giống hơn? (xếp theo khả thi + tác động)
✅ LÀM ĐƯỢC trên Kaggle T4
#	Task	Effort	Tác động	Giống thêm
1	Fix flow matching sign: target = action - noise	🟢 5 phút	Sửa bug	+3%
2	Action chunking (H=16): predict 16 actions/sample thay vì 1	🟡 2-3 giờ	Rất lớn — đúng bản chất GR00T	+10%
3	Phase 1 VLM encode: load model NVIDIA → freeze → encode frames → save .npy	🟡 4-8 giờ GPU	Rất lớn — có VL features	+15%
4	Thêm cross-attention vào DiT: DiT attend vào VL features (từ bước 3)	🟡 3-4 giờ code	Lớn — đúng kiến trúc	+8%
5	Beta noise schedule thay Uniform	🟢 10 phút	Nhỏ	+2%
6	Position embedding cho action tokens	🟢 15 phút	Nhỏ	+1%
7	K-step Euler ODE inference (K=4) thay vì 1-step	🟡 1-2 giờ	Lớn — đúng inference	+5%
8	Timestep discretization (1000 buckets)	🟢 30 phút	Nhỏ	+1%
Làm hết #1→#7 → giống ~70-75%

❌ KHÔNG THỂ LÀM trên Kaggle T4
#	Task	Lý do không thể	Ảnh hưởng
A	Train VLM backbone (Eagle-2, 1.34B)	Cần ~40GB+ VRAM, multi-GPU	Bạn chỉ freeze + encode, không train VLM
B	Train action head gốc (1.7B params)	Cần ~16GB+ VRAM (T4 chỉ có 15GB)	Phải dùng mini DiT (3.3M-50M)
C	CategorySpecificMLP per-embodiment	Cần train trên nhiều embodiment cùng lúc	Bạn chỉ train 1-3 embodiment
D	Full dataset (1.5TB, 113 subsets)	Kaggle disk ~57GB	Chỉ dùng 3-10 subsets
E	DeepSpeed ZeRO + multi-GPU training	Kaggle chỉ cho 1-2 T4	Dùng single-GPU, AMP
F	State dropout + noise augmentation	Cần tune hyperparams trên dataset lớn	Minor, bỏ qua được
Roadmap cụ thể: từ 25% → 70%
Hiện tại: ~25%
    │
    ▼ [Fix #1: flow matching sign] ──────────────── ~28%
    ▼ [Fix #2: action chunking H=16] ───────────── ~38%
    ▼ [Fix #3: Phase 1 VLM encode → save .npy] ── ~53%
    ▼ [Fix #4: DiT + cross-attention vào VL] ───── ~61%
    ▼ [Fix #7: K-step Euler ODE inference] ──────── ~66%
    ▼ [Fix #5,6,8: Beta noise + pos embed] ─────── ~70%
    │
    ▼ CEILING trên Kaggle T4: ~70-75%
    │
    ║ (không thể vượt qua vì constraint phần cứng)
    ║
    ▼ 100% = Full GR00T: multi-GPU, 1.7B action head, 1.5TB data
Tóm gọn
Hiện tại	Sau khi fix	Ceiling T4
Giống official	~25%	~70%	~75% max
Effort cần	—	~15-20 giờ GPU + code	—
Phần quan trọng nhất cần làm	—	Phase 1 VLM encode + cross-attention DiT	—
Phần KHÔNG BAO GIỜ làm được trên T4	—	—	Train VLM, train 1.7B action head, full dataset