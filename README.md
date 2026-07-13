# ML Seminar: Isaac GR00T Official-Mini

**English summary:** This repository contains a Strong Official-Mini reproduction of the Isaac GR00T N1/N1.6 pipeline for an ML seminar project. It uses real GR00T-X subset data, frozen Eagle2-2B visual-language features, and a DiT/action head trained from scratch. This is **not** a full paper-scale GR00T reproduction.

## Thông Tin Đề Tài

**Topic No.:** 9  
**Paper:** [GR00T N1: An Open Foundation Model for Generalist Humanoid Robots](https://arxiv.org/abs/2503.14734)  
**Final report:** [report/final_report.md](report/final_report.md)  

## Team Members

- Trần Hữu Kim Thành - 23120166
- Nguyễn Hoàng Minh Trí - 23120179
- Lê Thái Ngọc - 23122012

## Pipeline Tóm Tắt

```text
GR00T-X subset data
-> prepare state/action chunks
-> frozen Eagle2-2B visual-language encoding
-> train DiT/action head
-> denormalized evaluation
-> ablation
```

Pipeline này là bản **official-mini**: nhóm không train full VLM/backbone như paper gốc. Eagle2-2B được frozen để trích xuất visual-language features, còn DiT/action head là phần được train from scratch.

## Kết Quả Chính

- Dataset: NVIDIA GR00T-X Embodiment Sim.
- Data scale: 20 subsets, khoảng 119.36 GB.
- Train/test: 3,022,069 train samples và 755,702 test samples.
- Main tensor shape: state `[N, 1, 44]`, action `[N, 16, 44]`, mask `[N, 16, 44]`.
- Best checkpoint: `posttrain_checkpoint.pt`.
- Main evaluation: posttrain raw MSE `0.4085`, tốt hơn baseline và pretrain-only.

## Cấu Trúc Thư Mục

```text
isaac-groot/
├── 01_download_subset/        # 20 subset download notebooks
├── 02_prepare/                # prepare state/action/mask batches
├── 03_train_dit/              # Eagle2 encode + DiT/action head training
├── 04_evaluation/             # denormalized evaluation
├── 05_ablation/               # ablation study
├── report/                    # final report
└── README.md
```

## Notebook Run Order

```text
01_download_subset
-> 02_prepare
-> 03a_cache / 03a_encode
-> 03b_train_dit
-> 04_eval
-> 05_ablation
```

## Limitations

- Không phải full GR00T paper-scale reproduction.
- Không train/fine-tune VLM; Eagle2-2B được frozen.
- Chỉ dùng partial ego-view video, không full multi-camera.
- Evaluation là offline open-loop action prediction, chưa phải robot rollout.
- NVIDIA GR00T zero-shot baseline được attempted nhưng skipped.

## References

- Paper: [GR00T N1: An Open Foundation Model for Generalist Humanoid Robots](https://arxiv.org/abs/2503.14734)
- Original repository: [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)
