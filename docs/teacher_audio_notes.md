# Ghi Chú Từ Hai File Script Audio Của Giảng Viên

## 1. Mục Đích File

File này tóm tắt các ý giảng viên đã nói trong:

- `script_audio_seminar_1.txt`
- `script_audio_seminar_2.txt`

Mục tiêu là dùng làm context khi chuyển sang repo khác để tiếp tục đồ án cuối kỳ đề tài:

```text
Isaac GR00T: N1 through N1.6
```

## 2. Yêu Cầu Quan Trọng Về Đồ Án

### 2.1 Implementation

Giảng viên chia implementation thành 3 mức:

#### Implementation = Yes

Nhóm phải:

- Train toàn bộ pipeline.
- Tác giả train bao nhiêu epoch thì nhóm phải train tương tự.
- Tác giả dùng dataset nào thì nhóm phải dùng dataset đó.
- Bổ sung thêm 2 dataset mới ngoài dataset gốc của paper.

#### Implementation = Mini

Nhóm phải:

- Không cần bổ sung dataset mới.
- Chỉ cần train trên dataset gốc mà tác giả dùng.
- Train ít nhất **1 epoch**.
- Chứng minh được code chạy training thật, không chỉ inference.

Đây là mức áp dụng cho đề tài GR00T của nhóm.

#### Implementation = No

Không cần train, chỉ cần phân tích/evaluate/fine-tune tùy yêu cầu.

### 2.2 Áp Dụng Cho Đề Tài GR00T

Với đề tài GR00T, nhóm hiểu yêu cầu Mini theo hướng:

- Không train full VLA model từ đầu vì tài nguyên sinh viên không đủ.
- Dùng pretrained/frozen VLM backbone.
- Train from scratch phần **DiT/action head**.
- Cần ghi rõ trong báo cáo:

```text
Do giới hạn tài nguyên, nhóm tái hiện mini pipeline của GR00T:
VLM dùng pretrained/frozen backbone, còn DiT/action head được khởi tạo và train từ đầu trên GR00T-X-Embodiment-Sim.
```

## 3. Evaluation

Giảng viên nhấn mạnh evaluation không chỉ là đưa số kết quả.

Khi trình bày metric, phải nói rõ:

- Metric là gì.
- Metric được tính như thế nào.
- Ý nghĩa của metric.
- Giá trị nào là tốt, giá trị nào là xấu.
- Vì sao metric phù hợp với bài toán.

Ví dụ giảng viên dùng AUC/ROC để minh họa rằng người trình bày phải hiểu metric, không chỉ đọc số.

### Áp Dụng Cho GR00T

Với GR00T/action prediction, nên dùng:

- MSE
- MAE
- test loss nếu có

Cần giải thích:

- MSE đo trung bình bình phương sai số giữa action dự đoán và action thật.
- MAE đo trung bình trị tuyệt đối sai số.
- Với MSE/MAE, **càng thấp càng tốt**.
- Evaluation phải chạy trên **test split**, không dùng lại train split.

## 4. Ablation Study

Giảng viên nói ablation study là phần bắt buộc.

Ablation nghĩa là:

- Cắt bỏ một thành phần.
- Thay thế một thành phần.
- Thay đổi một thiết lập.
- Quan sát ảnh hưởng đến kết quả.

Ví dụ giảng viên nêu:

- batch size
- learning rate
- optimizer
- số block/layer
- kích thước đầu vào
- attention mechanism
- activation function
- quantization level

Điểm quan trọng:

Nếu implementation là `Mini` hoặc `Yes`, nhóm không nên chỉ ablation hyperparameter. Cần có ít nhất một ablation ở cấp model/module.

### Áp Dụng Cho GR00T

Các ablation hợp lý:

```text
LR: 1e-4 vs 5e-5
DiT layers: 4 vs 8
hidden size khác nhau nếu kịp
optimizer: AdamW vs Apollo nếu kịp
inference denoising steps K nếu eval pipeline sẵn
action horizon H nếu eval pipeline sẵn
```

Ưu tiên:

1. Một ablation hyperparameter.
2. Một ablation model-level, ví dụ số layer DiT.

## 5. Sử Dụng AI / Hình Ảnh Trong Slide

Giảng viên cho phép dùng công cụ AI tạo hình như Stable Diffusion, Midjourney hoặc công cụ tương tự.

Nhưng yêu cầu:

- Phải kiểm tra lại nội dung hình.
- Không được đưa hình AI lên slide nếu không hiểu hình đó nói gì.
- Hình AI có thể sai, thiếu thông tin hoặc hallucination.
- Nếu dùng hình AI, phải chỉnh và kiểm chứng.

### Áp Dụng Cho GR00T

Nếu dùng hình kiến trúc GR00T:

- Nên ưu tiên tự vẽ lại từ paper/official repo.
- Phải thể hiện đúng:
  - VLM / Eagle backbone
  - DiT/action head
  - state/action encoder
  - action decoder
  - flow matching
  - action chunk
  - cross-attention với VLM tokens

## 6. Code Và Repository

Giảng viên yêu cầu:

- Code phải push lên GitHub.
- Repository phải để **public**.
- Nộp link GitHub qua form.
- Nếu repo private thì giảng viên không chấm được phần code.

Áp dụng:

- Repo cuối cùng nên public.
- Không commit HuggingFace token, Kaggle secret, hoặc file chứa credential.
- Notebook nên sạch output lớn trước khi commit.

## 7. Slide Giữa Kỳ Và Cuối Kỳ

### Giữa kỳ

Tập trung vào:

- Method.
- Ý tưởng của paper.
- Kiến trúc mô hình.
- Paper giải quyết bài toán gì.
- Vì sao method hoạt động.

Chưa cần:

- Chạy code.
- Evaluation.
- Ablation.

### Cuối kỳ

Phải có:

- Method.
- Implementation.
- Training setup.
- Evaluation.
- Ablation study.
- Kết quả thực nghiệm.

Slide nên nộp:

- `.pptx`, hoặc
- Google Slides link.

Không nên nộp PDF nếu có animation/link vì giảng viên có thể không xem được.

## 8. Về Benchmark Và Điểm Tối Đa

Giảng viên nói nếu nhóm muốn đạt mức rất cao bằng cách vượt benchmark, thì phải:

- Vượt benchmark ở full scale của paper.
- Vượt trên các metric chính của paper.
- Kết quả phải reproducible.
- Không được so sánh lệch scale, ví dụ chỉ so với bản quantized yếu hơn.

Giảng viên cho phép đề xuất method mới, nhưng cảnh báo đây là rủi ro:

- Nếu method mới tốt hơn paper gốc, có thể được điểm cao.
- Nếu method mới tệ hoặc không chứng minh được hiệu quả, có thể bị trừ nặng.

Áp dụng cho nhóm GR00T:

- Không nên đặt mục tiêu vượt full benchmark GR00T.
- Nên làm chắc yêu cầu Mini:
  - train thật
  - eval thật
  - ablation thật

## 9. Về Quantization / Model Lớn

Giảng viên nói với model lớn có thể:

- Dùng quantization.
- Dùng LoRA / QLoRA nếu fine-tune.
- Evaluate model gốc hoặc model quantized.

Nếu dùng quantization, phải ghi rõ:

- Quantization level là bao nhiêu bit.
- Kết quả được đo trên bản quantized nào.
- Nếu được, so sánh nhiều mức như 8-bit, 4-bit, 3-bit, 2-bit.

Áp dụng cho GR00T:

- Nếu dùng model/VLM quantized, phải nói rõ trong báo cáo.
- Quantization có thể là một hướng ablation nếu còn thời gian.

## 10. Nội Dung Ít Liên Quan Trực Tiếp Đến CK

Trong file audio thứ hai, giảng viên nói thêm về lab:

- Lab 3 sẽ có Kaggle ranking chung.
- Lab tính điểm theo ranking.
- Ranking dùng để phân hóa điểm.
- Đồ án thì không giống lab hoàn toàn, đồ án có xu hướng giúp kéo điểm lên nếu làm nghiêm túc.

Nội dung này không ảnh hưởng trực tiếp đến pipeline GR00T, nhưng cho thấy giảng viên quan tâm đến:

- kết quả chạy thật
- benchmark thật
- nỗ lực có thể kiểm chứng

## 11. Checklist Áp Dụng Cho Đồ Án GR00T

Để bám đúng lời giảng viên, đồ án CK nên có:

- [ ] Code public GitHub.
- [ ] Dataset gốc GR00T-X-Embodiment-Sim.
- [ ] Train/test split rõ ràng.
- [ ] Train DiT/action head from scratch ít nhất 1 epoch.
- [ ] Ghi rõ VLM pretrained/frozen do giới hạn tài nguyên.
- [ ] Training log/loss curve.
- [ ] Evaluation trên test split.
- [ ] Giải thích metric MSE/MAE.
- [ ] Ablation hyperparameter.
- [ ] Ablation model-level, ví dụ số layer DiT.
- [ ] Slide CK có Method, Implementation, Evaluation, Ablation.
- [ ] Không dùng hình AI/hình architecture sai hoặc không kiểm chứng.

## 12. Tóm Tắt Một Câu

Theo lời giảng viên, nhóm không cần làm quá tham, nhưng phải chứng minh được pipeline chạy thật: **train được, evaluate được, hiểu metric, và có ablation hợp lý**.
