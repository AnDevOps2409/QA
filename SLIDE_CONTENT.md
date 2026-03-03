# SLIDE THUYẾT TRÌNH — Vietnamese Extractive Question Answering with XLM-RoBERTa
---

## 01 — INTRODUCTION

### Slide 1: What is Question Answering?

- **Question Answering (QA)** là bài toán trong NLP: Cho một đoạn văn bản (Context) và một câu hỏi (Question), hệ thống tự động trích xuất câu trả lời (Answer) từ trong Context.
- Hai dạng chính:
  - **Extractive QA:** Câu trả lời là một đoạn text con nằm nguyên vẹn trong Context ← *Dự án này*
  - **Abstractive QA:** Máy tự sinh câu trả lời mới (paraphrase)
- Câu hỏi có thể **có đáp án** (answerable) hoặc **không có đáp án** (unanswerable) trong Context
- Tham khảo: [Papers With Code — QA](https://paperswithcode.com/task/question-answering)

> **Speaker Notes:** Giới thiệu bài toán QA tổng quan. Nhấn mạnh nhóm tập trung vào dạng Extractive (trích xuất), tức máy không tự bịa câu trả lời mà phải tìm đúng đoạn text trong đoạn văn.

---

### Slide 2: UIT-ViQuAD 2.0

- Bộ dữ liệu **Hỏi đáp Tiếng Việt** do UIT xây dựng, cấu trúc tương tự SQuAD 2.0 (Stanford)
- Tổng: **~35,000 QA pairs**
  - `train.json`: 28,457 cặp hỏi-đáp
  - `test.json`: 3,821 cặp hỏi-đáp
- Bao gồm cả câu hỏi **có đáp án** và **không có đáp án** (unanswerable)
- Định dạng JSON:
  ```json
  {
    "context": "Đoạn văn bản...",
    "question": "Câu hỏi?",
    "answers": { "text": ["Câu trả lời"], "answer_start": [42] }
  }
  ```

> **Speaker Notes:** Dataset chuẩn cho bài toán QA tiếng Việt. Cấu trúc giống SQuAD nên có thể tận dụng toàn bộ pipeline xử lý của HuggingFace.

---

### Slide 3: Metrics

- **F1-Score:** Đo độ trùng khớp giữa Prediction và Ground Truth ở mức **từ (word-level overlap)**
  - Precision = (số từ đúng) / (tổng từ predicted)
  - Recall = (số từ đúng) / (tổng từ ground truth)
  - F1 = 2 × (P × R) / (P + R)
- **Exact Match (EM):** Prediction phải trùng **100% character-by-character** với Ground Truth
  - Được 1 điểm nếu match hoàn hảo, 0 điểm nếu sai bất kỳ ký tự nào
- F1 "dễ tính" hơn EM vì cho phép đúng một phần

> **Speaker Notes:** F1 phản ánh chất lượng tổng thể (trùng bao nhiêu từ), EM phản ánh độ chính xác tuyệt đối. Thường F1 cao hơn EM rất nhiều.

---

### Slide 4: Baseline System

- **Baseline model:** mBERT (Multilingual BERT)
  - F1 = **63.03%**
  - EM = **53.54%**
- **Human Performance** (con người trả lời):
  - F1 và EM cao hơn đáng kể
- → Mục tiêu của nhóm: **vượt qua mBERT Baseline** bằng mô hình mạnh hơn

> **Speaker Notes:** mBERT là mốc tham chiếu. Bất kỳ model nào nhóm train mà F1 > 63% là đã vượt baseline. Đây là động lực để nhóm tìm kiếm model tốt hơn.

---

## 02 — APPROACH

### Slide 5: Research — Khảo sát các Mô hình

- Nhóm đã nghiên cứu các paper về:
  - **BERT** — Bidirectional Encoder (Google, 2018)
  - **PhoBERT** — Pre-trained cho tiếng Việt (VinAI)
  - **XLM-RoBERTa** — Multilingual RoBERTa (Facebook AI / Meta)
  - **T5** — Text-to-Text Transformer (Google)
  - **BARTpho** — BART cho tiếng Việt (VinAI)
- Ban đầu nhóm dự định chọn **BARTpho** vì:
  1. Hứng thú với kiến trúc BART (Seq2Seq denoising)
  2. BARTpho chưa được đánh giá trên task QA
- **Tuy nhiên, gặp rào cản kỹ thuật nghiêm trọng** → Chuyển hướng sang **XLM-RoBERTa** *(giải thích ở slide sau)*

> **Speaker Notes:** Đây là điểm quan trọng nhất khi thầy hỏi "tại sao lại chọn model này?". Nhóm không chọn bừa, mà đã nghiên cứu nhiều model rồi mới quyết định dựa trên thực nghiệm.

---

### Slide 6: Tại sao KHÔNG dùng được BARTpho?

- Bài toán **Extractive QA** yêu cầu Tokenizer phải hỗ trợ:
  - `return_offsets_mapping=True` → Ánh xạ ngược từ sub-word token về vị trí ký tự gốc trong văn bản
  - Chỉ có **Fast Tokenizer** (viết bằng Rust) mới hỗ trợ tính năng này
- **Vấn đề:** Thư viện HuggingFace Transformers **KHÔNG có Fast Tokenizer cho BARTpho**
  - `BartphoTokenizerFast` → **không tồn tại**
  - Slow Tokenizer (Python) → không hỗ trợ `offset_mapping`
- → **Không thể xác định được tọa độ Start/End của câu trả lời** → Loss function không tính được → Không thể train

> **Speaker Notes:** Đây là bài học thực tế rất giá trị. Trên paper thì model nào cũng hay, nhưng khi code thực tế thì phải phụ thuộc vào hệ sinh thái thư viện. BARTpho thiếu Fast Tokenizer nên không dùng được cho Extractive QA.

---

### Slide 7: What is XLM-RoBERTa?

- **XLM-RoBERTa** (Cross-lingual Language Model — RoBERTa) do **Facebook AI / Meta** phát hành
- Pre-trained trên **2.5 TB CommonCrawl** data, bao gồm **100 ngôn ngữ** (có **Tiếng Việt**)
- Kiến trúc: **Transformer Encoder-only** (12 layers, 768 hidden, 125M params cho bản `base`)
- **Ưu điểm quyết định:**
  - ✅ Hỗ trợ **Fast Tokenizer** đầy đủ (SentencePiece + Rust backend)
  - ✅ Hỗ trợ `return_offsets_mapping=True` → hoàn hảo cho Extractive QA
  - ✅ Đã pre-train trên lượng lớn text tiếng Việt → hiểu ngữ nghĩa sâu
  - ✅ State-of-the-art trên nhiều benchmark đa ngữ (XNLI, XQuAD, MLQA)

> **Speaker Notes:** XLM-R giải quyết được tất cả vấn đề mà BARTpho gặp phải. Nó vừa mạnh, vừa tương thích hoàn hảo với pipeline Extractive QA của HuggingFace.

---

## 03 — IMPLEMENTATION

### Slide 8: How we Tokenize?

- Sử dụng `AutoTokenizer.from_pretrained("xlm-roberta-base")`
- Thuật toán: **SentencePiece** (Unigram subword)
- Ví dụ tokenization:
  ```
  Input:  "Khang Hi trị vì 61 năm"
  Tokens: ["▁Kh", "ang", "▁Hi", "▁trị", "▁v", "ì", "▁61", "▁năm"]
  ```
- **Offset Mapping:** Mỗi token được ánh xạ về `(start_char, end_char)` trong text gốc
  ```
  Token "▁61" → offset (17, 19) → text gốc: "61"
  ```
- Kỹ thuật **Sliding Window** cho đoạn văn dài:
  - `max_length = 384` tokens (giới hạn Transformer)
  - `stride = 128` tokens (phần overlap giữa các chunk)
  - Đoạn văn 800 chữ → tự động tách thành nhiều chunk gối đầu nhau

> **Speaker Notes:** Sliding Window rất quan trọng. Nếu context dài hơn 384 token mà không có kỹ thuật này, câu trả lời nằm ở cuối đoạn văn sẽ bị cắt mất.

---

### Slide 9: Dataset Split

- Dữ liệu gốc: `train.json` (28,457 samples)
- Chia lại thành:
  - `new_train`: **~25,600 samples** (90%) → dùng để huấn luyện
  - `valid`: **~2,857 samples** (10%) → dùng để đánh giá sau mỗi epoch
- Lưu dưới dạng HuggingFace Dataset (`load_from_disk("data/hf_dataset")`)
- Sau khi tokenize + sliding window, số lượng features tăng lên (vì 1 sample dài → nhiều chunk)

> **Speaker Notes:** Nhóm tách 10% từ training set để làm validation, đánh giá F1/EM sau mỗi epoch và lưu checkpoint tốt nhất.

---

### Slide 10: Experimental Setup

| Tham số | Giá trị |
|---------|---------|
| **Model** | `xlm-roberta-base` (125M params) |
| **Optimizer** | AdamW |
| **Learning Rate** | 2e-5 |
| **LR Scheduler** | Linear decay with warmup |
| **Batch Size (physical)** | 16 |
| **Gradient Accumulation** | 4 steps → effective batch = 64 |
| **Epochs** | 5 |
| **Max Sequence Length** | 384 tokens |
| **Stride** | 128 tokens |
| **n_best** | 20 |
| **Max Answer Length** | 200 tokens |
| **Mixed Precision** | AMP (torch.autocast → Float16) |

> **Speaker Notes:** So với bản BARTpho cũ (batch_size=2, max_length=1024, epochs=15), XLM-R chạy nhanh hơn rất nhiều nhờ max_length ngắn hơn và AMP. Gradient Accumulation giúp giả lập batch lớn trên GPU giới hạn VRAM.

---

### Slide 11: Kỹ thuật Tối ưu Phần cứng

- **Automatic Mixed Precision (AMP):**
  - Dùng `torch.autocast(device_type="cuda", dtype=torch.float16)`
  - Nén tham số từ Float32 → Float16
  - Tăng tốc ~40%, giảm VRAM ~50%
- **Gradient Accumulation:**
  - GPU chỉ chạy được batch_size = 16
  - Tích lũy gradient qua 4 bước → effective batch size = 64
  - Loss ổn định hơn, gradient mượt hơn
- **GradScaler:**
  - Kết hợp với AMP để tránh underflow khi dùng Float16

> **Speaker Notes:** Hai kỹ thuật này là chìa khóa để train model 125M params trên GPU cá nhân. Không có chúng, sẽ bị Out Of Memory hoặc train rất chậm.

---

### Slide 12: Evaluation Strategy

- Sau mỗi epoch, chạy **evaluation** trên validation set
- Cách chọn answer span tốt nhất:
  1. Model trả về `start_logits` và `end_logits` cho mỗi token
  2. Lấy **top-20 (n_best)** vị trí start và end có logit cao nhất
  3. Tính **score = start_logit + end_logit** (không dùng softmax)
  4. Chọn span có score cao nhất, thỏa mãn: `start ≤ end` và `length ≤ max_answer_length`
- **Save best checkpoint:** So sánh F1 sau mỗi epoch, nếu F1 mới > F1 cũ → ghi đè model vào `checkpoints/`

> **Speaker Notes:** Cách chọn span bằng logit score (không softmax) là kỹ thuật chuẩn từ paper SQuAD gốc. n_best=20 giúp xem xét nhiều ứng viên trước khi chọn đáp án cuối cùng.

---

## 04 — RESULT

### Slide 13: Quantitative Results

| Model | F1 | EM |
|-------|----|----|
| mBERT (Baseline) | 63.03% | 53.54% |
| **XLM-RoBERTa-base** *(hiện tại)* | **Điền sau** | **Điền sau** |

> **Speaker Notes:** Bảng so sánh giữa các cấu hình. Kết quả XLM-R cần được cập nhật sau khi train xong. Dựa trên các benchmark quốc tế, XLM-R-base thường đạt F1 ~75-82% trên ViQuAD.

---

### Slide 14: Demo

- **Backend:** Flask (Python) — chạy trên `localhost:5000`
- **Frontend:** Giao diện Web Dark theme hiện đại
  - Trang chủ: Hiển thị kết quả gần đây với animation
  - Trang Đặt câu hỏi: Form nhập Context + Question, có Loading spinner
- **Workflow demo:**
  1. Dán đoạn văn (Context) vào ô textarea
  2. Nhập câu hỏi (Question)
  3. Bấm "Tìm câu trả lời" → AI trích xuất ngay đáp án
  4. Hiển thị Answer Badge với kết quả

*(Chèn hình chụp giao diện Web ở đây)*

> **Speaker Notes:** Demo trực tiếp trên trình duyệt. Tốc độ inference dưới 1 giây cho đoạn văn 800 chữ, cho thấy model hoàn toàn khả thi để deploy ứng dụng thực tế.

---

## 05 — PROBLEMS & LIMITATIONS

### Slide 15: Các vấn đề gặp phải

- **Paraphrase (Diễn đạt lại):** Nếu câu hỏi viết khác cách so với context, model có thể bị nhầm
  - VD: Context viết "61 năm", Question hỏi "bao lâu" → model vẫn cần hiểu "bao lâu" = "bao nhiêu năm"
- **Từ xuất hiện nhiều lần:** Nếu một keyword xuất hiện ở nhiều vị trí trong context, model có thể chọn sai vị trí
- **No Answer (Câu hỏi không có đáp án):** Model hiện tại luôn cố gắng trả lời, chưa xử lý tốt trường hợp "không có đáp án trong đoạn văn"
- **Encoding tiếng Việt:** Windows mặc định dùng `cp1252`, cần luôn chỉ định `encoding="utf-8"` khi đọc/ghi file
- **Xung đột thư viện:** OpenMP runtime conflict trên Windows (`OMP Error #15`)

> **Speaker Notes:** Các vấn đề này là common issues trong Extractive QA. Paraphrase có thể cải thiện bằng data augmentation. No answer cần thêm threshold cho null score.

---

## 06 — SUMMARY

### Slide 16: Tổng kết

- ✅ Xây dựng thành công hệ thống **Extractive QA cho Tiếng Việt** end-to-end
- ✅ Chuyển đổi từ BARTpho → **XLM-RoBERTa** do giới hạn kỹ thuật (Fast Tokenizer)
- ✅ Áp dụng các kỹ thuật tối ưu: **AMP, Gradient Accumulation, Sliding Window**
- ✅ Deploy thành công trên **Flask Web App** với giao diện hiện đại

**Hướng phát triển tương lai:**
- Train thêm epoch / dùng `xlm-roberta-large` (355M params)
- Xử lý câu hỏi "unanswerable" (không có đáp án)
- Kết hợp **Retriever + Reader** (mô hình 2 giai đoạn) cho kho tài liệu lớn
- Deploy lên **HuggingFace Spaces** hoặc Cloud (AWS/GCP)

> **Speaker Notes:** Nhấn mạnh bài học từ việc chuyển model. Đây là giá trị thực tiễn của đồ án: không chỉ biết dùng model, mà còn biết debug và adapt khi gặp vấn đề kỹ thuật thực tế.

---

## 📋 TÓM TẮT CẤU TRÚC SLIDE

| # | Slide | Mục đích |
|---|-------|----------|
| 1 | What is QA? | Giới thiệu bài toán |
| 2 | UIT-ViQuAD 2.0 | Giới thiệu dataset |
| 3 | Metrics (F1, EM) | Cách đánh giá |
| 4 | Baseline (mBERT) | Mốc so sánh |
| 5 | Research & Model Selection | Tại sao chọn XLM-R |
| 6 | Tại sao không dùng BARTpho | Giải thích rào cản kỹ thuật |
| 7 | What is XLM-RoBERTa? | Giới thiệu model đã chọn |
| 8 | How we Tokenize? | Tiền xử lý + Sliding Window |
| 9 | Dataset Split | Chia dữ liệu |
| 10 | Experimental Setup | Bảng tham số |
| 11 | Hardware Optimization | AMP + Gradient Accumulation |
| 12 | Evaluation Strategy | Cách chọn answer span |
| 13 | Results (F1/EM) | Bảng kết quả |
| 14 | Demo | Web app demo |
| 15 | Problems & Limitations | Các hạn chế |
| 16 | Summary & Future Work | Tổng kết |
