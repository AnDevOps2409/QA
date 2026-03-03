# ViQA — 🧠 Hệ thống Hỏi đáp Tiếng Việt (Vietnamese Question Answering)

Dự án Xử lý Ngôn ngữ Tự nhiên (NLP) áp dụng bài toán **Extractive Question Answering** (Hỏi đáp trích xuất), sử dụng mô hình ngôn ngữ lớn **XLM-RoBERTa** được tinh chỉnh (fine-tuned) trên bộ dữ liệu Tiếng Việt.

---

## 🚀 Hướng dẫn Cài đặt & Khởi chạy

### 1. Yêu cầu Hệ thống
- Python 3.11+
- Môi trường ảo (Conda/Venv)
- Card đồ họa (NVIDIA GPU) được khuyến khích để tăng tốc huấn luyện (Training) hoặc suy luận (Inference), tuy nhiên vẫn có thể chạy trên CPU.

### 2. Cài đặt Thư viện
Mở Terminal, trỏ vào thư mục dự án và chạy lệnh sau để cài đặt các Dependency:
```bash
pip install -r requirements.txt
# Nếu chưa có file requirements, hãy chạy lệnh dưới đây:
pip install torch transformers datasets flask uvicorn numpy==1.26.4
```

### 3. Huấn luyện Mô hình (Training)
Nếu bạn muốn huấn luyện lại mô hình để tăng độ chính xác, hãy chạy lệnh:
```bash
python train.py
```
*Lưu ý: Thời gian huấn luyện phụ thuộc lớn vào sức mạnh GPU. Mặc định mã nguồn sẽ chạy 5 vòng lặp (epochs).* Mọi Checkpoint tốt nhất sẽ tự động lưu vào thư mục `checkpoints/`.

### 4. Khởi chạy Web Server (Inference)
Để chạy giao diện Web tương tác trực quan (Sử dụng Flask Backend), gõ lệnh:
```bash
python api.py
```
Truy cập vào địa chỉ: **[http://127.0.0.1:5000](http://127.0.0.1:5000)** trên trình duyệt.

---

## 📑 Báo Cáo Dự Án Cuối Kỳ

### 1. Định nghĩa NLP cho dự án này
**Xử lý Ngôn ngữ Tự nhiên (NLP - Natural Language Processing)** là lĩnh vực giao thoa giữa Khoa học Máy tính và Trí tuệ Nhân tạo, giúp máy tính hiểu, diễn dịch và xử lý ngôn ngữ con người.
Trong dự án này, nhóm áp dụng bài toán **Machine Reading Comprehension (Đọc hiểu máy)** – cụ thể là **Extractive Question Answering (Hỏi đáp trích xuất)**. Hệ thống sẽ nhận đầu vào gồm 1 đoạn văn (Context) và 1 câu hỏi (Question), sau đó AI sẽ "đọc hiểu" và dự đoán chính xác vị trí bắt đầu (start token) và vị trí kết thúc (end token) của câu trả lời nằm ngay bên trong đoạn văn bản đó.

### 2. Mục đích sử dụng
- **Tự động hóa tra cứu:** Giúp người dùng hoặc khách hàng không phải đọc toàn bộ tài liệu dài (như văn bản luật, chính sách bảo hiểm, sách giáo khoa) mà nhận ngay được câu trả lời trúng đích.
- **Tích hợp Chatbot:** Là module lõi ("não nội") để gắn vào các kịch bản Chatbot chăm sóc khách hàng, hệ thống giải đáp tự động của doanh nghiệp ở các website.

### 3. Các bước tiến hành và ví dụ thực tiễn
Quá trình xây dựng một hệ thống QA từ con số 0 gồm 5 bước chính:
- **Bước 1 - Chuẩn bị Dữ liệu:** Sử dụng bộ dữ liệu Tiếng Việt (định dạng SQuAD JSON). Dữ liệu bao gồm các cặp `[context, question, answer_text, answer_start]`.
- **Bước 2 - Tiền xử lý (Preprocessing):** Sử dụng `AutoTokenizer` thuộc hệ sinh thái Hugging Face. Do đặc thù Tiếng Việt, nhóm chọn mô hình có bộ từ vựng đa ngữ mạnh là **XLM-RoBERTa**. Cắt văn bản ra các token nhỏ (Subword) và áp dụng kỹ thuật trượt cửa sổ (`stride=128`, `max_length=384`) để xử lý các đoạn văn cực dài.
- **Bước 3 - Fine-tuning (Huấn luyện):** Code vòng lặp huấn luyện bằng nền tảng PyTorch. Nhóm cấu hình thuật toán tối ưu `AdamW` với learning rate `2e-5`, và áp dụng phương pháp **Gradient Accumulation** (Tích lũy dốc) để giả lập môi trường huấn luyện RAM lớn trên máy tính cá nhân.
- **Bước 4 - Đánh giá (Evaluation):** Tính chuẩn đánh giá **F1 Score** (Độ giao thoa tập hợp từ) và **Exact Match** (Trùng khớp 100%).
- **Bước 5 - Triển khai (Deployment):** Gói model vào Flask Backend `api.py`. Xây dựng REST API cục bộ và giao diện Frontend với HTML/CSS/JavaScript thân thiện để người dùng tương tác trực tiếp qua trình duyệt.

### 4. Vấn đề và Cách khắc phục phát sinh
Trong quá trình code và tuning, nhóm gặp phải và giải quyết một số vấn đề thực tế:
- **Giới hạn VRAM:** Việc hạ `max_length = 384` kết hợp **Automatic Mixed Precision (AMP - Torch Autocast)** giúp tận dụng Tensor Cores trên card NVIDIA RTX 2060, giúp tăng tốc tiến trình Train lên đến 40% và tránh lỗi OOM (Out Of Memory).
- **Lỗi Encoding tiếng Việt:** Môi trường Windows mặc định định dạng chuỗi là hệ chuẩn Mỹ (`cp1252`), gây ra `UnicodeEncodeError` khi lưu JSON chứa dấu câu Việt. Xử lý triệt để bằng tham số `encoding="utf-8"`.
- **Xung đột Library:** PyTorch và Intel MKL xung đột OpenMP trên hệ điều hành cục bộ. Bypass bằng biến môi trường `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"`.

### 5. Ví dụ minh họa thực tiễn
- **Input Context:** Thanh Thánh Tổ (Khang Hi) là vị hoàng đế... [dài 500 chữ]... trị vì tổng cộng 61 năm...
- **Input Question:** Khang Hi trị vì bao nhiêu năm?
- **Tiến trình NLP:** Flow đi từ: Text $\rightarrow$ Subword Tokenizer $\rightarrow$ chạy qua 12 lớp Transformer của XLM-RoBERTa $\rightarrow$ Hàm Softmax trả về Logit xác suất cực đại tại chuỗi con `["6", "1", " năm"]`.
- **Output (Web UI hiển thị):** Máy tính trả về Badge nổi bật **"61 năm"**.

### 6. Case study: Tình huống ứng dụng thực tế
**Bài toán:** Bệnh viện X nhận được quá nhiều cuộc gọi từ bệnh nhân hỏi về các quy trình khám BHYT, bảng giá dịch vụ vốn đã nằm rải rác trong file PDF trên website.
**Giải pháp áp dụng Model QA:** 
- Toàn bộ nội dung website và sổ tay bệnh viện được nạp vào cơ sở dữ liệu làm **Context**.
- Bệnh nhân chat qua Fanpage/Zalo OA: *"Thẻ BHYT trái tuyến được hưởng bao nhiêu % điều trị nội trú?"*
- Model NLP QA của dự án sẽ tìm kiếm đoạn tài liệu liên quan, **trích xuất trực tiếp ngắn gọn** đáp án *"40% chi phí"* trả cho người bệnh ngay lập tức, thay vì bắt bệnh nhân tự tải file cẩm nang 50 trang về đọc. 
$\Rightarrow$ Tiết kiệm thời gian tổng đài viên, tăng cực lớn độ hài lòng của khách hàng (UX).