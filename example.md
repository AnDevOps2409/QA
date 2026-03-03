# 📘 VIETNAMESE EXTRACTIVE QA (XLM-RoBERTa)

------------------------------------------------------------------------

## 🎯 1. Mục tiêu của Project

Cho:

-   📄 **Context (Đoạn văn)**
-   ❓ **Question (Câu hỏi)**

Hệ thống phải:

👉 Tìm đúng đoạn trong Context để trả lời.

⚠️ Máy **không tự bịa câu mới**\
⚠️ Chỉ chọn lại đoạn có sẵn trong văn bản

------------------------------------------------------------------------

## 🧠 2. Ví dụ cụ thể

**Context:**\
\> Khang Hi trị vì 61 năm.

**Question:**\
\> Khang Hi trị vì bao lâu?

**Đáp án mong muốn:**\
\> 61 năm

------------------------------------------------------------------------

## 🔹 3. Tokenize là gì?

Model không đọc theo từ như con người.\
Nó cắt câu thành các mảnh nhỏ gọi là **token (subword)**.

Ví dụ sau khi tokenize:

    ["▁Kh", "ang", "▁Hi", "▁trị", "▁vì", "▁61", "▁năm"]

Dấu `▁` nghĩa là bắt đầu một từ mới (có khoảng trắng phía trước).

------------------------------------------------------------------------

## 🔹 4. Offset Mapping là gì?

Offset giúp biết mỗi token nằm ở vị trí nào trong câu gốc.

Ví dụ đánh số từng ký tự:

    K  h  a  n  g  ␣  H  i  ␣  t  r  ị  ␣  v  ì  ␣  6  1  ␣  n  ă  m
    0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22

Offset của một số token:

  Token   Offset    Nghĩa
  ------- --------- -------------------
  ▁61     (16,18)   Lấy ký tự 16 → 17
  ▁năm    (19,22)   Lấy ký tự 19 → 21

Python slice hoạt động dạng:

    text[start:end]  # end không được lấy

------------------------------------------------------------------------

## 🔹 5. Model hoạt động sau khi Fine-tune

Sau khi được train bằng UIT-ViQuAD, model học được cách:

-   Nhìn câu hỏi
-   Tìm đoạn trong context phù hợp nhất
-   Chọn vị trí bắt đầu (start)
-   Chọn vị trí kết thúc (end)

Ví dụ model dự đoán:

    start = token của "▁61"
    end   = token của "▁năm"

Dùng offset:

    context[16:22]

👉 Kết quả: **61 năm**

------------------------------------------------------------------------

## 🚀 6. Toàn bộ Workflow

    Context + Question
            ↓
    Tokenizer (cắt token + offset)
            ↓
    XLM-RoBERTa (đã fine-tune)
            ↓
    Dự đoán start & end
            ↓
    Dùng offset cắt lại text gốc
            ↓
    Answer

------------------------------------------------------------------------

## 🏁 7. Tóm lại

Project của bạn là:

> Hệ thống đọc hiểu tiếng Việt, sử dụng XLM-RoBERTa fine-tuned trên
> UIT-ViQuAD để dự đoán vị trí bắt đầu và kết thúc của câu trả lời trong
> đoạn văn.

------------------------------------------------------------------------
