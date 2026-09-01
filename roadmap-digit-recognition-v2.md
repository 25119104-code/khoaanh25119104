# Roadmap: Nhận diện chữ số viết tay (Handwritten Digit Recognition)

> ⚠️ **Đây là giả thuyết ban đầu của mình, chưa qua Thầy duyệt.** Chỉ Phase 0 là chắc chắn (đã làm/đang làm theo đúng yêu cầu Thầy vừa đưa). Từ Phase 1 trở đi sẽ cụ thể hoá dần sau mỗi buổi gặp Thầy — không cam kết trước.

## 1. Mục tiêu nhiệm vụ

Project đầu tiên trong 2-3 project của năm, đi qua 3 mảng **AI – IC – ES**, để năm 3 chọn mảng đào sâu. Project này ở lại trong AI/DL, không lấn IC/ES.

- Làm quen quy trình build 1 model Deep Learning hoàn chỉnh: dữ liệu → model → train → đánh giá → lưu model.
- Tập thói quen tự học qua AI: gặp khái niệm lạ → hỏi → hiểu → áp dụng.
- Tập viết report kỹ thuật + dùng Git/GitHub quản lý version.
- **Mới**: commit + report hàng tuần lên GitHub là điều kiện bắt buộc, không phải tuỳ chọn (yêu cầu của Thầy).

Bài toán cụ thể: cho 1 ảnh chữ số viết tay (0–9), model phải đoán đúng đó là số mấy.

## 2. Giải thích thuật ngữ chuyên ngành

| Thuật ngữ | Giải thích ngắn gọn |
|---|---|
| **Deep Learning (DL)** | Nhánh ML dùng mạng neural nhiều lớp để tự học đặc trưng từ dữ liệu. |
| **MNIST** | 70.000 ảnh chữ số viết tay (28×28, ảnh xám), bài toán phân loại ảnh chuẩn. |
| **CNN** | Kiến trúc mạng chuyên xử lý ảnh, dùng tích chập phát hiện đặc trưng cục bộ. |
| **Convolution** | Cửa sổ nhỏ (kernel) trượt qua ảnh, tạo feature map. |
| **Pooling** | Thu nhỏ ảnh sau tích chập, giữ thông tin quan trọng. |
| **Epoch** | 1 lần mạng học qua toàn bộ tập train. |
| **Batch / batch size** | Lô dữ liệu đưa vào mạng cùng lúc mỗi bước cập nhật trọng số. |
| **Loss function** | Đo mức sai của model so với nhãn thật. |
| **Backpropagation** | Tính gradient của loss theo từng trọng số. |
| **Optimizer (Adam)** | Dùng gradient cập nhật trọng số để loss giảm dần. |
| **Learning rate** | Bước nhảy khi cập nhật trọng số. |
| **Overfitting** | Model học thuộc train, dự đoán kém trên dữ liệu mới. |
| **Dropout** | Tắt ngẫu nhiên neuron khi train, chống overfitting. |
| **Accuracy** | Tỷ lệ dự đoán đúng — thước đo phổ biến cho phân loại. |
| **GPU / CUDA** | Tăng tốc tính toán song song. |
| **Inference** | Dùng model đã train để dự đoán trên dữ liệu mới. |

## 3. Ứng dụng thực tế

- OCR: số hóa văn bản viết tay/in.
- Ngân hàng: đọc số tiền viết tay trên séc.
- Bưu chính: đọc mã bưu điện viết tay (gốc gác của MNIST, từ USPS).
- Chấm điểm/nhập liệu tự động: phiếu trắc nghiệm, khảo sát viết tay.
- Nền tảng mở rộng: biển số xe, khuôn mặt, computer vision công nghiệp.
- (Ngoài phạm vi project này) MNIST-CNN cũng là bài tập kinh điển cho TinyML — hướng cho project ES riêng sau này.

## 4. Kế hoạch theo phase (cuốn chiếu, không khoá cứng theo tuần)

**Nguyên tắc:** chỉ phase hiện tại + phase kế tiếp là có nội dung cụ thể. Phần sau là backlog định hướng, sẽ viết lại sau mỗi buổi gặp Thầy (thứ 4 hàng tuần) dựa trên feedback thật.

### Phase 0 — Baseline (đã xong tuần 1)
- Setup Git/GitHub, commit đều hàng tuần theo đúng yêu cầu Thầy.
- Build & chạy được CNN nhận diện MNIST (`mnist_digit_recognition.py`), ghi nhận accuracy thật.
- Hiểu chắc, tự giải thích lại được các khái niệm nền ở mục 2.
- **Kết quả đạt được:** model chạy được, accuracy 99.13%, repo GitHub + report tuần đầu tiên.

### Phase 1 — Đánh giá đúng cách (kế tiếp, sẽ cụ thể hoá sau khi gặp Thầy)
- Cải thiện model: thử data augmentation, điều chỉnh hyperparameter (learning rate, batch size).
- Đánh giá đúng cách: confusion matrix, precision/recall, phân tích số hay bị đoán nhầm.
- **Kết quả dự kiến:** bảng so sánh phiên bản model, report phân tích lỗi (không chỉ nhìn accuracy tổng).

### Phase 2 — So sánh & mở rộng (backlog, chưa cam kết)
- So sánh kiến trúc: CNN hiện tại vs MLP thuần vs CNN sâu hơn; thử optimizer/activation khác.
- Mở rộng dữ liệu: EMNIST, hoặc tự viết/chụp chữ số của chính mình để test.
- Transfer learning / thử framework thứ 2 (PyTorch ↔ Keras) để so sánh trải nghiệm.

### Phase 3 — Đóng gói & tối ưu (backlog, chưa cam kết)
- Đóng gói demo dùng được: web app đơn giản (Flask/Streamlit/Gradio) vẽ tay + nhận diện trực tiếp.
- Tối ưu model: quantization/pruning cơ bản, đo đánh đổi kích thước vs độ chính xác — nền cho project ES sau này.

### Phase cuối — Tổng kết
- Hoàn thiện repo GitHub, gộp report từng tuần thành báo cáo tổng, chuẩn bị trình bày với Thầy trước khi qua project mảng tiếp theo.

## 5. Cách vận hành thực tế

- Repo GitHub: `/notebooks`, `/reports` (mỗi buổi/tuần 1 file `.md`), `/models`.
- Mỗi buổi làm: commit, không đợi hết tuần.
- Mỗi thứ 4: report có đủ 3 phần (kết quả – vấn đề – kế hoạch), rồi mới viết lại "1 phase/tuần tới" — không dựa vào bảng phase này để đoán trước quá xa.
- Gặp khái niệm mới → hỏi AI → hiểu → áp dụng vào code, không học chay lý thuyết.
