# Báo cáo tuần 02 — Nhận diện chữ số viết tay (MNIST)

**Ngày:** 05/09/2026
**Tuần:** 2/9

**Lưu ý:** tuần này em làm khác kế hoạch ban đầu (data augmentation/hyperparameter tuning) — thay vào đó em tự làm thêm phần đánh giá lỗi (Phase 1) và demo thực tế (Phase 3) sớm hơn dự kiến, sau khi xem 1 video tham khảo và so sánh với project của mình. Data augmentation/hyperparameter tuning em dời sang tuần sau.

## 1. Kết quả / tiến độ

- [x] Viết `error_analysis.py`: tính confusion matrix 10×10 trên test set, tìm cặp số hay bị nhầm nhất, accuracy theo từng chữ số.
  - Accuracy tổng: **99.21%**
  - Top 3 cặp hay nhầm: số thật "2" đoán nhầm "7" (8 lần), "4" đoán nhầm "9" (6 lần), "9" đoán nhầm "7" (4 lần)
  - Số yếu nhất: số 9 (98.61%), số 2 (98.74%)
  - Xuất `error_analysis.png` (lưới ảnh sai) và `confusion_matrix.png` (heatmap)
- [x] Viết `demo_app.py`: demo tương tác bằng Gradio — vẽ tay 1 chữ số trên canvas, model đoán trực tiếp kèm % tin cậy từng lớp (0-9)
- [x] Phát hiện và tự sửa 1 lỗi thật khi test demo: vẽ số 0/1/8 rất rõ ràng nhưng model đoán sai gần hết

## 2. Vấn đề đang gặp

- **Lỗi tiền xử lý ảnh vẽ tay (đã tự sửa)**: code ban đầu chỉ `resize()` nguyên cả canvas về 28×28. Canvas trên UI lớn hơn nhiều so với chữ số vẽ ở giữa, nên sau khi resize, chữ số bị co lại thành 1 vệt nhỏ trong ảnh 28×28 — model chưa từng thấy tỉ lệ đó lúc train (MNIST gốc luôn chuẩn hóa chữ số chiếm ~20/28 pixel, căn giữa khung). Sửa bằng cách thêm bước: tìm bounding box của nét vẽ → cắt sát → scale giữ tỉ lệ vào khung 20×20 → dán vào giữa nền đen 28×28 — đúng quy trình chuẩn hóa gốc của MNIST. Test lại với 3 số (0, 1, 8) đều đoán đúng >99%.
- **Lỗ hổng phương pháp luận phát hiện được khi ôn lại code**: project hiện chỉ có train/test set, không có validation set riêng. Việc gọi `evaluate()` trên test set sau mỗi epoch trong lúc train khiến test set không còn hoàn toàn "sạch" — bất kỳ quyết định nào sau này (chọn số epoch, hyperparameter) bị ảnh hưởng bởi việc quan sát kết quả test set, dù chỉ bằng mắt, cũng là 1 dạng rò rỉ thông tin (data leakage) nhẹ. Chưa sửa code, dự kiến tách validation set ở tuần sau.

## 3. Kiến thức mới học được

- **Chuẩn hóa ảnh input đúng cách cho inference**: khác với training (dữ liệu đã được chuẩn hóa sẵn bởi `torchvision.datasets.MNIST`), dữ liệu thực tế (ảnh vẽ tay) cần tự tiền xử lý đúng chuẩn — đảo màu, bounding-box crop, scale giữ tỉ lệ, normalize cùng mean/std lúc train — nếu thiếu 1 bước, model vẫn chạy không lỗi nhưng cho kết quả sai hoàn toàn.
- **Phân biệt train vs inference (predict)**: dùng model đã train xong để dự đoán dữ liệu mới không cập nhật trọng số, khác hoàn toàn với train (không có `backward()`, `optimizer.step()`).
- **Batch dimension trong PyTorch**: model luôn kỳ vọng input/output có chiều batch (`[batch_size, ...]`) kể cả khi chỉ xử lý 1 mẫu — cần `unsqueeze(0)` khi đưa vào và `[0]` khi lấy kết quả ra.
- **Confidence (độ tin cậy) không đồng nghĩa với đúng**: softmax cho biết model "tự tin" đến đâu dựa trên những gì nó học được, không phản ánh việc dự đoán có đúng thực tế hay không — model có thể tự tin 100% và vẫn sai, đặc biệt với input khác phân phối dữ liệu train (out-of-distribution).
- **Vì sao CNN thắng MLP thuần cho ảnh** — 2 lý do: (1) parameter sharing — dùng lại 1 bộ kernel nhỏ ở mọi vị trí thay vì mỗi neuron 1 bộ trọng số riêng (ít hơn ~90 lần số tham số so với MLP tương đương); (2) translation invariance — kernel phát hiện được 1 đặc trưng dù nó nằm ở vị trí nào trên ảnh. Đồng thời nhận ra: model của mình thực chất là CNN + MLP kết hợp (2 lớp cuối `fc1`, `fc2` vẫn là fully-connected/MLP) — CNN lo trích đặc trưng, MLP lo phân loại cuối.
- **Data leakage / thiếu validation set**: hiểu vì sao dùng test set để theo dõi trong lúc train (dù không sửa code trực tiếp dựa vào đó) vẫn làm giảm độ tin cậy của con số accuracy cuối cùng.

## 4. Kế hoạch tuần sau

- Tách validation set riêng từ tập train (sửa `mnist_digit_recognition.py`, dùng `random_split`)
- Data augmentation (xoay/dịch/zoom ảnh) — dời từ tuần 2 (chưa làm)
- Hyperparameter tuning (learning rate, batch size) — dời từ tuần 2 (chưa làm)

## 5. Câu hỏi cho Thầy (nếu có)

- Project hiện chưa có validation set, dùng test set để theo dõi accuracy trong lúc train — em có cần làm lại toàn bộ quá trình train với validation set trước khi tiếp tục các phần sau, hay có thể vừa làm vừa sửa dần?
