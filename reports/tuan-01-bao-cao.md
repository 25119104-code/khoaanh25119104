# Báo cáo tuần 01 — Nhận diện chữ số viết tay (MNIST)

**Ngày:** 01/09/2026
**Tuần:** 1/9 — mục tiêu: chạy được model + hiểu khái niệm liên quan

## 1. Kết quả / tiến độ
- [x] Chạy trọn vẹn `mnist_digit_recognition.py` (train 8 epoch)
- [x] Accuracy trên test set: **99.13%** (epoch 8; cao nhất đạt được là 99.21% ở epoch 7)
- [x] Hiểu luồng code: load data → build CNN → train → evaluate → save model
- Kiến trúc: CNN 2 lớp conv (1→16→32 kênh) + maxpool + 2 lớp fully-connected, Adam, 8 epoch
- Loss trung bình giảm dần đều: 0.1898 (epoch 1) → 0.0178 (epoch 8)
- Model đã lưu vào `digit_cnn.pth`

## 2. Vấn đề đang gặp
- Setup môi trường local (macOS, Apple Silicon M2): dùng conda thay vì venv thuần vì máy đã có sẵn Anaconda base — tạo env riêng `mnist` (Python 3.11) để tránh xung đột với `base`.
- File code lúc đầu tải về nằm ở `~/Downloads`, phải tự `mv` sang đúng thư mục project trước khi chạy được (lỗi `No such file or directory` khi chạy nhầm thư mục).
- Máy M2 không có CUDA (đó là công nghệ GPU của NVIDIA) — code gốc mặc định chạy CPU (`device = "cuda" if ... else "cpu"`), có thể đổi sang `mps` để tận dụng GPU tích hợp của Apple Silicon, nhưng MNIST nhẹ nên CPU vẫn đủ nhanh, không bắt buộc đổi.
- Push lên GitHub bị lỗi 403 do máy có 2 tài khoản GitHub xung đột (credential cũ trong Keychain khác với tài khoản sở hữu repo) — xử lý bằng cách xoá credential cũ (`git credential-osxkeychain erase`) và tạo Personal Access Token mới để xác thực lại.

## 3. Kiến thức mới học được
- Luồng chuẩn của 1 pipeline Deep Learning: chuẩn bị dữ liệu (Dataset/DataLoader, transform, normalize) → định nghĩa model (Conv2d, MaxPool2d, ReLU, Dropout, Linear) → loss (CrossEntropyLoss) + optimizer (Adam) → vòng lặp train (`zero_grad → forward → backward → step`) → evaluate (`model.eval()`, `torch.no_grad()`) → lưu model (`state_dict`).
- Quan sát thực tế: accuracy đã vượt 99% từ epoch 3, và epoch 7 (99.21%) cao hơn epoch 8 (99.13%) — dấu hiệu model có thể bắt đầu overfit nhẹ dù loss vẫn giảm đều. Đây là điểm sẽ đào sâu ở tuần 2 (confusion matrix, phân tích lỗi thay vì chỉ nhìn accuracy tổng).
- Phân biệt môi trường thực thi: cần phân biệt rõ "chạy trên máy thật của mình" (conda env, có Terminal, cần tự cài đặt) khác với môi trường notebook có sẵn (Colab/Kaggle) — đánh đổi giữa tiện lợi tức thời và việc tự làm chủ môi trường phát triển lâu dài.
- Xác thực Git qua HTTPS cần Personal Access Token (không dùng mật khẩu tài khoản trực tiếp), và credential được cache riêng trên máy (Keychain) tách biệt với đăng nhập trình duyệt.

## 4. Kế hoạch tuần sau (Tuần 2 — theo roadmap)
- Data augmentation (xoay/dịch ảnh)
- Điều chỉnh hyperparameter (learning rate, batch size)
- So sánh accuracy trước/sau, bắt đầu hiểu overfitting/underfitting — đối chiếu với dấu hiệu overfit nhẹ đã thấy ở tuần 1 (epoch 7 vs 8)

## 5. Câu hỏi cho Thầy (nếu có)
- [chưa có — sẽ ghi nếu tuần 2 gặp vướng mắc chưa tự giải quyết được với AI]
