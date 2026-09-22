# ============================================================
# models.py — NOI DUY NHAT dinh nghia kien truc model
# ============================================================
# Ly do ton tai: truoc day class DigitCNN bi chep tay o 4 file
# (mnist_digit_recognition.py, model_comparison.py, demo_app.py,
# error_analysis.py). Doi kien truc la phai sua 4 cho; quen 1 cho
# thi hoac loi shape, hoac te hon la chay duoc nhung sai.
#
# Tu nay moi file import tu day:
#     from models import SmallCNN
#
# Ngoai le co y: mnist_digit_recognition.py (baseline tuan 1) van giu
# ban chep rieng cua no, de dong bang lam moc so sanh — khong sua.
#
# KIEN TRUC DA CHOT: SmallCNN (5.018 params, test acc 98,82%).
# DigitCNN (206.922 params, 99,05%) chi giu de doi chieu.
# ============================================================

import torch.nn as nn


# ------------------------------------------------------------
# [2] KIẾN TRÚC A — DigitCNN (bản gốc tuần 1, 206,922 params)
# ------------------------------------------------------------
class DigitCNN(nn.Module):
    """Kiến trúc ban đầu. Nút thắt: fc1 chiếm 97% tổng params.

    Nguyên nhân: sau 2 lần pool, feature map còn 32x7x7 = 1568 giá trị,
    nối thẳng vào lớp ẩn 128 neuron -> 1568 x 128 = 200,704 trọng số.
    """

    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(1, 16, kernel_size=3, padding=1)   #     160
        self.conv2   = nn.Conv2d(16, 32, kernel_size=3, padding=1)  #   4,640
        self.pool    = nn.MaxPool2d(2, 2)
        self.fc1     = nn.Linear(32 * 7 * 7, 128)                   # 200,832  <-- nút thắt
        self.fc2     = nn.Linear(128, 10)                           #   1,290
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # 28 -> 14
        x = self.pool(self.relu(self.conv2(x)))   # 14 -> 7
        x = x.flatten(1)                          # -> 1568
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ------------------------------------------------------------
# [3] KIẾN TRÚC B — SmallCNN (5,018 params)
# ------------------------------------------------------------
class SmallCNN(nn.Module):
    """Kiến trúc gọn, port từ bản Keras của anh năm 4 (5,018 params, 98.88%).

    Hai chiêu chính để giết nút thắt FC:
      1. Thêm conv3 + pool3 -> ép spatial 7x7 xuống 3x3 TRƯỚC khi Flatten.
         Flatten ra 144 thay vì 1568 (giảm 11 lần ngay tại đầu vào FC).
      2. Bỏ hẳn lớp FC ẩn -> đi thẳng Dense(144 -> 10).

    Cộng thêm channel nhỏ hơn (8/16/16 thay vì 16/32).
    Không dùng Dropout: 5k params thì overfit khó xảy ra, và bỏ đi thì
    Phase 2B (xác định model inference) nhẹ hơn.

    Lưu ý khi đối chiếu với bản Keras:
      - Keras padding='same' + kernel 3x3  ==  PyTorch padding=1
      - Keras MaxPooling2D padding='valid' ==  PyTorch MaxPool2d(2,2), 7//2 = 3
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,  8,  kernel_size=3, padding=1)   #    80
        self.conv2 = nn.Conv2d(8,  16, kernel_size=3, padding=1)   # 1,168
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)   # 2,320
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc    = nn.Linear(16 * 3 * 3, 10)                     # 1,450
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # 28 -> 14
        x = self.pool(self.relu(self.conv2(x)))   # 14 -> 7
        x = self.pool(self.relu(self.conv3(x)))   # 7  -> 3
        x = x.flatten(1)                          # -> 144
        return self.fc(x)                         # logits, chưa softmax
