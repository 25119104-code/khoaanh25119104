# ============================================================
# TUẦN 3 — Validation set + So sánh kiến trúc (206k vs 5k params)
# Project: Nhận diện chữ số viết tay (AI - Project 1/3)
# Framework: PyTorch | Chạy trên CPU (MPS có bug MPSFloatType)
# ============================================================
#
# File này giải quyết 2 việc Thầy yêu cầu ở feedback tuần 2:
#   1. Thêm validation set  -> hết data leakage, test set chỉ chạm 1 lần
#   2. Giảm params          -> so sánh DigitCNN (206,922) vs SmallCNN (5,018)
#
# CHẠY:  python model_comparison.py
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ------------------------------------------------------------
# [0] CẤU HÌNH
# ------------------------------------------------------------
SEED = 42
NUM_EPOCHS = 8
BATCH_SIZE = 64
LR = 0.001

# CPU cố định: MPS (GPU Apple Silicon) gây RuntimeError: MPSFloatType
device = torch.device("cpu")


def set_seed(seed=SEED):
    """Cố định seed để 2 model được so sánh trong cùng điều kiện.

    Không có bước này thì chênh lệch accuracy giữa 2 kiến trúc có thể
    chỉ là may rủi khởi tạo trọng số, không phải do kiến trúc.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# [1] DỮ LIỆU — train / val / test
# ------------------------------------------------------------
# KHÁI NIỆM MỚI: vì sao cần validation set
#   Code cũ dùng test set để theo dõi accuracy sau mỗi epoch, rồi lấy luôn
#   con số đó làm kết quả cuối. Đó là DATA LEAKAGE: mình đã "nhìn" test set
#   nhiều lần để ra quyết định (chọn epoch nào, kiến trúc nào), nên con số
#   cuối cùng không còn phản ánh khả năng tổng quát hóa thật.
#
#   Cách đúng:
#     - train (50k): dùng để cập nhật trọng số
#     - val   (10k): dùng để theo dõi, chọn epoch tốt nhất, so sánh kiến trúc
#     - test  (10k): CHỈ chạm đúng 1 lần, ở cuối cùng, để báo cáo
#
#   Tỷ lệ 50k/10k/10k ≈ 71/14/14, khớp với tỷ lệ ~70/15/15 Thầy đưa.

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

full_train = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_set   = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Generator riêng -> cả 2 model dùng CHUNG một cách chia, so sánh mới công bằng
split_gen = torch.Generator().manual_seed(SEED)
train_set, val_set = random_split(full_train, [50_000, 10_000], generator=split_gen)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=1000, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=1000, shuffle=False)

print(f"Train: {len(train_set):,} | Val: {len(val_set):,} | Test: {len(test_set):,}")


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


# ------------------------------------------------------------
# [4] ĐẾM PARAMS — theo từng lớp
# ------------------------------------------------------------
# KHÁI NIỆM MỚI: params vs FLOPs
#   params = số trọng số phải LƯU  -> quyết định BỘ NHỚ (vấn đề của FPGA)
#   FLOPs  = số phép tính mỗi lần suy luận -> quyết định TỐC ĐỘ / NĂNG LƯỢNG
#   Hai thứ này không đi cùng nhau: fc1 chiếm 97% params nhưng phần lớn
#   FLOPs lại nằm ở các lớp conv (vì conv dùng lại 1 kernel trên cả ảnh).

def param_table(model, name):
    print(f"\n{'=' * 58}")
    print(f"{name}")
    print(f"{'=' * 58}")
    print(f"{'Lớp':<12}{'Params':>12}{'% tổng':>12}")
    print("-" * 58)

    per_layer = []
    for layer_name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        if n > 0:
            per_layer.append((layer_name, n))

    total = sum(n for _, n in per_layer)
    for layer_name, n in per_layer:
        print(f"{layer_name:<12}{n:>12,}{100 * n / total:>11.2f}%")

    print("-" * 58)
    print(f"{'TỔNG':<12}{total:>12,}{100.0:>11.2f}%")
    print(f"{'Bộ nhớ':<12}{total * 4 / 1024:>11.1f} KB  (float32)")
    print(f"{'':<12}{total * 1 / 1024:>11.1f} KB  (nếu quantize int8)")
    return total


# ------------------------------------------------------------
# [5] TRAIN + ĐÁNH GIÁ
# ------------------------------------------------------------
def evaluate(model, loader):
    """Tính accuracy trên 1 loader bất kỳ. Không in gì, chỉ trả số."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return 100 * correct / total


def train_model(model, name, ckpt_path, epochs=NUM_EPOCHS):
    """Train và lưu checkpoint theo VAL accuracy cao nhất.

    KHÁI NIỆM MỚI: vì sao lưu theo best val acc
      Code cũ lưu state_dict sau vòng lặp -> luôn là epoch cuối.
      Nhưng epoch cuối chưa chắc tốt nhất (tuần 2: epoch 7 = 99.21%,
      epoch 8 = 99.13%, mà file .pth lại giữ epoch 8).
      Ở đây: mỗi epoch đo val acc, tốt hơn kỷ lục thì mới ghi đè file.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val = 0.0
    best_epoch = 0
    history = []

    print(f"\n--- Train {name} ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)
        history.append((epoch, avg_loss, val_acc))

        flag = ""
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), ckpt_path)
            flag = "  <- lưu checkpoint"

        print(f"Epoch {epoch}/{epochs} | loss {avg_loss:.4f} | val acc {val_acc:.2f}%{flag}")

    print(f"Tốt nhất: epoch {best_epoch} — val acc {best_val:.2f}% (đã lưu {ckpt_path})")

    # Nạp lại trọng số tốt nhất trước khi trả về, không dùng trọng số epoch cuối
    model.load_state_dict(torch.load(ckpt_path))
    return model, best_val, best_epoch, history


# ------------------------------------------------------------
# [6] CHẠY SO SÁNH
# ------------------------------------------------------------
if __name__ == "__main__":
    results = []

    # CHÚ Ý tên checkpoint: KHÔNG dùng "digit_cnn.pth".
    #   File đó do mnist_digit_recognition.py tạo (train trên 60k, lưu epoch cuối).
    #   Bản ở đây train trên 50k (đã tách 10k làm val) và lưu theo best val acc
    #   -> trọng số khác hẳn. Ghi đè lên file cũ sẽ làm demo_app.py và
    #   error_analysis.py âm thầm chạy bằng model khác mà không báo lỗi gì.
    for cls, name, ckpt in [
        (DigitCNN, "A. DigitCNN (bản gốc tuần 1)", "digit_cnn_val.pth"),
        (SmallCNN, "B. SmallCNN (kiến trúc gọn)",  "small_cnn.pth"),
    ]:
        set_seed()                      # cùng seed -> so sánh công bằng
        model = cls()
        n_params = param_table(model, name)
        model, best_val, best_epoch, _ = train_model(model, name, ckpt)
        results.append((name, n_params, best_val, best_epoch, model))

    # --------------------------------------------------------
    # TEST SET — chạm ĐÚNG 1 LẦN, ở đây, sau khi mọi quyết định đã chốt
    # --------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("KẾT QUẢ CUỐI — test set chạm đúng 1 lần")
    print(f"{'=' * 78}")
    print(f"{'Kiến trúc':<32}{'Params':>10}{'KB':>8}{'Val acc':>10}{'Test acc':>10}{'Epoch':>8}")
    print("-" * 78)

    baseline_params = baseline_test = None
    for name, n_params, best_val, best_epoch, model in results:
        test_acc = evaluate(model, test_loader)
        if baseline_params is None:
            baseline_params, baseline_test = n_params, test_acc
        print(f"{name:<32}{n_params:>10,}{n_params * 4 / 1024:>7.1f}K"
              f"{best_val:>9.2f}%{test_acc:>9.2f}%{best_epoch:>8}")

        if n_params != baseline_params:
            ratio = baseline_params / n_params
            delta = test_acc - baseline_test
            print(f"{'  -> so với bản gốc:':<32}{f'ít hơn {ratio:.1f}×':>10}"
                  f"{'':>8}{'':>10}{f'{delta:+.2f}%':>10}")

    print("-" * 78)
    print("Ghi chú cho report: con số đáng nói không phải accuracy cao hơn,")
    print("mà là ĐÁNH ĐỔI — giảm bao nhiêu lần params, mất bao nhiêu % accuracy.")
    print(f"{'=' * 78}")
