# ============================================================
# NHIỆM VỤ 01: Nhận diện chữ số viết tay (MNIST) bằng Deep Learning
# Framework: PyTorch
# Chạy trên: Google Colab hoặc Kaggle Notebook (có GPU free)
# ============================================================
#
# CÁCH DÙNG:
# - Copy từng khối "# %%" vào 1 cell riêng trên Colab/Kaggle, HOẶC
# - Chạy thẳng cả file này nếu máy cá nhân có Python + PyTorch.
#
# TRƯỚC KHI CHẠY (chỉ trên máy cá nhân, Colab/Kaggle đã có sẵn):
#   pip install torch torchvision matplotlib

# %% [1] IMPORT THƯ VIỆN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Khái niệm mới:
# - torch.nn: chứa các "khối xây dựng" mạng neural (layer, hàm loss...)
# - torchvision.datasets: có sẵn bộ dữ liệu MNIST, không cần tự tải/xử lý
# - DataLoader: chia dữ liệu thành từng "lô" (batch) để train hiệu quả hơn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Đang chạy trên:", device)
# -> Nhớ bật GPU: Colab: Runtime > Change runtime type > GPU
#    Kaggle: Settings > Accelerator > GPU

# %% [2] TẢI VÀ CHUẨN BỊ DỮ LIỆU
transform = transforms.Compose([
    transforms.ToTensor(),                    # ảnh -> tensor, giá trị pixel [0,1]
    transforms.Normalize((0.1307,), (0.3081,)) # chuẩn hóa theo mean/std chuẩn của MNIST
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Khái niệm mới:
# - MNIST: 70.000 ảnh chữ số viết tay 0-9, kích thước 28x28 pixel, ảnh xám (1 kênh màu)
# - Normalize: đưa dữ liệu về phân phối chuẩn -> mạng học nhanh và ổn định hơn
# - batch_size: số ảnh xử lý cùng lúc mỗi bước train (đánh đổi tốc độ / bộ nhớ)
# - shuffle=True: xáo trộn dữ liệu mỗi epoch để tránh mạng học theo thứ tự

# Xem thử vài ảnh mẫu (không bắt buộc, chỉ để hiểu dữ liệu)
examples = enumerate(train_loader)
_, (example_data, example_targets) = next(examples)
fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(example_data[i][0], cmap="gray")
    plt.title(f"Label: {example_targets[i].item()}")
    plt.axis("off")
plt.tight_layout()
plt.show()

# %% [3] XÂY DỰNG MODEL (CNN - Convolutional Neural Network)
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 1 kênh -> 16 kênh đặc trưng
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 16 -> 32 kênh đặc trưng
        self.pool  = nn.MaxPool2d(2, 2)                          # giảm kích thước ảnh đi 1 nửa
        self.fc1   = nn.Linear(32 * 7 * 7, 128)                  # lớp fully-connected
        self.fc2   = nn.Linear(128, 10)                          # 10 lớp output = 10 chữ số (0-9)
        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(self.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = x.view(x.size(0), -1)                # "duỗi phẳng" tensor thành vector
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # trả về 10 điểm số (logits), chưa qua softmax

# Khái niệm mới:
# - Conv2d (tích chập): quét một "cửa sổ nhỏ" qua ảnh để phát hiện đặc trưng
#   (cạnh, góc, nét cong...) -> rất hiệu quả cho ảnh, hơn hẳn fully-connected thuần
# - MaxPool2d: giữ lại giá trị lớn nhất trong 1 vùng -> giảm kích thước, giữ đặc trưng quan trọng
# - ReLU: hàm kích hoạt phi tuyến, giúp mạng học được quan hệ phức tạp
# - Dropout: ngẫu nhiên "tắt" một số neuron khi train -> chống overfitting
# - Linear (fully-connected): lớp kết nối đầy đủ, ra quyết định cuối cùng
# - forward(): định nghĩa dữ liệu đi qua mạng theo thứ tự nào

model = DigitCNN().to(device)
print(model)

# %% [4] HÀM LOSS VÀ OPTIMIZER
criterion = nn.CrossEntropyLoss()               # hàm mất mát cho bài toán phân loại nhiều lớp
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Khái niệm mới:
# - CrossEntropyLoss: đo "mạng sai bao nhiêu" khi dự đoán 1 trong 10 lớp
# - Adam: thuật toán tối ưu, tự điều chỉnh tốc độ học (learning rate) cho từng tham số
# - lr (learning rate): bước nhảy khi cập nhật trọng số; quá lớn -> không hội tụ,
#   quá nhỏ -> học chậm

# %% [5] VÒNG LẶP TRAIN
def train_one_epoch(epoch):
    model.train()  # bật chế độ train (Dropout hoạt động)
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()        # xóa gradient cũ trước mỗi bước
        output = model(data)         # forward pass: dự đoán
        loss = criterion(output, target)
        loss.backward()              # backpropagation: tính gradient
        optimizer.step()             # cập nhật trọng số theo gradient

        total_loss += loss.item()
        if batch_idx % 200 == 0:
            print(f"Epoch {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}] "
                  f"Loss: {loss.item():.4f}")
    print(f"==> Epoch {epoch} - Loss trung bình: {total_loss/len(train_loader):.4f}")

# Khái niệm mới:
# - epoch: 1 lần mạng "nhìn qua" toàn bộ dữ liệu train
# - zero_grad / backward / step: 3 bước cố định của mọi vòng train PyTorch
#   (xóa gradient cũ -> tính gradient mới -> cập nhật trọng số)
# - model.train() vs model.eval(): bật/tắt các layer chỉ dùng khi train (như Dropout)

# %% [6] ĐÁNH GIÁ TRÊN TẬP TEST
def evaluate():
    model.eval()  # tắt Dropout khi đánh giá
    correct = 0
    total = 0
    with torch.no_grad():  # không cần tính gradient khi test -> tiết kiệm bộ nhớ, nhanh hơn
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)  # chọn lớp có điểm cao nhất làm dự đoán
            correct += (pred == target).sum().item()
            total += target.size(0)
    acc = 100 * correct / total
    print(f"Độ chính xác trên tập test: {acc:.2f}%")
    return acc

# %% [7] CHẠY TRAIN (5-10 epoch theo yêu cầu)
NUM_EPOCHS = 8
for epoch in range(1, NUM_EPOCHS + 1):
    train_one_epoch(epoch)
    evaluate()

# %% [8] LƯU MODEL
torch.save(model.state_dict(), "digit_cnn.pth")
print("Đã lưu model vào digit_cnn.pth")

# Khi Thầy gửi model nhỏ hơn để train nhanh, bạn chỉ cần thay class DigitCNN
# bằng kiến trúc Thầy đưa, phần còn lại (loop train/test) giữ nguyên.

# %% [9] THỬ DỰ ĐOÁN 1 ẢNH BẤT KỲ TỪ TẬP TEST
model.eval()
sample_data, sample_target = next(iter(test_loader))
sample_data, sample_target = sample_data.to(device), sample_target.to(device)
with torch.no_grad():
    output = model(sample_data)
    pred = output.argmax(dim=1)

plt.imshow(sample_data[0][0].cpu(), cmap="gray")
plt.title(f"Dự đoán: {pred[0].item()} | Thật: {sample_target[0].item()}")
plt.axis("off")
plt.show()
