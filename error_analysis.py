# ============================================================
# PHÂN TÍCH LỖI (Error Analysis) — sau khi đã có digit_cnn.pth
# Mục đích: không chỉ nhìn accuracy tổng, mà xem CỤ THỂ model
# nhầm số nào với số nào, và nhìn trực tiếp các ảnh bị đoán sai.
# ============================================================
#
# CÁCH DÙNG: chạy sau khi đã có sẵn digit_cnn.pth trong cùng thư mục
#   python3 error_analysis.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [1] LOAD LẠI ĐÚNG KIẾN TRÚC MODEL (phải giống hệt lúc train)
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(32 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)
        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = DigitCNN().to(device)
model.load_state_dict(torch.load("digit_cnn.pth", map_location=device))
model.eval()  # tắt Dropout — nhớ lý do đã học: cần kết quả ổn định, không ngẫu nhiên

# %% [2] LOAD TẬP TEST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# %% [3] MA TRẬN NHẦM LẪN (Confusion Matrix)
# confusion[i][j] = số lần model đoán là "j" khi đáp án thật là "i"
confusion = np.zeros((10, 10), dtype=int)
wrong_examples = {d: [] for d in range(10)}  # lưu ảnh sai theo từng số thật

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)

        for i in range(len(target)):
            true_label = target[i].item()
            pred_label = pred[i].item()
            confusion[true_label][pred_label] += 1
            if true_label != pred_label and len(wrong_examples[true_label]) < 10:
                wrong_examples[true_label].append((data[i].cpu(), true_label, pred_label))

# %% [4] IN MA TRẬN NHẦM LẪN RA TERMINAL
print("Ma trận nhầm lẫn (hàng = đáp án thật, cột = model đoán):")
print("     " + " ".join(f"{j:4d}" for j in range(10)))
for i in range(10):
    print(f"{i:3d}: " + " ".join(f"{confusion[i][j]:4d}" for j in range(10)))

worst_pairs = []
for i in range(10):
    for j in range(10):
        if i != j and confusion[i][j] > 0:
            worst_pairs.append((confusion[i][j], i, j))
worst_pairs.sort(reverse=True)

print("\nTop 5 cặp số hay bị nhầm nhất:")
for count, true_l, pred_l in worst_pairs[:5]:
    print(f"  Số thật '{true_l}' bị đoán nhầm thành '{pred_l}': {count} lần")

# %% [5] TÍNH ACCURACY THEO TỪNG SỐ
print("\nAccuracy theo từng chữ số:")
for d in range(10):
    total_d = confusion[d].sum()
    correct_d = confusion[d][d]
    acc_d = 100 * correct_d / total_d if total_d > 0 else 0
    print(f"  Số {d}: {acc_d:.2f}% ({correct_d}/{total_d})")

overall_acc = 100 * np.trace(confusion) / confusion.sum()
print(f"\nAccuracy tổng: {overall_acc:.2f}%")

# %% [6] VẼ LƯỚI ẢNH BỊ ĐOÁN SAI
fig, axes = plt.subplots(10, 10, figsize=(15, 15))
for d in range(10):
    examples = wrong_examples[d]
    for col in range(10):
        ax = axes[d][col]
        ax.axis("off")
        if col < len(examples):
            img, true_l, pred_l = examples[col]
            ax.imshow(img[0], cmap="gray")
            ax.set_title(f"T:{true_l} P:{pred_l}", fontsize=8, color="red")
    axes[d][0].set_ylabel(f"Số {d}", fontsize=10, rotation=0, labelpad=20)

plt.suptitle("Các ảnh bị đoán sai theo từng chữ số thật (T=thật, P=model đoán)")
plt.tight_layout()
plt.savefig("error_analysis.png", dpi=120)
print("\nĐã lưu lưới ảnh sai vào error_analysis.png")

# %% [7] VẼ HEATMAP MA TRẬN NHẦM LẪN
fig2, ax2 = plt.subplots(figsize=(8, 7))
im = ax2.imshow(confusion, cmap="Blues")
ax2.set_xticks(range(10))
ax2.set_yticks(range(10))
ax2.set_xlabel("Model đoán")
ax2.set_ylabel("Đáp án thật")
ax2.set_title("Confusion Matrix")
for i in range(10):
    for j in range(10):
        color = "white" if confusion[i][j] > confusion.max()/2 else "black"
        ax2.text(j, i, confusion[i][j], ha="center", va="center", color=color, fontsize=8)
plt.colorbar(im, ax=ax2)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
print("Đã lưu confusion matrix vào confusion_matrix.png")
