# ============================================================
# PHÂN TÍCH LỖI (Error Analysis)
# Mục đích: không chỉ nhìn accuracy tổng, mà xem CỤ THỂ model
# nhầm số nào với số nào, và nhìn trực tiếp các ảnh bị đoán sai.
# ============================================================
#
# CÁCH DÙNG:
#   python3 error_analysis.py                  # SmallCNN (kiến trúc đã chốt)
#   python3 error_analysis.py --model digit    # DigitCNN tuần 1, để đối chiếu
#
# Tên hình xuất ra phụ thuộc model, nên chạy model này KHÔNG ghi đè hình của
# model kia — tránh chuyện hình trong slide bị thay bằng hình của model khác.

import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Kiến trúc nạp từ models.py — không chép lại class ở đây nữa.
from models import DigitCNN, SmallCNN

import os
os.makedirs("models", exist_ok=True)   # noi chua .pth va .onnx
os.makedirs("figures", exist_ok=True)  # noi chua .png


device = torch.device("cpu")  # MPS bị lỗi (RuntimeError MPSFloatType), quay lại CPU cho chắc

# %% [1] CHỌN MODEL — mặc định là kiến trúc đã chốt
ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["small", "digit"], default="small",
                help="small = SmallCNN 5.018 params (đã chốt); digit = DigitCNN tuần 1")
args = ap.parse_args()

if args.model == "small":
    model_cls, ckpt = SmallCNN, "models/small_cnn.pth"
    label = "SmallCNN (5.018 params)"
    out_conf, out_wrong = "figures/confusion_matrix_small.png", "figures/error_analysis_small.png"
else:
    model_cls, ckpt = DigitCNN, "models/digit_cnn_val.pth"
    label = "DigitCNN (206.922 params)"
    out_conf, out_wrong = "figures/confusion_matrix.png", "figures/error_analysis.png"

print(f"Model: {label}  |  checkpoint: {ckpt}")

model = model_cls().to(device)
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()  # tắt Dropout (DigitCNN có, SmallCNN không) — cần kết quả ổn định

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

plt.suptitle(f"Ảnh bị đoán sai — {label} (T=thật, P=model đoán)")
plt.tight_layout()
plt.savefig(out_wrong, dpi=120)
print(f"\nĐã lưu lưới ảnh sai vào {out_wrong}")

# %% [7] VẼ HEATMAP MA TRẬN NHẦM LẪN
fig2, ax2 = plt.subplots(figsize=(8, 7))
im = ax2.imshow(confusion, cmap="Blues")
ax2.set_xticks(range(10))
ax2.set_yticks(range(10))
ax2.set_xlabel("Model đoán")
ax2.set_ylabel("Đáp án thật")
ax2.set_title(f"Confusion Matrix — {label}")
for i in range(10):
    for j in range(10):
        color = "white" if confusion[i][j] > confusion.max()/2 else "black"
        ax2.text(j, i, confusion[i][j], ha="center", va="center", color=color, fontsize=8)
plt.colorbar(im, ax=ax2)
plt.tight_layout()
plt.savefig(out_conf, dpi=120)
print(f"Đã lưu confusion matrix vào {out_conf}")
