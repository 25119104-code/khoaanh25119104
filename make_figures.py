# ============================================================
# SINH HÌNH CHO SLIDE VÀ TÀI LIỆU — chạy sau khi đã có small_cnn.pth
#
# Tạo ra 4 hình + 1 file ONNX:
#   confusion_matrix_small.png   ma trận nhầm lẫn của SmallCNN
#   error_analysis_small.png     lưới ảnh SmallCNN đoán sai
#   mnist_samples.png            ảnh mẫu đầu vào (cho slide bước 1)
#   per_class_accuracy.png       accuracy theo từng chữ số, so 2 model
#   small_cnn.onnx               để xem sơ đồ kiến trúc bằng Netron
#
# CHẠY:  conda activate mnist && python make_figures.py
# ============================================================

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")                      # không mở cửa sổ, chỉ ghi file
import matplotlib.pyplot as plt

# Nạp lại class model và test_loader từ model_comparison thay vì chép lần nữa.
# Class DigitCNN hiện đã bị chép ở 4 file — đây là lý do nên gom vào models.py.
from model_comparison import DigitCNN, SmallCNN, test_loader, device

MEAN, STD = 0.1307, 0.3081                 # để đảo chuẩn hoá khi hiển thị ảnh


def analyze(model, ckpt):
    """Chạy toàn bộ test set, trả về ma trận nhầm lẫn và vài ảnh đoán sai."""
    model = model.to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    conf = np.zeros((10, 10), dtype=int)
    wrong = {d: [] for d in range(10)}

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            for i in range(len(target)):
                t, p = target[i].item(), pred[i].item()
                conf[t][p] += 1
                if t != p and len(wrong[t]) < 10:
                    wrong[t].append((data[i].cpu(), t, p))
    return conf, wrong


def top_pairs(conf, k=5):
    pairs = [(conf[i][j], i, j) for i in range(10) for j in range(10)
             if i != j and conf[i][j] > 0]
    pairs.sort(reverse=True)
    return pairs[:k]


def plot_confusion(conf, title, path):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(conf, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("Model doan"); ax.set_ylabel("Dap an that")
    ax.set_title(title)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, conf[i][j], ha="center", va="center", fontsize=8,
                    color="white" if conf[i][j] > conf.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print("  ->", path)


def plot_wrong(wrong, title, path):
    fig, axes = plt.subplots(10, 10, figsize=(14, 14))
    for d in range(10):
        for c in range(10):
            ax = axes[d][c]
            ax.axis("off")
            if c < len(wrong[d]):
                img, t, p = wrong[d][c]
                ax.imshow(img[0], cmap="gray")
                ax.set_title(f"T:{t} P:{p}", fontsize=8, color="red")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)
    print("  ->", path)


def plot_samples(path):
    """Lưới 5x8 ảnh đầu vào — dùng cho slide bước 1 (chuẩn bị dữ liệu)."""
    data, target = next(iter(test_loader))
    fig, axes = plt.subplots(5, 8, figsize=(9, 6))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        ax.imshow(data[i][0] * STD + MEAN, cmap="gray")   # đảo chuẩn hoá về 0-1
        ax.set_title(str(target[i].item()), fontsize=10)
    plt.suptitle("Anh dau vao MNIST 28x28 grayscale", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print("  ->", path)


def plot_per_class(conf_a, conf_b, path):
    """Accuracy theo từng chữ số, đặt cạnh nhau 2 model."""
    acc = lambda c: [100 * c[d][d] / c[d].sum() for d in range(10)]
    a, b = acc(conf_a), acc(conf_b)
    x = np.arange(10); w = 0.38

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(x - w / 2, a, w, label="DigitCNN (206.922 params)", color="#B8C9D4")
    ax.bar(x + w / 2, b, w, label="SmallCNN (5.018 params)", color="#1C7293")
    ax.set_xticks(x); ax.set_xlabel("Chu so"); ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(96, 100.3)
    ax.set_title("Accuracy theo tung chu so")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#DCE5EB")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print("  ->", path)


if __name__ == "__main__":
    print("\n[1/4] Phan tich SmallCNN tren test set...")
    conf_s, wrong_s = analyze(SmallCNN(), "small_cnn.pth")

    print("[2/4] Phan tich DigitCNN de doi chieu...")
    conf_d, _ = analyze(DigitCNN(), "digit_cnn_val.pth")

    # ---- so sánh cặp hay nhầm ----
    print("\n" + "=" * 62)
    print("CAP CHU SO HAY NHAM NHAT — co doi khi thu gon kien truc khong?")
    print("=" * 62)
    print(f"{'':4}{'DigitCNN':<26}{'SmallCNN':<26}")
    print("-" * 62)
    for i, (pd, ps) in enumerate(zip(top_pairs(conf_d), top_pairs(conf_s)), 1):
        print(f"{i:<4}{f'{pd[1]} -> {pd[2]}  ({pd[0]} lan)':<26}"
              f"{f'{ps[1]} -> {ps[2]}  ({ps[0]} lan)':<26}")
    print("-" * 62)
    print(f"{'Tong sai':<10}{10000 - np.trace(conf_d):<20}{10000 - np.trace(conf_s):<20}")
    print(f"{'Accuracy':<10}{100*np.trace(conf_d)/10000:<20.2f}{100*np.trace(conf_s)/10000:<20.2f}")
    print("=" * 62)

    print("\n[3/4] Ve hinh...")
    plot_confusion(conf_s, "Confusion Matrix — SmallCNN (5.018 params)",
                   "confusion_matrix_small.png")
    plot_wrong(wrong_s, "Anh SmallCNN doan sai (T = that, P = doan)",
               "error_analysis_small.png")
    plot_samples("mnist_samples.png")
    plot_per_class(conf_d, conf_s, "per_class_accuracy.png")

    print("\n[4/4] Xuat ONNX de xem bang Netron (netron.app)...")
    model = SmallCNN().to(device)
    model.load_state_dict(torch.load("small_cnn.pth", map_location=device))
    model.eval()
    torch.onnx.export(
        model, torch.randn(1, 1, 28, 28, device=device), "small_cnn.onnx",
        input_names=["input"], output_names=["logits"], opset_version=18,
        # opset 18: torch moi chi co implementation cho >=18. De 13 van xuat duoc
        # nhung in ra mot traceback "No Adapter From Version 14 for Relu" gay hoang mang.
    )
    print("  -> small_cnn.onnx")
    print("\nXong. Keo small_cnn.onnx vao netron.app de chup so do kien truc.")
