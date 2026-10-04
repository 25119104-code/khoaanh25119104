# ============================================================
# THÍ NGHIỆM: thêm BatchNorm vào SmallCNN — được gì, mất gì?
#
# Câu hỏi:
#   1. Không có BatchNorm thì SmallCNN thiệt gì? (tốc độ hội tụ, accuracy)
#   2. Có BatchNorm thì GỘP (fold) vào Conv được không — kết quả có giữ nguyên?
#   3. Sau khi gộp, trọng số có còn vừa int8 / Q1.7 không? (chuẩn bị cho quantization)
#
# Script này là THÍ NGHIỆM, KHÔNG thay model chính:
#   - Không sửa models.py, không ghi đè models/small_cnn.pth.
#   - Checkpoint riêng: models/study_small_cnn_lai.pth, models/study_small_cnn_bn.pth
#   - Kết quả: figures/thi_nghiem_batchnorm.png + study/ket_qua_batchnorm.json
#
# So sánh công bằng: cả hai model train LẠI trong cùng một lần chạy, cùng seed,
# cùng cách chia 50k/10k/10k, cùng Adam lr=0.001, batch 64, 30 epoch —
# dùng lại đúng hàm train_model() của model_comparison.py.
#
# CHẠY:  conda activate mnist && python study/thu_batchnorm.py
#        (khoảng vài chục phút trên CPU — train 2 model × 30 epoch)
#        python study/thu_batchnorm.py --epochs 5   # chạy thử nhanh
# ============================================================

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)   # model_comparison dùng đường dẫn tương đối: ./data, models/

from model_comparison import (EPOCHS, device, evaluate, set_seed, test_loader,  # noqa: E402
                              train_model)
from models import SmallCNN  # noqa: E402

Q17_MAX = 127 / 128          # trần Q1.7 = 0,9921875


# ------------------------------------------------------------
# [1] SmallCNN có BatchNorm
# ------------------------------------------------------------
class SmallCNN_BN(nn.Module):
    """Giống hệt SmallCNN, chỉ chèn BatchNorm2d NGAY SAU mỗi Conv, TRƯỚC ReLU.

    Thứ tự Conv → BN → ReLU là bắt buộc nếu muốn gộp BN vào Conv:
    Conv và BN (lúc inference) đều tuyến tính nên gộp được; ReLU thì không.

    Conv đặt bias=False: BN đã có β làm nhiệm vụ dịch, bias của Conv sẽ bị
    trừ mất ở bước (x − μ) nên giữ lại chỉ thừa tham số.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 3 * 3, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))   # 28 -> 14
        x = self.pool(self.relu(self.bn2(self.conv2(x))))   # 14 -> 7
        x = self.pool(self.relu(self.bn3(self.conv3(x))))   # 7  -> 3
        return self.fc(x.flatten(1))


# ------------------------------------------------------------
# [2] GỘP BatchNorm vào Conv
# ------------------------------------------------------------
def fold_bn(model_bn):
    """Trả về một SmallCNN BÌNH THƯỜNG (không có BN) cho kết quả y hệt model_bn.

    Lúc inference, BN của mỗi kênh là:   y = γ·(x − μ)/√(σ² + ε) + β  =  a·x + c
        a = γ / √(σ² + ε)
        c = β − a·μ
    Conv (không bias) ra x = Σ w·in. Thay vào:
        y = Σ (a·w)·in + c   →   w_mới = a·w ,  bias_mới = c

    Toàn bộ √ và phép chia được tính MỘT LẦN trên PC ở đây. Chip chỉ thấy
    một Conv bình thường — đúng định dạng export_params.py đang dùng.
    """
    folded = SmallCNN()
    he_so_a = {}
    with torch.no_grad():
        for i in (1, 2, 3):
            conv = getattr(model_bn, f"conv{i}")
            bn = getattr(model_bn, f"bn{i}")
            a = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            getattr(folded, f"conv{i}").weight.copy_(conv.weight * a[:, None, None, None])
            getattr(folded, f"conv{i}").bias.copy_(bn.bias - a * bn.running_mean)
            he_so_a[f"conv{i}"] = (a.min().item(), a.max().item())
        folded.fc.load_state_dict(model_bn.fc.state_dict())
    return folded.eval(), he_so_a


# ------------------------------------------------------------
# [3] Đo đạc
# ------------------------------------------------------------
def so_sanh_logit(m1, m2):
    """Chạy cả test set qua 2 model, trả về (lệch logit lớn nhất, số ảnh đoán khác nhau)."""
    m1.eval(); m2.eval()
    max_diff, khac = 0.0, 0
    with torch.no_grad():
        for data, _ in test_loader:
            z1, z2 = m1(data.to(device)), m2(data.to(device))
            max_diff = max(max_diff, (z1 - z2).abs().max().item())
            khac += (z1.argmax(1) != z2.argmax(1)).sum().item()
    return max_diff, khac


def thong_ke_trong_so(model):
    """max|w| từng lớp và số trọng số vượt trần Q1.7."""
    out = {}
    for name in ("conv1", "conv2", "conv3", "fc"):
        w = getattr(model, name).weight.detach()
        out[name] = {
            "max_abs": w.abs().max().item(),
            "vuot_Q17": int((w.abs() > Q17_MAX).sum().item()),
            "so_trong_so": w.numel(),
        }
    return out


def epoch_dau_tien_dat(history, nguong):
    for epoch, _, val in history:
        if val >= nguong:
            return epoch
    return None


def ve_hinh(h_lai, h_bn, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    for h, ten, mau in [(h_lai, "SmallCNN (không BN)", "#B0B7BF"), (h_bn, "SmallCNN + BN", "#1C7293")]:
        ep = [e for e, _, _ in h]
        ax1.plot(ep, [l for _, l, _ in h], marker="o", ms=3, color=mau, label=ten)
        ax2.plot(ep, [v for _, _, v in h], marker="o", ms=3, color=mau, label=ten)
    ax1.set_title("Loss train theo epoch"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax2.set_title("Val accuracy theo epoch"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val acc (%)")
    lo = min(v for h in (h_lai, h_bn) for _, _, v in h)
    ax2.set_ylim(max(lo - 0.3, 90), 100)
    for ax in (ax1, ax2):
        ax.grid(color="#E1E9EE"); ax.legend(frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Thí nghiệm BatchNorm — cùng seed, cùng dữ liệu, cùng optimizer", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS["SmallCNN"])
    args = ap.parse_args()
    t0 = time.time()

    # ---- Train 2 model, cùng điều kiện ----
    set_seed()
    m_lai, val_lai, ep_lai, h_lai = train_model(
        SmallCNN(), "SmallCNN (train lại, không BN)", "models/study_small_cnn_lai.pth", args.epochs)
    set_seed()
    m_bn, val_bn, ep_bn, h_bn = train_model(
        SmallCNN_BN(), "SmallCNN + BatchNorm", "models/study_small_cnn_bn.pth", args.epochs)

    # ---- Gộp BN và kiểm chứng ----
    m_fold, he_so_a = fold_bn(m_bn)
    diff, khac = so_sanh_logit(m_bn, m_fold)

    # ---- Test set: chạm 1 lần cho mỗi model, chỉ để báo cáo ----
    acc = {
        "lai": evaluate(m_lai, test_loader),
        "bn": evaluate(m_bn, test_loader),
        "fold": evaluate(m_fold, test_loader),
    }

    dem = lambda m: sum(p.numel() for p in m.parameters())
    tk_lai, tk_bn, tk_fold = thong_ke_trong_so(m_lai), thong_ke_trong_so(m_bn), thong_ke_trong_so(m_fold)

    # ---- In kết quả ----
    print("\n" + "=" * 78)
    print("KẾT QUẢ THÍ NGHIỆM BATCHNORM")
    print("=" * 78)
    print(f"{'':34}{'Không BN':>14}{'Có BN':>14}{'BN đã gộp':>14}")
    print("-" * 78)
    print(f"{'Tham số học được':34}{dem(m_lai):>14,}{dem(m_bn):>14,}{dem(m_fold):>14,}")
    print(f"{'Val acc tốt nhất':34}{val_lai:>13.2f}%{val_bn:>13.2f}%{'—':>14}")
    print(f"{'Epoch tốt nhất':34}{f'{ep_lai}/{args.epochs}':>14}{f'{ep_bn}/{args.epochs}':>14}{'—':>14}")
    for ng in (98.0, 98.5):
        a, b = epoch_dau_tien_dat(h_lai, ng), epoch_dau_tien_dat(h_bn, ng)
        print(f"{f'Epoch đầu tiên val ≥ {ng}%':34}{str(a or 'không đạt'):>14}{str(b or 'không đạt'):>14}{'—':>14}")
    print(f"{'Test acc':34}{acc['lai']:>13.2f}%{acc['bn']:>13.2f}%{acc['fold']:>13.2f}%")
    print("-" * 78)
    print(f"Gộp BN → Conv: lệch logit lớn nhất {diff:.2e}, số ảnh đoán khác nhau {khac}/10000")
    print("-" * 78)
    print(f"{'max|w| (trần Q1.7 = 0,9922)':28}{'Không BN':>14}{'BN trước gộp':>14}{'BN sau gộp':>14}{'  hệ số a (min…max)':>20}")
    for name in ("conv1", "conv2", "conv3", "fc"):
        l, b, f = tk_lai[name], tk_bn[name], tk_fold[name]
        a_txt = f"{he_so_a[name][0]:.2f}…{he_so_a[name][1]:.2f}" if name in he_so_a else "—"
        print(f"{name:28}{l['max_abs']:>9.4f} ({l['vuot_Q17']:>2}){b['max_abs']:>9.4f} ({b['vuot_Q17']:>2})"
              f"{f['max_abs']:>9.4f} ({f['vuot_Q17']:>2}){a_txt:>20}")
    print("  (số trong ngoặc = số trọng số vượt trần Q1.7)")
    print(f"\nThời gian chạy: {(time.time() - t0) / 60:.1f} phút")

    # ---- Lưu kết quả ----
    os.makedirs("figures", exist_ok=True)
    ve_hinh(h_lai, h_bn, "figures/thi_nghiem_batchnorm.png")
    with open("study/ket_qua_batchnorm.json", "w", encoding="utf-8") as f:
        json.dump({
            "epochs": args.epochs,
            "khong_bn": {"params": dem(m_lai), "best_val": val_lai, "best_epoch": ep_lai,
                         "test": acc["lai"], "history": h_lai, "trong_so": tk_lai},
            "co_bn": {"params": dem(m_bn), "best_val": val_bn, "best_epoch": ep_bn,
                      "test": acc["bn"], "history": h_bn, "trong_so_truoc_gop": tk_bn},
            "bn_da_gop": {"params": dem(m_fold), "test": acc["fold"], "lech_logit_max": diff,
                          "so_anh_khac": khac, "trong_so": tk_fold, "he_so_a": he_so_a},
        }, f, ensure_ascii=False, indent=1)
    print("Đã lưu: figures/thi_nghiem_batchnorm.png, study/ket_qua_batchnorm.json")
