# ============================================================
# ĐẾM SỐ ẢNH CÓ HAI LOGIT BẰNG NHAU — câu 3 mục "Tự kiểm tra" của argmax.md
#
# Câu hỏi gốc: đổi '>' thành '>=' trong argmax thì bao nhiêu ảnh đổi kết quả?
# Chỉ những ảnh có GIÁ TRỊ LỚN NHẤT đạt tại >= 2 lớp mới đổi.
#   '>'  giữ chỉ số NHỎ NHẤT   (giống torch.argmax)
#   '>=' giữ chỉ số LỚN NHẤT
#
# Script CHỈ ĐỌC. Không train, không ghi đè checkpoint, không ghi file.
#
# CHẠY:  conda activate mnist && python study/dem_hoa_logit.py
# ============================================================

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from model_comparison import SmallCNN, test_loader, device
from xem_quantization import quantize_model   # dùng lại đúng hàm của bảng 3


def argmax_lon_nhat(z):
    """Mô phỏng code C viết '>=': giữ chỉ số LỚN NHẤT trong các cực đại."""
    n = z.shape[1]
    idx_dao = torch.flip(z, dims=[1]).argmax(dim=1)   # argmax trên vector đảo ngược
    return (n - 1) - idx_dao


def do_mot_kieu(ten, lay_logit, lam_tron=None):
    hoa = 0          # số ảnh có >= 2 lớp cùng đạt giá trị lớn nhất
    doi_kq = 0       # số ảnh mà '>' và '>=' cho kết quả KHÁC nhau
    gan_hoa = 0      # khoảng cách giữa hai logit cao nhất < 1e-6
    tong = 0
    khoang_cach_min = float("inf")

    with torch.no_grad():
        for data, _ in test_loader:
            z = lay_logit(data.to(device))
            if lam_tron is not None:
                z = lam_tron(z)

            top2 = z.topk(2, dim=1).values
            khoang = top2[:, 0] - top2[:, 1]

            hoa += (khoang == 0).sum().item()
            gan_hoa += ((khoang > 0) & (khoang < 1e-6)).sum().item()
            khoang_cach_min = min(khoang_cach_min, khoang.min().item())

            doi_kq += (z.argmax(dim=1) != argmax_lon_nhat(z)).sum().item()
            tong += z.size(0)

    print(f"{ten:<34}{hoa:>8}{gan_hoa:>12}{doi_kq:>14}{khoang_cach_min:>16.3e}")
    return hoa, doi_kq


model = SmallCNN().to(device)
model.load_state_dict(torch.load("models/small_cnn.pth", map_location=device))
model.eval()

# Bản int8 weight-only — DÙNG ĐÚNG hàm của xem_quantization.py (bảng 3)
model_i8 = quantize_model(model, mode="deq", bits=8).to(device)
model_i8.eval()

print("=" * 84)
print("SỐ ẢNH CÓ HAI LOGIT BẰNG NHAU — 10.000 ảnh test")
print("=" * 84)
print(f"{'Kiểu số':<34}{'Hoà':>8}{'Gần hoà':>12}{'Đổi kết quả':>14}{'Khoảng cách min':>16}")
print("-" * 84)

do_mot_kieu("A. float32 gốc", lambda x: model(x))
do_mot_kieu("B. int8 weight-only (bảng 3)", lambda x: model_i8(x))

# C. Mô phỏng đầu ra 8-bit: ép chính LOGIT về lưới 256 mức.
#    Đây KHÔNG phải pipeline int8 đầy đủ, chỉ là mô phỏng cho thấy chuyện gì
#    xảy ra khi logit trở thành số nguyên thay vì số thực.
def ve_luoi_8bit(z):
    scale = z.abs().amax(dim=1, keepdim=True) / 127.0
    return torch.round(z / scale) * scale

do_mot_kieu("C. ép LOGIT về lưới 8-bit (mô phỏng)", lambda x: model(x), lam_tron=ve_luoi_8bit)

print("-" * 84)
print()
print("CÁCH ĐỌC:")
print("  Hoà         = số ảnh có >= 2 lớp cùng đạt giá trị lớn nhất -> '>' và '>=' khác nhau")
print("  Gần hoà     = chưa bằng nhau nhưng cách nhau < 1e-6, tức chỉ cần nhiễu nhỏ là thành hoà")
print("  Đổi kết quả = đếm trực tiếp, phải bằng cột 'Hoà'")
print()
print("  Dòng A và B gần giống nhau là ĐÚNG, không phải lỗi:")
print("  weight-only chỉ ép TRỌNG SỐ về lưới rồi giải lượng tử về float32,")
print("  nên LOGIT vẫn là float32 đầy đủ độ phân giải. Bảng 3 đếm trùng trên")
print("  trọng số, không phải trên logit.")
print("  Dòng C mới là lúc chuồng bồ câu áp lên logit.")
