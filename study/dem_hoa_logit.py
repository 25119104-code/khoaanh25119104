# ============================================================
# ĐẾM SỐ ẢNH CÓ HAI LOGIT BẰNG NHAU — câu 3 mục "Tự kiểm tra" của argmax.md
#
# Câu hỏi goc: doi '>' thanh '>=' trong argmax thi bao nhieu anh doi ket qua?
# Chi nhung anh co GIA TRI LON NHAT dat tai >= 2 lop moi doi.
#   '>'  giu chi so NHO NHAT   (giong torch.argmax)
#   '>=' giu chi so LON NHAT
#
# Script CHI DOC. Khong train, khong ghi de checkpoint, khong ghi file.
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
from xem_quantization import quantize_model   # dung lai dung ham cua bang 3


def argmax_lon_nhat(z):
    """Mo phong code C viet '>=': giu chi so LON NHAT trong cac cuc dai."""
    n = z.shape[1]
    idx_dao = torch.flip(z, dims=[1]).argmax(dim=1)   # argmax tren vector dao nguoc
    return (n - 1) - idx_dao


def do_mot_kieu(ten, lay_logit, lam_tron=None):
    hoa = 0          # so anh co >= 2 lop cung dat gia tri lon nhat
    doi_kq = 0       # so anh ma '>' va '>=' cho ket qua KHAC nhau
    gan_hoa = 0      # khoang cach giua hai logit cao nhat < 1e-6
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

# Ban int8 weight-only — DUNG DUNG ham cua xem_quantization.py (bang 3)
model_i8 = quantize_model(model, mode="deq", bits=8).to(device)
model_i8.eval()

print("=" * 84)
print("SO ANH CO HAI LOGIT BANG NHAU — 10.000 anh test")
print("=" * 84)
print(f"{'Kieu so':<34}{'Hoa':>8}{'Gan hoa':>12}{'Doi ket qua':>14}{'Khoang cach min':>16}")
print("-" * 84)

do_mot_kieu("A. float32 goc", lambda x: model(x))
do_mot_kieu("B. int8 weight-only (bang 3)", lambda x: model_i8(x))

# C. Mo phong dau ra 8-bit: ep chinh LOGIT ve luoi 256 muc.
#    Day KHONG phai pipeline int8 day du, chi la mo phong cho thay chuyen gi
#    xay ra khi logit tro thanh so nguyen thay vi so thuc.
def ve_luoi_8bit(z):
    scale = z.abs().amax(dim=1, keepdim=True) / 127.0
    return torch.round(z / scale) * scale

do_mot_kieu("C. ep LOGIT ve luoi 8-bit (mo phong)", lambda x: model(x), lam_tron=ve_luoi_8bit)

print("-" * 84)
print()
print("CACH DOC:")
print("  Hoa         = so anh co >= 2 lop cung dat gia tri lon nhat -> '>' va '>=' khac nhau")
print("  Gan hoa     = chua bang nhau nhung cach nhau < 1e-6, tuc chi can nhieu nho la thanh hoa")
print("  Doi ket qua = dem truc tiep, phai bang cot 'Hoa'")
print()
print("  Dong A va B gan giong nhau la DUNG, khong phai loi:")
print("  weight-only chi ep TRONG SO ve luoi roi giai luong tu ve float32,")
print("  nen LOGIT van la float32 day du do phan giai. Bang 3 dem trung tren")
print("  trong so, khong phai tren logit.")
print("  Dong C moi la luc chuong bo cau ap len logit.")
