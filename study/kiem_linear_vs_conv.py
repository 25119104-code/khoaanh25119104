# ============================================================
# KIỂM: Linear(144,10) có thật sự bằng Conv2d(16→10, k=3, p=0) không?
#
# Mục 12 của Word bước 4 khẳng định hai lớp này tương đương. Script này
# ĐO chứ không lập luận: dựng lớp conv từ chính trọng số fc đã train,
# chạy cả hai trên 10.000 ảnh test, so từng phần tử logit.
#
# Script CHỈ ĐỌC. Không train, không ghi đè checkpoint, không ghi file.
#
# CHẠY:  conda activate mnist && python study/kiem_linear_vs_conv.py
#        (script tự trỏ về thư mục gốc project nên đứng ở đâu chạy cũng được)
# ============================================================

import os
import sys

import torch
import torch.nn as nn

# Python đặt THƯ MỤC CHỨA SCRIPT (study/) vào sys.path, không phải thư mục
# đang đứng. Phải tự thêm thư mục gốc project vào thì mới import được.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)   # model_comparison dung duong dan tuong doi: ./data, models/

from model_comparison import SmallCNN, test_loader, device


# ------------------------------------------------------------
# [1] Nạp model đã train
# ------------------------------------------------------------
model = SmallCNN().to(device)
model.load_state_dict(torch.load("models/small_cnn.pth", map_location=device))
model.eval()

fc = model.fc                       # Linear(144, 10)
OUT, IN = fc.weight.shape           # [10, 144]
C, H, W = 16, 3, 3                  # feature map ngay truoc flatten
assert IN == C * H * W, f"{IN} != {C}*{H}*{W}"


# ------------------------------------------------------------
# [2] Dựng lớp conv tương đương — KHÔNG train lại một epoch nào
# ------------------------------------------------------------
# fc.weight  co shape [10, 144]
# conv.weight can shape [10, 16, 3, 3]
# 144 = 16*3*3, va flatten danh chi so theo NCHW: i = c*(H*W) + h*W + w
# -> dung bang thu tu chieu cua conv.weight, nen .view() la du.
conv = nn.Conv2d(C, OUT, kernel_size=H, padding=0, bias=True).to(device)
with torch.no_grad():
    conv.weight.copy_(fc.weight.view(OUT, C, H, W))
    conv.bias.copy_(fc.bias)
conv.eval()


def dem(module):
    return sum(p.numel() for p in module.parameters())


print("=" * 62)
print("SO THAM SO")
print("=" * 62)
print(f"  Linear({IN},{OUT})                : {dem(fc):,} params")
print(f"  Conv2d({C}->{OUT}, k={H}, p=0)      : {dem(conv):,} params")
print(f"  Cong thuc Linear : {IN} x {OUT} + {OUT} = {IN * OUT + OUT:,}")
print(f"  Cong thuc Conv2d : {H} x {W} x {C} x {OUT} + {OUT} = {H * W * C * OUT + OUT:,}")
print(f"  => Bang nhau: {dem(fc) == dem(conv)}")


# ------------------------------------------------------------
# [3] Lấy feature map ngay trước flatten
# ------------------------------------------------------------
def feature_map(x):
    """Chay lai dung forward cua SmallCNN, dung ngay truoc flatten."""
    x = model.pool(model.relu(model.conv1(x)))   # 28 -> 14
    x = model.pool(model.relu(model.conv2(x)))   # 14 -> 7
    x = model.pool(model.relu(model.conv3(x)))   # 7  -> 3
    return x                                      # [B, 16, 3, 3]


# ------------------------------------------------------------
# [4] Chạy cả hai trên toàn bộ test set, so từng phần tử
# ------------------------------------------------------------
max_diff = 0.0
tong_diff = 0.0
so_phan_tu = 0
lech_du_doan = 0
so_anh = 0
shape_conv = None
max_logit = 0.0          # do lon logit -> de biet 2e-4 la lon hay binh thuong
vuot_nguong = 0          # so phan tu lech qua 1e-4

with torch.no_grad():
    for data, _ in test_loader:
        data = data.to(device)
        feat = feature_map(data)

        z_linear = fc(feat.flatten(1))            # [B, 10]
        out_conv = conv(feat)                     # [B, 10, 1, 1]
        if shape_conv is None:
            shape_conv = tuple(out_conv.shape[1:])
        z_conv = out_conv.flatten(1)              # [B, 10]

        d = (z_linear - z_conv).abs()
        max_diff = max(max_diff, d.max().item())
        tong_diff += d.sum().item()
        so_phan_tu += d.numel()

        max_logit = max(max_logit, z_linear.abs().max().item())
        vuot_nguong += (d > 1e-4).sum().item()

        lech_du_doan += (z_linear.argmax(1) != z_conv.argmax(1)).sum().item()
        so_anh += data.size(0)

print()
print("=" * 62)
print(f"SO SANH LOGIT TREN {so_anh:,} ANH TEST")
print("=" * 62)
print(f"  Shape dau ra Linear : (10,)")
print(f"  Shape dau ra Conv2d : {shape_conv}   (flatten lai thanh (10,))")
print(f"  So phan tu da so    : {so_phan_tu:,}")
print(f"  Sai lech LON NHAT   : {max_diff:.3e}")
print(f"  Sai lech trung binh : {tong_diff / so_phan_tu:.3e}")
print(f"  Logit lon nhat      : {max_logit:.3f}")
print(f"  Sai lech TUONG DOI  : {max_diff / max_logit:.3e}  (= lech lon nhat / logit lon nhat)")
print(f"  Float32 eps         : {torch.finfo(torch.float32).eps:.3e}")
print(f"  Tuong duong         : {max_diff / max_logit / torch.finfo(torch.float32).eps:.0f} lan eps")
print(f"  So phan tu lech >1e-4: {vuot_nguong:,} / {so_phan_tu:,}"
      f"  ({100 * vuot_nguong / so_phan_tu:.3f}%)")
print(f"  So anh doan KHAC nhau: {lech_du_doan} / {so_anh}")

print()
if lech_du_doan == 0:
    print("  => Cung du doan tren toan bo test set.")
else:
    print(f"  => CO {lech_du_doan} anh doan khac nhau — xem lai thu tu chieu trong .view()")

if max_diff == 0.0:
    print("  => Giong nhau TUNG BIT tren may nay.")
else:
    print(f"  => Khong bit-exact: lech toi da {max_diff:.3e}.")
    print("     Do la sai so lam tron float32 vi thu tu cong don khac nhau")
    print("     (Linear chay GEMM, Conv2d chay im2col), khong phai loi cong thuc.")
    print(f"     Nguong verify TUYET DOI 1e-4 -> {'DAT' if max_diff < 1e-4 else 'KHONG DAT'}.")
    print()
    print("  BAI HOC CHO PHASE 2D (quan trong hon ca muc 12):")
    print("     PyTorch chay CUNG mot phep toan bang hai cach da lech nhu tren.")
    print("     Nen 'khop PyTorch trong 1e-4 tuyet doi' la muc tieu KHONG dat duoc,")
    print("     ke ca khi golden model C viet dung 100%. Tieu chi nen dung:")
    print("       1. So anh du doan khac nhau tren 10.000 anh test (o day: %d)" % lech_du_doan)
    print("       2. Sai so TUONG DOI, khong phai tuyet doi")
    print("     Sai lech logit khong doi du doan thi khong phai bug.")

print()
print("=" * 62)
print("KET LUAN: hai lop dung CHUNG mot mang so, chi khac cach danh chi so.")
print("Doi qua lai chi can .view(), khong can train lai.")
print("Van giu Linear trong project: cung 1.450 params va 1.440 MAC,")
print("nhung golden model C chi phai viet 2 vong lap thay vi 6.")
print("=" * 62)
