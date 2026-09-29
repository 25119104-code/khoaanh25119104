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
os.chdir(ROOT)   # model_comparison dùng đường dẫn tương đối: ./data, models/

from model_comparison import SmallCNN, test_loader, device


# ------------------------------------------------------------
# [1] Nạp model đã train
# ------------------------------------------------------------
model = SmallCNN().to(device)
model.load_state_dict(torch.load("models/small_cnn.pth", map_location=device))
model.eval()

fc = model.fc                       # Linear(144, 10)
OUT, IN = fc.weight.shape           # [10, 144]
C, H, W = 16, 3, 3                  # feature map ngay trước flatten
assert IN == C * H * W, f"{IN} != {C}*{H}*{W}"


# ------------------------------------------------------------
# [2] Dựng lớp conv tương đương — KHÔNG train lại một epoch nào
# ------------------------------------------------------------
# fc.weight  có shape [10, 144]
# conv.weight cần shape [10, 16, 3, 3]
# 144 = 16*3*3, và flatten đánh chỉ số theo NCHW: i = c*(H*W) + h*W + w
# -> đúng bằng thứ tự chiều của conv.weight, nên .view() là đủ.
conv = nn.Conv2d(C, OUT, kernel_size=H, padding=0, bias=True).to(device)
with torch.no_grad():
    conv.weight.copy_(fc.weight.view(OUT, C, H, W))
    conv.bias.copy_(fc.bias)
conv.eval()


def dem(module):
    return sum(p.numel() for p in module.parameters())


print("=" * 62)
print("SỐ THAM SỐ")
print("=" * 62)
print(f"  Linear({IN},{OUT})                : {dem(fc):,} params")
print(f"  Conv2d({C}->{OUT}, k={H}, p=0)      : {dem(conv):,} params")
print(f"  Công thức Linear : {IN} x {OUT} + {OUT} = {IN * OUT + OUT:,}")
print(f"  Công thức Conv2d : {H} x {W} x {C} x {OUT} + {OUT} = {H * W * C * OUT + OUT:,}")
print(f"  => Bằng nhau: {dem(fc) == dem(conv)}")


# ------------------------------------------------------------
# [3] Lấy feature map ngay trước flatten
# ------------------------------------------------------------
def feature_map(x):
    """Chạy lại đúng forward của SmallCNN, dừng ngay trước flatten."""
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
max_logit = 0.0          # độ lớn logit -> để biết 2e-4 là lớn hay bình thường
vuot_nguong = 0          # số phần tử lệch quá 1e-4

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
print(f"SO SÁNH LOGIT TRÊN {so_anh:,} ẢNH TEST")
print("=" * 62)
print(f"  Shape đầu ra Linear : (10,)")
print(f"  Shape đầu ra Conv2d : {shape_conv}   (flatten lại thành (10,))")
print(f"  Số phần tử đã so    : {so_phan_tu:,}")
print(f"  Sai lệch LỚN NHẤT   : {max_diff:.3e}")
print(f"  Sai lệch trung bình : {tong_diff / so_phan_tu:.3e}")
print(f"  Logit lớn nhất      : {max_logit:.3f}")
print(f"  Sai lệch TƯƠNG ĐỐI  : {max_diff / max_logit:.3e}  (= lệch lớn nhất / logit lớn nhất)")
print(f"  Float32 eps         : {torch.finfo(torch.float32).eps:.3e}")
print(f"  Tương đương         : {max_diff / max_logit / torch.finfo(torch.float32).eps:.0f} lần eps")
print(f"  Số phần tử lệch >1e-4: {vuot_nguong:,} / {so_phan_tu:,}"
      f"  ({100 * vuot_nguong / so_phan_tu:.3f}%)")
print(f"  Số ảnh đoán KHÁC nhau: {lech_du_doan} / {so_anh}")

print()
if lech_du_doan == 0:
    print("  => Cùng dự đoán trên toàn bộ test set.")
else:
    print(f"  => CÓ {lech_du_doan} ảnh đoán khác nhau — xem lại thứ tự chiều trong .view()")

if max_diff == 0.0:
    print("  => Giống nhau TỪNG BIT trên máy này.")
else:
    print(f"  => Không bit-exact: lệch tối đa {max_diff:.3e}.")
    print("     Đó là sai số làm tròn float32 vì thứ tự cộng dồn khác nhau")
    print("     (Linear chạy GEMM, Conv2d chạy im2col), không phải lỗi công thức.")
    print(f"     Ngưỡng verify TUYỆT ĐỐI 1e-4 -> {'ĐẠT' if max_diff < 1e-4 else 'KHÔNG ĐẠT'}.")
    print()
    print("  BÀI HỌC CHO PHASE 2D (quan trọng hơn cả mục 12):")
    print("     PyTorch chạy CÙNG một phép toán bằng hai cách đã lệch như trên.")
    print("     Nên 'khớp PyTorch trong 1e-4 tuyệt đối' là mục tiêu KHÔNG đạt được,")
    print("     kể cả khi golden model C viết đúng 100%. Tiêu chí nên dùng:")
    print("       1. Số ảnh dự đoán khác nhau trên 10.000 ảnh test (ở đây: %d)" % lech_du_doan)
    print("       2. Sai số TƯƠNG ĐỐI, không phải tuyệt đối")
    print("     Sai lệch logit không đổi dự đoán thì không phải bug.")

print()
print("=" * 62)
print("KẾT LUẬN: hai lớp dùng CHUNG một mảng số, chỉ khác cách đánh chỉ số.")
print("Đổi qua lại chỉ cần .view(), không cần train lại.")
print("Vẫn giữ Linear trong project: cùng 1.450 params và 1.440 MAC,")
print("nhưng golden model C chỉ phải viết 2 vòng lặp thay vì 6.")
print("=" * 62)
