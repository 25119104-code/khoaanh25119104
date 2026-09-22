# ============================================================
# In bảng kết quả cuối từ checkpoint đã có — KHÔNG train lại.
# Dùng khi model_comparison.py đã train xong nhưng crash ở bước in bảng.
#
# CHẠY:  python final_table.py
# ============================================================
#
# Lưu ý: file này import class model qua model_comparison.py (vì cần dùng
# cả val_loader và evaluate ở đó). Kiến trúc nay chỉ định nghĩa MỘT lần
# trong models.py; model_comparison.py import lại và xuất ra tên cũ nên
# dòng import phía dưới vẫn chạy y như trước.
# Ngoại lệ có chủ ý: mnist_digit_recognition.py (baseline tuần 1) giữ bản
# chép riêng của nó để đóng băng làm mốc so sánh.

import torch

from model_comparison import (
    DigitCNN,
    SmallCNN,
    evaluate,
    val_loader,
    test_loader,
    device,
)

import os
os.makedirs("models", exist_ok=True)   # noi chua .pth va .onnx
os.makedirs("figures", exist_ok=True)  # noi chua .png

# (class, tên hiển thị, file checkpoint, số epoch đã chạy)
RUNS = [
    (DigitCNN, "A. DigitCNN (bản gốc tuần 1)", "models/digit_cnn_val.pth", 8),
    (SmallCNN, "B. SmallCNN (kiến trúc gọn)",  "models/small_cnn.pth",     30),
]


def n_params_of(model):
    return sum(p.numel() for p in model.parameters())


print(f"\n{'=' * 78}")
print("KẾT QUẢ CUỐI — nạp từ checkpoint, test set chạm đúng 1 lần")
print(f"{'=' * 78}")
print(f"{'Kiến trúc':<32}{'Params':>10}{'KB':>8}{'Val acc':>10}{'Test acc':>10}{'Epoch':>8}")
print("-" * 78)

baseline_params = baseline_test = None

for cls, name, ckpt, n_epochs in RUNS:
    model = cls().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    n_params = n_params_of(model)
    val_acc = evaluate(model, val_loader)
    test_acc = evaluate(model, test_loader)

    if baseline_params is None:
        baseline_params, baseline_test = n_params, test_acc

    print(f"{name:<32}{n_params:>10,}{n_params * 4 / 1024:>7.1f}K"
          f"{val_acc:>9.2f}%{test_acc:>9.2f}%{n_epochs:>8}")

    if n_params != baseline_params:
        ratio = baseline_params / n_params
        delta = test_acc - baseline_test
        print(f"{'  -> so với bản gốc:':<32}{f'ít hơn {ratio:.1f}×':>10}"
              f"{'':>8}{'':>10}{f'{delta:+.2f}%':>10}")

print("-" * 78)
print("Ghi chú cho report: con số đáng nói không phải accuracy cao hơn,")
print("mà là ĐÁNH ĐỔI — giảm bao nhiêu lần params, mất bao nhiêu % accuracy.")
print(f"{'=' * 78}")
