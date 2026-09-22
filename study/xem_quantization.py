# ============================================================
# XEM QUANTIZATION LÀM GÌ VỚI TRỌNG SỐ THẬT CỦA MÌNH
#
# Script này CHỈ ĐỌC. Không train, không ghi đè checkpoint, không đụng
# tới pipeline. Mục đích là nhìn thấy con số thật thay vì đọc lý thuyết.
#
# CHẠY:  conda activate mnist && python study/xem_quantization.py
#        (script tự trỏ về thư mục gốc project nên đứng ở đâu chạy cũng được)
# ============================================================

import copy
import os
import sys

import torch

# Python đặt THƯ MỤC CHỨA SCRIPT (study/) vào sys.path, không phải thư mục
# đang đứng. Nên phải tự thêm thư mục gốc project vào thì mới import được
# model_comparison.py nằm ở đó.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)   # model_comparison dung duong dan tuong doi: ./data, models/

from model_comparison import SmallCNN, evaluate, test_loader, device


# ------------------------------------------------------------
# Hai kiểu lượng tử hoá
# ------------------------------------------------------------
def quant_deq(w, bits):
    """Quantization đối xứng: scale là số thực, chọn sao cho vừa khít khoảng giá trị.

    Đây là kiểu PyTorch/ONNX dùng. Trả về (trọng số đã làm tròn về lưới, scale).
    """
    qmax = 2 ** (bits - 1) - 1              # int8 -> 127, int16 -> 32767
    scale = w.abs().max().item() / qmax
    if scale == 0:
        return w.clone(), 0.0
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    return q * scale, scale


def quant_deq_fixed(w, frac_bits=7):
    """Fixed-point Qm.n: scale BẮT BUỘC là luỹ thừa 2.

    Q1.7 = 1 bit nguyên + 7 bit thập phân -> scale = 1/128.
    Nhân/chia scale trên phần cứng chỉ là dịch bit, gần như miễn phí.
    Đổi lại: không khít khoảng giá trị bằng cách trên.
    """
    scale = 1.0 / (2 ** frac_bits)
    q = torch.clamp(torch.round(w / scale), -128, 127)
    return q * scale, scale


def quantize_model(model, mode, bits=8):
    """Trả về bản sao của model với MỌI trọng số đã đi qua lưới lượng tử hoá."""
    m = copy.deepcopy(model)
    with torch.no_grad():
        for p in m.parameters():
            if mode == "fixed":
                p.copy_(quant_deq_fixed(p.data)[0])
            else:
                p.copy_(quant_deq(p.data, bits)[0])
    return m


# ------------------------------------------------------------
if __name__ == "__main__":
    model = SmallCNN().to(device)
    model.load_state_dict(torch.load(os.path.join(ROOT, "models/small_cnn.pth"), map_location=device))
    model.eval()

    # ========== 1. Trọng số thật trải trong khoảng nào ==========
    print("\n" + "=" * 78)
    print("1. KHOẢNG GIÁ TRỊ THẬT CỦA TỪNG LỚP")
    print("=" * 78)
    print(f"{'Lớp':<10}{'Số ts':>8}{'nhỏ nhất':>11}{'lớn nhất':>11}"
          f"{'scale int8':>13}{'bước nhảy':>13}")
    print("-" * 78)

    layers = [(n, p) for n, p in model.named_parameters() if p.dim() > 1]
    for name, p in layers:
        _, sc = quant_deq(p.data, 8)
        print(f"{name.replace('.weight',''):<10}{p.numel():>8,}"
              f"{p.min().item():>11.4f}{p.max().item():>11.4f}"
              f"{sc:>13.6f}{sc:>13.6f}")
    print("-" * 78)
    print("Bước nhảy = khoảng cách giữa 2 giá trị liền nhau mà int8 biểu diễn được.")
    print("Hai trọng số chênh nhau ÍT HƠN bước nhảy sẽ bị gộp thành cùng một số.")

    # ========== 2. Năm trọng số cụ thể, xem chúng biến thành gì ==========
    print("\n" + "=" * 78)
    print("2. NĂM TRỌNG SỐ ĐẦU TIÊN CỦA conv1 — trước và sau")
    print("=" * 78)

    w = model.conv1.weight.data.flatten()[:5]
    qmax = 127
    scale = model.conv1.weight.data.abs().max().item() / qmax
    print(f"scale của lớp này = {scale:.6f}   (lưu 1 lần cho cả lớp)\n")
    print(f"{'gốc (float32)':>16}{'chia scale':>14}{'làm tròn q':>13}"
          f"{'đọc lại':>13}{'sai lệch':>13}")
    print("-" * 78)
    for v in w:
        v = v.item()
        raw = v / scale
        q = max(-qmax, min(qmax, round(raw)))
        back = q * scale
        print(f"{v:>16.6f}{raw:>14.2f}{q:>13d}{back:>13.6f}{back - v:>+13.6f}")
    print("-" * 78)
    print("Cột 'làm tròn q' là thứ DUY NHẤT được lưu vào file — đúng 1 byte mỗi số.")

    # ========== 3. Bao nhiêu giá trị bị gộp lại ==========
    print("\n" + "=" * 78)
    print("3. ĐỘ PHÂN GIẢI: bao nhiêu giá trị khác nhau còn sót lại")
    print("=" * 78)
    print(f"{'Lớp':<10}{'float32':>14}{'int8':>10}{'int16':>10}{'Q1.7':>10}")
    print("-" * 78)
    for name, p in layers:
        n_f32 = len(torch.unique(p.data))
        n_i8 = len(torch.unique(quant_deq(p.data, 8)[0]))
        n_i16 = len(torch.unique(quant_deq(p.data, 16)[0]))
        n_fx = len(torch.unique(quant_deq_fixed(p.data)[0]))
        print(f"{name.replace('.weight',''):<10}{n_f32:>14,}{n_i8:>10,}"
              f"{n_i16:>10,}{n_fx:>10,}")
    print("-" * 78)
    print("float32 gần như mỗi trọng số một giá trị riêng. int8 tối đa 255 giá trị.")
    print("Q1.7 còn ít hơn vì scale bị ép là 1/128, không khít khoảng giá trị thật.")

    # ========== 4. Sai số ==========
    print("\n" + "=" * 78)
    print("4. SAI SỐ TRÊN TRỌNG SỐ")
    print("=" * 78)
    print(f"{'Lớp':<10}{'int8 t.bình':>14}{'int8 lớn nhất':>16}"
          f"{'int16 lớn nhất':>17}{'Q1.7 lớn nhất':>16}")
    print("-" * 78)
    for name, p in layers:
        e8 = (quant_deq(p.data, 8)[0] - p.data).abs()
        e16 = (quant_deq(p.data, 16)[0] - p.data).abs()
        ef = (quant_deq_fixed(p.data)[0] - p.data).abs()
        print(f"{name.replace('.weight',''):<10}{e8.mean():>14.6f}{e8.max():>16.6f}"
              f"{e16.max():>17.8f}{ef.max():>16.6f}")
    print("-" * 78)

    # ========== 5. Accuracy thật ==========
    print("\n" + "=" * 78)
    print("5. ACCURACY TRÊN TEST SET — cái giá thật sự")
    print("=" * 78)

    base = evaluate(model, test_loader)
    rows = [("float32 (gốc)", base, 4)]
    for label, mode, bits, byt in [
        ("int16", "sym", 16, 2),
        ("int8", "sym", 8, 1),
        ("Q1.7 (fixed-point)", "fixed", 8, 1),
    ]:
        acc = evaluate(quantize_model(model, mode, bits), test_loader)
        rows.append((label, acc, byt))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{'Kiểu số':<22}{'Accuracy':>11}{'Chênh':>10}{'Bộ nhớ':>12}{'So gốc':>10}")
    print("-" * 78)
    for label, acc, byt in rows:
        kb = n_params * byt / 1024
        print(f"{label:<22}{acc:>10.2f}%{acc - base:>+9.2f}%"
              f"{kb:>11.1f}K{n_params*byt/(n_params*4):>9.2f}×")
    print("-" * 78)

    print("""
QUAN TRỌNG khi báo cáo — đừng nói quá:
  Script này chỉ lượng tử hoá TRỌNG SỐ, còn activation giữa các lớp vẫn là
  float32. Inference int8 thật trên phần cứng lượng tử hoá cả hai, và phải
  "calibrate" khoảng giá trị của activation bằng một ít dữ liệu. Con số ở
  bảng trên vì vậy LẠC QUAN hơn thực tế. Nó cho biết trọng số chịu được
  int8 tới đâu, không phải accuracy cuối cùng khi chạy trên chip.

Điểm chính không nằm ở bộ nhớ:
  4,9 KB so với 19,6 KB là phần nổi. Phần chìm là mỗi phép nhân float32
  trên FPGA cần cả một khối mạch (tách mũ, nhân định trị, chuẩn hoá), còn
  nhân int8 chỉ là 1 DSP slice. Cùng một con chip chạy được gấp nhiều lần
  phép tính — đó mới là lý do Thầy nói "tối ưu trọng số KHI ĐƯA VÀO CHIP".
""")
