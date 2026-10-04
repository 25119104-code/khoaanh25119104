# ============================================================
# kiem_tra_tinh_tay.py — Đáp án cho bài tính tay trong docs/on-tap-flow-7-buoc.md
#
# CÁCH DÙNG: làm bài trên giấy TRƯỚC, rồi mới chạy để so.
#   python study/kiem_tra_tinh_tay.py          # in đáp án tất cả các bài
#   python study/kiem_tra_tinh_tay.py 3        # chỉ bài 3
#
# Bài 2, 3 dùng SỐ THẬT của SmallCNN (đọc từ params/ và golden/), nên cần
# chạy export_params.py trước. Chỉ dùng numpy.
# ============================================================

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def tieu_de(n, ten):
    print(f"\n{'=' * 64}\nBÀI {n} — {ten}\n{'=' * 64}")


# ------------------------------------------------------------
def bai_1():
    tieu_de(1, "Backprop 1 neuron, 1 bước cập nhật (bước 1 của flow)")
    x = np.array([1.0, 2.0]); w = np.array([0.5, -0.3]); b = 0.1; y = 1.0; lr = 0.1
    z = w @ x + b
    loss = 0.5 * (z - y) ** 2
    print(f"Forward : ŷ = w1·x1 + w2·x2 + b = {w[0]}·{x[0]} + ({w[1]})·{x[1]} + {b} = {z:.4f}")
    print(f"Loss    : L = ½(ŷ − y)² = ½({z:.4f} − {y})² = {loss:.4f}")
    g = z - y
    dw, db = g * x, g
    print(f"Backprop: ∂L/∂ŷ = ŷ − y = {g:.4f}")
    print(f"          ∂L/∂w1 = (ŷ − y)·x1 = {dw[0]:.4f} ; ∂L/∂w2 = (ŷ − y)·x2 = {dw[1]:.4f} ; ∂L/∂b = {db:.4f}")
    w2, b2 = w - lr * dw, b - lr * db
    print(f"Cập nhật: w1 = {w[0]} − {lr}·({dw[0]:.1f}) = {w2[0]:.4f}")
    print(f"          w2 = {w[1]} − {lr}·({dw[1]:.1f}) = {w2[1]:.4f}")
    print(f"          b  = {b} − {lr}·({db:.1f}) = {b2:.4f}")
    z2 = w2 @ x + b2
    print(f"Kiểm tra: ŷ mới = {z2:.4f}, loss mới = {0.5 * (z2 - y) ** 2:.4f}  (giảm từ {loss:.4f})")


# ------------------------------------------------------------
def _so_that():
    W = np.loadtxt("params/weights.txt"); B = np.loadtxt("params/biases.txt")
    x = np.loadtxt("golden/input.txt").reshape(28, 28)
    c1 = np.loadtxt("golden/01_conv1.txt").reshape(8, 28, 28)
    return W, B, x, c1


def bai_2():
    tieu_de(2, "Tính 1 điểm conv1: output[0][26][12] (bước 3)")
    W, B, x, c1 = _so_that()
    p = np.round(x[25:28, 11:14], 4); k = np.round(W[0:9].reshape(3, 3), 4); b = round(B[0], 4)
    tich = p * k
    for i in range(3):
        print("  " + "   ".join(f"{p[i, j]:+.4f}×{k[i, j]:+.4f} = {tich[i, j]:+.6f}" for j in range(3)))
    print(f"Σ 9 tích = {tich.sum():+.6f}")
    print(f"+ bias   = {b:+.4f}")
    print(f"= {tich.sum() + b:+.6f}   (số làm tròn 4 chữ số)")
    print(f"PyTorch  = {c1[0, 26, 12]:+.6f}   (số đủ độ chính xác — lệch vì làm tròn đầu vào)")


def bai_3():
    tieu_de(3, "Điểm ở góc conv1: output[0][0][0] — vùng đệm (bước 3–4)")
    W, B, x, c1 = _so_that()
    pp = np.pad(x, 1)[0:3, 0:3]; k = W[0:9].reshape(3, 3)
    print("Cửa sổ trên ảnh đã đệm (0 = đệm, −0,4242 = nền ảnh):")
    print(np.round(pp, 4))
    tich = pp * k
    print(f"Số tap rơi vào vùng đệm: {(pp == 0).sum()}  → 9 phép nhân, {(pp == 0).sum()} phép nhân với 0")
    print(f"Σ = {tich.sum():+.6f} ; + bias {B[0]:+.6f} ; = {tich.sum() + B[0]:+.6f}  (PyTorch {c1[0, 0, 0]:+.6f})")
    sai = np.full((3, 3), -0.424212962) * k
    print(f"Nếu đệm SAI bằng màu nền −0,4242: = {sai.sum() + B[0]:+.6f} → lệch {sai.sum() - tich.sum():+.4f}")


def bai_4():
    tieu_de(4, "MaxPool, ReLU và đổi thứ tự relu ↔ pool (bước 3–5)")
    blk = np.array([[-1.2, 0.7], [0.3, -0.5]])
    print(f"Khối 2x2 = {blk.tolist()}")
    print(f"maxpool(relu(x)) = max(relu) = {np.maximum(blk, 0).max()}")
    print(f"relu(maxpool(x)) = relu(max) = {max(blk.max(), 0)}")
    blk2 = np.array([[-1.2, -0.7], [-0.3, -0.5]])
    print(f"Khối toàn âm {blk2.tolist()}: maxpool(relu) = {np.maximum(blk2, 0).max()} ; relu(maxpool) = {max(blk2.max(), 0)}")
    print("Đếm ReLU: trước pool = 8·28² + 16·14² + 16·7² =", 8 * 784 + 16 * 196 + 16 * 49,
          "; sau pool = 8·14² + 16·7² + 16·3² =", 8 * 196 + 16 * 49 + 16 * 9)
    print("Vì sao AvgPool KHÔNG đổi được: khối [−2, 2, 0, 0]:",
          "avg(relu) =", np.mean(np.maximum([-2, 2, 0, 0], 0)), "; relu(avg) =", max(np.mean([-2, 2, 0, 0]), 0))


def bai_5():
    tieu_de(5, "Chỉ số flatten — bẫy NCHW / NHWC (bước 3, 6)")
    c, y, xx = 5, 2, 1
    print(f"Phần tử pool3[c={c}][y={y}][x={xx}], tensor 16×3×3:")
    print(f"  NCHW (đúng, PyTorch): c·9 + y·3 + x = {c * 9 + y * 3 + xx}")
    print(f"  NHWC (sai):           (y·3 + x)·16 + c = {(y * 3 + xx) * 16 + c}")


def bai_6():
    tieu_de(6, "Đổi trọng số sang fixed-point Q1.7 (bước 4)")
    W = np.loadtxt("params/weights.txt")
    for v in [W[0], -0.7234, 1.03]:
        q = int(np.clip(np.round(v * 128), -128, 127))
        print(f"  w = {v:+.6f} → ×128 = {v * 128:+9.3f} → làm tròn/kẹp q = {q:+4d} → đọc lại q/128 = {q / 128:+.7f}"
              f" ; sai số {q / 128 - v:+.6f}")
    print("  Trần Q1.7 = 127/128 = 0.9921875 ; sàn = −128/128 = −1. Bước nhảy = 1/128 = 0.0078125")


def bai_7():
    tieu_de(7, "Bỏ phép chia: chuẩn hoá = nhân + cộng; chia cho 2^n = dịch bit (bước 4)")
    a = 1 / (255 * 0.3081); b = -0.1307 / 0.3081
    for px in (0, 128, 255):
        print(f"  pixel {px:3d}: (px/255 − 0,1307)/0,3081 = {(px / 255 - 0.1307) / 0.3081:+.6f} ;"
              f" a·px + b = {a * px + b:+.6f}")
    print(f"  a = 1/(255·0,3081) = {a:.8f} ; b = −0,1307/0,3081 = {b:.8f}  (tính sẵn 1 lần trên PC)")
    print(f"  Số nguyên: 200 / 8 = {200 // 8} ; 200 >> 3 = {200 >> 3}  (dịch phải 3 bit = chia 2³)")


def bai_8():
    tieu_de(8, "Đếm phép tính 1 ảnh (bước 2–3)")
    conv = [(1, 8, 28), (8, 16, 14), (16, 16, 7)]
    tong_mac = 0
    for ic, oc, h in conv:
        m = 9 * ic * oc * h * h
        tong_mac += m
        print(f"  conv {ic}→{oc}, {h}×{h}: 9·{ic}·{oc}·{h}² = {m:,}")
    relu = 8 * 784 + 16 * 196 + 16 * 49
    pool = 3 * (8 * 196 + 16 * 49 + 16 * 9)
    fc = 144 * 10
    print(f"  Tổng MAC conv = {tong_mac:,}")
    print(f"  ReLU = {relu:,} ; MaxPool = 3 phép so sánh × {8 * 196 + 16 * 49 + 16 * 9} ô = {pool:,} ;"
          f" FC = {fc:,} MAC ; argmax = 9 so sánh")
    print(f"  TỔNG = {tong_mac + relu + pool + fc + 9:,} phép / ảnh")


def bai_9():
    tieu_de(9, "Softmax vs argmax (bước 4–5)")
    z = np.array([2.0, 1.0, 0.1])
    e = np.exp(z); p = e / e.sum()
    print(f"  logit z = {z.tolist()}")
    print(f"  e^z = {np.round(e, 4).tolist()} ; tổng = {e.sum():.4f}")
    print(f"  softmax = {np.round(p, 4).tolist()} (tổng = 1)")
    print(f"  argmax(z) = {z.argmax()} ; argmax(softmax) = {p.argmax()} → giống nhau, bỏ được 3 phép exp và các phép chia")


def bai_10():
    tieu_de(10, "Gộp BatchNorm vào Conv (bước 5)")
    w, inp, gamma, beta, mu, var = 0.3, 2.0, 1.5, 0.1, 0.2, 0.25
    conv = w * inp
    bn = gamma * (conv - mu) / np.sqrt(var) + beta
    a = gamma / np.sqrt(var); w_new = a * w; b_new = beta - a * mu
    print(f"  Chưa gộp: conv = {w}·{inp} = {conv} ; BN = {gamma}·({conv} − {mu})/√{var} + {beta} = {bn:.4f}")
    print(f"  Gộp: a = γ/√σ² = {a} ; w_mới = a·w = {w_new:.4f} ; bias_mới = β − a·μ = {b_new:.4f}")
    print(f"  Sau gộp: {w_new:.4f}·{inp} + ({b_new:.4f}) = {w_new * inp + b_new:.4f}  → giống hệt, không còn √ và phép chia")


def bai_11():
    tieu_de(11, "Dropout lúc train và lúc inference (bước 5)")
    v = np.array([1.0, 2.0, 3.0, 4.0]); mask = np.array([1, 0, 1, 0]); p = 0.5
    print(f"  Train, p = {p}: v·mask/(1−p) = {(v * mask / (1 - p)).tolist()}  (giữ lại thì nhân 2 để kỳ vọng không đổi)")
    print(f"  Inference: giữ nguyên v = {v.tolist()} → Dropout là phép đồng nhất, XOÁ khỏi đường inference")


BAI = {1: bai_1, 2: bai_2, 3: bai_3, 4: bai_4, 5: bai_5, 6: bai_6,
       7: bai_7, 8: bai_8, 9: bai_9, 10: bai_10, 11: bai_11}

if __name__ == "__main__":
    chon = [int(a) for a in sys.argv[1:]] or list(BAI)
    for n in chon:
        BAI[n]()
