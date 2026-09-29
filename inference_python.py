# ============================================================
# inference_python.py — Inference SmallCNN bằng VÒNG LẶP THUẦN
# ============================================================
# Bản nháp của golden model C, viết bằng Python cho dễ debug.
#   - KHÔNG dùng PyTorch, KHÔNG dùng numpy để tính. Chỉ list + for + if.
#   - Trọng số đọc từ params/weights.txt + params/biases.txt.
#   - Mọi mảng là MẢNG 1 CHIỀU, tự tính chỉ số — đúng như trong C.
#     Dịch sang C gần như dòng-đổi-dòng.
#
# Chạy:
#   python export_params.py              # cần chạy trước 1 lần
#   python inference_python.py           # ảnh mẫu: so TỪNG LỚP với PyTorch
#   python inference_python.py --n 200   # thêm: đo accuracy trên 200 ảnh
#
# Đường inference (6 operator):
#   conv2d -> relu -> maxpool   (x3, kênh 1->8->16->16, spatial 28->14->7->3)
#   -> flatten (144) -> linear (144->10) -> argmax
# ============================================================

import argparse
import time

from mnist_io import load_test_set, normalize

# ------------------------------------------------------------
# [1] Bảng offset — PHẢI khớp params/README.md
# ------------------------------------------------------------
#          tên      in_ch out_ch  w_off  b_off
CONV = [("conv1",   1,    8,     0,     0),
        ("conv2",   8,    16,    72,    8),
        ("conv3",   16,   16,    1224,  24)]
FC_IN, FC_OUT, FC_W_OFF, FC_B_OFF = 144, 10, 3528, 40
K, PAD = 3, 1


def read_floats(path):
    """Giống fscanf("%f") trong vòng lặp ở C: mỗi dòng 1 số."""
    with open(path) as f:
        return [float(line) for line in f]


# ------------------------------------------------------------
# [2] Sáu operator
# ------------------------------------------------------------
# Quy ước chỉ số feature map [C][H][W] trải phẳng:
#     idx = (c * H + y) * W + x
# Quy ước chỉ số weight conv [OC][IC][K][K] trải phẳng:
#     idx = w_off + ((oc * IC + ic) * K + ky) * K + kx

mac_count = {"if": 0, "pad": 0}   # đếm số phép nhân-cộng THẬT SỰ thực hiện


def conv2d_if(inp, IC, H, W, wts, bias, OC, w_off, b_off):
    """Bản 1 — có `if` kiểm tra biên. Giống mã giả trong operators/conv2d.md.

    Tap nào rơi ra ngoài ảnh (vùng padding) thì BỎ QUA -> ít phép nhân hơn,
    nhưng mỗi tap phải qua 1 lần so sánh.
    """
    out = [0.0] * (OC * H * W)          # PAD=1, K=3, stride=1 -> H_out = H
    for oc in range(OC):
        for oy in range(H):
            for ox in range(W):
                acc = bias[b_off + oc]              # khởi tạo bằng BIAS
                for ic in range(IC):
                    for ky in range(K):
                        for kx in range(K):
                            iy = oy - PAD + ky
                            ix = ox - PAD + kx
                            if 0 <= iy < H and 0 <= ix < W:
                                acc += inp[(ic * H + iy) * W + ix] * \
                                       wts[w_off + ((oc * IC + ic) * K + ky) * K + kx]
                                mac_count["if"] += 1
                out[(oc * H + oy) * W + ox] = acc
    return out


def conv2d_pad(inp, IC, H, W, wts, bias, OC, w_off, b_off):
    """Bản 2 — đệm số 0 trước, rồi LUÔN tính đủ 9 tap. Không có `if` bên trong.

    Đây là cách phần cứng hay làm: line buffer đẩy số 0 ra ở biên, mọi vị trí
    đều là 9 MAC giống hệt nhau. Nhân với 0 là phí phép tính, nhưng không phí
    chu kỳ clock và không cần mạch so sánh.
    """
    HP, WP = H + 2 * PAD, W + 2 * PAD
    padded = [0.0] * (IC * HP * WP)                 # bước 1: tạo ảnh đã đệm 0
    for ic in range(IC):
        for y in range(H):
            for x in range(W):
                padded[(ic * HP + y + PAD) * WP + x + PAD] = inp[(ic * H + y) * W + x]

    out = [0.0] * (OC * H * W)
    for oc in range(OC):
        for oy in range(H):
            for ox in range(W):
                acc = bias[b_off + oc]
                for ic in range(IC):
                    for ky in range(K):
                        for kx in range(K):
                            acc += padded[(ic * HP + oy + ky) * WP + ox + kx] * \
                                   wts[w_off + ((oc * IC + ic) * K + ky) * K + kx]
                            mac_count["pad"] += 1
                out[(oc * H + oy) * W + ox] = acc
    return out


def relu(x):
    return [v if v > 0.0 else 0.0 for v in x]


def maxpool2d(inp, C, H, W):
    """Kernel 2, stride 2. H_out = floor(H/2): 7 -> 3, hàng/cột cuối bị bỏ."""
    HO, WO = H // 2, W // 2
    out = [0.0] * (C * HO * WO)
    for c in range(C):
        for oy in range(HO):
            for ox in range(WO):
                m = inp[(c * H + 2 * oy) * W + 2 * ox]
                for dy in range(2):
                    for dx in range(2):
                        v = inp[(c * H + 2 * oy + dy) * W + 2 * ox + dx]
                        if v > m:
                            m = v
                out[(c * HO + oy) * WO + ox] = m
    return out, HO, WO


def flatten(x):
    """Mảng đã trải phẳng theo [C][H][W] (NCHW) -> flatten không làm gì cả.

    Đây chính là lý do giữ layout giống PyTorch: fc.weight được train với
    thứ tự (c, y, x). Đổi sang (y, x, c) thì KHÔNG crash mà accuracy tụt.
    """
    return x


def linear(x, wts, bias):
    out = [0.0] * FC_OUT
    for o in range(FC_OUT):
        acc = bias[FC_B_OFF + o]
        for i in range(FC_IN):
            acc += x[i] * wts[FC_W_OFF + o * FC_IN + i]
        out[o] = acc
    return out


def argmax(z):
    """So sánh CHẶT (>) để khớp torch.argmax khi có logit bằng nhau."""
    best = 0
    for i in range(1, len(z)):
        if z[i] > z[best]:
            best = i
    return best


# ------------------------------------------------------------
# [3] Ghép lại — trả về từng bước để so với golden/
# ------------------------------------------------------------
def infer(x, wts, bias, conv_fn):
    steps = []
    C, H, W = 1, 28, 28
    for name, IC, OC, w_off, b_off in CONV:
        x = conv_fn(x, IC, H, W, wts, bias, OC, w_off, b_off); C = OC
        steps.append((name, x))
        x = relu(x);                   steps.append((name.replace("conv", "relu"), x))
        x, H, W = maxpool2d(x, C, H, W); steps.append((name.replace("conv", "pool"), x))
    x = flatten(x);                    steps.append(("flatten", x))
    z = linear(x, wts, bias);          steps.append(("fc_logits", z))
    return z, steps


# Lưu ý: float của Python là 64 bit, PyTorch tính 32 bit -> lệch ~1e-6..1e-5
# là do ĐỘ CHÍNH XÁC, không phải sai thuật toán. Sai thuật toán (quên bias,
# sai -PAD, sai layout) sẽ lệch từ ~1e-1 trở lên.
def max_abs_diff(a, b):
    return max(abs(p - q) for p, q in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="số ảnh test để đo accuracy (0 = bỏ qua)")
    ap.add_argument("--tol", type=float, default=1e-4, help="ngưỡng lệch cho phép mỗi lớp")
    args = ap.parse_args()

    wts = read_floats("params/weights.txt")
    bias = read_floats("params/biases.txt")
    assert len(wts) == 4968 and len(bias) == 50, "sai số lượng tham số"
    print(f"Đọc {len(wts)} weight + {len(bias)} bias = {len(wts) + len(bias)} params")

    # ---- (a) Ảnh mẫu: so TỪNG LỚP với PyTorch ----
    x0 = read_floats("golden/input.txt")
    label = int(open("golden/label.txt").read())
    golden_files = ["01_conv1", "02_relu1", "03_pool1", "04_conv2", "05_relu2", "06_pool2",
                    "07_conv3", "08_relu3", "09_pool3", "10_flatten", "11_fc_logits"]

    results = {}
    for tag, fn in [("if", conv2d_if), ("pad", conv2d_pad)]:
        mac_count[tag] = 0
        t = time.time()
        z, steps = infer(x0, wts, bias, fn)
        results[tag] = (z, steps, time.time() - t)

    print(f"\n[a] Ảnh mẫu (nhãn {label}) — lệch lớn nhất so với PyTorch, ngưỡng {args.tol:g}")
    print(f"  {'lớp':<11}{'số pt':>7}{'conv if':>13}{'conv pad':>13}")
    ok = True
    for gname, (name, _) in zip(golden_files, results["if"][1]):
        g = read_floats(f"golden/{gname}.txt")
        d_if = max_abs_diff(dict(results["if"][1])[name], g)
        d_pad = max_abs_diff(dict(results["pad"][1])[name], g)
        flag = "" if max(d_if, d_pad) < args.tol else "  <-- VƯỢT NGƯỠNG"
        ok &= not flag
        print(f"  {name:<11}{len(g):>7}{d_if:>13.2e}{d_pad:>13.2e}{flag}")

    z_if, z_pad = results["if"][0], results["pad"][0]
    print(f"\n  Đoán: bản if = {argmax(z_if)}, bản pad = {argmax(z_pad)}, nhãn đúng = {label}")
    print(f"  Hai bản conv lệch nhau tối đa: {max_abs_diff(z_if, z_pad):.2e} "
          f"(phải = 0: cộng thêm 0*w = 0.0 không đổi tổng)")
    print(f"\n  Số MAC conv / ảnh:  bản if = {mac_count['if']:,}   bản pad = {mac_count['pad']:,}"
          f"   -> pad làm thêm {mac_count['pad'] - mac_count['if']:,} phép nhân với 0 "
          f"({(mac_count['pad'] - mac_count['if']) / mac_count['pad'] * 100:.1f}%)")
    print(f"  Thời gian 1 ảnh: bản if {results['if'][2]:.2f}s, bản pad {results['pad'][2]:.2f}s")
    print(f"\n  KẾT LUẬN ảnh mẫu: {'KHỚP PyTorch ở mọi lớp' if ok else 'CÓ LỚP LỆCH — xem bảng trên'}")

    # ---- (b) Nhiều ảnh: accuracy + so dự đoán với PyTorch ----
    if args.n > 0:
        images, labels = load_test_set()
        with open("golden/logits_pytorch.txt") as f:
            pt = [[float(v) for v in line.split()] for line in f]
        n = min(args.n, len(images))
        correct = same_pred = 0
        worst = 0.0
        t = time.time()
        for i in range(n):
            x = [float(v) for v in normalize(images[i]).ravel()]
            z, _ = infer(x, wts, bias, conv2d_pad)
            p = argmax(z)
            correct += p == labels[i]
            same_pred += p == argmax(pt[i])
            worst = max(worst, max_abs_diff(z, pt[i]))
            if (i + 1) % 50 == 0:
                print(f"  ... {i + 1}/{n}", flush=True)
        print(f"\n[b] {n} ảnh đầu test set ({time.time() - t:.0f}s, bản conv pad)")
        print(f"  Accuracy bản Python: {correct / n * 100:.2f}%")
        print(f"  Trùng dự đoán PyTorch: {same_pred}/{n}")
        print(f"  Logit lệch lớn nhất: {worst:.2e}")


if __name__ == "__main__":
    main()
