# ============================================================
# make_conv_figures.py — Vẽ cách conv tính RA 1 ĐIỂM output
# ============================================================
# Trả lời câu Thầy hỏi: "chỉ tính 1 điểm trên 1 kênh output thì tính thế nào?"
# Dùng SỐ THẬT: trọng số từ params/, feature map từ golden/ (ảnh test số 0).
# Không cần PyTorch. Chạy sau export_params.py.
#
# Sinh 3 hình:
#   figures/conv_1diem_conv1.png  — 1 kênh vào: 9 phép nhân + bias
#   figures/conv_1diem_conv2.png  — 8 kênh vào: 72 phép nhân DỒN VÀO 1 SỐ
#   figures/conv_padding_goc.png  — điểm ở góc: 5/9 tap rơi vào vùng đệm 0
# ============================================================

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K, PAD = 3, 1


def load(path, shape):
    return np.loadtxt(path, dtype=np.float64).reshape(shape)


W = np.loadtxt("params/weights.txt")
B = np.loadtxt("params/biases.txt")
w1 = W[0:72].reshape(8, 1, 3, 3);       b1 = B[0:8]
w2 = W[72:1224].reshape(16, 8, 3, 3);   b2 = B[8:24]

x_in = load("golden/input.txt", (1, 28, 28))
conv1 = load("golden/01_conv1.txt", (8, 28, 28))
pool1 = load("golden/03_pool1.txt", (8, 14, 14))
conv2 = load("golden/04_conv2.txt", (16, 14, 14))


def pick_position(x, H):
    """Chọn vị trí ở giữa ảnh có cửa sổ 3x3 'nhiều thông tin' nhất (độ lệch lớn)."""
    best, pos = -1, (H // 2, H // 2)
    for y in range(1, H - 1):
        for xx in range(1, H - 1):
            s = x[:, y - 1:y + 2, xx - 1:xx + 2].std()
            if s > best:
                best, pos = s, (y, xx)
    return pos


def grid(ax, m, title, cmap="RdBu_r", fmt="{:.2f}", grey=None, vmax=None):
    """Vẽ ma trận 3x3 kèm số trong từng ô. grey = mask các ô vùng đệm."""
    vmax = vmax or max(abs(m).max(), 1e-9)
    ax.imshow(m, cmap=cmap, vmin=-vmax, vmax=vmax)
    for (i, j), v in np.ndenumerate(m):
        txt = "0\n(đệm)" if grey is not None and grey[i, j] else fmt.format(v)
        ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                color="black", fontweight="bold" if grey is None or not grey[i, j] else "normal")
        if grey is not None and grey[i, j]:
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#bbbbbb"))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)


# ------------------------------------------------------------
# Hình 1 — conv1: 1 kênh vào
# ------------------------------------------------------------
def fig_conv1(oc=0):
    oy, ox = pick_position(x_in, 28)
    patch = x_in[0, oy - 1:oy + 2, ox - 1:ox + 2]
    ker = w1[oc, 0]
    prod = patch * ker
    val = prod.sum() + b1[oc]

    fig = plt.figure(figsize=(14, 5.2))
    ax0 = fig.add_subplot(1, 5, 1)
    ax0.imshow(x_in[0], cmap="gray")
    ax0.add_patch(plt.Rectangle((ox - 1.5, oy - 1.5), 3, 3, fill=False, ec="red", lw=2))
    ax0.set_title(f"Ảnh vào 28×28\ncửa sổ 3×3 quanh ({oy},{ox})", fontsize=10)
    ax0.set_xticks([]); ax0.set_yticks([])

    grid(fig.add_subplot(1, 5, 2), patch, "① 9 pixel vào\n(đã chuẩn hoá)")
    grid(fig.add_subplot(1, 5, 3), ker, f"② Kernel conv1\nkênh ra oc={oc}")
    grid(fig.add_subplot(1, 5, 4), prod, "③ Nhân từng ô\n(9 phép nhân)")

    ax = fig.add_subplot(1, 5, 5); ax.axis("off")
    ax.text(0, .85, "④ Cộng dồn", fontsize=11, fontweight="bold")
    ax.text(0, .68, f"Σ 9 tích   = {prod.sum():+.5f}", fontsize=10, family="monospace")
    ax.text(0, .56, f"+ bias[{oc}]  = {b1[oc]:+.5f}", fontsize=10, family="monospace")
    ax.text(0, .42, f"= {val:+.5f}", fontsize=12, fontweight="bold", family="monospace")
    ax.text(0, .26, f"PyTorch:  {conv1[oc, oy, ox]:+.5f}", fontsize=10, family="monospace")
    ax.text(0, .08, f"→ output[{oc}][{oy}][{ox}]\n  của conv1 (8×28×28)", fontsize=10)

    fig.suptitle("conv1 — tính 1 điểm output: 9 MAC + 1 bias  (acc khởi tạo = bias)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("figures/conv_1diem_conv1.png", dpi=150)
    plt.close(fig)
    return oy, ox, val, conv1[oc, oy, ox]


# ------------------------------------------------------------
# Hình 2 — conv2: 8 kênh vào dồn vào 1 số
# ------------------------------------------------------------
def fig_conv2(oc=0):
    oy, ox = pick_position(pool1, 14)
    IC = 8
    fig, axes = plt.subplots(3, IC + 1, figsize=(17, 6.6),
                             gridspec_kw={"width_ratios": [1] * IC + [1.5]})
    partial = []
    for ic in range(IC):
        patch = pool1[ic, oy - 1:oy + 2, ox - 1:ox + 2]
        ker = w2[oc, ic]
        s = (patch * ker).sum()
        partial.append(s)
        grid(axes[0, ic], patch, f"vào ic={ic}", fmt="{:.1f}")
        grid(axes[1, ic], ker, f"kernel [{oc}][{ic}]", fmt="{:.2f}")
        axes[2, ic].axis("off")
        axes[2, ic].text(.5, .6, f"Σ9 = {s:+.3f}", ha="center", fontsize=10, family="monospace")

    val = sum(partial) + b2[oc]
    for r in range(3):
        axes[r, IC].axis("off")
    axes[0, IC].text(0, .5, "Hàng 1: cửa sổ 3×3\ncùng vị trí trên\n8 kênh pool1", fontsize=10)
    axes[1, IC].text(0, .5, "Hàng 2: 8 lát kernel\ncủa CÙNG 1 kênh ra\nw2[oc] = 8×3×3", fontsize=10)
    axes[2, IC].text(0, .75, f"Σ 8 kênh  = {sum(partial):+.4f}", fontsize=10)
    axes[2, IC].text(0, .55, f"+ bias      = {b2[oc]:+.4f}", fontsize=10)
    axes[2, IC].text(0, .30, f"= {val:+.4f}", fontsize=12, fontweight="bold", family="monospace")
    axes[2, IC].text(0, .08, f"PyTorch: {conv2[oc, oy, ox]:+.4f}", fontsize=10, family="monospace")

    fig.suptitle(f"conv2 — 1 điểm output[{oc}][{oy}][{ox}]: 8 kênh × 9 = 72 MAC + 1 bias → MỘT số duy nhất\n"
                 "(mỗi kênh vào KHÔNG ra output riêng — tất cả cộng dồn vào 1 accumulator)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig("figures/conv_1diem_conv2.png", dpi=150)
    plt.close(fig)
    return oy, ox, val, conv2[oc, oy, ox]


# ------------------------------------------------------------
# Hình 3 — điểm ở góc: vùng đệm 0
# ------------------------------------------------------------
def fig_padding(oc=0):
    padded = np.pad(x_in[0], PAD)                       # 30x30, viền = 0
    oy = ox = 0
    win = padded[oy:oy + 3, ox:ox + 3]
    mask = np.zeros((3, 3), bool); mask[0, :] = True; mask[:, 0] = True
    ker = w1[oc, 0]
    val = (win * ker).sum() + b1[oc]

    fig = plt.figure(figsize=(14, 5))
    ax0 = fig.add_subplot(1, 4, 1)
    show = padded.copy()
    ax0.imshow(show, cmap="gray")
    ax0.add_patch(plt.Rectangle((-.5, -.5), 30, 30, fill=False, ec="orange", lw=3, ls="--"))
    ax0.add_patch(plt.Rectangle((-.5, -.5), 3, 3, fill=False, ec="red", lw=2))
    ax0.set_title("Ảnh đã đệm 30×30 — viền = số 0\n(0 ≠ nền ảnh −0,42 sau chuẩn hoá!)", fontsize=10)
    ax0.set_xticks([]); ax0.set_yticks([])

    grid(fig.add_subplot(1, 4, 2), win, "Cửa sổ tại output (0,0)", grey=mask)
    grid(fig.add_subplot(1, 4, 3), ker, f"Kernel conv1 oc={oc}")

    ax = fig.add_subplot(1, 4, 4); ax.axis("off")
    lines = [
        ("5 / 9 tap rơi vào vùng đệm 0", True),
        ("", False),
        ("Bản có `if` (CPU):", True),
        ("  bỏ 5 tap → 4 phép nhân", False),
        ("  nhưng MỖI tap tốn 1 phép so sánh", False),
        ("", False),
        ("Bản đệm 0 (FPGA):", True),
        ("  luôn 9 phép nhân, 5 phép × 0", False),
        ("  9 bộ nhân song song → 1 chu kỳ", False),
        ("  bỏ bớt cũng KHÔNG nhanh hơn", False),
        ("", False),
        (f"Cả hai ra: {val:+.5f}", True),
        (f"PyTorch:   {conv1[oc, 0, 0]:+.5f}", False),
    ]
    for i, (t, bold) in enumerate(lines):
        ax.text(0, 1 - i * .075, t, fontsize=10, fontweight="bold" if bold else "normal")

    fig.suptitle("Điểm ở góc — vì sao phần cứng giữ đủ 9 phép thay vì dùng `if`",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("figures/conv_padding_goc.png", dpi=150)
    plt.close(fig)
    return val, conv1[oc, 0, 0]


if __name__ == "__main__":
    print("conv1 (oy, ox, tính tay, PyTorch):", fig_conv1())
    print("conv2 (oy, ox, tính tay, PyTorch):", fig_conv2())
    print("góc   (tính tay, PyTorch):        ", fig_padding())
    print("-> figures/conv_1diem_conv1.png, conv_1diem_conv2.png, conv_padding_goc.png")
