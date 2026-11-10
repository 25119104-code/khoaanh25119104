# ============================================================
# ve_song.py — Vẽ waveform từ file .vcd ra ảnh .png để dán vào report
# ============================================================
# Chỉ vẽ tín hiệu ở cấp testbench (không vẽ lại tín hiệu bên trong DUT, vì
# chúng trùng với dây testbench). Bỏ qua biến integer của vòng lặp.
#   1 bit  : đường bậc thang 0/1, x/z vẽ màu đỏ.
#   nhiều bit: dải bus, ghi giá trị thập phân (hoặc x).
#
# Chạy: python3 ve_song.py song.vcd song.png "Tiêu đề"
# Xem tương tác (phóng to, đo thời gian) thì dùng: surfer song.vcd
# ============================================================

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def doc_vcd(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        toks = f.read().split()
    tin_hieu, thu_tu, thay_doi = {}, [], {}
    scope, don_vi, i = [], "1ns", 0

    def den_end(k):
        return toks.index("$end", k)

    # --- Phần khai báo ---
    while i < len(toks):
        t = toks[i]
        if t == "$timescale":
            j = den_end(i); don_vi = "".join(toks[i + 1:j]); i = j + 1
        elif t == "$scope":
            scope.append(toks[i + 2]); i = den_end(i) + 1
        elif t == "$upscope":
            scope.pop(); i = den_end(i) + 1
        elif t == "$var":
            j = den_end(i)
            kieu, rong, ma, ten = toks[i + 1], int(toks[i + 2]), toks[i + 3], toks[i + 4]
            if ma not in tin_hieu:          # cùng mã = cùng dây (tb.A và dut.A)
                tin_hieu[ma] = dict(ten=ten, rong=rong, kieu=kieu, cap=len(scope))
                thu_tu.append(ma)
                thay_doi[ma] = []
            i = j + 1
        elif t == "$enddefinitions":
            i = den_end(i) + 1
            break
        elif t.startswith("$"):
            i = den_end(i) + 1
        else:
            i += 1

    # --- Phần giá trị theo thời gian ---
    tg = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("#"):
            tg = int(t[1:])
        elif t[0] in "bB":
            thay_doi[toks[i + 1]].append((tg, t[1:])); i += 1
        elif t[0] in "rR":
            i += 1                           # số thực: không vẽ
        elif t[0] in "01xXzZ" and t[1:] in thay_doi:
            thay_doi[t[1:]].append((tg, t[0]))
        i += 1
    return don_vi, tin_hieu, thu_tu, thay_doi, tg


def ve(path_vcd, path_png, tieu_de=""):
    don_vi, tin_hieu, thu_tu, thay_doi, t_cuoi = doc_vcd(path_vcd)
    # Icarus ghi thời gian theo độ phân giải (1ps). Đổi về ns cho dễ đọc.
    if don_vi == "1ps":
        thay_doi = {m: [(t // 1000, v) for t, v in ds] for m, ds in thay_doi.items()}
        t_cuoi //= 1000
        don_vi = "ns"
    ve_ma = [m for m in thu_tu
             if tin_hieu[m]["cap"] == 1 and tin_hieu[m]["kieu"] not in ("integer", "real", "parameter")]
    # reg của testbench (= đầu vào DUT) vẽ trước, wire (= đầu ra DUT) vẽ sau
    ve_ma.sort(key=lambda m: tin_hieu[m]["kieu"] != "reg")
    if t_cuoi == 0:
        t_cuoi = 1
    n = len(ve_ma)
    fig, ax = plt.subplots(figsize=(12, 0.7 * n + 1.2))
    for hang, m in enumerate(ve_ma):
        y0 = (n - 1 - hang) * 1.5
        ds = thay_doi[m] + [(t_cuoi, None)]
        rong = tin_hieu[m]["rong"]
        for (ta, v), (tb, _) in zip(ds[:-1], ds[1:]):
            if tb <= ta and tb != t_cuoi:
                continue
            if rong == 1:
                if v in "01":
                    y = y0 + int(v)
                    ax.plot([ta, tb], [y, y], color="#1f6fb4", lw=2)
                else:
                    ax.plot([ta, tb], [y0 + 0.5] * 2, color="red", lw=2)
            else:
                mau = "red" if any(c in "xXzZ" for c in v) else "#1f6fb4"
                ax.plot([ta, tb], [y0 + 0.9] * 2, color=mau, lw=1.5)
                ax.plot([ta, tb], [y0 + 0.1] * 2, color=mau, lw=1.5)
                ax.plot([ta, ta], [y0 + 0.1, y0 + 0.9], color=mau, lw=1.5)
                nhan = "x" if mau == "red" else str(int(v, 2))
                if (tb - ta) / t_cuoi > 0.02:
                    ax.text((ta + tb) / 2, y0 + 0.5, nhan, ha="center", va="center", fontsize=8)
        # nối các bước nhảy của tín hiệu 1 bit
        if rong == 1:
            for k in range(1, len(ds) - 1):
                t_k, v_k = ds[k]
                v_tr = ds[k - 1][1]
                if v_k in "01" and v_tr in "01":
                    ax.plot([t_k, t_k], [y0 + int(v_tr), y0 + int(v_k)], color="#1f6fb4", lw=2)
        ten = tin_hieu[m]["ten"] + (f"[{rong - 1}:0]" if rong > 1 else "")
        ax.text(-0.01 * t_cuoi, y0 + 0.5, ten, ha="right", va="center", fontsize=10)
    ax.set_xlim(0, t_cuoi)
    ax.set_ylim(-0.5, 1.5 * n)
    ax.set_yticks([])
    ax.set_xlabel(f"thời gian (đơn vị {don_vi})")
    ax.grid(axis="x", alpha=0.3)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    if tieu_de:
        ax.set_title(tieu_de)
    fig.tight_layout()
    fig.savefig(path_png, dpi=150)
    print(f"Đã vẽ {path_png} ({n} tín hiệu, 0 → {t_cuoi} {don_vi})")


if __name__ == "__main__":
    ve(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
