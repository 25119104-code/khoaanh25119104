# ============================================================
# ptq_int8.py — Post-Training Quantization INT8, SỐ NGUYÊN THUẦN
# ============================================================
# Mục tiêu: chạy SmallCNN chỉ bằng số nguyên, đúng như phần cứng sẽ làm:
#     int8 × int8 → cộng dồn int32 → requantize (nhân M0, dịch bit) → int8
# Không dùng torch.ao.quantization: bản đó là hộp đen (fbgemm/qnnpack),
# không cho ra số nguyên từng lớp để sau này so với C và RTL.
#
# Sơ đồ lượng tử (theo slide của Thầy, có bổ sung):
#   - Trọng số : int8 đối xứng, zp = 0, scale per-tensor = max|w| / 127
#   - Bias     : int32, scale = s_vào × s_w, zp = 0
#   - Activation: int8 bất đối xứng, (scale, zp) calibrate bằng min/max
#   - Requantize: M = s_vào·s_w / s_ra = M0 · 2^-shift
#                 q_ra = clamp(((acc·M0 + 2^(shift-1)) >> shift) + zp_ra, -128, 127)
#   - Lớp fc cuối KHÔNG requantize: argmax thẳng trên acc int32
#     (nhân với scale dương không đổi thứ tự → cùng kết quả).
#
# Quy tắc dữ liệu: mọi lựa chọn (calibrate bao nhiêu ảnh, ReLU kiểu nào,
# M0 bao nhiêu bit) so trên VALIDATION. Test set chỉ chạy 1 lần cho cấu
# hình đã chọn.
#
# Chạy:  python ptq_int8.py
# Ra:    params_int8/ (tham số số nguyên + ảnh mẫu từng lớp + tham chiếu 10.000 ảnh
#        cho golden C + ket_qua_ptq.json)
#
#        python ptq_int8.py --chi-tham-chieu
#        Không calibrate lại: đọc tham số ĐÃ XUẤT trong params_int8/, chỉ sinh lại
#        golden_int8/fc_acc_test.txt và README.md cho golden C (golden_c/golden_int8.c).
# ============================================================

import argparse
import json
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from models import SmallCNN
from mnist_io import load_test_set, MEAN, STD

INT8_MIN, INT8_MAX = -128, 127
CKPT = "models/small_cnn.pth"
OUT_DIR = "params_int8"
SEED = 42            # PHẢI giống model_comparison.py để lấy đúng tập val
SO_ANH_CALIB = 1000  # số ảnh val dùng để đo min/max


# ------------------------------------------------------------
# [1] DỮ LIỆU — đọc file raw, tách train/val giống hệt random_split
# ------------------------------------------------------------
def doc_idx(img_file, lbl_file):
    with open(img_file, "rb") as f:
        raw = f.read()
    n = int.from_bytes(raw[4:8], "big")
    imgs = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(n, 28, 28)
    with open(lbl_file, "rb") as f:
        lbls = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return imgs, lbls


def load_train_val():
    """Tách 60k ảnh train thành 50k train + 10k val GIỐNG HỆT model_comparison.py.

    random_split(…, [50_000, 10_000], generator=seed 42) bên trong chỉ gọi
    torch.randperm(60000, generator) rồi cắt 50k đầu / 10k cuối. Làm lại đúng
    vậy thì không cần torchvision mà vẫn ra đúng tập val model chưa từng train.
    """
    imgs, lbls = doc_idx("data/MNIST/raw/train-images-idx3-ubyte",
                         "data/MNIST/raw/train-labels-idx1-ubyte")
    perm = torch.randperm(60000, generator=torch.Generator().manual_seed(SEED)).numpy()
    tr, va = perm[:50_000], perm[50_000:]
    return (imgs[tr], lbls[tr]), (imgs[va], lbls[va])


def chuan_hoa(imgs_u8):
    """uint8 [N,28,28] → float32 [N,1,28,28], cùng thứ tự phép tính với mnist_io.normalize."""
    x = imgs_u8.astype(np.float32) / np.float32(255.0)
    return ((x - np.float32(MEAN)) / np.float32(STD))[:, None, :, :]


# ------------------------------------------------------------
# [2] HÀM LƯỢNG TỬ CƠ BẢN
# ------------------------------------------------------------
def lam_tron(x):
    """Làm tròn nửa lên: floor(x + 0,5). Dễ viết lại y hệt trong C.

    Không dùng np.round vì np.round làm tròn về số chẵn (2,5 → 2), C thì không.
    """
    return np.floor(np.asarray(x, dtype=np.float64) + 0.5)


def gioi_han(bits):
    """Dải số nguyên có dấu `bits` bit: 8 → (-128, 127), 4 → (-8, 7)."""
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def tham_so_bat_doi_xung(lo, hi, bits=8):
    """Từ dải thực [lo, hi] → (scale, zp) cho số nguyên `bits` bit (mặc định int8 [-128, 127]).

    Dải luôn được nới cho chứa 0, để số thực 0 (giá trị đệm, đầu ra ReLU)
    biểu diễn CHÍNH XÁC bằng một số nguyên (= zp).
    """
    qmin, qmax = gioi_han(bits)
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    s = (hi - lo) / (qmax - qmin)
    zp = int(np.clip(lam_tron(qmin - lo / s), qmin, qmax))
    return s, zp


def luong_tu_trong_so(w, bits=8):
    """Trọng số đối xứng: zp = 0, int8 dùng [-127, 127] (bỏ -128 cho đối xứng)."""
    qmax = gioi_han(bits)[1]
    s = float(np.abs(w).max()) / qmax
    q = np.clip(lam_tron(w / s), -qmax, qmax).astype(np.int64)
    return q, s


def tach_M(M, bits=31):
    """M (số thực) → (M0, shift) với M ≈ M0 · 2^-shift, M0 có đúng `bits` bit.

    Ví dụ bits=31: M0 nằm trong [2^30, 2^31) → vừa thanh ghi int32 không dấu bit.
    bits nhỏ hơn (15) → hợp bộ nhân DSP nhỏ trên FPGA nhưng kém chính xác hơn.
    """
    shift = 0
    while M * (1 << shift) < (1 << (bits - 1)):
        shift += 1
    M0 = int(lam_tron(M * (1 << shift)))
    if M0 == (1 << bits):          # làm tròn tràn lên bit thứ bits+1
        M0 >>= 1
        shift -= 1
    return M0, shift


def requantize(acc, M0, shift, zp, bits=8):
    """int32 → int8 chỉ bằng nhân số nguyên, cộng, dịch bit, kẹp.

    `>>` của numpy trên int64 là dịch SỐ HỌC (làm tròn xuống, kể cả số âm).
    Cộng 2^(shift-1) trước khi dịch → làm tròn nửa lên. Golden C phải làm
    y hệt (gcc/clang dịch số học với số âm, nhưng chuẩn C không bảo đảm).
    """
    tich = acc * M0
    assert np.abs(tich).max() < (1 << 62), "tràn int64 khi nhân acc × M0"
    q = ((tich + (1 << (shift - 1))) >> shift) + zp
    return np.clip(q, *gioi_han(bits))           # BƯỚC KẸP — slide thiếu


# ------------------------------------------------------------
# [3] FORWARD FLOAT CÓ GHI LẠI TỪNG LỚP (để calibrate)
# ------------------------------------------------------------
@torch.no_grad()
def forward_float(model, x):
    ghi, h = {}, x
    for i, conv in enumerate([model.conv1, model.conv2, model.conv3], 1):
        c = conv(h)
        r = torch.relu(c)
        ghi[f"conv{i}"], ghi[f"relu{i}"] = c, r
        h = F.max_pool2d(r, 2)
    return ghi, model.fc(h.flatten(1))


def calibrate(model, x_np):
    """Đo min/max của đầu vào, đầu ra conv (trước ReLU) và sau ReLU trên ảnh calib."""
    x = torch.from_numpy(x_np)
    ghi, _ = forward_float(model, x)
    dai = {"input": (float(x.min()), float(x.max()))}
    for k, v in ghi.items():
        dai[k] = (float(v.min()), float(v.max()))
    return dai


# ------------------------------------------------------------
# [4] DỰNG MÔ HÌNH SỐ NGUYÊN
# ------------------------------------------------------------
def dung_mo_hinh_int(sd, dai, relu="gop", bits_M0=31, bits=8):
    """sd: state_dict (numpy). dai: dải thực từng lớp. Trả về toàn bộ tham số số nguyên.

    relu="rieng": đúng slide của Thầy. Scale lấy theo dải đầu ra conv TRƯỚC ReLU
                  (có cả phần âm), rồi ReLU là phép q = max(q, zp).
    relu="gop"  : scale lấy theo dải SAU ReLU [0, max] → zp = -128. Khi đó bước
                  kẹp về [-128, 127] đã tự cắt phần âm → ReLU không tốn phép nào,
                  và 256 mức int8 dùng hết cho phần dương (độ phân giải gấp đôi).
    bits=8 là sản phẩm chính; bits=4 chỉ để thí nghiệm khi nào QAT mới cần.
    """
    s_in, zp_in = tham_so_bat_doi_xung(*dai["input"], bits=bits)
    mh = {"relu": relu, "bits_M0": bits_M0, "bits": bits, "s_in": s_in, "zp_in": zp_in, "conv": []}
    s_truoc, zp_truoc = s_in, zp_in
    for i in (1, 2, 3):
        qw, sw = luong_tu_trong_so(sd[f"conv{i}.weight"], bits)
        s_acc = s_truoc * sw
        qb = lam_tron(sd[f"conv{i}.bias"] / s_acc).astype(np.int64)
        lo, hi = dai[f"conv{i}"] if relu == "rieng" else (0.0, dai[f"relu{i}"][1])
        s_ra, zp_ra = tham_so_bat_doi_xung(lo, hi, bits)
        M = s_acc / s_ra
        M0, shift = tach_M(M, bits_M0)
        mh["conv"].append(dict(qw=qw, qb=qb, sw=sw, zp_vao=zp_truoc, s_ra=s_ra,
                               zp_ra=zp_ra, M=M, M0=M0, shift=shift))
        s_truoc, zp_truoc = s_ra, zp_ra
    qw, sw = luong_tu_trong_so(sd["fc.weight"], bits)
    s_acc = s_truoc * sw
    mh["fc"] = dict(qw=qw, qb=lam_tron(sd["fc.bias"] / s_acc).astype(np.int64),
                    sw=sw, zp_vao=zp_truoc, s_acc=s_acc)
    for L in mh["conv"] + [mh["fc"]]:
        assert np.abs(L["qb"]).max() < 2**31, "bias không vừa int32"
    return mh


# ------------------------------------------------------------
# [5] CÁC OPERATOR SỐ NGUYÊN
# ------------------------------------------------------------
def luong_tu_dau_vao(x_float, s_in, zp_in, bits=8):
    return np.clip(lam_tron(x_float / s_in) + zp_in, *gioi_han(bits)).astype(np.int64)


def conv3x3_int(q, zp_vao, qw, qb):
    """Conv 3×3, padding 1, đủ 9 tap. Trả acc int32 (chứa trong int64 để kiểm tràn).

    BẪY ĐỆM: số thực 0 trong miền int8 là zp, KHÔNG phải số nguyên 0.
    Ở đây trừ zp trước rồi mới đệm 0 → tương đương đệm giá trị zp.
    Đệm số nguyên 0 trực tiếp lên q là sai khi zp ≠ 0.
    """
    x = q - zp_vao
    N, C, H, W = x.shape
    OC = qw.shape[0]
    xp = np.zeros((N, C, H + 2, W + 2), dtype=np.int64)
    xp[:, :, 1:-1, 1:-1] = x
    cot = np.empty((N, C, 9, H, W), dtype=np.int64)
    for ky in range(3):
        for kx in range(3):
            cot[:, :, ky * 3 + kx] = xp[:, :, ky:ky + H, kx:kx + W]
    cot = cot.reshape(N, C * 9, H * W)
    acc = np.matmul(qw.reshape(OC, C * 9), cot) + qb[None, :, None]   # khởi tạo bằng bias
    return acc.reshape(N, OC, H, W)


def maxpool2_int(q):
    """MaxPool 2×2 trên int8: max không đổi scale/zp, chạy thẳng trên số nguyên, chính xác."""
    N, C, H, W = q.shape
    H2, W2 = H // 2, W // 2                      # floor: 7 → 3 cắt hàng/cột cuối
    return q[:, :, :H2 * 2, :W2 * 2].reshape(N, C, H2, 2, W2, 2).max(axis=(3, 5))


def suy_luan_int(mh, q_in, thong_ke=None, ghi=None):
    """q_in: int8 [N,1,28,28]. Trả acc int32 của fc [N,10] (logit chưa đổi scale)."""
    q = q_in
    for i, L in enumerate(mh["conv"], 1):
        acc = conv3x3_int(q, L["zp_vao"], L["qw"], L["qb"])
        if thong_ke is not None:
            thong_ke[f"conv{i}"] = max(thong_ke.get(f"conv{i}", 0), int(np.abs(acc).max()))
        q = requantize(acc, L["M0"], L["shift"], L["zp_ra"], mh["bits"])
        if mh["relu"] == "rieng":
            q = np.maximum(q, L["zp_ra"])        # ReLU đúng dấu: q < zp thì q = zp
        if ghi is not None:
            ghi[f"conv{i}_acc"], ghi[f"relu{i}"] = acc, q
        q = maxpool2_int(q)
        if ghi is not None:
            ghi[f"pool{i}"] = q
    fc = mh["fc"]
    v = q.reshape(q.shape[0], -1) - fc["zp_vao"]     # flatten NCHW, giống PyTorch
    acc = v @ fc["qw"].T + fc["qb"]
    if thong_ke is not None:
        thong_ke["fc"] = max(thong_ke.get("fc", 0), int(np.abs(acc).max()))
    if ghi is not None:
        ghi["fc_acc"] = acc
    return acc


def argmax_chat(a):
    """np.argmax trả chỉ số ĐẦU TIÊN khi bằng nhau — giống so sánh chặt '>' trong C."""
    return np.argmax(a, axis=1)


def chay_int(mh, x_float, lo=500, thong_ke=None):
    du_doan = []
    for k in range(0, len(x_float), lo):
        q_in = luong_tu_dau_vao(x_float[k:k + lo], mh["s_in"], mh["zp_in"], mh["bits"])
        du_doan.append(argmax_chat(suy_luan_int(mh, q_in, thong_ke)))
    return np.concatenate(du_doan)


@torch.no_grad()
def chay_float(model, x_float, lo=1000):
    return np.concatenate([model(torch.from_numpy(x_float[k:k + lo])).argmax(1).numpy()
                           for k in range(0, len(x_float), lo)])


# ------------------------------------------------------------
# [6] XUẤT THAM SỐ SỐ NGUYÊN — cho golden C int8 (golden_c/golden_int8.c)
# ------------------------------------------------------------
THU_TU_GOLDEN = ["conv1_acc", "relu1", "pool1", "conv2_acc", "relu2", "pool2",
                 "conv3_acc", "relu3", "pool3", "fc_acc"]


def ghi_mang(path, a):
    np.savetxt(path, np.asarray(a).reshape(-1), fmt="%d")


def xuat_tham_so(mh, x_mau, nhan_mau, thu_muc=OUT_DIR, nguon="ptq_int8.py"):
    os.makedirs(f"{thu_muc}/golden_int8", exist_ok=True)
    dong = [f"# Tham số lượng tử — sinh tự động bởi {nguon}, KHÔNG sửa tay",
            f"# relu={mh['relu']}  bits_M0={mh['bits_M0']}",
            f"input  s={mh['s_in']:.9g}  zp={mh['zp_in']}"]
    for i, L in enumerate(mh["conv"], 1):
        ghi_mang(f"{thu_muc}/conv{i}_w.txt", L["qw"])
        ghi_mang(f"{thu_muc}/conv{i}_b.txt", L["qb"])
        dong.append(f"conv{i}  s_w={L['sw']:.9g}  zp_vao={L['zp_vao']}  s_ra={L['s_ra']:.9g}  "
                    f"zp_ra={L['zp_ra']}  M={L['M']:.9g}  M0={L['M0']}  shift={L['shift']}")
    ghi_mang(f"{thu_muc}/fc_w.txt", mh["fc"]["qw"])
    ghi_mang(f"{thu_muc}/fc_b.txt", mh["fc"]["qb"])
    dong.append(f"fc  s_w={mh['fc']['sw']:.9g}  zp_vao={mh['fc']['zp_vao']}  "
                f"s_acc={mh['fc']['s_acc']:.9g}  (không requantize, argmax trên acc)")
    with open(f"{thu_muc}/tham_so.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(dong) + "\n")

    # Ảnh mẫu (test index 0) — đầu ra số nguyên từng lớp
    ghi = {}
    q_in = luong_tu_dau_vao(x_mau, mh["s_in"], mh["zp_in"])
    suy_luan_int(mh, q_in, ghi=ghi)
    ghi_mang(f"{thu_muc}/golden_int8/input.txt", q_in)
    for j, k in enumerate(THU_TU_GOLDEN, 1):
        ghi_mang(f"{thu_muc}/golden_int8/{j:02d}_{k}.txt", ghi[k])
    with open(f"{thu_muc}/golden_int8/label.txt", "w") as f:
        f.write(f"{int(nhan_mau)}\n")

    # Tự kiểm: đọc lại phải khớp từng số
    for i, L in enumerate(mh["conv"], 1):
        assert np.array_equal(np.loadtxt(f"{thu_muc}/conv{i}_w.txt", dtype=np.int64),
                              L["qw"].reshape(-1))
    return lines_readme(mh, THU_TU_GOLDEN, ghi, thu_muc)


def xuat_tham_chieu_test(mh, x_te, thu_muc=OUT_DIR):
    """10 acc lớp fc của cả 10.000 ảnh test → golden_int8/fc_acc_test.txt (10.000 dòng × 10 số).

    Golden C int8 so TỪNG BIT cả 10 số của từng ảnh, chặt hơn nhiều so với chỉ so dự đoán.
    """
    acc = np.concatenate([
        suy_luan_int(mh, luong_tu_dau_vao(x_te[k:k + 500], mh["s_in"], mh["zp_in"], mh["bits"]))
        for k in range(0, len(x_te), 500)])
    np.savetxt(f"{thu_muc}/golden_int8/fc_acc_test.txt", acc, fmt="%d")
    return acc


def lines_readme(mh, thu_tu, ghi, thu_muc=OUT_DIR):
    bang = "\n".join(f"| `{j:02d}_{k}.txt` | {list(ghi[k].shape[1:])} | {ghi[k][0].size} |"
                     for j, k in enumerate(thu_tu, 1))
    txt = f"""# {thu_muc}/ — tham số số nguyên của SmallCNN (sinh tự động, không sửa tay)

Sinh bởi `ptq_int8.py` (PTQ → `params_int8/`) hoặc `qat.py --bits 8 --xuat` (QAT → `params_int8_qat/`).
Cấu hình: ReLU **{mh['relu']}**, M0 **{mh['bits_M0']} bit**.

| File | Nội dung | Kiểu |
|---|---|---|
| `convN_w.txt` | trọng số [OC, IC, 3, 3] trải phẳng row-major | int8 đối xứng, zp = 0 |
| `convN_b.txt` | bias, scale = s_vào × s_w | int32 |
| `fc_w.txt`, `fc_b.txt` | [10, 144] và [10] | int8 / int32 |
| `tham_so.txt` | scale, zp, M0, shift từng lớp | |
| `golden_int8/` | ảnh test index 0, đầu ra số nguyên từng lớp | |
| `golden_int8/fc_acc_test.txt` | 10 acc lớp fc của cả 10.000 ảnh test (10.000 dòng × 10 số) | int32 |

Công thức mỗi lớp conv:

    acc  = bias + Σ (q_vào − zp_vào) · q_w          (int32, đệm 0 SAU khi trừ zp)
    q_ra = clamp(((acc · M0 + 2^(shift−1)) >> shift) + zp_ra, −128, 127)
    pool = max 2×2 trên q_ra

Lớp fc: `acc = bias + Σ (q − zp_vào) · q_w`, dự đoán = argmax(acc), so sánh chặt `>`.
Golden C int8 (`golden_c/golden_int8.c`) phải khớp **từng số** với `golden_int8/`.

| File golden | Shape | Số phần tử |
|---|---|---|
| `input.txt` | [1, 28, 28] | 784 |
{bang}
"""
    with open(f"{thu_muc}/README.md", "w", encoding="utf-8") as f:
        f.write(txt)


def doc_mo_hinh_tu_txt(thu_muc=OUT_DIR):
    """Dựng lại mô hình số nguyên từ file ĐÃ XUẤT: không calibrate lại, không cần checkpoint.

    Chỉ dùng cho int8 (tham_so.txt không ghi số bit, golden C cũng chỉ chạy int8).
    s_in trong file chỉ còn 9 chữ số có nghĩa — đủ, vì hàm gọi tự kiểm q_in = pixel − 128.
    """
    with open(f"{thu_muc}/tham_so.txt", encoding="utf-8") as f:
        txt = f.read()
    dong = {d.split()[0]: d for d in txt.splitlines() if d.strip() and not d.startswith("#")}

    def lay(ten_dong, khoa, kieu=int):
        return kieu(re.search(rf"\b{khoa}=(\S+)", dong[ten_dong]).group(1))

    mh = {"relu": re.search(r"relu=(\w+)", txt).group(1),
          "bits_M0": int(re.search(r"bits_M0=(\d+)", txt).group(1)), "bits": 8,
          "s_in": lay("input", "s", float), "zp_in": lay("input", "zp"), "conv": []}
    for i, (ic, oc) in enumerate([(1, 8), (8, 16), (16, 16)], 1):
        mh["conv"].append(dict(
            qw=np.loadtxt(f"{thu_muc}/conv{i}_w.txt", dtype=np.int64).reshape(oc, ic, 3, 3),
            qb=np.loadtxt(f"{thu_muc}/conv{i}_b.txt", dtype=np.int64),
            zp_vao=lay(f"conv{i}", "zp_vao"), zp_ra=lay(f"conv{i}", "zp_ra"),
            M0=lay(f"conv{i}", "M0"), shift=lay(f"conv{i}", "shift")))
    mh["fc"] = dict(qw=np.loadtxt(f"{thu_muc}/fc_w.txt", dtype=np.int64).reshape(10, 144),
                    qb=np.loadtxt(f"{thu_muc}/fc_b.txt", dtype=np.int64),
                    zp_vao=lay("fc", "zp_vao"))
    return mh


def chi_tham_chieu(thu_muc=OUT_DIR):
    """Sinh lại golden_int8/fc_acc_test.txt và README.md từ tham số ĐÃ XUẤT, không calibrate lại.

    Vì sao không chạy lại main(): calibrate dùng float của PyTorch, khác máy có thể lệch M0 vài
    đơn vị (đã gặp: conv3 M0 trên máy ảo Linux khác trên Mac). Bộ tham số đã chốt phải giữ nguyên.
    """
    mh = doc_mo_hinh_tu_txt(thu_muc)
    te_img, te_lbl = load_test_set()
    x_te = chuan_hoa(te_img)

    # Đầu vào: lượng tử hoá đầy đủ phải đúng bằng pixel − 128 (golden C dùng thẳng phép này)
    q_in = luong_tu_dau_vao(x_te, mh["s_in"], mh["zp_in"])
    so_khac = int((q_in != te_img.astype(np.int64)[:, None] - 128).sum())
    assert so_khac == 0, f"{so_khac} pixel có q_in ≠ pixel − 128: golden C sẽ sai ở đầu vào"

    # Ảnh mẫu dựng lại từ file phải khớp golden_int8/ đã có (kiểm doc_mo_hinh_tu_txt đọc đúng)
    ghi = {}
    suy_luan_int(mh, q_in[:1], ghi=ghi)
    khop = sum(np.array_equal(np.loadtxt(f"{thu_muc}/golden_int8/{j:02d}_{k}.txt", dtype=np.int64),
                              ghi[k].reshape(-1))
               for j, k in enumerate(THU_TU_GOLDEN, 1))
    assert khop == len(THU_TU_GOLDEN), "mô hình đọc từ file không tái tạo được golden_int8/"

    acc = xuat_tham_chieu_test(mh, x_te, thu_muc)
    lines_readme(mh, THU_TU_GOLDEN, ghi, thu_muc)
    print(f"{thu_muc}/: tham số đọc từ file, không calibrate lại")
    print(f"  Pixel test có q_in ≠ (pixel − 128): {so_khac} / {q_in.size}")
    print(f"  Ảnh mẫu dựng lại khớp golden_int8/: {khop}/{len(THU_TU_GOLDEN)} file")
    print(f"  INT8 trên 10.000 ảnh test: {(argmax_chat(acc) == te_lbl).mean() * 100:.2f}%")
    print(f"  Đã ghi {thu_muc}/golden_int8/fc_acc_test.txt và {thu_muc}/README.md")


# ------------------------------------------------------------
# [7] CHƯƠNG TRÌNH CHÍNH
# ------------------------------------------------------------
def nap_model():
    model = SmallCNN()
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()
    sd = {k: v.numpy().astype(np.float64) for k, v in model.state_dict().items()}
    return model, sd


def main():
    t0 = time.time()
    model, sd = nap_model()
    _, (va_img, va_lbl) = load_train_val()
    te_img, te_lbl = load_test_set()
    x_va, x_te = chuan_hoa(va_img), chuan_hoa(te_img)
    ket_qua = {}

    print("=" * 66)
    print("PTQ INT8 — SmallCNN (5.018 params), số nguyên thuần")
    print("=" * 66)

    pred_float_va = chay_float(model, x_va)
    acc_float_va = (pred_float_va == va_lbl).mean() * 100
    print(f"\nFloat32 trên val (10.000 ảnh): {acc_float_va:.2f}%")

    # ---- Thí nghiệm trên VAL ----
    dai_1000 = calibrate(model, x_va[:SO_ANH_CALIB])
    dai_100 = calibrate(model, x_va[:100])
    cau_hinh = [
        ("A. Slide Thầy: ReLU riêng, calib 1000, M0 31 bit", dai_1000, "rieng", 31),
        ("B. ReLU gộp vào kẹp (zp=-128), calib 1000, M0 31 bit", dai_1000, "gop", 31),
        ("C. Như B, calib chỉ 100 ảnh", dai_100, "gop", 31),
        ("D. Như B, M0 chỉ 15 bit (vừa DSP nhỏ)", dai_1000, "gop", 15),
        ("E. Như B, M0 chỉ 8 bit", dai_1000, "gop", 8),
        ("F. Thử int4 (W4A4): như B nhưng 4 bit", dai_1000, "gop", 31, 4),
    ]
    print("\nSo cấu hình trên VAL:")
    print(f"{'Cấu hình':58s} {'Acc':>7s} {'Khác float':>11s}")
    ket_qua["val"] = []
    for ten, dai, relu, bits_M0, *b in cau_hinh:
        mh = dung_mo_hinh_int(sd, dai, relu, bits_M0, b[0] if b else 8)
        pred = chay_int(mh, x_va)
        acc = (pred == va_lbl).mean() * 100
        khac = int((pred != pred_float_va).sum())
        print(f"{ten:58s} {acc:6.2f}% {khac:7d} ảnh")
        ket_qua["val"].append(dict(ten=ten, acc=round(acc, 2), khac_float=khac))
    ket_qua["float_val"] = round(acc_float_va, 2)

    # ---- Cấu hình chọn: B ----
    mh = dung_mo_hinh_int(sd, dai_1000, "gop", 31)

    print("\nDải calibrate (1000 ảnh val) và tham số requantize — cấu hình B:")
    print(f"  input: dải [{dai_1000['input'][0]:.4f}, {dai_1000['input'][1]:.4f}]  "
          f"s={mh['s_in']:.6f}  zp={mh['zp_in']}")
    ket_qua["lop"] = []
    for i, L in enumerate(mh["conv"], 1):
        lo, hi = dai_1000[f"conv{i}"]
        print(f"  conv{i}: trước ReLU [{lo:.3f}, {hi:.3f}]  s_w={L['sw']:.6f}  s_ra={L['s_ra']:.6f}  "
              f"zp_ra={L['zp_ra']}  M={L['M']:.6f} = {L['M0']}·2^-{L['shift']}")
        ket_qua["lop"].append(dict(lop=f"conv{i}", lo=round(lo, 3), hi=round(hi, 3),
                                   s_w=L["sw"], s_ra=L["s_ra"], zp_ra=L["zp_ra"],
                                   M=L["M"], M0=L["M0"], shift=L["shift"]))

    # ---- Đầu vào: lượng tử hoá có bằng đúng (pixel − 128) không? ----
    q_in = luong_tu_dau_vao(x_te, mh["s_in"], mh["zp_in"])
    p_tru_128 = te_img.astype(np.int64)[:, None] - 128
    so_khac = int((q_in != p_tru_128).sum())
    print(f"\nĐầu vào: s_in = {mh['s_in']:.8f}  (1/(255·std) = {1/(255*STD):.8f}),  zp_in = {mh['zp_in']}")
    print(f"  Số pixel test có q_in ≠ (pixel − 128): {so_khac} / {q_in.size}")
    ket_qua["dau_vao"] = dict(s_in=mh["s_in"], zp_in=mh["zp_in"], so_pixel_khac=so_khac,
                              tong_pixel=int(q_in.size))

    # ---- TEST: chạy đúng 1 lần cho cấu hình B ----
    thong_ke = {}
    pred_float_te = chay_float(model, x_te)
    pred_int_te = chay_int(mh, x_te, thong_ke=thong_ke)
    acc_f = (pred_float_te == te_lbl).mean() * 100
    acc_q = (pred_int_te == te_lbl).mean() * 100
    khac = int((pred_int_te != pred_float_te).sum())
    dung_thanh_sai = int(((pred_float_te == te_lbl) & (pred_int_te != te_lbl)).sum())
    sai_thanh_dung = int(((pred_float_te != te_lbl) & (pred_int_te == te_lbl)).sum())
    print(f"\nTEST (10.000 ảnh) — cấu hình B:")
    print(f"  Float32: {acc_f:.2f}%   INT8: {acc_q:.2f}%   chênh {acc_q-acc_f:+.2f} điểm %")
    print(f"  Dự đoán khác float: {khac} ảnh  (float đúng→int8 sai: {dung_thanh_sai}, "
          f"float sai→int8 đúng: {sai_thanh_dung})")
    ket_qua["test"] = dict(float=round(acc_f, 2), int8=round(acc_q, 2), khac=khac,
                           dung_thanh_sai=dung_thanh_sai, sai_thanh_dung=sai_thanh_dung)

    # ---- Accumulator cần bao nhiêu bit ----
    print("\nAccumulator lớn nhất trên 10.000 ảnh test (|acc|) → số bit có dấu cần:")
    ket_qua["acc_bit"] = {}
    for k, v in thong_ke.items():
        bit = int(v).bit_length() + 1
        print(f"  {k:6s}: {v:>9,d}  → {bit} bit")
        ket_qua["acc_bit"][k] = dict(max=v, bit=bit)

    # ---- Bộ nhớ tham số ----
    n_w = sum(L["qw"].size for L in mh["conv"]) + mh["fc"]["qw"].size
    n_b = sum(L["qb"].size for L in mh["conv"]) + mh["fc"]["qb"].size
    byte_f = (n_w + n_b) * 4
    byte_q = n_w * 1 + n_b * 4
    print(f"\nBộ nhớ tham số: float32 {byte_f:,} byte  →  int8/int32 {byte_q:,} byte  "
          f"(nhỏ hơn {byte_f/byte_q:.2f} lần; chưa tính vài hằng số M0/shift/zp)")
    ket_qua["bo_nho"] = dict(n_w=int(n_w), n_b=int(n_b), float_byte=int(byte_f), int_byte=int(byte_q))

    # ---- Xuất tham số + tham chiếu 10.000 ảnh cho golden C ----
    xuat_tham_so(mh, x_te[:1], te_lbl[0])
    xuat_tham_chieu_test(mh, x_te)
    with open(f"{OUT_DIR}/ket_qua_ptq.json", "w", encoding="utf-8") as f:
        json.dump(ket_qua, f, ensure_ascii=False, indent=2)
    print(f"\nĐã ghi {OUT_DIR}/  ({time.time()-t0:.0f} giây)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chi-tham-chieu", action="store_true",
                    help="không calibrate lại: đọc params_int8/ đã xuất, chỉ sinh lại "
                         "golden_int8/fc_acc_test.txt và README.md")
    if ap.parse_args().chi_tham_chieu:
        chi_tham_chieu()
    else:
        main()
