# ============================================================
# export_params.py — Phase 2B: trích tham số SmallCNN ra file .txt
# ============================================================
# Chạy:  python export_params.py            (mặc định ảnh test số 0)
#        python export_params.py --idx 5    (lấy ảnh test số 5 làm mẫu)
#
# Sinh ra:
#   params/weights.txt   4.968 số — TẤT CẢ trọng số, 4 lớp nối tiếp nhau
#   params/biases.txt       50 số — TẤT CẢ bias, cùng thứ tự lớp
#   params/README.md     bảng offset: lớp nào nằm từ dòng nào tới dòng nào
#
#   golden/input.txt           784 số — ảnh mẫu ĐÃ chuẩn hoá
#   golden/label.txt           nhãn đúng của ảnh mẫu
#   golden/NN_<lớp>.txt        đầu ra TỪNG LỚP của PyTorch cho ảnh mẫu
#   golden/logits_pytorch.txt  10.000 dòng x 10 logit — cả test set
#   golden/README.md           shape của từng file
#
# Vì sao 2 file (theo Thầy) mà không 8 file?
#   Trên FPGA, trọng số nằm trong MỘT bộ nhớ (ROM/BRAM). Mỗi lớp chỉ là
#   một "địa chỉ gốc" (base address) trong bộ nhớ đó. Đọc 1 file tuần tự
#   + bảng offset chính là mô phỏng cách phần cứng truy cập bộ nhớ.
#
# Định dạng: 1 số / dòng, in "%.9g".
#   9 chữ số có nghĩa là TỐI THIỂU để float32 -> chữ -> float32 ra đúng
#   từng bit. In "%.4f" thì C sẽ lệch PyTorch ngay từ bước đọc file.
#   Cuối script có bước TỰ KIỂM: đọc lại file, so bit-for-bit.
# ============================================================

import argparse
import os

import numpy as np
import torch

from models import SmallCNN
from mnist_io import load_test_set, normalize

CKPT = "models/small_cnn.pth"
PARAM_DIR = "params"
GOLDEN_DIR = "golden"

# Thứ tự lớp trong file — C đọc theo ĐÚNG thứ tự này. Đổi ở đây = đổi ở C.
LAYERS = ["conv1", "conv2", "conv3", "fc"]


def write_floats(path, arr):
    """Trải phẳng theo thứ tự C (row-major) rồi ghi 1 số / dòng."""
    flat = np.ascontiguousarray(arr, dtype=np.float32).ravel()
    with open(path, "w") as f:
        for v in flat:
            f.write("%.9g\n" % v)
    return flat.size


def read_floats(path):
    with open(path) as f:
        return np.array([float(line) for line in f], dtype=np.float32)


def export_params(sd):
    os.makedirs(PARAM_DIR, exist_ok=True)

    # Ghép weight các lớp nối tiếp nhau. PyTorch lưu conv là
    # [out_ch][in_ch][ky][kx], fc là [out][in] — giữ NGUYÊN thứ tự này.
    w_parts, b_parts, rows = [], [], []
    w_off = b_off = 0
    for name in LAYERS:
        w = sd[f"{name}.weight"].numpy()
        b = sd[f"{name}.bias"].numpy()
        w_parts.append(w.ravel())
        b_parts.append(b.ravel())
        rows.append((name, list(w.shape), w_off, w.size, b_off, b.size))
        w_off += w.size
        b_off += b.size

    n_w = write_floats(f"{PARAM_DIR}/weights.txt", np.concatenate(w_parts))
    n_b = write_floats(f"{PARAM_DIR}/biases.txt", np.concatenate(b_parts))

    # README bảng offset — C sẽ chép bảng này thành hằng số.
    lines = [
        "# params/ — tham số SmallCNN cho golden model C",
        "",
        f"Sinh bởi `export_params.py` từ `{CKPT}`. **Không sửa tay.**",
        "",
        "- `weights.txt`: %d số, `biases.txt`: %d số, tổng **%d** params." % (n_w, n_b, n_w + n_b),
        "- 1 số float32 / dòng, định dạng `%.9g` (đọc lại ra đúng từng bit).",
        "- Thứ tự trong mỗi lớp giữ nguyên PyTorch: conv `[out_ch][in_ch][ky][kx]`, fc `[out][in]`.",
        "",
        "| Lớp | Shape weight | Offset weight | Số weight | Offset bias | Số bias |",
        "|---|---|---|---|---|---|",
    ]
    for name, shape, wo, wn, bo, bn in rows:
        lines.append(f"| `{name}` | {shape} | {wo} | {wn} | {bo} | {bn} |")
    lines += [
        "",
        "Offset tính từ 0. Ví dụ phần tử `conv2.weight[oc][ic][ky][kx]` nằm ở dòng (tính từ 0):",
        "",
        "```",
        "72 + ((oc*8 + ic)*3 + ky)*3 + kx",
        "```",
    ]
    with open(f"{PARAM_DIR}/README.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # TỰ KIỂM: đọc lại phải giống bit-for-bit
    for fname, parts in [("weights.txt", w_parts), ("biases.txt", b_parts)]:
        back = read_floats(f"{PARAM_DIR}/{fname}")
        orig = np.concatenate(parts).astype(np.float32)
        same = np.array_equal(back.view(np.uint32), orig.view(np.uint32))
        print(f"  {fname:12s} {back.size:5d} số   đọc lại khớp từng bit: {same}")
        assert same, f"{fname} không khớp — kiểm tra định dạng in"
    return rows


def forward_by_layer(model, x):
    """Chạy từng bước, trả về list (tên, tensor) — để C so sánh TỪNG LỚP.

    Không dùng hook vì SmallCNN dùng chung 1 module relu/pool cho 3 lần.
    """
    out = []
    for i, conv in enumerate([model.conv1, model.conv2, model.conv3], start=1):
        x = conv(x);        out.append((f"conv{i}", x))
        x = model.relu(x);  out.append((f"relu{i}", x))
        x = model.pool(x);  out.append((f"pool{i}", x))
    x = x.flatten(1);       out.append(("flatten", x))
    x = model.fc(x);        out.append(("fc_logits", x))
    return out


def export_golden(model, images, labels, idx):
    os.makedirs(GOLDEN_DIR, exist_ok=True)

    x = normalize(images[idx])                      # float32 [28,28]
    write_floats(f"{GOLDEN_DIR}/input.txt", x)
    with open(f"{GOLDEN_DIR}/label.txt", "w") as f:
        f.write(f"{labels[idx]}\n")

    readme = [
        "# golden/ — đầu ra chuẩn của PyTorch để đối chiếu",
        "",
        f"Ảnh mẫu: test set index **{idx}**, nhãn đúng **{labels[idx]}**. Sinh bởi `export_params.py`.",
        "",
        "Mỗi file: 1 số / dòng, trải phẳng row-major theo shape (bỏ chiều batch).",
        "",
        "| File | Shape | Số phần tử |",
        "|---|---|---|",
        f"| `input.txt` | [1, 28, 28] | 784 |",
    ]
    with torch.no_grad():
        t = torch.from_numpy(x).reshape(1, 1, 28, 28)
        for k, (name, y) in enumerate(forward_by_layer(model, t), start=1):
            y = y[0].numpy()
            fname = f"{k:02d}_{name}.txt"
            write_floats(f"{GOLDEN_DIR}/{fname}", y)
            readme.append(f"| `{fname}` | {list(y.shape)} | {y.size} |")
        pred = int(y.argmax())
    print(f"  ảnh mẫu idx={idx}: nhãn {labels[idx]}, PyTorch đoán {pred}")

    # Logit cả test set — để đo accuracy và so từng ảnh với bản Python/C
    with torch.no_grad():
        xs = torch.from_numpy(np.stack([normalize(im) for im in images])).unsqueeze(1)
        logits = model(xs).numpy()
    with open(f"{GOLDEN_DIR}/logits_pytorch.txt", "w") as f:
        for row in logits:
            f.write(" ".join("%.9g" % v for v in row) + "\n")
    acc = (logits.argmax(1) == labels).mean() * 100
    readme += [
        f"| `logits_pytorch.txt` | [{len(images)}, 10] | 1 dòng / ảnh, 10 số cách nhau dấu cách |",
        "",
        f"Accuracy PyTorch trên {len(images)} ảnh test: **{acc:.2f}%**.",
    ]
    with open(f"{GOLDEN_DIR}/README.md", "w", encoding="utf-8") as f:
        f.write("\n".join(readme) + "\n")
    print(f"  logits_pytorch.txt: {len(images)} ảnh, accuracy {acc:.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=0, help="index ảnh test làm mẫu")
    args = ap.parse_args()

    model = SmallCNN()
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()
    sd = model.state_dict()

    print("[1] Trích tham số ->", PARAM_DIR + "/")
    rows = export_params(sd)
    for name, shape, wo, wn, bo, bn in rows:
        print(f"  {name:6s} weight {str(shape):16s} offset {wo:5d} (+{wn:4d})   bias offset {bo:2d} (+{bn})")

    print("[2] Đầu ra chuẩn PyTorch ->", GOLDEN_DIR + "/")
    images, labels = load_test_set()
    export_golden(model, images, labels, args.idx)


if __name__ == "__main__":
    main()
