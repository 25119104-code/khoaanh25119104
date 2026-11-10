# ============================================================
# qat.py — Quantization-Aware Training (fake-quant + STE)
# ============================================================
# Ý tưởng: lúc train, forward "giả vờ" lượng tử (làm tròn trọng số và
# activation về lưới int rồi đổi lại float), để model học cách chịu sai số
# làm tròn. Backward dùng STE (Straight-Through Estimator): coi phép làm tròn
# như phép đồng nhất, vì đạo hàm thật của round() bằng 0 gần như khắp nơi.
#
#     x_fq = x + (lượng_tử(x) − x).detach()
#        forward : ra đúng lượng_tử(x)
#        backward: đạo hàm = 1, gradient chảy qua như không có round
#
# Sơ đồ lượng tử Y HỆT ptq_int8.py (cấu hình B): weight đối xứng per-tensor,
# activation sau ReLU dải [0, max] → zp = qmin. Train xong, đưa trọng số và
# dải activation học được vào ĐÚNG đường số nguyên của ptq_int8.py để đo —
# không đo trên model fake-quant, vì chip chạy số nguyên chứ không chạy fake-quant.
#
# Chạy:  python qat.py --bits 8      (sản phẩm chính)
#        python qat.py --bits 4      (thí nghiệm: QAT có ích khi nào)
#        python qat.py --bits 8 --xuat
#            Không train: nạp models/small_cnn_qat_int8.pth đã train, xuất tham số số
#            nguyên ra params_int8_qat/ (cùng định dạng params_int8/) cho golden C.
# ============================================================

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import SmallCNN
from mnist_io import load_test_set
import ptq_int8 as ptq


class QATSmallCNN(SmallCNN):
    """SmallCNN + fake-quant. Cùng tên tham số → nạp thẳng small_cnn.pth được."""

    def __init__(self, bits, dai_vao, hi_dau):
        super().__init__()
        self.bits = bits
        self.qmin, self.qmax = ptq.gioi_han(bits)
        self.dai_vao = dai_vao
        # max sau ReLU của 3 lớp conv, cập nhật bằng trung bình trượt (EMA) lúc train
        self.register_buffer("hi", torch.tensor(hi_dau, dtype=torch.float32))
        self.momentum = 0.01

    def fq_w(self, w):
        """Fake-quant trọng số đối xứng, scale = max|w| / qmax (tính lại mỗi bước)."""
        s = w.abs().max().detach() / self.qmax
        q = torch.clamp(torch.floor(w / s + 0.5), -self.qmax, self.qmax) * s
        return w + (q - w).detach()

    def fq_a(self, x, lo, hi):
        s, zp = ptq.tham_so_bat_doi_xung(lo, hi, self.bits)
        q = (torch.clamp(torch.floor(x / s + 0.5) + zp, self.qmin, self.qmax) - zp) * s
        return x + (q - x).detach()

    def forward(self, x):
        x = self.fq_a(x, *self.dai_vao)
        for i, conv in enumerate([self.conv1, self.conv2, self.conv3]):
            r = torch.relu(F.conv2d(x, self.fq_w(conv.weight), conv.bias, padding=1))
            if self.training:
                with torch.no_grad():
                    self.hi[i] = (1 - self.momentum) * self.hi[i] + self.momentum * r.max()
            r = self.fq_a(r, 0.0, float(self.hi[i]))
            x = F.max_pool2d(r, 2)
        return F.linear(x.flatten(1), self.fq_w(self.fc.weight), self.fc.bias)

    def sang_so_nguyen(self):
        """Lấy trọng số + dải đã học → mô hình số nguyên của ptq_int8."""
        sd = {k: v.detach().numpy().astype(np.float64)
              for k, v in self.state_dict().items() if k != "hi"}
        dai = {"input": self.dai_vao}
        for i in range(3):
            dai[f"relu{i+1}"] = (0.0, float(self.hi[i]))
        return ptq.dung_mo_hinh_int(sd, dai, relu="gop", bits_M0=31, bits=self.bits)


def acc_int(mh, x, y):
    return (ptq.chay_int(mh, x) == y).mean() * 100


def xuat_qat(bits, thu_muc="params_int8_qat"):
    """Không train. Nạp checkpoint QAT tốt nhất đã lưu → tham số số nguyên → thu_muc/.

    Cùng định dạng params_int8/ nên golden C chạy được ngay:
        ./golden_c/golden_int8 10000 params_int8_qat
    """
    if bits != 8:
        raise SystemExit("Golden C chỉ chạy int8, nên chỉ xuất --bits 8")
    model_f, _ = ptq.nap_model()
    _, (va_img, _) = ptq.load_train_val()
    te_img, te_lbl = load_test_set()
    x_va, x_te = ptq.chuan_hoa(va_img), ptq.chuan_hoa(te_img)
    # Dải đầu vào lấy đúng như lúc train (calibrate PTQ trên 1000 ảnh val)
    dai = ptq.calibrate(model_f, x_va[:ptq.SO_ANH_CALIB])
    model = QATSmallCNN(bits, dai["input"], [0.0, 0.0, 0.0])
    model.load_state_dict(torch.load(f"models/small_cnn_qat_int{bits}.pth"))   # có cả buffer hi
    model.eval()
    mh = model.sang_so_nguyen()
    ptq.xuat_tham_so(mh, x_te[:1], te_lbl[0], thu_muc, nguon=f"qat.py --bits {bits} --xuat")
    acc = ptq.xuat_tham_chieu_test(mh, x_te, thu_muc)
    a = (ptq.argmax_chat(acc) == te_lbl).mean() * 100
    print(f"QAT int{bits} số nguyên, test 10.000 ảnh: {a:.2f}%  "
          f"(phải bằng số 'qat' trong params_int8/ket_qua_qat_int{bits}.json)")
    print(f"Dải sau ReLU học được: {[round(float(h), 3) for h in model.hi]}")
    print(f"Đã ghi {thu_muc}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--xuat", action="store_true",
                    help="không train: xuất tham số số nguyên của checkpoint QAT ra params_int8_qat/")
    args = ap.parse_args()
    if args.xuat:
        xuat_qat(args.bits)
        return
    torch.manual_seed(ptq.SEED)
    t0 = time.time()

    model_f, sd_f = ptq.nap_model()
    (tr_img, tr_lbl), (va_img, va_lbl) = ptq.load_train_val()
    te_img, te_lbl = load_test_set()
    x_tr, x_va, x_te = ptq.chuan_hoa(tr_img), ptq.chuan_hoa(va_img), ptq.chuan_hoa(te_img)
    y_tr = torch.from_numpy(tr_lbl.astype(np.int64))

    # Khởi đầu: dải lấy từ calibrate PTQ (1000 ảnh val), trọng số từ small_cnn.pth
    dai = ptq.calibrate(model_f, x_va[:ptq.SO_ANH_CALIB])
    mh_ptq = ptq.dung_mo_hinh_int(sd_f, dai, "gop", 31, args.bits)
    acc_ptq_va = acc_int(mh_ptq, x_va, va_lbl)

    model = QATSmallCNN(args.bits, dai["input"], [dai[f"relu{i}"][1] for i in (1, 2, 3)])
    model.load_state_dict(model_f.state_dict(), strict=False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    print("=" * 62)
    print(f"QAT int{args.bits} — fine-tune từ small_cnn.pth, {args.epochs} epoch, lr {args.lr}")
    print("=" * 62)
    print(f"Epoch 0 (= PTQ int{args.bits}) | val acc số nguyên {acc_ptq_va:.2f}%")

    best, best_ep, lich_su = acc_ptq_va, 0, [(0, None, round(acc_ptq_va, 2))]
    ckpt = f"models/small_cnn_qat_int{args.bits}.pth"
    torch.save(model.state_dict(), ckpt)
    x_tr_t = torch.from_numpy(x_tr)
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(x_tr_t))
        tong = 0.0
        for k in range(0, len(perm), 64):
            idx = perm[k:k + 64]
            loss = loss_fn(model(x_tr_t[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tong += loss.item() * len(idx)
        model.eval()
        a = acc_int(model.sang_so_nguyen(), x_va, va_lbl)
        moi = ""
        if a > best:
            best, best_ep, moi = a, ep, "  ← tốt nhất, lưu"
            torch.save(model.state_dict(), ckpt)
        lich_su.append((ep, round(tong / len(perm), 4), round(a, 2)))
        print(f"Epoch {ep} | loss {tong/len(perm):.4f} | val acc số nguyên {a:.2f}%{moi}  "
              f"({time.time()-t0:.0f}s)")

    # Nạp epoch tốt nhất (epoch 0 = giữ nguyên PTQ nếu QAT không hơn)
    model.load_state_dict(torch.load(ckpt))
    mh_qat = model.sang_so_nguyen()

    # TEST — chạy 1 lần
    pred_f = ptq.chay_float(model_f, x_te)
    pred_p = ptq.chay_int(mh_ptq, x_te)
    pred_q = ptq.chay_int(mh_qat, x_te)
    a_f, a_p, a_q = [(p == te_lbl).mean() * 100 for p in (pred_f, pred_p, pred_q)]
    print(f"\nTEST 10.000 ảnh: float32 {a_f:.2f}% | PTQ int{args.bits} {a_p:.2f}% | "
          f"QAT int{args.bits} {a_q:.2f}% (epoch {best_ep})")
    print(f"Dải sau ReLU học được: {[round(float(h), 3) for h in model.hi]} "
          f"(PTQ calibrate: {[round(dai[f'relu{i}'][1], 3) for i in (1, 2, 3)]})")

    kq = dict(bits=args.bits, epochs=args.epochs, lr=args.lr, lich_su=lich_su, best_epoch=best_ep,
              val=dict(ptq=round(acc_ptq_va, 2), qat=round(best, 2)),
              test=dict(float=round(a_f, 2), ptq=round(a_p, 2), qat=round(a_q, 2)),
              hi_qat=[float(h) for h in model.hi],
              hi_ptq=[dai[f"relu{i}"][1] for i in (1, 2, 3)],
              giay=round(time.time() - t0))
    with open(f"params_int8/ket_qua_qat_int{args.bits}.json", "w", encoding="utf-8") as f:
        json.dump(kq, f, ensure_ascii=False, indent=2)
    print(f"Đã ghi {ckpt} và params_int8/ket_qua_qat_int{args.bits}.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
