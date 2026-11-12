# Chép tay — INT8 mục 5–7: PTQ, QAT và Golden C int8

Đi tiếp `docs/chep-tay-int8-muc1-4.md`. Đi kèm `docs/on-tap-int8-ptq-qat-c.md` (mục 5–7).
Mọi số lấy từ code và file kết quả thật (`params_int8/ket_qua_*.json`), đã đọc lại 11/11/2026.
Mỗi bước có dòng **✍️ Chép** ở cuối — chỉ chép các dòng đó.

**Cần nắm trước:** công thức một lớp conv của file mục 1–4:

    acc   = q_b + Σ (q_in − zp_in) · q_w
    q_out = kẹp( ((acc · M0 + 2^(shift−1)) >> shift) + zp_out,  −128, 127 )

---

## Bước 0 — Mục tiêu và bức tranh chung

Mục 1–4 dạy **một lớp** chạy bằng số nguyên. Mục 5–7 trả lời ba câu tiếp theo:

| Mục | Câu hỏi | Trả lời ngắn | File |
|---|---|---|---|
| 5. PTQ | Lấy **cả model** đã train, ra bộ tham số int8 thế nào? | Calibrate → quantize → chạy thử → xuất file. **Không train lại** | `ptq_int8.py` |
| 6. QAT | Nếu int8 làm tụt accuracy thì sao? | Train tiếp vài epoch với fake-quant, để model **quen sai số làm tròn** | `qat.py` |
| 7. Golden C | Lấy gì làm chuẩn để kiểm RTL? | Bản C số nguyên thuần, **khớp Python từng bit** | `golden_c/golden_int8.c` |

Luồng file:

    models/small_cnn.pth (float32, 98,82%)
        │
        ├── ptq_int8.py ───────────────────────────────→ params_int8/      ─┐
        │   (calibrate + quantize, không train)                             │
        │                                                                   ├→ golden_int8.c → (RTL)
        └── qat.py (fine-tune fake-quant) → small_cnn_qat_int8.pth          │
                └── qat.py --xuat ─────────────────────→ params_int8_qat/ ─┘

✍️ **Chép:** PTQ = quantize model đã train, không train lại. QAT = train tiếp với fake-quant để model quen sai số.
Golden C = bản C số nguyên làm chuẩn kiểm RTL. Cả hai bộ tham số dùng chung một golden C.

---

# PHẦN A — PTQ (mục 5, `ptq_int8.py`)

## Bước A1 — PTQ là gì

**P**ost-**T**raining **Q**uantization: lượng tử **sau khi train xong**. Thực chất là **áp mục 1–4 cho cả model**:
tính thước (s, zp) cho mọi tensor, quantize trọng số + bias, rồi chạy công thức một lớp cho conv1, conv2, conv3, fc.

Theo cách chia của video: project là **PTQ static** (cả weight và activation int8, cần dữ liệu calibrate)
và **integer-only** (không còn float trên chip).

✍️ **Chép:** PTQ = áp mục 1–4 cho cả model sau khi train. Project: PTQ static + integer-only.

---

## Bước A2 — 6 bước trong code

| # | Làm gì | Hàm / dòng trong `ptq_int8.py` |
|---|---|---|
| 1 | Nạp model float `small_cnn.pth` | `nap_model()` |
| 2 | **Calibrate**: chạy 1.000 ảnh **val**, đo min/max đầu vào, trước ReLU, sau ReLU của 3 lớp | `calibrate()` dòng 164, gọi ở dòng 469 |
| 3 | Từ min/max → s, zp. Quantize trọng số (đối xứng), bias (int32). Tính M, M0, shift | `dung_mo_hinh_int()` dòng 177 |
| 4 | Chạy inference **số nguyên thuần** (tự viết) | `suy_luan_int()` dòng 245 |
| 5 | So **6 cấu hình** trên **val**, chọn 1. Chạy **test đúng 1 lần** | `main()` dòng 468–530 |
| 6 | Xuất `params_int8/`: trọng số, bias, `tham_so.txt`, ảnh mẫu từng lớp, tham chiếu 10.000 ảnh | `xuat_tham_so()`, `xuat_tham_chieu_test()` |

Hai quy tắc:
- **Calibrate trên val, không dùng test.** Calibrate cũng là ra quyết định → dùng test là data leakage.
- **Không dùng `torch.ao.quantization`.** Đó là hộp đen, không cho số nguyên từng lớp → không có gì để so với C và RTL.

✍️ **Chép:** PTQ: nạp model → calibrate 1.000 ảnh val → tính s, zp, M0 → chạy số nguyên → chọn cấu hình trên val
→ test 1 lần → xuất params_int8/. Tự viết, không dùng torch.ao (hộp đen, không so được với C/RTL).

---

## Bước A3 — So 6 cấu hình: mỗi cái để trả lời một câu hỏi

| Cấu hình (val) | Acc | Khác float | Câu hỏi nó trả lời | Kết luận |
|---|---|---|---|---|
| A. ReLU riêng (đúng slide Thầy), M0 31 bit | 98,78% | 9 | Mốc chuẩn | — |
| **B. ReLU gộp vào kẹp**, M0 31 bit | 98,77% | 9 | Gộp ReLU có mất gì không? | Chỉ chênh 1 ảnh → **chọn B**, bỏ được bộ so sánh |
| C. Như B, calibrate 100 ảnh | 98,76% | 9 | Cần bao nhiêu ảnh calibrate? | 100 ảnh cũng đủ |
| D. Như B, M0 15 bit | 98,77% | 9 | M0 ngắn hơn được không? | 15 bit = 31 bit → vừa 1 DSP48 (25 × 18) |
| E. Như B, M0 8 bit | 98,76% | 11 | Ngắn nữa thì sao? | Bắt đầu lệch |
| F. int4 | 97,69% | 171 | Xuống 4 bit thì sao? | Tụt rõ (~1,1 điểm) → chỗ QAT có ích |

Float trên val: 98,79%.

✍️ **Chép:** Chọn B (ReLU gộp vào kẹp): mất 1 ảnh so với A mà bỏ được bộ so sánh. M0 15 bit đủ (vừa 1 DSP).
100 ảnh calibrate đủ. int4 tụt rõ.

---

## Bước A4 — Kết quả test (chạy 1 lần)

| | Float32 | PTQ int8 |
|---|---|---|
| Test acc | 98,82% | **98,84%** |

Dự đoán khác float **9/10.000 ảnh**: 3 ảnh float đúng → int8 sai, 5 ảnh float sai → int8 đúng,
1 ảnh cả hai cùng sai nhưng đoán ra số khác nhau.

Đọc cho đúng: int8 cao hơn 0,02 điểm (= 2 ảnh) **không có nghĩa int8 tốt hơn** — chỉ là dao động do làm tròn.
Kết luận đúng là: **int8 không làm tụt accuracy**.

✍️ **Chép:** PTQ int8 test 98,84% (float 98,82%), khác float 9 ảnh. Cao hơn 2 ảnh là dao động, kết luận = không tụt.

---

## Bước A5 — Integer-only: chip chỉ còn số nguyên

**1. Định nghĩa.** Inference chỉ dùng **nhân, cộng, dịch bit, kẹp**. Không float, kể cả bước chuyển lớp.
Số thực (s, M) chỉ dùng trên máy tính **trước khi** lên chip. Bằng chứng: golden C không có `math.h`.

| Kiểu | Lưu | Tính lúc chạy | Có float lúc chạy? |
|---|---|---|---|
| Weight-only | Trọng số int8 | Đổi ra float rồi nhân float | Có |
| int8, đổi thước bằng float | int8 | Nhân int8, nhưng `acc × s_in·s_w/s_out` bằng số thực | Có, ở bước chuyển lớp |
| **Integer-only** (project) | int8 | Nhân int8, cộng int32, chuyển lớp bằng M0 + dịch | **Không** |

**2. Mấu chốt ở bước chuyển lớp.** M = s_in·s_w/s_out là số thực → **M ≈ M0 / 2^shift**
(conv1: 2.094.719.601 / 2^39). Nhân M0 rồi dịch phải là xong.

**3. Cần PTQ static.** Thước activation tính sẵn bằng calibrate, cố định lúc chạy.

| | Static (project) | Dynamic |
|---|---|---|
| Thước activation tính lúc nào | Trước khi chạy (calibrate 1.000 ảnh val) | Lúc chạy, cho từng ảnh |
| Chip phải làm thêm | Không | Tìm min/max, chia số thực, tính M0 |
| Hợp FPGA | **Có** | Không |
| Đánh đổi | Giá trị vượt dải calibrate bị kẹp | — |

**4. Activation = output các lớp** (dữ liệu chạy qua mạng), khác trọng số (cố định).
Activation function là hàm (ReLU). Maxpool là pooling, không phải activation function, nhưng output của nó vẫn là
activation và dùng chung thước với output ReLU.
Cùng một activation: đường float lưu 1,8441, đường int8 lưu −47, nối bằng r = s·(q − zp).

**5. Kiểu số trên đường INT8 — ReLU nằm trong bước kẹp**

    pixel → q = pixel − 128                              (int8)
    lặp 3 lần:
        conv:       acc = bias + Σ x·w                    (int32)
        requantize: ((acc·M0 + 2^38) >> 39) + zp_out
        kẹp [−128, 127]  ← ReLU nằm ở đây                 (int8)
        maxpool                                           (int8)
    fc: acc = bias + Σ x·w                                (int32)
    argmax                                                (0–9)

- Giữa các lớp là **int8**; bên trong mỗi lớp acc là **int32**; fc giữ int32 rồi argmax.
- **Phải có ngoặc** quanh `(… >> 39)` trước khi cộng zp_out: trong C và Verilog, `+` ưu tiên hơn `>>`,
  viết `… >> 39 + zp_out` sẽ thành dịch `39 + zp_out` bit — sai.
- Cả 3 lớp conv đều shift = 39, zp_out = −128; chỉ M0 khác.

**Vì sao kẹp = ReLU:** zp_out = −128 = số thực 0 ở đáy thước → acc âm sau requantize ra số < −128 → kẹp lên −128 = 0.
Ví dụ: acc −4.247 → −16 − 128 = −144 → **−128**.

| | Cấu hình A (slide Thầy) | Cấu hình B (đang dùng) |
|---|---|---|
| ReLU | Bước riêng `q = max(q, zp)` | Gộp vào kẹp |
| Phần cứng | Thêm bộ so sánh | Không tốn thêm |
| Val acc | 98,78% | 98,77% |

**6. Số tích = kích thước kernel × số kênh vào** (= số phép nhân để ra 1 điểm output)

    conv1: 3 × 3 × 1  = 9      (ảnh xám 1 kênh)
    conv2: 3 × 3 × 8  = 72     (8 feature map của conv1)
    conv3: 3 × 3 × 16 = 144
    fc:    144 tích / 1 output

Mọi tích cộng chung vào **một** acc → mỗi điểm output vẫn chỉ ra một số.

**7. Độ rộng bit**

| | Bit |
|---|---|
| x conv1 / các lớp sau | 9 có dấu ([−33, 222]) / 8 không dấu ([0, 255]) |
| Trọng số | 8 ([−127, 127]) |
| Một tích | ≤ 16 (255 × 127 = 32.385) |
| **acc** | **20** |
| acc × M0 | 51 → int64 trong C |

**8. Vì sao acc 20 bit** (hàm `chan_tren` trong `golden_int8.c`): mỗi trọng số ghép với đầu mút x cho tích lớn nhất
(trọng số dương ↔ x lớn nhất, âm ↔ x nhỏ nhất), cộng bias → chặn trên đúng cho **mọi** ảnh.

| Lớp | Số tích | Chặn trên \|acc\| | Bit có dấu |
|---|---|---|---|
| conv1 | 9 | 86.624 | 18 |
| conv2 | 72 | 357.383 | 20 |
| conv3 | 144 | **518.164** | **20** |
| fc | 144 | 435.342 | 20 |

    2^18 = 262.144 < 518.164 < 2^19 = 524.288  →  19 bit độ lớn + 1 bit dấu = 20 bit

| Cách tính | Lớp lớn nhất | Bit |
|---|---|---|
| Thô (mọi trọng số ±127) | 4.664.259 | 24 — phí |
| **Từ trọng số thật** | 518.164 | **20** |
| Đo trên 10.000 ảnh test | — | 17–19 — không dùng làm chuẩn được |

Chặn trên gắn với bộ trọng số: train lại hoặc đổi trọng số thì phải tính lại (bộ QAT cũng ra tối đa 20 bit).

✍️ **Chép:** Integer-only = chỉ nhân, cộng, dịch, kẹp; M thực → M0/2^shift; cần PTQ static (thước tính sẵn).
pixel − 128 → [conv acc int32 → ((acc·M0 + 2^38) >> 39) + zp_out → kẹp (= ReLU) int8 → maxpool int8] × 3 → fc int32 → argmax.
Số tích = 3×3×kênh vào (9/72/144). acc 20 bit: chặn trên conv3 518.164 < 2^19.

---

# PHẦN B — QAT (mục 6, `qat.py`)

## Bước B1 — Ý tưởng

PTQ làm tròn **sau** khi train → model chưa từng "thấy" sai số làm tròn.
QAT cho model **thấy sai số ngay lúc train**, để nó tự chỉnh trọng số cho chịu được.

Giống tập chạy với tạ chân: lúc thi bỏ tạ (chip chạy int8) thì đã quen.

✍️ **Chép:** QAT = train tiếp mà forward giả vờ lượng tử, để model học cách chịu sai số làm tròn.

---

## Bước B2 — Fake-quant = quantize rồi dequantize ngay

Trong forward, mỗi trọng số và activation đi qua 2 bước liền nhau:

    quantize   : 0,3618 → làm_tròn(0,3618 / 0,006848) = 53
    dequantize : 53 × 0,006848 = 0,3629

Giá trị **vẫn là float** (0,3629), nhưng đã bị **ép đúng vào một nấc** → model thấy đúng sai số int8 sẽ gây ra.

| Hàm trong `qat.py` | Fake-quant cái gì | Thước |
|---|---|---|
| `fq_w` (dòng 52) | Trọng số | Đối xứng, s = max\|w\| / 127, tính lại mỗi bước |
| `fq_a` (dòng 57) | Input và output sau ReLU | Bất đối xứng, như PTQ cấu hình B |

✍️ **Chép:** Fake-quant = quantize + dequantize liền nhau. Giá trị vẫn float nhưng nằm đúng trên nấc.
Ví dụ 0,3618 → 53 → 0,3629.

---

## Bước B3 — Vấn đề đạo hàm và STE

`làm_tròn()` là hàm **bậc thang**: phẳng ở mọi chỗ, nhảy ở mép nấc → **đạo hàm = 0 gần như khắp nơi**.
Backprop nhân với 0 → gradient chết → **model không học được gì**.

**STE** (Straight-Through Estimator): lúc backward **giả vờ làm_tròn là đường thẳng y = x**, đạo hàm = 1.

Code viết bằng một dòng:

    x_fq = x + (lượng_tử(x) − x).detach()

| | Giá trị | Vì sao |
|---|---|---|
| Forward | x + lượng_tử(x) − x = **lượng_tử(x)** | Ra đúng giá trị đã làm tròn |
| Backward | Đạo hàm của x = **1** | `.detach()` = "phần này không tính đạo hàm", nên chỉ còn x |

✍️ **Chép:** làm_tròn() có đạo hàm 0 → gradient chết. STE: backward coi làm_tròn như y = x, đạo hàm 1.
Code: x + (q(x) − x).detach() → forward ra q(x), backward ra 1.

---

## Bước B4 — Dải sau ReLU học dần bằng EMA

PTQ đo dải một lần. QAT **cập nhật dần** max sau ReLU mỗi batch (trung bình trượt, momentum 0,01):

    hi_mới = 0,99 × hi_cũ + 0,01 × max(batch)

Kết quả thật:

| Lớp | hi PTQ (calibrate) | hi QAT (học được) |
|---|---|---|
| conv1 | 5,833 | 5,808 |
| conv2 | 18,126 | 15,934 |
| conv3 | 32,002 | 30,283 |

QAT học ra dải **hẹp hơn** → s = hi / 255 nhỏ hơn → **nấc mịn hơn**.
(Số dương cực lớn hiếm gặp thì chịu bị kẹp, đổi lại các số thường gặp được làm tròn chính xác hơn.)

✍️ **Chép:** QAT cập nhật dải sau ReLU bằng EMA (momentum 0,01). Học ra dải hẹp hơn PTQ → nấc mịn hơn.

---

## Bước B5 — Quy trình train

| Thiết lập | Giá trị |
|---|---|
| Khởi đầu | Trọng số `small_cnn.pth` + dải từ calibrate PTQ |
| Optimizer | Adam, lr 1e-4 (nhỏ, vì chỉ tinh chỉnh) |
| Số epoch | 3, batch 64 |
| Đo sau mỗi epoch | **Đường số nguyên thật** của `ptq_int8.py` trên **val** |
| Giữ | Epoch có val cao nhất (epoch 0 = PTQ nếu QAT không hơn) |

| Epoch | Val acc số nguyên |
|---|---|
| 0 (= PTQ) | 98,77% |
| **1** | **98,80%** ← giữ |
| 2 | 98,75% |
| 3 | 98,80% (bằng, không hơn → không lưu) |

**Vì sao đo bằng đường số nguyên, không đo trên model fake-quant:** chip chạy số nguyên (có M0, dịch bit, làm tròn
requantize), fake-quant chỉ là float mô phỏng. Đo cái chip thật sự chạy mới có ý nghĩa.

Train xong: `qat.py --bits 8 --xuat` → `params_int8_qat/` (cùng định dạng `params_int8/`, không train lại).

✍️ **Chép:** QAT: khởi đầu từ model float + dải PTQ, Adam lr 1e-4, 3 epoch. Mỗi epoch đo bằng đường SỐ NGUYÊN
trên val, giữ epoch tốt nhất (epoch 1). Đo số nguyên vì chip chạy số nguyên, không chạy fake-quant.

---

## Bước B6 — Kết quả và đọc cho đúng

| Test | Float | PTQ | QAT |
|---|---|---|---|
| int8 | 98,82% | 98,84% | **98,91%** |
| int4 (thí nghiệm) | 98,82% | 98,00% | 98,72% |

- **int8: QAT gần như không thêm gì**, vì PTQ vốn không tụt. Hơn PTQ 7 ảnh, nhưng QAT được train thêm 1 epoch —
  **chưa có đối chứng** (train thêm 1 epoch không fake-quant) nên chưa tách được nhờ "biết lượng tử" hay nhờ train thêm.
- QAT int8 khác float **45 ảnh** (PTQ chỉ 9): QAT là **model đã đổi**, không phải bản lượng tử của cùng model.
- **int4: QAT có ích rõ** — kéo lại 0,72 điểm, còn cách float 0,10.
- Train có ngẫu nhiên, con số dao động cỡ ±0,2.

✍️ **Chép:** int8: QAT 98,91% vs PTQ 98,84% — gần như không thêm, chưa có đối chứng. int4: QAT kéo lại 0,72 điểm.
Kết luận: QAT chỉ đáng làm khi PTQ tụt rõ (bit thấp).

---

# PHẦN C — Golden C int8 (mục 7, `golden_c/golden_int8.c`)

## Bước C1 — Vai trò

**Chuẩn đối chiếu cho RTL.** Sau này RTL ra số nào khác golden C ⇒ **bug phần cứng**, không cần bàn.

Ba nguyên tắc của file:
1. Chỉ `stdio.h`, `stdlib.h`, `stdint.h`. Không `math.h`, **không float trên đường inference**.
2. **Không tính lại s, zp, M0** — đọc hết từ thư mục tham số.
3. Cùng code chạy được cả PTQ (`params_int8/`) và QAT (`params_int8_qat/`), chỉ đổi tham số dòng lệnh.

✍️ **Chép:** Golden C = chuẩn đối chiếu RTL. Chỉ số nguyên, không math.h. Đọc tham số từ file, không tính lại.

---

## Bước C2 — Code = đúng 6 operator của công thức

| Hàm (`golden_int8.c`) | Làm gì | Ứng với |
|---|---|---|
| `dau_vao` | q = pixel − 128 | Input (mục 4) |
| `conv3x3_int` | Trừ zp → đệm 0 → acc = bias + Σ x·w (đủ 9 tap) | Dòng 1 công thức |
| `requantize` | ((acc·M0 + 2^(shift−1)) >> shift) + zp_out, kẹp | Dòng 2 công thức |
| `maxpool2_int8` | Max 2×2 trên int8 (max không đổi thước nên chạy thẳng) | Pool |
| `linear_int` | acc fc = bias + Σ (q − zp)·w, **không requantize** | fc |
| `argmax_int` | So sánh chặt `>` | Kết quả |

Đường đi một ảnh (`infer_int`):

    q = pixel − 128
    lặp 3 lần:  acc = conv3x3_int(q)  →  q = requantize(acc)  →  q = maxpool(q)
    acc_fc = linear_int(q)  →  argmax

fc không cần requantize vì nhân cùng một số dương không đổi thứ tự → argmax trên acc cũng ra đúng.

✍️ **Chép:** Golden C: pixel − 128 → [conv (trừ zp, đệm 0, acc = bias) → requantize → maxpool] × 3 → fc → argmax.
fc không requantize vì nhân số dương không đổi argmax.

---

## Bước C3 — Kiểm chứng: khớp TỪNG BIT

| Kiểm | PTQ | QAT |
|---|---|---|
| [a] Ảnh mẫu: 11 file từng lớp | 0 số khác | 0 số khác |
| [b] 10.000 ảnh: cả 10 acc lớp fc | 10.000/10.000 | 10.000/10.000 |
| Accuracy | 98,84% | 98,91% |

**Vì sao bản số nguyên phải khớp từng bit, bản float thì chỉ cần lệch ≤ 1e-4:**
- Cộng số thực theo thứ tự khác → khác chữ số cuối (float làm tròn sau mỗi phép cộng).
- Cộng số nguyên (không tràn) → thứ tự nào cũng ra **đúng một số**.
- ⇒ Ở int8, **lệch 1 đơn vị là lỗi**, không có dung sai.

✍️ **Chép:** Golden C khớp Python từng bit: 11/11 file ảnh mẫu, 10.000/10.000 ảnh, cả PTQ lẫn QAT.
Số nguyên cộng thứ tự nào cũng ra một số → lệch 1 là lỗi.

---

## Bước C4 — Ba bẫy chỉ bản số nguyên mới có ✍️

**Bẫy 1 — Đệm.** Số thực 0 ở miền int8 là **zp**, không phải số nguyên 0.
→ Trừ zp **trước** (x = q − zp), **rồi** mới đệm 0 lên x.

**Bẫy 2 — Dịch phải số âm.**

| −7 chia 2 (đáp số −3,5) | Kết quả | Ai dùng |
|---|---|---|
| Làm tròn về 0 | −3 | C `/` |
| Làm tròn xuống (floor) | **−4** | Python `>>`, Verilog `>>>`, cần cho requantize |
| C `>>` trên số âm | Tuỳ trình biên dịch | **Không dùng được** (implementation-defined) |

→ Golden C tự viết `dich_phai_so_hoc` (dòng 169): số âm thì `−((−x − 1) >> n) − 1`.
Thử: −7, n = 1 → −((6) >> 1) − 1 = −3 − 1 = **−4** ✓

**Bẫy 3 — acc × M0 phải 64 bit.** acc 20 bit × M0 31 bit = **51 bit** → tràn int32 → dùng `int64_t`.

✍️ **Chép:** 3 bẫy: (1) trừ zp rồi mới đệm 0; (2) dịch phải số âm phải floor — C tự viết, Verilog dùng >>>;
(3) acc × M0 = 51 bit → int64.

---

## Bước C5 — Hai phát hiện cho phần cứng ✍️

**1. Thanh ghi acc 20 bit có dấu đủ cho MỌI ảnh.**
Chặn trên tính từ trọng số thật (hàm `chan_tren`), không chỉ đo trên test. Đo trên test chỉ 17–19 bit.

**2. Requantize đặt SAU maxpool được — giống hệt từng bit.**
- Requantize là **hàm không giảm**: acc lớn hơn thì q_out không nhỏ hơn.
- Với hàm không giảm: `max(rq(a), rq(b)) = rq(max(a, b))` → lấy max trước hay sau đều ra một số.
- Số lần requantize mỗi ảnh: **10.192 → 2.496** (ít hơn 4,08 lần). Kiểm: 0/10.000 ảnh khác, cả PTQ và QAT.
- Giá phải trả: maxpool so sánh trên acc **20 bit** thay vì 8 bit.
- Cùng lý do với đổi ReLU ↔ maxpool ở bản float.

✍️ **Chép:** acc 20 bit đủ cho mọi ảnh. Requantize không giảm → đổi chỗ với maxpool được, 10.192 → 2.496 lần/ảnh,
giá: maxpool so sánh 20 bit.

---

## Phải nắm để làm tiếp RTL ✍️

| # | Điều cần nắm | Vì sao |
|---|---|---|
| 1 | RTL dùng tham số từ `params_int8/` (PTQ) — đang đề xuất, chờ Thầy chốt | Chọn bộ tham số nạp ROM |
| 2 | Kiểm RTL = so với `golden_int8/` từng lớp, từng số | Lệch 1 là bug |
| 3 | 3 bẫy ở Bước C4 | Lỗi hay gặp nhất khi port sang Verilog |
| 4 | acc 20 bit; acc × M0 51 bit; M0 15 bit vừa DSP48 | Chọn độ rộng thanh ghi, bộ nhân |
| 5 | Có 2 cách sắp: requantize → pool, hoặc pool → requantize | Cách 2 ít khối requantize hơn 4 lần |
| 6 | Khối RTL đầu tiên sẽ làm: **requantize** | Theo kế hoạch sau 10 bài Verilog |

---

## Bỏ qua — không cần đi sâu lúc này

| Thứ | Vì sao bỏ được |
|---|---|
| Chi tiết dòng code `main()` của `ptq_int8.py` | Chỉ cần nhớ 6 bước ở A2 |
| Thuộc hết số của cấu hình A–F | Chỉ cần kết luận: chọn B, M0 15 bit đủ |
| Train lại QAT, chạy nhiều seed | Việc tồn, chưa cần cho RTL |
| int4 | Thí nghiệm để hiểu khi nào QAT có ích, RTL dùng int8 |
| Đọc hết 577 dòng `golden_int8.c` | Chỉ cần 6 hàm ở C2 + hàm dịch số học |
| Code `chan_tren` (tính chặn trên acc) | Chỉ nhớ kết luận: 20 bit |
| Lệnh demo (mục 8 tài liệu ôn) | Đọc khi chuẩn bị gặp Thầy |

---

## Tự kiểm (tự làm rồi mới mở đáp án)

1. Vì sao calibrate trên val chứ không phải test? Vì sao không dùng `torch.ao.quantization`?
2. Vì sao chọn cấu hình B dù acc val thấp hơn A 1 ảnh?
3. Fake-quant trọng số 0,3618 (s_w = 0,006848) ra số nào? Số đó là int hay float?
4. Giải thích STE trong một câu. Vì sao đo QAT bằng đường số nguyên chứ không đo trên model fake-quant?
5. QAT int8 hơn PTQ 7 ảnh. Kết luận được "QAT tốt hơn" chưa? Vì sao?
6. Vì sao golden C int8 phải khớp từng bit, còn golden C float chỉ cần lệch ≤ 1e-4?
7. Tính −7 >> 1 theo kiểu floor. Vì sao không dùng `>>` của C?
8. Vì sao requantize đổi chỗ được với maxpool mà không đổi được với avgpool?

<details>
<summary>Đáp án</summary>

1. Calibrate là ra quyết định (dải → scale) → dùng test là data leakage. `torch.ao` là hộp đen, không cho số nguyên từng lớp để so với C/RTL.
2. A và B chỉ chênh 1 ảnh (trong dao động), mà B bỏ được bộ so sánh ReLU trên chip.
3. 0,3618 / 0,006848 = 52,83 → 53 → 53 × 0,006848 = **0,3629**. Là **float**, nhưng nằm đúng trên một nấc.
4. STE: lúc backward coi làm_tròn() như y = x (đạo hàm 1) để gradient không chết. Đo bằng số nguyên vì chip chạy số nguyên (M0, dịch bit), fake-quant chỉ là float mô phỏng.
5. Chưa. QAT được train thêm 1 epoch, chưa chạy đối chứng train thêm không fake-quant; 7 ảnh cũng nằm trong dao động ±0,2 điểm.
6. Cộng số nguyên (không tràn) thứ tự nào cũng ra một số → lệch 1 là lỗi. Cộng float thì thứ tự khác ra khác chữ số cuối.
7. −3,5 làm tròn xuống = **−4**. `>>` của C trên số âm là implementation-defined (tuỳ trình biên dịch), `/` của C lại ra −3.
8. Max đổi chỗ được với **mọi hàm không giảm** (requantize là hàm không giảm). Trung bình chỉ đổi chỗ được với hàm tuyến tính; requantize có làm tròn và kẹp nên không tuyến tính → avg(rq) ≠ rq(avg).

</details>
