# Chép tay — INT8 mục 1–4: một lớp conv chạy bằng số nguyên

Đi kèm `docs/on-tap-int8-ptq-qat-c.md` (mục 1–4). Mọi số lấy từ file thật, đã chạy lại 11/11/2026.
Mỗi bước có dòng **✍️ Chép** ở cuối — chỉ cần chép các dòng đó là đủ một trang A4.
Thứ tự đọc: Bước 0 → 9, mỗi bước dựa trên bước trước.

---

## Bước 0 — Mục tiêu

**Mục 1–4 trả lời một câu:** làm sao để một lớp conv chạy **chỉ bằng số nguyên** (nhân, cộng, dịch bit, kẹp)
mà kết quả vẫn gần như float32. Đó là đúng thứ RTL sẽ làm trên chip.

| Mục tài liệu | Câu hỏi | Bước trong file này |
|---|---|---|
| 1 | Vì sao INT8? | Bước 1 |
| 2 | Quantize số thực thành int8 thế nào? | Bước 2–5 |
| 3 | Một lớp conv chạy bằng số nguyên thế nào? | Bước 6–9 |
| 4 | Input xử lý thế nào? | Bước 3 |

Project thuộc loại **PTQ static + integer-only**: trọng số và activation đều int8, cần dữ liệu để calibrate,
và trên chip **không còn một phép float nào** (video tổng quan trên YouTube dừng ở PTQ static, không nói phần integer-only).

✍️ **Chép:** Mục tiêu = conv chỉ bằng số nguyên (nhân, cộng, dịch, kẹp). Project = PTQ static + integer-only.

---

## Bước 1 — Vì sao INT8 (mục 1)

- FPGA nhân số nguyên rẻ (khối DSP), float tốn rất nhiều LUT.
- Bộ nhớ: float32 5.018 × 4 = 20.072 byte → int8 4.968 × 1 + 50 × 4 = 5.168 byte → **giảm 3,88×**.
- Không đúng 4× vì **50 bias giữ int32** (lý do ở Bước 7).
- int8 **có dấu** = −128 … 127. (0 … 255 là uint8 — chỉ pixel gốc dùng.)

✍️ **Chép:** INT8 vì FPGA nhân số nguyên rẻ. Bộ nhớ giảm 3,88× (không phải 4× vì bias int32). int8 = −128…127.

---

## Bước 2 — Cây thước: q, s, zp, r

Mỗi tensor (input, trọng số, output mỗi lớp) có **một cây thước 256 nấc**, mô tả bằng 2 số:

| Ký hiệu | Tên | Nghĩa trên thước |
|---|---|---|
| q | quantized | Nấc đang đứng — số nguyên chip lưu |
| s | scale | Độ dài 1 nấc |
| zp | zero-point | Nấc nằm đúng ở số thực 0 |
| r | real | Số thực |

Hai chiều đổi:

| | Công thức | Nghĩa | Ai làm |
|---|---|---|---|
| **Quantize** | q = làm_tròn(r / s) + zp | Số thực rơi vào nấc nào | Máy tính, **1 lần**, trước khi lên chip |
| **Dequantize** | r = s · (q − zp) | Nấc này là số thực bao nhiêu | **Chỉ người** — để hiểu và kiểm. Chip không làm |

`q − zp` = số nấc tính từ số 0.

**làm_tròn(x) = floor(x + 0,5)** (`ptq_int8.py` dòng 87): về số nguyên gần nhất, đúng nửa thì làm tròn lên.

| x | floor(x) — làm tròn xuống | **floor(x + 0,5)** — project dùng | np.round |
|---|---|---|---|
| 52,83 | 52 | **53** | 53 |
| −94,67 | −95 | **−95** | −95 |
| 2,5 | 2 | **3** | 2 (về số chẵn) |
| −2,5 | −3 | **−2** | −2 |

Không dùng `np.round` vì nó làm tròn số đúng nửa về số chẵn, C và phần cứng không làm vậy → lệch nhau.
Requantize `+ 2^38` rồi `>> 39` là cùng quy tắc này viết bằng số nguyên.

✍️ **Chép:** r = s·(q − zp) (dequantize); q = làm_tròn(r/s) + zp (quantize). s = độ dài nấc, zp = nấc của số 0.
làm_tròn(x) = floor(x + 0,5): gần nhất, đúng nửa thì lên — không phải làm tròn xuống, không dùng np.round.
Quantize làm 1 lần trên máy tính; dequantize chỉ để người kiểm, chip không làm.

---

## Bước 3 — Thước input (mục 4)

**s_in — độ dài nấc.** Lấy cả đoạn số thực chia cho số khoảng.

    pixel 0   → (0 − 0,1307)/0,3081 = −0,4242   (lo)
    pixel 255 → (1 − 0,1307)/0,3081 =  2,8215   (hi)
    s_in = (hi − lo) / 255 = 3,2457 / 255 = 0,012728

255 vì 256 nấc có 255 khoảng. Mỗi nấc đúng bằng 1 mức pixel.

**zp_in — nấc của số 0.** Đếm từ đáy lên.

    từ −0,4242 lên 0: 0,4242 / 0,012728 = 33,33 nấc
    zp_in = −128 + 33,33 = −94,67 → −95

**Hệ quả — chuẩn hoá biến mất:**

    q_in = pixel − 128      (đảo bit cao nhất, không tốn phép tính)
    x    = q_in − zp_in = pixel − 33      ∈ [−33, 222] → bộ nhân 9 bit có dấu

x là số **thật sự đi vào bộ nhân**. Nền đen (pixel 0) có x = −33, **không phải 0**.

lo, hi đo bằng **calibrate**: chạy 1.000 ảnh **val** qua model float, lấy min/max. Dải luôn nới cho chứa 0.

✍️ **Chép:** s_in = (hi − lo)/255 = 0,012728. zp_in = −128 + 33,33 → −95.
Chip nhận q_in = pixel − 128; vào bộ nhân x = pixel − 33. Calibrate = đo min/max trên 1.000 ảnh val.

**Hỏi thêm — nếu chip vẫn tự chuẩn hoá thì sao?** Ra **y hệt** pixel − 128, chỉ tốn thêm phép số thực.

    r      = (pixel/255 − 0,1307) / 0,3081               ← chuẩn hoá
    r/s_in = (pixel/255 − 0,1307) / 0,3081 × 255 × 0,3081
           = pixel − 0,1307 × 255 = pixel − 33,33          ← 0,3081 và 255 triệt tiêu
    q_in   = làm_tròn(pixel − 33,33) + (−95) = (pixel − 33) − 95 = pixel − 128

| | Có chuẩn hoá trên chip | Không (chọn) |
|---|---|---|
| Mỗi pixel | Nhân, trừ, chia, làm tròn, cộng zp | Đảo bit cao nhất |
| Phần cứng | Bộ nhân + cộng + làm tròn (fixed-point) | Một cổng NOT |
| Kết quả | pixel − 128 | pixel − 128 |

Mẹo chỉ đúng khi dải input = đúng pixel 0 … 255 và có 255 khoảng (int8). Đổi dải calibrate hoặc xuống int4 là hỏng.
Golden C có chốt kiểm: zp_in ≠ −95 thì báo lỗi.

✍️ **Chép:** Có chuẩn hoá vẫn ra pixel − 128 (r/s_in = pixel − 33,33 → −33, cộng −95). Chuẩn hoá trên chip là thừa.
Chỉ đúng khi dải = pixel 0…255, 255 khoảng.

---

## Bước 4 — Thước trọng số (đối xứng)

Trọng số cân quanh 0 → đặt số 0 ngay nấc 0 → **zp_w = 0**. Dùng nấc −127 … 127 (bỏ −128 cho cân).

    max|w| conv1 = 0,869651794
    s_w = max|w| / 127 = 0,006848

Ví dụ: 0,3618 / 0,006848 = 52,83 → **q_w = 53**.

Vì sao zp_w = 0: tích `(q_in − zp_in)·(q_w − zp_w)` còn `(q_in − zp_in)·q_w` → **mỗi MAC bớt một phép trừ**.

✍️ **Chép:** Trọng số đối xứng: s_w = max|w|/127 = 0,006848, zp_w = 0 → MAC khỏi trừ zp_w.

**Hỏi thêm — số trong kernel lấy ở đâu? Có phải pixel không?**

Không phải pixel. Kernel = **trọng số**, cố định cho mọi vị trí, mọi ảnh. Pixel thì lấy từ ảnh, 9 ô nằm dưới kernel.

    khởi tạo ngẫu nhiên → train (backprop + Adam) học ra → models/small_cnn.pth
      → export_params.py → params/weights.txt (float)
      → ptq_int8.py: ÷ s_w, làm tròn → params_int8/conv1_w.txt (int8)

Kernel kênh 0 = 9 số đầu (thứ tự [kênh ra][kênh vào][ky][kx]):

| Ô kernel | Float | ÷ 0,006848 | q_w |
|---|---|---|---|
| (0,0) | 0,3191 | 46,61 | 47 |
| (0,1) | 0,2397 | 35,01 | 35 |
| (0,2) | −0,3606 | −52,66 | −53 |
| (1,0) | 0,3299 | 48,18 | 48 |
| (1,1) | 0,2259 | 32,99 | 33 |
| (1,2) | 0,3618 | 52,83 | 53 |
| (2,0) | −0,3522 | −51,44 | −51 |
| (2,1) | 0,0247 | 3,60 | 4 |
| (2,2) | 0,2187 | 31,93 | 32 |

conv1 có **8 kernel** (8 kênh ra) × 9 = 72 trọng số + 8 bias = 80 params. Cùng một kernel trượt khắp ảnh (dùng chung trọng số).

**Hỏi thêm — max|w| = 0,8697 lấy ở đâu? Kernel kênh 0 lớn nhất chỉ 0,3618.**

**Per-tensor:** max lấy trên **cả 72 trọng số conv1** (đủ 8 kernel), không riêng kênh 0.
Số −0,869651794 nằm ở vị trí 50 trong `weights.txt` → 50 = 5 × 9 + 5 → **kênh 5**, ô (1,2). Nó nằm đúng đáy thước (−127).
Cả 8 kernel chung một thước 0,006848 → kernel kênh 0 chỉ dùng tới khoảng ±53 trong ±127 nấc.

| | Per-tensor (đang dùng) | Per-channel |
|---|---|---|
| Thước | 1 cho cả lớp | 1 cho mỗi kênh ra |
| s_w kênh 0 | 0,006848 | 0,3618 / 127 ≈ 0,00285 (tính tay), mịn hơn ~2,4× |
| Số M0 cần | 1 / lớp | 8 / lớp (conv1) |
| Trạng thái | Đang dùng | Câu hỏi chờ Thầy |

Code: `ptq_int8.py` dòng 116, `s = np.abs(w).max() / qmax` với w là cả tensor của lớp.

✍️ **Chép:** Kernel = trọng số (train học ra), không phải pixel. weights.txt (float) → ÷ s_w → conv1_w.txt. Kênh 0 = 9 số đầu.
s_w per-tensor: max|w| trên cả 72 trọng số conv1 (số lớn nhất ở kênh 5) → cả 8 kernel chung thước 0,006848.

---

## Bước 5 — Thước output sau ReLU (bất đối xứng)

Sau ReLU chỉ còn số ≥ 0: dải conv1 = [0; 5,833] — **lệch hẳn một phía**.

| | Đối xứng | Bất đối xứng (chọn) |
|---|---|---|
| s | 5,833 / 127 = 0,0459 | 5,833 / 255 = **0,022875** |
| Nấc dùng | 0 … 127, bỏ phí −127 … −1 | −128 … 127, dùng hết |
| zp | 0 | **−128** (số 0 nằm ở đáy) |

→ Bất đối xứng **độ phân giải gấp đôi** với cùng 8 bit.

| Thước conv1 | s | zp | Kiểu |
|---|---|---|---|
| Input | 0,012728 | −95 | Bất đối xứng |
| Trọng số | 0,006848 | 0 | Đối xứng |
| Output (sau ReLU) | 0,022875 | −128 | Bất đối xứng |

✍️ **Chép:** Activation lệch một phía → bất đối xứng, dùng đủ 256 nấc, độ phân giải ×2.
Output sau ReLU: s = 0,022875, zp = −128. (Chép luôn bảng 3 thước.)

**Hỏi thêm — zp luôn là −95 à?** Không. **Mỗi thước có zp riêng**; −95 chỉ là input conv1.

| Lớp | zp_in | zp_out | zp trọng số |
|---|---|---|---|
| conv1 | **−95** (ảnh vào) | −128 | 0 |
| conv2 | **−128** | −128 | 0 |
| conv3 | **−128** | −128 | 0 |
| fc | **−128** | không có (argmax trên acc) | 0 |

Vì sao từ conv2 là −128: input conv2 = output conv1 sau ReLU + maxpool.
Sau ReLU số 0 ở đáy → zp = −128; **maxpool không đổi thước** (chỉ chọn số lớn nhất, không nhân chia)
→ zp_out lớp trước = zp_in lớp sau (golden C kiểm điều này).

| | conv1 | conv2, conv3, fc |
|---|---|---|
| x = q − zp_in | pixel − 33 | q + 128 |
| Dải x | [−33, 222] | [0, 255] |
| Bộ nhân | 9 bit có dấu | 8 bit không dấu |
| Trừ zp | Cần bộ cộng | Miễn phí (đảo bit cao nhất) |

Nếu không gộp ReLU (cấu hình A), thước output lấy theo dải trước ReLU [−6,749; 5,833] → zp_out conv1 ≈ +9
(tính tay, file không lưu) → mỗi lớp zp khác nhau, trừ zp không còn là đảo bit. Thêm một lý do chọn B.

✍️ **Chép:** zp riêng từng thước. Input conv1 −95; từ conv2 trở đi −128 (0 ở đáy sau ReLU, maxpool không đổi thước);
trọng số 0. conv1 nhân 9 bit có dấu; các lớp sau 8 bit, trừ zp = đảo bit.

**Hỏi thêm — conv2, conv3 sau ReLU vẫn bất đối xứng?** Đúng. Cả 3 lớp cùng zp_out = −128, chỉ khác s:

| Lớp | Dải sau ReLU | s_out = hi / 255 | zp_out |
|---|---|---|---|
| conv1 | [0; 5,833] | 0,022875 | −128 |
| conv2 | [0; 18,126] | 0,071083 | −128 |
| conv3 | [0; 32,002] | 0,125496 | −128 |

Bất đối xứng với lo = 0, zp = −128 = **uint8 trá hình**: q − zp = q + 128 ∈ [0, 255].

| Tensor | Kiểu |
|---|---|
| Input conv1 (ảnh) | Bất đối xứng, zp −95 |
| Output conv1, conv2, conv3 (sau ReLU) | Bất đối xứng, zp −128 |
| Trọng số mọi lớp | Đối xứng, zp 0 |
| Output fc | Không quantize — acc int32 → argmax |

✍️ **Chép:** Sau ReLU cả 3 lớp bất đối xứng, [0, hi], zp −128, s = hi/255 (0,0229 / 0,0711 / 0,1255).
lo = 0, zp = −128 = uint8 trá hình.

---

## Bước 6 — Nhân một cặp: tách số nguyên ra

Nhắc lại ký hiệu (từ Bước 3): **x = q_in − zp_in** = số nấc tính từ số 0 trên thước input.
Ở conv1: x = q_in + 95 = pixel − 33. Từ Bước 2: r_in = s_in · (q_in − zp_in) = **s_in · x**.

    r_in · r_w = (s_in · x) · (s_w · q_w) = (s_in · s_w) · (x · q_w)
                                             └── thước ──┘  └ số nguyên ┘

- `x · q_w`: **chip tính**. Ví dụ 152 × 53 = 8.056.
- `s_in · s_w = 0,0000871585`: thước mới của tích — **chỉ ghi trên giấy**, chip không tính.
- Giống đổi đơn vị: cm × cm = cm².

**Nhìn bằng số thực — vì sao chia cho thước lại ra đúng số nguyên**

s_in = 0,012728234 · s_w = 0,00684765192 · thước của tích s_in · s_w = 0,0000871585

| Ô | x | r_in = s_in · x | q_w | r_w = s_w · q_w | Tích thực r_in · r_w | ÷ 0,0000871585 | x · q_w (chip) |
|---|---|---|---|---|---|---|---|
| (7,7) | 152 | 1,934692 | 53 | 0,362926 | 0,702149 | **8.056,000** | 8.056 |
| (6,5) | −33 | −0,420032 | 47 | 0,321840 | −0,135183 | **−1.551,000** | −1.551 |
| (8,7) | 221 | 2,812940 | 32 | 0,219125 | 0,616385 | **7.072,000** | 7.072 |

Chia tích thực cho thước ra **đúng số nguyên, không dư** — vì:

    r_in · r_w = (s_in · x) · (s_w · q_w)
    ÷ (s_in · s_w)  →  s_in, s_w triệt tiêu  →  còn x · q_w  (số nguyên × số nguyên)

Cả r_in và r_w đều **đã nằm đúng trên nấc** (x và q_w là số nguyên), nên tích của chúng nằm đúng trên nấc của thước mới.

**So với trọng số float gốc (chưa quantize):**

| Ô | Trọng số float | r_in × w_float | ÷ 0,0000871585 | Nằm trên nấc? |
|---|---|---|---|---|
| (7,7) | 0,361772 | 0,699918 | 8.030,40 | Không — lẻ 0,40 |
| (6,5) | 0,319135 | −0,134047 | −1.537,97 | Không |
| (8,7) | 0,218657 | 0,615069 | 7.056,90 | Không |

→ Trọng số float **không nằm trên nấc** nên tích ra số lẻ. Quantize (0,3618 → 53 → 0,3629) chính là **ép trọng số lên nấc**
để tích ra số nguyên. Khoảng chênh 8.056 so với 8.030 chính là **sai số lượng tử** của trọng số này.

✍️ **Chép:** Ví dụ ô (7,7): 1,934692 × 0,362926 = 0,702149; ÷ 0,0000871585 = 8.056,000 = 152 × 53 — đúng số nguyên.
Vì r_in và r_w đều đã nằm trên nấc → tích nằm trên nấc thước s_in·s_w. Trọng số float chưa quantize thì ra lẻ (8.030,40).

✍️ **Chép:** r_in·r_w = (s_in·s_w)·(x·q_w). Chip chỉ nhân số nguyên x·q_w. s_in·s_w = 0,0000871585 là thước của tích.

---

## Bước 7 — Cộng 9 tích + bias → acc

9 tích cùng thước `s_in · s_w` → **cộng thẳng số nguyên**.

Bias cũng phải về thước này mới cộng được:

    q_b = 0,401263 / 0,0000871585 = 4.604     → quá int8 (cần 14 bit) → giữ int32

Ví dụ conv1, kênh 0, hàng 7, cột 6 (dòng 203 trong file — file đánh dòng từ 1):

| Ô | pixel | x = pixel − 33 | q_w | x · q_w |
|---|---|---|---|---|
| (6,5) | 0 | −33 | 47 | −1.551 |
| (6,6) | 0 | −33 | 35 | −1.155 |
| (6,7) | 0 | −33 | −53 | 1.749 |
| (7,5) | 0 | −33 | 48 | −1.584 |
| (7,6) | 84 | 51 | 33 | 1.683 |
| (7,7) | 185 | 152 | 53 | 8.056 |
| (8,5) | 0 | −33 | −51 | 1.683 |
| (8,6) | 222 | 189 | 4 | 756 |
| (8,7) | 254 | 221 | 32 | 7.072 |
| | | | **Σ** | **16.709** |

    acc = 4.604 + 16.709 = 21.313      (01_conv1_acc.txt, dòng 203)
    acc ra số thực: 21.313 × 0,0000871585 = 1,8576

acc **khởi tạo bằng bias** rồi cộng dồn. Thanh ghi acc **20 bit có dấu** là đủ cho mọi ảnh.

✍️ **Chép:** acc = q_b + Σ x·q_w, cùng thước s_in·s_w. Bias đổi về thước này → số lớn (4.604) → int32.
Ví dụ: 4.604 + 16.709 = 21.313 ↔ 1,8576. acc khởi tạo = bias, 20 bit.

**Hỏi thêm — vì sao bias giữ int32, không ép int8?**

1. **Bias cộng thẳng vào acc → phải cùng thước với acc:** s_bias = s_in · s_w = 0,012728234 × 0,00684765192 = 0,0000871585.
2. **Thước rất mịn → số nguyên lớn:** q_b = 0,401263237 / 0,0000871585 = 4.603,83 → **4.604**
   (bias thực: `params/biases.txt` dòng 1; s_in, s_w: `tham_so.txt`).

| Lớp | conv1 | conv2 | conv3 | fc |
|---|---|---|---|---|
| Bias lớn nhất (số nguyên) | 4.604 | 2.422 | −819 | 147 |
| Cần (bit có dấu) | 14 | 13 | 11 | 9 |

   → Lớp nào cũng vượt [−128, 127]. Ép int8 thì 4.604 bị kẹp còn 127: bias 0,40 chỉ còn 0,011.
3. **Dùng int8 với thước riêng cho bias?** Được, nhưng trước khi cộng phải đổi thước → thêm nhân + dịch mỗi điểm ra → mất lợi.
4. **Vì sao 32 bit dù 14 bit đủ?** Khớp kiểu acc int32 trong phần mềm (quy ước TFLite), cộng thẳng không ép kiểu.
   Chỉ tốn 50 × 4 = 200 byte.
5. **Hệ quả:** bộ nhớ giảm 3,88× chứ không phải 4× (Bước 1).
6. **Ghi chú RTL:** phần cứng không bị buộc 32 bit — ROM bias lưu 14 bit, mở rộng dấu khi nạp vào acc 20 bit.

✍️ **Chép:** Bias cộng thẳng vào acc → cùng thước s_in·s_w (rất mịn) → số lớn (4.604, cần 14 bit) → int32.
Ép int8 thì kẹp 127 (0,40 → 0,011). 32 bit là quy ước khớp acc, chỉ 200 byte. RTL lưu 14 bit là đủ.

---

## Bước 8 — Requantize: đổi acc sang thước output

Muốn: **dequantize** acc ra số thực rồi **quantize** vào thước output. Nhưng chip không có số thực → gộp thành một phép số nguyên.

    q_out = r / s_out + zp_out
         = acc · (s_in · s_w / s_out) + zp_out
         = acc · M + zp_out

**M = tỷ lệ thước cũ / thước mới** = 0,0000871585 / 0,022875 = 0,00381027 (lấy từ `tham_so.txt`).

M là số thực → viết thành **M0 / 2^shift**:

    M ≈ 2.094.719.601 / 2^39       (M0 31 bit, shift = 39)

Nhân M0 = nhân số nguyên; chia 2^39 = dịch phải 39 bit (gần như miễn phí).
Làm tròn: cộng nửa nấc 2^38 trước khi dịch (giống +0,5 rồi bỏ lẻ).

| Bước | Phép tính | Kết quả |
|---|---|---|
| Nhân M0 | 21.313 × 2.094.719.601 | 44.644.758.856.113 (51 bit → cần biến 64 bit) |
| Làm tròn | + 2^38 | 44.919.636.763.057 |
| Dịch 39 | 81,208 + 0,5 → bỏ lẻ | **81** |
| + zp_out | 81 − 128 | **−47** |
| Kẹp [−128, 127] | | **−47** (`02_relu1.txt`, dòng 203) |

Kiểm: dequantize −47 = 0,022875 × (−47 + 128) = 1,8528. PyTorch float = 1,8441. Lệch 0,009 do làm tròn — bình thường.

### Bảng nhân – dịch chi tiết

**1. Vì sao "nhân M0 rồi dịch 39" = "nhân M"**

Dịch phải n bit = chia 2^n (bỏ phần lẻ). Nên:

    acc × M0 >> 39  ≈  acc × M0 / 2^39  =  acc × (2.094.719.601 / 549.755.813.888)  =  acc × 0,0038102728  =  acc × M

| Lũy thừa | Giá trị | Dùng ở đâu |
|---|---|---|
| 2^30 | 1.073.741.824 | Cận dưới M0 (31 bit) |
| 2^31 | 2.147.483.648 | Cận trên M0 |
| 2^38 | 274.877.906.944 | Nửa nấc, cộng vào để làm tròn |
| 2^39 | 549.755.813.888 | Số chia (dịch 39 bit) |

**2. Tìm shift: tăng dần tới khi M × 2^shift lọt vào [2^30, 2^31)** (hàm `tach_M`, dòng 121)

| shift | M × 2^shift | Lọt [1.073.741.824 ; 2.147.483.648)? |
|---|---|---|
| 37 | 523.679.900 | Chưa — nhỏ quá |
| 38 | 1.047.359.800 | Chưa — vẫn dưới 2^30 |
| **39** | **2.094.719.600,77** | **Lọt** → làm tròn **M0 = 2.094.719.601** |
| 40 | 4.189.439.201 | Quá 2^31 |

M0 càng nhiều bit thì M0/2^shift càng sát M: ở đây lệch chỉ 4 × 10⁻¹³.

**3. M0, shift của cả 3 lớp** (`tham_so.txt`)

| Lớp | M | M0 | shift |
|---|---|---|---|
| conv1 | 0,00381027276 | 2.094.719.601 | 39 |
| conv2 | 0,00248648071 | 1.366.957.226 | 39 |
| conv3 | 0,00351455811 | 1.932.148.756 | 39 |
| fc | — | — | không requantize |

Lấy M (9 chữ số) × 2^39 tự bấm máy sẽ lệch M0 vài đơn vị (ví dụ conv3 ra 1.932.148.754) — **luôn lấy M0 từ file**.

**4. Ví dụ acc âm** (conv1, vị trí 176, `01_conv1_acc.txt` dòng 177)

| Bước | Phép tính | Kết quả |
|---|---|---|
| Nhân M0 | −4.247 × 2.094.719.601 | −8.896.274.145.447 |
| Làm tròn | + 2^38 | −8.621.396.238.503 |
| Dịch 39 (floor) | −15,68 → làm tròn **xuống** | **−16** (không phải −15) |
| + zp_out | −16 − 128 | −144 |
| Kẹp | dưới −128 | **−128** = số thực 0 (ReLU) |

Ở đây thấy rõ vì sao dịch số âm phải là **floor**: −15,68 → −16. Nếu làm tròn về 0 (kiểu `/` của C) sẽ ra −15 — sai 1.
(Lần này kẹp nuốt mất sai số, nhưng ở chỗ không bị kẹp thì lệch 1 là lỗi.)

✍️ **Chép:** Dịch phải n bit = chia 2^n. acc·M0 >> 39 = acc·M0/2^39 = acc·M.
shift: tăng tới khi M·2^shift ∈ [2^30, 2^31) → 39, M0 = làm_tròn(M·2^39). Cả 3 lớp shift = 39.
Acc âm: −4.247 → −15,68 → floor −16 → −144 → kẹp −128.

✍️ **Chép:** Requantize = dequantize + quantize gộp bằng số nguyên. M = s_in·s_w/s_out (tỷ lệ 2 thước) ≈ M0/2^shift.
Nhân M0, cộng 2^(shift−1), dịch phải shift, cộng zp_out, kẹp. Ví dụ: 21.313 → 81 → −47.

---

## Bước 9 — Kẹp chính là ReLU

zp_out = −128 = số thực 0. Kẹp dưới tại −128 ⇔ mọi số âm về 0 ⇔ **ReLU**.

Ví dụ vị trí 176: acc = −4.247 → … → −144 → **kẹp lên −128** = số thực 0.

→ Không cần bộ so sánh ReLU riêng (cấu hình B).

✍️ **Chép:** zp_out = −128 nên kẹp dưới −128 = ReLU. acc −4.247 → −144 → −128 (= 0).

---

## Công thức tổng quát ✍️ (chép nguyên khung)

    acc   = q_b + Σ (q_in − zp_in) · q_w
    q_out = kẹp( ((acc · M0 + 2^(shift−1)) >> shift) + zp_out,  −128, 127 )

    M   = s_in · s_w / s_out  ≈  M0 / 2^shift
    q_b = làm_tròn( bias / (s_in · s_w) )      (int32)

- conv2, conv3: cùng công thức, zp_in = −128, cộng 72 / 144 tích.
- fc: dừng ở acc, argmax luôn (không requantize).

---

## Bảng ký hiệu (tra, không cần chép)

Trong `tham_so.txt` code đặt tên tiếng Việt: `zp_vao` = zp_in, `s_ra` = s_out, `zp_ra` = zp_out.

| Ký hiệu | Nghĩa | conv1 | Lấy ở đâu |
|---|---|---|---|
| s_in, zp_in | Thước input | 0,012728 / −95 | `tham_so.txt` |
| q_in | Nấc input = pixel − 128 | 57 (ô 7,7) | `golden_int8/input.txt` |
| x | q_in − zp_in = pixel − 33 | 152 | Tính |
| s_w | Thước trọng số (zp = 0) | 0,006848 | `tham_so.txt` |
| q_w | Trọng số dạng nấc | 53 | `conv1_w.txt` |
| s_in · s_w | Thước của tích và acc | 0,0000871585 | Tính |
| q_b | Bias trên thước s_in·s_w | 4.604 | `conv1_b.txt` |
| acc | Tổng cộng dồn | 21.313 | `01_conv1_acc.txt` |
| s_out, zp_out | Thước output | 0,022875 / −128 | `tham_so.txt` |
| M | Tỷ lệ 2 thước | 0,00381027 | `tham_so.txt` |
| M0, shift | M dạng số nguyên + dịch | 2.094.719.601 / 39 | `tham_so.txt` |
| q_out | Output int8 | −47 | `02_relu1.txt` |

---

## Phải nắm để làm tiếp RTL ✍️

| # | Điều cần nắm | Vì sao |
|---|---|---|
| 1 | Công thức tổng quát ở trên | Chính là mạch cần viết |
| 2 | Bit: x 9, q_w 8, tích 16, **acc 20**, acc × M0 **51** | Chọn độ rộng thanh ghi, bộ nhân |
| 3 | Dịch phải số âm = **floor** → Verilog `>>>` trên `signed` | Sai thì lệch 1 đơn vị |
| 4 | Làm tròn = cộng 2^(shift−1) rồi dịch | Phải khớp từng bit với golden C |
| 5 | acc khởi tạo bằng bias | Sai thì lệch đúng bằng bias |
| 6 | Đệm: **trừ zp trước, rồi đệm 0** (ô đệm x = 0, khác nền đen x = −33) | Sai thì viền ảnh lệch |
| 7 | zp = −128 → `q − zp` = đảo bit cao nhất | Không cần bộ cộng |
| 8 | ReLU = kẹp dưới −128 | Không cần bộ so sánh riêng |
| 9 | fc không requantize, argmax trên acc, so sánh chặt `>` | Khối cuối |
| 10 | s, zp, M0, shift **đọc từ `params_int8/`**, không tự tính | Tính lại lệch M0 → không khớp golden |

---

## Bỏ qua — không cần đi sâu lúc này

| Thứ | Vì sao bỏ được |
|---|---|
| Tự tính s, zp, M0 cho mọi lớp | Có sẵn trong `tham_so.txt`; đã hiểu cách ra qua conv1 |
| Thuật toán `tach_M` chọn shift | Chỉ nhớ: M0 ∈ [2^30, 2^31) — 31 bit |
| Chi tiết code `calibrate()` | Chạy 1 lần, đã xuất file |
| Cấu hình A–F, int4 | Thuộc mục 5, đã chốt B |
| Per-channel | Project dùng per-tensor; đang chờ Thầy |
| Năm chỗ slide Thầy lược bớt | Code đã sửa; 2 cái quan trọng (kẹp, đệm) đã nằm trong bảng RTL |
| PTQ dynamic, FP16 (trong video) | Project không dùng |

---

## Tự kiểm (tự làm rồi mới mở đáp án)

1. Vì sao bộ nhớ chỉ giảm 3,88×, không phải 4×?
2. Vì sao phải nới dải cho chứa 0?
3. acc = 0 ở conv1 → q_out = ? Ứng với số thực nào?
4. Pixel = 200 → q_in = ? x vào bộ nhân conv1 = ?
5. Vì sao trọng số zp = 0 mà activation lại có zp?
6. M dùng để làm gì? Vì sao phải đổi thành M0 và shift?

<details>
<summary>Đáp án</summary>

1. 50 bias giữ int32: 4.968 × 1 + 50 × 4 = 5.168 byte so với 20.072 → 3,88×. Bias nằm trên thước s_in·s_w rất mịn → số lớn (tới 4.604).
2. Đệm và output ReLU là số thực 0. Có 0 trong dải thì 0 trùng đúng một nấc (= zp), không bị làm tròn lệch.
3. (0 + 2^38) >> 39 = 0 → 0 − 128 = **−128** → số thực 0,022875 × 0 = **0**.
4. q_in = 200 − 128 = **72**; x = 72 + 95 = **167** (= 200 − 33).
5. Trọng số cân quanh 0 → đối xứng không phí, zp = 0 bớt phép trừ trong MAC. Activation lệch một phía (sau ReLU ≥ 0) → bất đối xứng để dùng đủ 256 nấc.
6. M đổi acc từ thước s_in·s_w sang thước output s_out. M là số thực, chip không nhân số thực → M ≈ M0/2^shift: nhân số nguyên + dịch bit.

</details>
