# Ôn tập INT8: PTQ, QAT (Python) và Golden C int8

Tài liệu ôn để báo cáo Thầy tuần 09/11/2026. Mọi số ở đây lấy từ code và kết quả chạy thật trong project.
Đọc theo thứ tự 1 → 8. Mỗi mục có **câu hỏi tự kiểm** ở cuối, đáp án ở mục 9.

| File cần mở khi ôn | Nội dung |
|---|---|
| `ptq_int8.py` | PTQ số nguyên thuần |
| `qat.py` | QAT fake-quant + STE |
| `params_int8/tham_so.txt` | scale, zp, M0, shift từng lớp |
| `golden_c/golden_int8.c` | Golden model C int8 |
| `reports/report-tuan06-20261014.md` | Report PTQ/QAT |
| `reports/report-dot2-tuan01-20261104.md` | Report golden C int8 + Vivado |
| `docs/slide-quantization-tuan06.pptx` | Slide PTQ/QAT nộp Thầy |

---

## 1. Vì sao phải chuyển sang INT8

- FPGA làm phép **nhân số nguyên** rẻ (khối DSP), còn số thực float32 thì tốn rất nhiều LUT.
- Bộ nhớ tham số: float32 20.072 byte → int8/int32 5.168 byte, **nhỏ hơn 3,88 lần**.
  Không phải đúng 4 lần vì 50 bias vẫn giữ 32 bit.
- Mục tiêu: đường suy luận **không còn phép float nào**, chỉ còn nhân, cộng, dịch bit, kẹp.

**Tự kiểm 1.** Vì sao bộ nhớ chỉ giảm 3,88 lần mà không phải 4 lần?

---

## 2. Công thức lượng tử

Mỗi số thực viết lại thành:

    r = s · (q − zp)

| Ký hiệu | Ý nghĩa | Ví dụ thật |
|---|---|---|
| r | số thực | |
| q | số nguyên int8 lưu trên chip, [−128, 127] | |
| s | scale: bước nhảy giữa 2 mức int | ảnh vào: s = 0,012728 |
| zp | zero-point: số nguyên biểu diễn số thực 0 | ảnh vào: zp = −95 |

Từ dải thực [lo, hi] (dải luôn được nới cho chứa 0):

    s  = (hi − lo) / 255
    zp = −128 − lo / s      (làm tròn)

Hai kiểu:

| | Activation (dữ liệu chạy qua mạng) | Trọng số |
|---|---|---|
| Kiểu | **Bất đối xứng**, có zp | **Đối xứng**, zp = 0 |
| Dải int | [−128, 127] | [−127, 127] |
| Scale | (hi − lo)/255, đo bằng calibrate | max\|w\| / 127 |
| Vì sao | sau ReLU toàn số ≥ 0, dải lệch hẳn một phía | zp = 0 thì khỏi trừ zp trong vòng MAC |

**Bias:** int32, scale = s_vào × s_w, zp = 0 (cộng thẳng vào acc được).

**Tự kiểm 2.** Vì sao phải nới dải cho chứa 0? (Gợi ý: đệm ảnh và đầu ra ReLU đều là số thực 0.)

---

## 3. Một lớp conv chạy bằng số nguyên

    acc  = bias + Σ (q_vào − zp_vào) · q_w                         ← int32, cộng dồn
    q_ra = kẹp( ((acc · M0 + 2^(shift−1)) >> shift) + zp_ra, −128, 127 )   ← requantize
    pool = max 2×2 trên q_ra

Vì sao có M0 và shift:

- Đổi acc (scale s_vào·s_w) sang scale đầu ra s_ra cần nhân **M = s_vào·s_w / s_ra**. M là số thực, cỡ 0,0038.
- Viết M ≈ **M0 · 2^−shift**, với M0 là số nguyên 31 bit. Nhân số thực biến thành **nhân số nguyên + dịch phải**.
- conv1: M = 0,00381027276 ≈ 2.094.719.601 · 2^−39.

### Ví dụ tính tay (số thật, conv1, ảnh mẫu, vị trí 202)

| Bước | Phép tính | Kết quả |
|---|---|---|
| acc (từ `01_conv1_acc.txt`) | | 21.313 |
| Nhân M0 | 21.313 × 2.094.719.601 | 44.644.758.856.113 |
| Cộng 2^38 (làm tròn) | + 274.877.906.944 | 44.919.636.763.057 |
| Dịch phải 39 bit | ÷ 2^39 rồi floor: 81,708 → 81 | **81** (= 81,208 làm tròn gần nhất) |
| Cộng zp_ra | 81 + (−128) | **−47** |
| Kẹp [−128, 127] | | −47, khớp `02_relu1.txt` ✓ |

Ví dụ thứ hai, acc âm (vị trí 176): acc = −4.247 → (…) >> 39 = −16 → −16 − 128 = −144 → **kẹp lên −128**.
−128 chính là số thực 0: **bước kẹp đã làm luôn ReLU**.

### ReLU gộp vào bước kẹp (cấu hình B)

Lấy scale đầu ra theo dải **sau** ReLU [0, max] thì zp_ra = −128. Mọi giá trị âm bị kẹp về −128 = số thực 0.
Không tốn bộ so sánh ReLU nào, và 256 mức int8 dùng hết cho phần dương (độ phân giải gấp đôi).

### Năm chỗ slide của Thầy lược bớt (đã sửa trong code)

1. Thiếu **bước kẹp** về [−128, 127] sau khi cộng zp → tràn int8.
2. `Σ input·weight` chỉ đúng khi zp_vào = 0. Ở đây zp_vào = −95 (conv1) và −128 → phải tính `Σ (q − zp)·w`.
3. ReLU ghi "if (>zp), set to zp" bị **ngược dấu**. Đúng: nếu q < zp thì q = zp.
4. **Đệm bằng zp, không đệm số 0**: trừ zp trước rồi mới đệm 0.
5. **Làm tròn và dịch bit:** không dùng `np.round` (làm tròn về số chẵn, 2,5 → 2). Dùng `floor(x + 0,5)` ở mọi chỗ.

**Tự kiểm 3.** Tính tay requantize cho acc = 0 ở conv1. Ra q bằng bao nhiêu, và nghĩa là số thực bao nhiêu?

---

## 4. Đầu vào: chuẩn hoá biến mất

- Dải ảnh sau chuẩn hoá [−0,4242; 2,8215] đúng bằng pixel 0 và 255 → s_in = 1/(255·std) = 0,012728, zp_in = −95.
- Ghép lại: **q_in = pixel − 128**, khớp **7.840.000/7.840.000** pixel test.
- Trừ 128 trên số 8 bit = **đảo bit cao nhất**. Chip nhận thẳng uint8, không cần phép chia, không cần `a·x + b`.
  Câu hỏi "chuẩn hoá làm ở đâu" từ tuần 4 tự có lời giải.
- conv1 nhận `x = q − zp = pixel − 33`, nằm trong [−33, 222]: cần bộ nhân 9 bit có dấu.

**Tự kiểm 4.** Pixel = 200 thì q_in bằng bao nhiêu, và x đưa vào bộ nhân conv1 bằng bao nhiêu?

---

## 5. PTQ — Post-Training Quantization (`ptq_int8.py`)

**Ý nghĩa:** lấy model **đã train xong**, không train lại, chỉ đo dải giá trị rồi đổi sang int8.

Các bước trong code:

1. Nạp `models/small_cnn.pth` (float32, 98,82%).
2. **Calibrate** trên 1.000 ảnh **validation** (không dùng test): đo min/max đầu vào và đầu ra từng lớp.
3. Lượng tử trọng số (đối xứng), bias (int32), tính s, zp, M0, shift từng lớp.
4. Chạy suy luận **số nguyên thuần** (tự viết, không dùng `torch.ao.quantization` vì đó là hộp đen, không cho số
   nguyên từng lớp để so với C và RTL).
5. So 6 cấu hình trên **val**, chọn 1, chạy **test đúng 1 lần**.
6. Xuất `params_int8/`: trọng số, bias, `tham_so.txt`, ảnh mẫu từng lớp.

| Cấu hình (trên val) | Acc | Khác float |
|---|---|---|
| A. Đúng slide Thầy: ReLU riêng, M0 31 bit | 98,78% | 9 |
| **B. ReLU gộp vào kẹp, M0 31 bit** (chọn) | 98,77% | 9 |
| C. Như B, calibrate chỉ 100 ảnh | 98,76% | 9 |
| D. Như B, M0 chỉ 15 bit | 98,77% | 9 |
| E. Như B, M0 chỉ 8 bit | 98,76% | 11 |
| F. Thử int4 | 97,69% | 171 |

Chọn B vì A và B chỉ chênh 1 ảnh mà B bỏ được bộ so sánh ReLU.

**Kết quả test (chạy 1 lần):** float32 98,82% → **PTQ int8 98,84%**. Khác float 9/10.000 ảnh
(3 ảnh float đúng → int8 sai, 5 ảnh ngược lại).

**Tự kiểm 5.** Vì sao calibrate trên val chứ không phải test? Vì sao không dùng `torch.ao.quantization`?

---

## 6. QAT — Quantization-Aware Training (`qat.py`)

**Ý nghĩa:** train tiếp vài epoch mà **forward giả vờ lượng tử**, để model học cách chịu sai số làm tròn.

    x_fq = x + (lượng_tử(x) − x).detach()

- **Forward:** ra đúng giá trị đã làm tròn về lưới int (fake-quant).
- **Backward:** STE (Straight-Through Estimator). Đạo hàm của làm_tròn() bằng 0 gần như khắp nơi, nên coi làm_tròn như
  phép đồng nhất, đạo hàm = 1, gradient chảy qua như không có làm_tròn.
- Dải sau ReLU cập nhật bằng trung bình trượt (EMA, momentum 0,01).
- Train xong, đưa trọng số và dải học được vào **đúng đường số nguyên của PTQ** để đo, vì chip chạy số nguyên chứ
  không chạy fake-quant.

| | Float | PTQ | QAT |
|---|---|---|---|
| int8 (test) | 98,82% | 98,84% | 98,91% |
| int4 (thí nghiệm) | 98,82% | 98,00% | 98,72% |

Đọc kết quả cho đúng:

- **Ở int8, QAT gần như không thêm gì** vì PTQ không tụt. QAT hơn PTQ 7 ảnh, nhưng QAT train thêm 1 epoch, chưa
  tách được là nhờ "biết lượng tử" hay chỉ nhờ train thêm (chưa chạy đối chứng). QAT int8 khác float ở 45 ảnh, PTQ
  chỉ 9: QAT là một model đã đổi, không phải bản lượng tử của cùng model.
- **Ở int4 thì QAT có ích:** kéo lại 0,72 điểm %, khoảng cách với float từ 0,82 còn 0,10.
  Train có ngẫu nhiên nên con số dao động cỡ ±0,2 giữa các lần chạy.
- Kết luận: **QAT chỉ đáng làm khi PTQ tụt rõ**, thường là khi xuống bit thấp.

**Tự kiểm 6.** Giải thích STE trong một câu. Vì sao đo QAT bằng đường số nguyên chứ không đo trên model fake-quant?

---

## 7. Golden C int8 (`golden_c/golden_int8.c`)

**Ý nghĩa:** chuẩn đối chiếu cho RTL. RTL phải ra **đúng từng số** như file này.

- Chỉ `stdio.h`, `stdlib.h`, `stdint.h`. Không `math.h`, không float trên đường suy luận.
- **Không tính lại scale/M0**: đọc hết từ `params_int8/`. Cùng code chạy được cả PTQ và QAT.

| Kiểm tra | PTQ | QAT |
|---|---|---|
| Ảnh mẫu: 11 file từng lớp | 0 số khác | 0 số khác |
| 10.000 ảnh: cả 10 acc lớp fc | 10.000/10.000 khớp từng bit | 10.000/10.000 |
| Accuracy | 98,84% | 98,91% |

Vì sao bản số nguyên phải khớp **từng bit** còn bản float chỉ cần lệch ≤ 1e-4: cộng số thực theo thứ tự khác ra kết quả
khác ở chữ số cuối; cộng số nguyên (không tràn) thì thứ tự nào cũng ra đúng một số. Lệch 1 đơn vị là **lỗi**.

Ba bẫy chỉ bản số nguyên mới có:

1. **Đệm:** trừ zp trước, rồi mới đệm 0.
2. **Dịch phải số âm:** C `>>` trên số âm là implementation-defined; C `/` làm tròn về 0. Phải tự viết `floor`.

   | −7 chia 2 (đáp số −3,5) | Kết quả | Ai dùng |
   |---|---|---|
   | Làm tròn về 0 | −3 | C `/` |
   | Làm tròn xuống | −4 | Python `>>`, Verilog `>>>` |
   | Làm tròn nửa lên (+2^(n−1) rồi floor) | −3 | requantize |

3. **acc × M0 phải 64 bit:** acc cỡ 20 bit nhân M0 31 bit ra 51 bit.

Hai phát hiện cho phần cứng:

- **Thanh ghi acc 20 bit có dấu là đủ cho MỌI ảnh** (chặn trên tính từ trọng số thật, không chỉ đo trên test).
  Đo trên test chỉ 17–19 bit.
- **Requantize đặt SAU maxpool được, giống hệt từng bit:** requantize là hàm không giảm nên
  `max(rq(a), rq(b)) = rq(max(a, b))`. Số lần requantize mỗi ảnh 10.192 → 2.496 (ít hơn 4,08 lần), kiểm 0/10.000 ảnh khác.
  Cùng lý do với đổi ReLU ↔ maxpool ở bản float.

**Tự kiểm 7.** Vì sao requantize đổi chỗ được với maxpool mà không đổi được với avgpool?

---

## 8. Lệnh để demo khi báo cáo

```bash
cd ~/mnist-digit-project && conda activate mnist
python ptq_int8.py                     # PTQ: bảng A–F trên val + test 1 lần (~20 giây)
python qat.py --bits 8                 # QAT int8 (~1 phút, số có thể lệch ±0,2 vì train ngẫu nhiên)
cc -O2 -std=c99 -Wall -Wextra -o golden_c/golden_int8 golden_c/golden_int8.c
./golden_c/golden_int8                 # golden C: ảnh mẫu + 10.000 ảnh, khớp từng bit
./golden_c/golden_int8 10000 params_int8_qat
```

**Lưu ý:** chạy lại `ptq_int8.py` sẽ ghi đè `params_int8/`. Trên Mac thì ra đúng bộ số đã chốt; đừng chạy trên máy khác.
Chỉ cần demo thì chạy golden C là đủ, nó không ghi đè gì.

---

## 9. Đáp án tự kiểm

1. 4.968 trọng số giảm từ 4 byte xuống 1 byte, nhưng 50 bias vẫn 4 byte: (4.968·4 + 50·4)/(4.968 + 50·4) = 20.072/5.168 = 3,88.
2. Để số thực 0 biểu diễn **chính xác** bằng một số nguyên (= zp). Nếu không, ô đệm và đầu ra ReLU = 0 sẽ bị làm tròn lệch,
   sai số dồn qua mọi lớp.
3. acc = 0 → (0 + 2^38) >> 39 = 0 → 0 + (−128) = **−128**, tức số thực **0**. Hợp lý: acc = 0 nghĩa là tổng = 0.
4. q_in = 200 − 128 = **72**. x = q − zp = 72 + 95 = **167** (= 200 − 33).
5. Calibrate là một bước **ra quyết định** (chọn dải). Dùng test để quyết định thì accuracy test không còn khách quan.
   `torch.ao` là hộp đen (fbgemm/qnnpack): không cho số nguyên từng lớp để so với C và RTL.
6. STE: coi phép làm tròn như phép đồng nhất khi lan truyền ngược, vì đạo hàm thật của nó bằng 0 gần như khắp nơi.
   Đo bằng đường số nguyên vì chip chạy số nguyên; fake-quant vẫn là phép tính float, có thể lệch so với chip.
7. Requantize là hàm **không giảm** nên giao hoán với max. AvgPool lấy **trung bình**: requantize có làm tròn và kẹp
   (phi tuyến), nên trung bình của các giá trị đã làm tròn khác làm tròn của trung bình.

---

## 10. Câu Thầy có thể hỏi (trả lời miệng ~30 giây)

| Câu hỏi | Ý chính để trả lời |
|---|---|
| PTQ và QAT khác nhau thế nào? | PTQ: không train, chỉ đo dải rồi đổi. QAT: train tiếp với forward giả lượng tử, backward STE |
| Vì sao em chọn PTQ cho phần cứng? | Int8 không tụt (98,84% so với 98,82%), chỉ khác float 9 ảnh. QAT hơn 7 ảnh nhưng chưa có đối chứng |
| QAT có ích khi nào? | Khi PTQ tụt rõ, ví dụ int4: QAT kéo lại 0,72 điểm % |
| M0 và shift là gì? | M = s_vào·s_w/s_ra là số thực; viết thành M0·2^−shift để chỉ còn nhân số nguyên và dịch bit |
| ReLU nằm ở đâu? | Gộp vào bước kẹp: zp_ra = −128 = số thực 0, giá trị âm bị kẹp về −128 |
| Chuẩn hoá ảnh làm ở đâu? | Biến mất: q_in = pixel − 128 = đảo bit cao nhất |
| Thanh ghi acc bao nhiêu bit? | 20 bit đủ cho mọi ảnh (chặn trên tính từ trọng số), đo trên test 17–19 bit |
| Golden C đúng tới mức nào? | Khớp từng bit: 11/11 file ảnh mẫu, 10 acc fc của 10.000/10.000 ảnh |
| Slide lượng tử của Thầy thiếu gì? | 5 chỗ ở mục 3 |
