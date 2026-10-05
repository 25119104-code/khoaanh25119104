# Ôn tập theo flow 7 bước của Thầy — từ AI model tới FPGA

> Bám đúng slide "FLOW XỬ LÝ: TỪ AI MODEL ĐẾN FPGA/EMBEDDED SYSTEM" (video Step2 của Thầy).
> Mục tiêu: **hiểu ngọn ngành từng phép toán, tự tính được bằng tay, nói được vì sao chọn/bỏ**,
> để báo cáo thứ Tư 07/10/2026.
>
> Cách dùng: đọc phần giải thích → làm **bài tập trên giấy** (ô ✏️) → chạy
> `python study/kiem_tra_tinh_tay.py <số bài>` để chấm. Đáp án nằm ở phụ lục cuối file —
> **đừng mở trước khi làm**.

---

## 0. Bản đồ: thư mục đang ở đâu trên flow

| Bước | Trạng thái | File trong project |
|---|---|---|
| 1. AI / Algorithm | ✅ | `models.py`, `model_comparison.py`, `final_table.py`, `reports/` |
| 2. Phân tích model | ✅ | `docs/buoc3-giai-thich-y-nghia.docx`, `study/bai-tap-01-conv-params.md` |
| 3. Phân tích operator | ✅ | `operators/` (6 file × 7 mục), `figures/conv_*.png` |
| 4. Hardware-friendly | 🟡 | Đã bỏ softmax, đổi relu/pool, đệm 0. **Chưa INT8** (`study/xem_quantization.py` mới khảo sát) |
| 5. Inference model | ✅ | Không Dropout, không BN; `study/thi-nghiem-batchnorm.md` (đã thử gộp BN) |
| 6. Tham số + golden model C | ✅ **mới xong** | `export_params.py`, `params/`, `golden/`, `inference_python.py`, **`golden_c/golden_model.c`** |
| 7. FPGA / RTL | ⬜ | Chưa làm — bắt đầu bằng giải pháp 1 (mỗi layer 1 khối) theo video Step3 |

## Lịch ôn 3 buổi

| Buổi | Nội dung | Bài tập |
|---|---|---|
| **Thứ Hai 05/10** | Bước 1, 2, 3 — train/inference, backprop, shape, 6 operator | Bài 1, 2, 3, 4, 5, 8 |
| **Thứ Ba 06/10** | Bước 4, 5 — fixed-point, bỏ phép chia/exp, gộp BN, Dropout | Bài 6, 7, 9, 10, 11 |
| **Thứ Tư 07/10 (sáng)** | Bước 6, 7 + tự trả lời 15 câu hỏi ở mục 9 thành tiếng | Chạy golden model C trên máy mình |

---

## 1. Bước 1 — AI / Algorithm: xây và huấn luyện model

### 1.1 Train và inference khác nhau ở đâu

| | Train | Inference |
|---|---|---|
| Trọng số | Thay đổi sau mỗi batch | **Cố định**, chỉ đọc |
| Chiều tính | Forward → loss → **backward** → cập nhật | **Chỉ forward** |
| Cần | Nhãn, loss, gradient, learning rate, optimizer | Ảnh + trọng số |
| Chạy ở đâu | PC, 1 lần | Chip, mãi mãi |

Inference là một hàm cố định `y = f(x; W)`. **FPGA chỉ làm inference.**

### 1.2 Vòng train — 4 khái niệm phải nói được

1. **Forward:** đưa ảnh qua mạng, ra 10 logit.
2. **Loss** (hàm mất mát): đo dự đoán sai bao nhiêu. Project dùng `CrossEntropyLoss` = softmax + log, chỉ tồn tại lúc train.
3. **Backprop** (lan truyền ngược): dùng quy tắc đạo hàm chuỗi tính `∂L/∂w` cho **mọi** trọng số — trọng số đó góp bao nhiêu vào cái sai.
4. **Cập nhật** (gradient descent): `w ← w − lr · ∂L/∂w`. Đi ngược hướng đạo hàm để loss giảm. Project dùng Adam — một biến thể tự điều chỉnh bước nhảy cho từng trọng số, nhưng ý tưởng giống hệt.

Thêm: **epoch** = đi hết 50.000 ảnh train 1 lần; **batch** = 64 ảnh tính chung 1 lần cập nhật; **validation set** = 10.000 ảnh để chọn epoch tốt nhất; **test set** chỉ chạm 1 lần cuối.

### 1.3 ✏️ Bài 1 — Backprop bằng tay (bài Thầy hay cho)

Một neuron tuyến tính `ŷ = w1·x1 + w2·x2 + b`, loss `L = ½(ŷ − y)²`.
Cho `x = (1, 2)`, `w = (0,5 ; −0,3)`, `b = 0,1`, nhãn `y = 1`, learning rate `lr = 0,1`.

a) Tính `ŷ` và `L`.
b) Tính `∂L/∂w1`, `∂L/∂w2`, `∂L/∂b`. *(Gợi ý: `∂L/∂ŷ = ŷ − y`; `∂ŷ/∂w1 = x1`.)*
c) Cập nhật 1 bước, tính `ŷ` và `L` mới. Loss có giảm không?

> Điều cần hiểu: trọng số nào gắn với input lớn (`x2 = 2`) thì bị chỉnh mạnh hơn. CNN thật
> làm đúng việc này cho 5.018 trọng số cùng lúc, lặp hàng nghìn lần.

---

## 2. Bước 2 — Phân tích model: kiến trúc và kích thước tensor

### 2.1 SmallCNN — đi từng lớp

| Lớp | Tensor ra | Params | Công thức params |
|---|---|---|---|
| input | 1×28×28 | 0 | |
| conv1 + ReLU | 8×28×28 | 80 | 3·3·1·8 + 8 |
| maxpool | 8×14×14 | 0 | |
| conv2 + ReLU | 16×14×14 | 1.168 | 3·3·8·16 + 16 |
| maxpool | 16×7×7 | 0 | |
| conv3 + ReLU | 16×7×7 | 2.320 | 3·3·16·16 + 16 |
| maxpool | 16×3×3 | 0 | 7 → 3 do `floor(7/2)` |
| flatten | 144 | 0 | |
| fc | 10 | 1.450 | 144·10 + 10 |
| **Tổng** | | **5.018** | |

Hai công thức nền:

```
H_out = floor((H_in + 2·pad − kernel) / stride) + 1
params_conv = k·k·C_in·C_out + C_out          ← KHÔNG có kích thước ảnh
MAC_conv    = k·k·C_in·C_out·H_out·W_out      ← CÓ kích thước ảnh
```

**Params quyết định bộ nhớ, MAC quyết định tốc độ** — hai trục khác nhau. conv1 ít hơn conv3 29 lần params nhưng chỉ ít hơn 2 lần MAC.

### 2.2 ✏️ Bài 8 — Đếm phép tính 1 ảnh

Tính số MAC của conv1, conv2, conv3, số phép ReLU, số phép so sánh của MaxPool (mỗi ô 2×2 cần 3 phép so sánh), MAC của FC, so sánh của argmax. Tổng phải ra con số đã biết của project.

Bài shape/params nâng cao: `study/bai-tap-01-conv-params.md` (5 bài, có bẫy `floor`).

---

## 3. Bước 3 — Phân tích operator

Đường inference chỉ có **6 operator**:

    conv2d → relu → maxpool  (×3)  → flatten → linear → argmax

Chi tiết đủ 7 mục mỗi operator ở `operators/`. Phần dưới là cốt lõi phải thuộc.

### 3.1 Conv2d — tính 1 điểm output

```
out[oc][oy][ox] = bias[oc] + Σ_ic Σ_ky Σ_kx  in[ic][oy−1+ky][ox−1+kx] · w[oc][ic][ky][kx]
```

> **Quy ước:** `in[ic][r][c] = 0` khi `r` hoặc `c` nằm ngoài 0…H−1 (zero-padding).
> Cài đặt bằng mảng đệm (golden C): `in[ic][oy−1+ky][ox−1+kx] = padded[ic][oy+ky][ox+kx]`.
> `−1` ở đây và `+1` lúc chép ảnh vào `padded` triệt tiêu nhau, nên trong code không còn `−1`.
> Công thức toán không đổi; chỉ cách đánh chỉ số khác.

- conv1: 1·9 = **9 MAC**/điểm; conv2: 8·9 = **72**; conv3: 16·9 = **144**.
- **Mọi kênh vào dồn vào MỘT accumulator** → ra một số duy nhất. Kênh vào không cho output riêng.
- Accumulator **khởi tạo bằng bias**, không phải 0.
- Hình minh hoạ số thật: `figures/conv_1diem_conv1.png`, `conv_1diem_conv2.png`.

✏️ **Bài 2.** Ảnh test số 0 (chữ "7"), kênh ra 0 của conv1, vị trí (26, 12). Cửa sổ 3×3 ảnh vào (đã chuẩn hoá) và kernel:

```
ảnh vào                          kernel conv1[0]                 bias[0] = 0,4013
 2,8088   2,8088   2,3633         0,3191   0,2397  −0,3606
 2,8088   2,2105  −0,1951         0,3299   0,2259   0,3618
−0,4242  −0,4242  −0,4242        −0,3522   0,0247   0,2187
```

Tính 9 tích, cộng lại, cộng bias. So với PyTorch `+2,520520`. Vì sao lệch nhẹ?

✏️ **Bài 3.** Điểm ở góc (0, 0) cùng kernel. Ảnh đã đệm 0 nên cửa sổ là:

```
 0        0        0
 0       −0,4242  −0,4242
 0       −0,4242  −0,4242
```

a) Bao nhiêu tap rơi vào vùng đệm? b) Tính output (cần kernel đủ độ chính xác → dùng 4 số lẻ ở trên).
c) Nếu đệm **sai** bằng màu nền −0,4242 thay vì 0 thì lệch bao nhiêu?

### 3.2 ReLU

`relu(x) = max(0, x)` — chỉ **1 phép so sánh**, không nhân. Rẻ nhất trên FPGA (bit dấu = 1 thì xuất 0).
Vì sao cần: không có ReLU thì nhiều lớp conv chồng lên nhau vẫn chỉ là **một** phép tuyến tính lớn — mạng sâu vô nghĩa.

### 3.3 MaxPool 2×2

Lấy số lớn nhất trong mỗi ô 2×2, bước 2 → kích thước giảm một nửa. **3 phép so sánh/ô, 0 tham số.**
Bẫy: 7×7 → 3×3 (hàng/cột cuối bị bỏ), không phải 4×4.

✏️ **Bài 4.** a) Khối `[[−1,2 ; 0,7] ; [0,3 ; −0,5]]`: tính `maxpool(relu(x))` và `relu(maxpool(x))`. Thử lại với khối toàn âm.
b) Đếm số phép ReLU nếu đặt ReLU **trước** pool và **sau** pool.
c) Vì sao mẹo đổi thứ tự **sai** với AvgPool? Thử khối `[−2, 2, 0, 0]`.

### 3.4 Flatten

**Không tốn phép tính nào** — dữ liệu đã nằm liền trong bộ nhớ, chỉ đọc tiếp như vector 144.
Nhưng **thứ tự** phải đúng `[c][y][x]` (NCHW) như lúc train.

✏️ **Bài 5.** Phần tử `pool3[c = 5][y = 2][x = 1]` nằm ở vị trí bao nhiêu trong vector 144? Nếu viết C theo kiểu NHWC thì nó bị đọc ở vị trí nào?

> Sai layout ở đây **không crash**, chỉ làm accuracy tụt. Chỉ phát hiện được bằng cách so từng phần tử.

### 3.5 Linear (fully connected)

`z[o] = bias[o] + Σ_i x[i] · w[o][i]` — 144 × 10 = **1.440 MAC**, nhưng **1.440 trọng số chỉ dùng 1 lần mỗi cái**.
Conv2 dùng lại mỗi trọng số 196 lần (14×14 vị trí). Vì vậy FC **nghẽn băng thông bộ nhớ**, không nghẽn phép nhân: thêm bộ nhân không làm FC nhanh hơn.

### 3.6 Argmax

Chọn chỉ số logit lớn nhất → chữ số dự đoán. **9 phép so sánh.** Dùng so sánh **chặt `>`** để khớp `torch.argmax` khi hai logit bằng nhau.

---

## 4. Bước 4 — Hardware-friendly AI model

Ý tưởng chung: **mọi thứ tính trước được thì tính trên PC, chip chỉ còn +, ×, so sánh.**
Slide của Thầy có 4 phép biến đổi:

### 4.1 Floating point → Fixed point / INT8

**Float32** = 1 bit dấu + 8 bit mũ + 23 bit định trị. Phép nhân float cần tách mũ, nhân định trị, chuẩn hoá lại → tốn nhiều LUT và vài chu kỳ.
**Fixed-point** = số nguyên, dấu phẩy nằm ở vị trí cố định do mình quy ước. Nhân fixed-point chỉ là nhân số nguyên → 1 khối DSP.

**Q1.7** (8 bit): 1 bit dấu/phần nguyên + 7 bit thập phân. Giá trị thật = `q / 128`.

| | Giá trị |
|---|---|
| Bước nhảy nhỏ nhất | 1/128 = 0,0078125 |
| Lớn nhất | 127/128 = 0,9921875 |
| Nhỏ nhất | −128/128 = −1 |

Đổi: `q = làm_tròn(w · 128)`, rồi **kẹp** vào [−128, 127] (bão hoà — saturation).

✏️ **Bài 6.** Đổi sang Q1.7 và tính sai số: `w = 0,319135` (trọng số đầu tiên của conv1), `w = −0,7234`, `w = 1,03`. Số nào bị kẹp?

> Liên hệ project: max|w| của conv2 = 0,9813, chỉ cách trần 1,1%. Thí nghiệm BN cho thấy
> gộp BN làm conv1 lên 1,12 — **vượt trần**. Vì vậy quantization thật phải chọn scale
> **riêng từng lớp**, không cố định Q1.7. *(Thầy dặn chưa cần làm phần này.)*

### 4.2 Complex Activation → ReLU / LUT / xấp xỉ

Sigmoid `1/(1 + e^−x)` và tanh cần `exp` và phép chia — rất đắt trên phần cứng. Cách xử lý:
- **Thay bằng ReLU** (rẻ nhất) — **SmallCNN đã dùng ReLU từ đầu**, nên không cần làm gì.
- **LUT** (Look-Up Table): tính sẵn kết quả cho mọi giá trị input (ví dụ 256 giá trị với int8), lưu vào bảng; chip chỉ cần **tra bảng** thay vì tính.
- **Xấp xỉ từng đoạn** (piecewise linear): thay đường cong bằng vài đoạn thẳng.

### 4.3 Division → Shift / Multiply

- Chia cho `2^n` = **dịch phải n bit**: `200 / 8 = 200 >> 3 = 25`. Gần như miễn phí trong phần cứng (chỉ là nối dây).
- Chia cho hằng số bất kỳ = **nhân với nghịch đảo tính sẵn**.
- Trong INT8 thật, phép đổi scale `× 0,00123…` được viết thành **nhân số nguyên rồi dịch bit** (`multiplier` + `shift` mà Thầy nhắc trong video).

✏️ **Bài 7.** Chuẩn hoá ảnh `(px/255 − 0,1307)/0,3081` có 2 phép chia. Viết lại dạng `a·px + b`, tính `a`, `b`. Kiểm tra với pixel 0, 128, 255.

### 4.4 Exp() → LUT / xấp xỉ — hoặc bỏ hẳn

Softmax `p_i = e^{z_i} / Σ e^{z_j}` có 10 `exp` + phép chia. Nhưng `exp` **đơn điệu tăng** và mẫu số chung cho mọi lớp, nên **thứ tự không đổi**: `argmax(softmax(z)) = argmax(z)`. → **Bỏ hẳn softmax**, giống hệt bit-for-bit, không cần LUT.

✏️ **Bài 9.** Logit `z = (2,0 ; 1,0 ; 0,1)`. Tính softmax (cho sẵn `e² ≈ 7,3891`, `e¹ ≈ 2,7183`, `e^0,1 ≈ 1,1052`). So argmax của `z` và của softmax.

### 4.5 Những gì project đã / chưa làm ở bước 4

| Biến đổi trên slide | SmallCNN |
|---|---|
| Float → INT8 | ⬜ chưa (Thầy: chưa cần) |
| Complex activation → ReLU | ✅ dùng ReLU từ đầu |
| Division → shift/multiply | ✅ trên chip không còn phép chia; chuẩn hoá đã viết được dạng `a·x + b` (câu hỏi đang chờ Thầy: làm trên PC hay chip) |
| Exp → LUT | ✅ bỏ hẳn softmax |
| Thêm: đổi thứ tự relu ↔ pool | ✅ 10.192 → 2.496 phép ReLU (4,08 lần) |
| Thêm: đệm 0 thay `if` | ✅ luôn 9 tap — đều, không cần mạch so sánh biên |

**Vì sao đệm 0 thay vì `if`** (Thầy góp ý 23/09): conv 3×3 trên FPGA là 9 bộ nhân **song song**, 1 điểm/1 chu kỳ. Bỏ 5 tap ở góc thì 5 bộ nhân ngồi không — **không nhanh hơn**, lại phải thêm bộ so sánh `0 ≤ iy < H` + MUX cho từng tap (tốn LUT). Giá: 395.136 thay vì 351.008 MAC (+11,2%), kết quả giống hệt.

---

## 5. Bước 5 — Xây dựng inference model

Mục tiêu: lấy model lúc train, **gỡ mọi thứ chỉ có nghĩa lúc train**, gộp những gì gộp được.

### 5.1 Dropout — xoá

Lúc train, Dropout tắt ngẫu nhiên p% neuron để model không học thuộc lòng; neuron còn lại nhân `1/(1−p)` để giá trị trung bình không đổi. **Lúc inference Dropout là phép đồng nhất** (không làm gì) → xoá khỏi đường inference.
SmallCNN **không có Dropout** (DigitCNN cũ có `Dropout(0.25)`).

✏️ **Bài 11.** `v = (1, 2, 3, 4)`, mask `(1, 0, 1, 0)`, `p = 0,5`. Tính đầu ra lúc train và lúc inference.

### 5.2 BatchNorm — gộp vào Conv

BN mỗi kênh: `y = γ·(x − μ)/√(σ² + ε) + β`. Lúc inference `μ, σ²` cố định → BN chỉ còn `a·x + c`:

```
a        = γ / √(σ² + ε)
w_mới    = a · w
bias_mới = β − a·μ        (+ a·bias_cũ nếu conv có bias)
```

Gộp xong **BN biến mất**: không `√`, không phép chia, không thêm MAC.

✏️ **Bài 10.** Conv `w = 0,3`, input 2,0; BN `γ = 1,5`, `β = 0,1`, `μ = 0,2`, `σ² = 0,25` (bỏ ε). Tính kết quả chưa gộp và sau khi gộp — phải bằng nhau.

> Project: SmallCNN không có BN. Đã **tự thử** ở `study/thi-nghiem-batchnorm.md`: thêm BN không
> tăng accuracy (val 98,80% vs 98,79%, test 98,71% vs 98,82%) nhưng train nhanh gấp đôi; gộp xong
> **0/10.000 ảnh khác**, nhưng conv1 to lên 1,12 — vượt trần Q1.7.

### 5.3 Fusion (gộp) khác trên slide

- **Conv + Activation fusion:** áp ReLU **ngay khi accumulator xong**, không ghi kết quả conv ra bộ nhớ rồi đọc lại. Trong phần cứng = nối thẳng khối ReLU sau bộ cộng dồn → tiết kiệm 1 lượt đọc/ghi bộ nhớ.
- **FC + BN fusion:** giống gộp Conv + BN, áp dụng cho lớp fully connected.

### 5.4 Softmax → ArgMax

Xem 4.4. Kết quả: đường inference **không còn hàm siêu việt** nào — golden model C không cần `math.h`.

---

## 6. Bước 6 — Trích xuất tham số và golden model C

### 6.1 Trích tham số (`export_params.py`)

- **2 file** như Thầy dặn: `params/weights.txt` (4.968 số), `params/biases.txt` (50 số), 4 lớp nối tiếp. Giống phần cứng: mọi trọng số trong **một** bộ nhớ, mỗi lớp là một địa chỉ gốc (offset).
- Bảng offset trong `params/README.md`: conv1 từ 0, conv2 từ 72, conv3 từ 1.224, fc từ 3.528.
- In `%.9g` — 9 chữ số là **tối thiểu** để float32 → chữ → float32 không mất bit. Đọc lại khớp **từng bit**.
- Kèm `golden/`: đầu ra PyTorch **từng lớp** của 1 ảnh mẫu + logit của 10.000 ảnh test → chuẩn để so.

### 6.2 Golden model là gì, để làm gì

**Golden model** = chương trình **chuẩn đối chiếu**: viết lại inference bằng vòng lặp thuần, mọi phép toán lộ ra rõ ràng. Khi viết RTL, mỗi khối Verilog phải ra **đúng** số của golden model → tìm lỗi phần cứng bằng cách so từng lớp.

Thứ tự đã làm: PyTorch → **Python thuần** (`inference_python.py`, dễ debug) → **C** (`golden_c/golden_model.c`).

### 6.3 Kết quả golden model C (float32)

```
cc -O2 -std=c99 -Wall -o golden_c/golden_model golden_c/golden_model.c
./golden_c/golden_model
```

| Kiểm tra | Kết quả |
|---|---|
| Thư viện dùng | chỉ `stdio.h`, `stdlib.h` — không `math.h` |
| Số MAC conv / ảnh | 395.136 (đúng công thức, đệm 0 luôn 9 tap) |
| Lệch so PyTorch từng lớp (ảnh mẫu) | ≤ 3e-06 ở các lớp conv/pool, 1,9e-05 ở logit |
| 10.000 ảnh test | **98,82%**, trùng dự đoán PyTorch **10.000/10.000**, logit lệch ≤ 3,05e-05 |
| Thời gian | ~3 giây cho 10.000 ảnh |

Lệch ~1e-05 là do **thứ tự cộng float32** khác PyTorch, không phải sai thuật toán. Sai thuật toán (quên bias, sai `−PAD`, sai layout) sẽ lệch từ ~0,1 trở lên.

### 6.4 Ba bẫy khi viết C (đều không crash, chỉ ra số sai)

1. Layout flatten NCHW vs NHWC (bài 5).
2. Thứ tự chiều trọng số: PyTorch `[out][in][ky][kx]`, Keras ngược lại.
3. Accumulator khởi tạo bằng 0 thay vì bias → lệch đúng bằng bias.

---

## 7. Bước 7 — Vi mạch / FPGA (xem trước)

- **Từ vựng FPGA:** LUT (khối logic nhỏ, làm phép so sánh/MUX), **DSP** (khối nhân-cộng cứng), **BRAM** (bộ nhớ trên chip — chứa trọng số), **chu kỳ clock**, **pipeline** (chia việc thành tầng chạy gối nhau).
- **Ba giải pháp thiết kế** (video Step3): (1) mỗi layer 1 khối riêng — dễ, tốn diện tích, **Thầy khuyên làm trước**; (2) 1 khối tính dùng chung + FSM điều khiển; (3) accelerator chạy theo tập lệnh.
- **Golden model C dùng thế nào:** testbench Verilog nạp cùng ảnh, cùng trọng số; so đầu ra từng module với `golden/` / golden C.

---

## 8. Bảng từ khoá Thầy dùng

| Từ khoá | Nghĩa ngắn |
|---|---|
| Operator | Một phép toán cơ bản của mạng (conv, relu, pool…) |
| MAC | Multiply-Accumulate: `acc += a·b` — đơn vị đếm phép tính |
| Tensor | Mảng nhiều chiều, ví dụ 8×28×28 |
| Weight / Bias | Tham số học được: hệ số nhân / số cộng thêm |
| Inference | Chỉ chạy forward với trọng số cố định |
| Golden model | Chương trình chuẩn để đối chiếu phần cứng |
| Hardware-friendly | Biến đổi model để phần cứng làm rẻ (không chia, không exp, số nguyên) |
| Quantization | Đổi float sang số nguyên ít bit (int8) |
| Fixed-point | Số nguyên với dấu phẩy ở vị trí quy ước, ví dụ Q1.7 |
| LUT (toán) | Bảng tra kết quả tính sẵn |
| Fusion / Folding | Gộp 2 phép thành 1 (Conv + BN, Conv + ReLU) |
| Co-design | Thiết kế model AI và phần cứng cùng lúc, cái này ảnh hưởng cái kia |
| Saturation | Kẹp giá trị vào khoảng biểu diễn được |
| RTL | Register-Transfer Level — mô tả phần cứng bằng Verilog/VHDL |

---

## 9. Câu Thầy có thể hỏi — tự trả lời thành tiếng

1. Train và inference khác nhau ở đâu? FPGA làm phần nào?
2. Tính 1 điểm output của conv2 thế nào? Bao nhiêu phép nhân?
3. Vì sao accumulator khởi tạo bằng bias?
4. Vì sao conv1 ít params mà vẫn nhiều phép tính?
5. Vì sao bỏ được softmax? Có phải xấp xỉ không?
6. Vì sao đổi relu ↔ maxpool được? Sao không đúng 4 lần? Sao AvgPool không làm được?
7. Vì sao đệm 0 tốt hơn `if` trên FPGA, dù nhiều phép nhân hơn?
8. Đệm bằng giá trị gì? Vì sao không phải màu nền?
9. BatchNorm là gì? Gộp vào Conv thế nào? Model em có BN không?
10. Dropout lúc inference làm gì?
11. Fixed-point Q1.7 là gì? Đổi 0,319 sang Q1.7?
12. Chia cho hằng số trên phần cứng làm thế nào?
13. Vì sao 2 file tham số mà không tách theo lớp? Vì sao in `%.9g`?
14. Golden model C kiểm chứng thế nào? Lệch 1e-05 có phải lỗi không?
15. Bước tiếp theo là gì? (quantization hay RTL trước — câu hỏi mở cho Thầy)

---

## Phụ lục — Đáp án (chỉ mở sau khi làm xong)

Chạy `python study/kiem_tra_tinh_tay.py` để xem đầy đủ từng bước.

| Bài | Đáp án |
|---|---|
| 1 | ŷ = 0 ; L = 0,5 ; ∂L/∂w1 = −1, ∂L/∂w2 = −2, ∂L/∂b = −1 ; w = (0,6 ; −0,1), b = 0,2 ; ŷ mới = 0,6, L mới = 0,08 |
| 2 | Σ 9 tích = 2,118892 ; + 0,4013 = **2,520192** ; PyTorch 2,520520 — lệch do làm tròn đầu vào còn 4 chữ số |
| 3 | 5 tap vùng đệm ; output = **0,048754** (khớp PyTorch) ; đệm sai bằng nền → −0,025918, lệch −0,0747 |
| 4 | Cả hai thứ tự đều 0,7 ; khối toàn âm cả hai = 0 ; ReLU 10.192 → 2.496 ; AvgPool: avg(relu) = 0,5 ≠ relu(avg) = 0 |
| 5 | NCHW: 5·9 + 2·3 + 1 = **52** ; NHWC sai: (2·3 + 1)·16 + 5 = **117** |
| 6 | 0,319135 → q = 41 → 0,3203125 (sai số +0,0012) ; −0,7234 → −93 → −0,7265625 ; 1,03 → **kẹp 127** → 0,9921875 (sai số −0,0378) |
| 7 | a = 1/(255·0,3081) = 0,01272823 ; b = −0,42421292 ; pixel 0 → −0,424213, 128 → 1,205001, 255 → 2,821487 |
| 8 | 56.448 + 225.792 + 112.896 = 395.136 MAC conv ; ReLU 10.192 ; pool 7.488 ; FC 1.440 ; argmax 9 ; **tổng 414.265** |
| 9 | softmax ≈ (0,659 ; 0,2424 ; 0,0986) ; argmax đều = 0 |
| 10 | Chưa gộp: 1,5·(0,6 − 0,2)/0,5 + 0,1 = 1,3 ; a = 3, w_mới = 0,9, bias_mới = −0,5 ; 0,9·2 − 0,5 = **1,3** |
| 11 | Train: (2, 0, 6, 0) ; inference: (1, 2, 3, 4) |
