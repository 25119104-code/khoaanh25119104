# Bảng thuật ngữ — Project nhận diện chữ số viết tay (AI → IC → ES)

Gom thuật ngữ của cả project vào một chỗ. Ví dụ số lấy từ code và file tham số thật
(`params_int8/tham_so.txt`, `docs/on-tap-int8-ptq-qat-c.md`), không lấy số minh hoạ.

Nguồn cũ vẫn còn, file này **gộp và thay thế** để tra cho nhanh:
`archive/roadmap-digit-recognition-v2.md` (mục 2), `roadmap-digit-recognition-v3.md` (mục 4),
`docs/on-tap-flow-7-buoc.md` (mục 8).

| Nhóm | Dùng khi |
|---|---|
| [A. Nền Deep Learning](#a-nền-deep-learning) | Train model, đánh giá |
| [B. CNN và operator](#b-cnn-và-operator) | Hiểu từng lớp của `SmallCNN` |
| [C. Đưa model lên phần cứng](#c-đưa-model-lên-phần-cứng) | Golden model, FPGA, đếm phép tính |
| [D. INT8 — PTQ, QAT](#d-int8--ptq-qat) | **Đang ôn** — đi kèm `docs/on-tap-int8-ptq-qat-c.md` |
| [E. Verilog và công cụ IC](#e-verilog-và-công-cụ-ic) | 10 bài nhập môn, RTL sau này |

---

## A. Nền Deep Learning

| Thuật ngữ | Giải thích | Trong project |
|---|---|---|
| **Deep Learning** | Nhánh ML dùng mạng neural nhiều lớp, tự học đặc trưng từ dữ liệu | PyTorch |
| **MNIST** | 70.000 ảnh chữ số viết tay 28×28, ảnh xám, 10 lớp (0–9) | Chia 50k train / 10k val / 10k test |
| **Train / Inference** | Train: forward + loss + backward + cập nhật trọng số. Inference: chỉ forward với trọng số cố định | FPGA chỉ làm inference |
| **Epoch** | Một lượt học qua toàn bộ tập train | |
| **Batch size** | Số ảnh đưa vào cùng lúc trong một bước cập nhật | |
| **Loss function** | Đo mức sai so với nhãn thật. Chỉ tồn tại lúc train | `CrossEntropyLoss` — không nằm trên đường inference |
| **Backpropagation** | Tính gradient của loss theo từng trọng số (quy tắc dây chuyền) | |
| **Optimizer** | Dùng gradient để cập nhật trọng số (Adam, SGD) | Chỉ lúc train |
| **Learning rate** | Bước nhảy mỗi lần cập nhật trọng số | |
| **Overfitting** | Học thuộc tập train, đoán kém trên dữ liệu mới | |
| **Dropout** | Tắt ngẫu nhiên neuron khi train. Lúc inference không làm gì | |
| **Validation set** | Tập riêng để ra quyết định (chọn epoch, chọn kiến trúc, chọn cấu hình lượng tử) | Thầy bắt buộc. 10k ảnh |
| **Test set** | Chỉ chạm **một lần** cuối để báo kết quả | |
| **Data leakage** | Dùng test để ra quyết định → con số cuối không còn khách quan | Lý do phải có val |
| **Checkpoint** | File lưu trọng số. Lưu theo **val acc cao nhất**, không phải epoch cuối | `models/*.pth` |
| **Accuracy** | Tỷ lệ đoán đúng | `SmallCNN` 98,82% |
| **Confusion matrix** | Bảng nhãn thật × nhãn đoán, thấy model hay nhầm số nào với số nào | `error_analysis.py` |
| **Seed** | Hạt giống ngẫu nhiên. Đổi seed → kết quả train dao động | QAT chưa chạy nhiều seed |

---

## B. CNN và operator

Đường inference của `SmallCNN` (5.018 params):

    conv2d → relu → maxpool  (×3, kênh 1→8→16→16, kích thước 28→14→7→3)
           → flatten (144) → linear (144→10) → argmax

| Thuật ngữ | Giải thích | Trong project |
|---|---|---|
| **CNN** | Mạng tích chập, chuyên cho ảnh | |
| **Operator** | Một phép biến đổi trong model. Tầng AI là 1 dòng thư viện, tầng phần cứng là 1 khối mạch | 6 operator, `operators/` |
| **Tensor** | Mảng nhiều chiều | 8×28×28 sau conv1 |
| **Channel (kênh)** | Chiều "chiều sâu" của tensor; mỗi kênh là một feature map | 1→8→16→16 |
| **Feature map** | Ảnh đầu ra của một kênh conv | |
| **Kernel** | Cửa sổ trọng số trượt trên ảnh | 3×3 |
| **Conv2d** | Mỗi điểm ra = `bias + Σ_kênh vào Σ_ky Σ_kx in·w` | conv1 9 MAC, conv2 72, conv3 144 / điểm |
| **Padding** | Thêm viền quanh ảnh để giữ kích thước | padding = 1 |
| **Zero-padding** | Đệm bằng **số thực 0 sau chuẩn hoá**, không phải màu nền (−0,4242) | Luôn đủ 9 tap, không dùng `if` |
| **Stride** | Bước trượt của kernel | |
| **ReLU** | `max(0, x)` | Ở INT8 gộp vào bước kẹp |
| **MaxPool2d** | Lấy max trong khối 2×2, giảm kích thước một nửa (`floor`) | 7→3 do `floor` cắt hàng/cột cuối |
| **AvgPool** | Lấy trung bình khối. **Không** đổi chỗ được với ReLU | Không dùng |
| **Flatten** | Duỗi tensor thành vector | 16×3×3 = 144 |
| **NCHW / NHWC** | Thứ tự lưu: kênh trước hay kênh sau. Sai layout → acc rơi ~10% mà **không crash** | Dùng NCHW như PyTorch |
| **Linear / FC** | Lớp kết nối đầy đủ: `y = W·x + b` | 144→10 |
| **Logit** | 10 số ra của lớp cuối, trước softmax | |
| **Softmax** | Đổi logit thành xác suất, dùng `exp`. **Bỏ được** khi inference | `argmax(softmax(z)) = argmax(z)` |
| **Argmax** | Chỉ số của giá trị lớn nhất = chữ số dự đoán. So sánh chặt `>` | |
| **BatchNorm** | Chuẩn hoá đầu ra lớp lúc train. Khi inference gộp được vào Conv | Model chính không có BN |
| **Fold / Fusion** | Gộp 2 phép thành 1 (Conv + BN, Conv + ReLU), kết quả không đổi | `study/thu_batchnorm.py` |

---

## C. Đưa model lên phần cứng

| Thuật ngữ | Giải thích | Trong project |
|---|---|---|
| **AI → IC → ES** | Flow 3 mảng: model AI → vi mạch (RTL) → hệ nhúng | Project đi full flow |
| **Params** | Số trọng số phải lưu → quyết định **bộ nhớ** | 5.018 (giảm 41,2× so với 206.922) |
| **FLOPs / số phép** | Số phép tính mỗi ảnh → quyết định **tốc độ, năng lượng** | 414.265 phép/ảnh |
| **MAC** | Multiply-Accumulate: `acc += a·b`. Đơn vị đếm phép conv | 395.136 MAC conv/ảnh |
| **Accumulator (acc)** | Thanh ghi cộng dồn kết quả MAC. **Khởi tạo bằng bias**, không phải 0 | INT8: 20 bit có dấu là đủ |
| **Mã giả** | Mô tả operator bằng vòng lặp + phép cơ bản, cầu nối toán ↔ C/RTL | `operators/*.md` |
| **Golden model** | Chương trình C thuần làm **chuẩn đối chiếu**. RTL lệch golden = bug phần cứng | `golden_c/` |
| **Bit-exact (khớp từng bit)** | Hai bản ra **đúng cùng một số nguyên**, không chỉ "gần bằng" | Golden C int8 ↔ Python |
| **Hardware-friendly** | Phép rẻ trên phần cứng: cộng, nhân, so sánh, dịch bit. Đắt: chia, `exp`, `sqrt`, `log` | Thầy: tránh `exp`, `sqrt` |
| **Hàm siêu việt** | `exp`, `log`, `sin`, sigmoid, tanh… | Đường inference không còn hàm nào |
| **Co-design** | Thiết kế model và phần cứng cùng lúc, cái này ảnh hưởng cái kia | |
| **FPGA** | Chip có mạch lập trình lại được bằng cách nạp cấu hình | Board chưa chốt |
| **LUT (FPGA)** | Ô logic cơ bản của FPGA, làm hàm logic nhỏ | `if` bỏ padding tốn thêm LUT |
| **LUT (toán)** | Bảng tra kết quả tính sẵn | Khác nghĩa với LUT FPGA |
| **DSP / DSP48** | Khối nhân cứng trong FPGA, nhân 25 × 18 bit | M0 15 bit vừa 1 DSP48 |
| **Băng thông bộ nhớ** | Tốc độ đọc dữ liệu. FC nghẽn băng thông (mỗi trọng số dùng 1 lần), không nghẽn phép nhân | |
| **Fixed-point (Qm.n)** | Số nguyên với dấu phẩy ở vị trí quy ước. Q1.7: 1 bit nguyên (dấu), 7 bit lẻ | Đã thay bằng int8 có scale |
| **Saturation** | Kẹp giá trị vào khoảng biểu diễn được thay vì để tràn | = bước kẹp ở INT8 |

---

## D. INT8 — PTQ, QAT

Cột "Mục" chỉ chỗ đọc trong `docs/on-tap-int8-ptq-qat-c.md`.

### D1. Khái niệm chung

| Thuật ngữ | Giải thích | Ví dụ thật | Mục |
|---|---|---|---|
| **Quantization (lượng tử hoá)** | Đổi số thực float32 sang số nguyên ít bit | float32 → int8 | 1 |
| **int8 / uint8** | Số nguyên 8 bit có dấu [−128, 127] / không dấu [0, 255] | Pixel là uint8 | 1, 4 |
| **int32** | Số nguyên 32 bit, dùng cho bias | 50 bias giữ 32 bit → bộ nhớ chỉ giảm 3,88× | 1, 2 |
| **PTQ** | Post-Training Quantization: lấy model **đã train xong**, đo dải, đổi sang int8. **Không train lại** | 98,82% → 98,84% | 5 |
| **QAT** | Quantization-Aware Training: train tiếp với forward **giả vờ lượng tử** để model quen sai số làm tròn | int8 98,91%; có ích rõ ở int4 | 6 |
| **Calibrate** | Chạy model float qua một ít ảnh để đo min/max từng lớp → tính scale, zp | 1.000 ảnh **val**, không dùng test | 5 |
| **Cấu hình A–F** | 6 cách lượng tử đem so trên val | Chọn **B** | 5 |

### D2. Công thức `r = s · (q − zp)`

| Thuật ngữ | Giải thích | Ví dụ thật | Mục |
|---|---|---|---|
| **r** | Số thực | | 2 |
| **q** | Số nguyên int8 lưu trên chip | | 2 |
| **Scale (s)** | Bước nhảy giữa 2 mức int liền nhau, tính bằng `(hi − lo)/255` | Ảnh vào s = 0,012728 | 2 |
| **Zero-point (zp)** | Số nguyên đại diện cho **số thực 0** | Ảnh vào zp = −95; sau ReLU zp = −128 | 2 |
| **Dải [lo, hi]** | Min/max số thực đo khi calibrate. Luôn **nới cho chứa 0** | | 2 |
| **Bất đối xứng** | Có zp ≠ 0, dải int [−128, 127]. Dùng cho **activation** | | 2 |
| **Đối xứng** | zp = 0, dải [−127, 127], `s = max\|w\|/127`. Dùng cho **trọng số** → khỏi trừ zp trong vòng MAC | conv1 s_w = 0,00684765 | 2 |
| **Activation** | Dữ liệu chạy qua mạng (đầu ra từng lớp), khác với trọng số | | 2 |
| **Per-tensor** | Một scale cho **cả** tensor (cả lớp) | Project dùng kiểu này | — |
| **Per-channel** | Mỗi kênh ra một scale riêng. Chính xác hơn, tốn thêm tham số và phần cứng | Chưa dùng, câu hỏi chờ Thầy | — |
| **Bias int32** | Scale = s_vào × s_w, zp = 0 → cộng thẳng vào acc | | 2 |

### D3. Một lớp chạy bằng số nguyên

    acc  = bias + Σ (q_vào − zp_vào) · q_w
    q_ra = kẹp( ((acc · M0 + 2^(shift−1)) >> shift) + zp_ra, −128, 127 )

| Thuật ngữ | Giải thích | Ví dụ thật | Mục |
|---|---|---|---|
| **Requantize** | Đưa acc (int32, scale s_vào·s_w) về int8 với scale đầu ra s_ra | Cả dòng `q_ra = …` ở trên | 3 |
| **M** | Hệ số đổi scale: `M = s_vào · s_w / s_ra`, là số thực rất nhỏ | conv1 M = 0,00381027276 | 3 |
| **M0** | Số nguyên thay cho M: `M ≈ M0 · 2^−shift` | conv1 M0 = 2.094.719.601 (31 bit) | 3 |
| **Shift** | Số bit dịch phải đi kèm M0 | 39 ở cả 3 lớp conv | 3 |
| **Dịch phải `>>`** | Chia cho 2^n. Trên số âm phải là **làm tròn xuống (floor)** | Python `>>`, Verilog `>>>` | 3, 7 |
| **Làm tròn nửa lên** (`làm_tròn`) | `floor(x + 0,5)`: gần nhất, đúng nửa thì lên. Dùng ở MỌI chỗ (quantize, zp, M0, bias, requantize = cộng `2^(shift−1)` rồi dịch). **Không** dùng `np.round`/`torch.round`/`round()` của Python (2,5 → 2) | conv1 cộng 2^38; `ptq_int8.py` dòng 87 | 3 |
| **Kẹp (clamp)** | Ép về [−128, 127]. Thiếu bước này → tràn int8 | | 3 |
| **ReLU gộp vào kẹp** | Scale theo dải sau ReLU → zp_ra = −128 = số thực 0. Kẹp `max(q, −128)` chính là ReLU | Cấu hình B | 3 |
| **Đệm ở INT8** | **Trừ zp trước, rồi đệm 0.** Đệm số 0 thẳng lên q là sai khi zp ≠ 0 | | 3, 7 |
| **Lớp fc** | Không requantize, argmax chạy luôn trên acc | `tham_so.txt` dòng fc | 7 |

### D4. Đầu vào

| Thuật ngữ | Giải thích | Ví dụ thật | Mục |
|---|---|---|---|
| **Chuẩn hoá (normalize)** | `(x − mean)/std` lúc train. Ở INT8 **biến mất vào scale/zp** | | 4 |
| **q_in = pixel − 128** | Công thức đầu vào chip, khớp 7.840.000/7.840.000 pixel test | pixel 0 → q_in = −128 | 4 |
| **Đảo bit cao nhất** | Trừ 128 trên số 8 bit = lật bit 7. Không tốn bộ cộng | | 4 |
| **x = pixel − 33** | Giá trị thật vào bộ nhân conv1 (= q_in − zp_in), dải [−33, 222] → bộ nhân **9 bit có dấu** | pixel 0 → x = −33 | 4 |

### D5. QAT

| Thuật ngữ | Giải thích | Trong code | Mục |
|---|---|---|---|
| **Fake-quant** | Forward làm tròn về lưới int rồi trả lại số thực. Model vẫn chạy float nhưng "thấy" sai số lượng tử | `x + (lượng_tử(x) − x).detach()` | 6 |
| **STE** | Straight-Through Estimator: coi `làm_tròn()` có đạo hàm = 1 lúc backward, để gradient chảy qua | `.detach()` tạo ra hiệu ứng này | 6 |
| **EMA** | Trung bình trượt, cập nhật dần dải sau ReLU khi train | momentum 0,01 | 6 |
| **Đối chứng (control)** | Train thêm cùng số epoch **không** fake-quant, để biết QAT tốt hơn là nhờ "biết lượng tử" hay chỉ nhờ train thêm | Chưa chạy | 6 |

### D6. Golden C int8

| Thuật ngữ | Giải thích | Trong project | Mục |
|---|---|---|---|
| **Implementation-defined** | Hành vi C để trình biên dịch tự quyết. `>>` trên số âm thuộc loại này → phải tự viết floor | | 7 |
| **Làm tròn về 0 (truncation)** | Phép `/` của C: −7/2 = −3 | Khác floor (−4) | 7 |
| **acc × M0 cần 64 bit** | 20 bit × 31 bit = 51 bit, tràn int32 | `int64_t` trong C | 7 |
| **Hàm không giảm** | `a ≤ b ⇒ f(a) ≤ f(b)`. Requantize có tính chất này nên **đổi chỗ được với max** | Requantize sau maxpool: 10.192 → 2.496 lần/ảnh | 7 |
| **Tham chiếu (reference)** | Đầu ra Python để C/RTL so | `ptq_int8.py --chi-tham-chieu` sinh lại, **không calibrate lại** | 7 |

---

## E. Verilog và công cụ IC

| Thuật ngữ | Giải thích | Trong project |
|---|---|---|
| **RTL** | Register-Transfer Level: mô tả phần cứng bằng thanh ghi + logic giữa chúng, viết bằng Verilog/VHDL | `rtl/` |
| **Verilog** | Ngôn ngữ mô tả phần cứng (HDL) | 10 bài nhập môn |
| **Module** | Một khối mạch có cổng vào/ra, tương tự hàm nhưng là phần cứng chạy song song | |
| **Testbench** | Module không có cổng, tạo tín hiệu vào cho mạch cần thử và kiểm tra đầu ra. Chỉ dùng mô phỏng, không thành mạch | Mỗi bài có testbench tự kiểm |
| **DUT** | Device Under Test: mạch đang được thử trong testbench | |
| **Testbench tự kiểm** | Testbench tự so đầu ra với bảng mong đợi, in PASS/FAIL | Bài 1 PASS 8/8 |
| **Bảng chân trị** | Liệt kê đầu ra cho mọi tổ hợp đầu vào | Bạn lập trước mỗi bài |
| **Mạch tổ hợp** | Đầu ra chỉ phụ thuộc đầu vào hiện tại, không có nhớ | Bài 1, Full Adder |
| **Mạch tuần tự** | Có thanh ghi/flip-flop, đầu ra phụ thuộc cả trạng thái trước | Các bài sau |
| **`#10`** | Chờ 10 đơn vị thời gian mô phỏng | |
| **`>>>`** | Dịch phải số học (giữ bit dấu) = floor với số có dấu | Dùng cho requantize |
| **Mô phỏng (simulation)** | Chạy RTL trên máy tính để xem tín hiệu, chưa thành mạch thật | Icarus Verilog |
| **Icarus Verilog** | Trình mô phỏng Verilog miễn phí, chạy được trên Mac M2 | Đã cài (10/11) |
| **VCD** | File ghi lại giá trị tín hiệu theo thời gian | `*.vcd`, không commit |
| **Waveform (dạng sóng)** | Hình hiển thị tín hiệu theo thời gian | Surfer; `tools/ve_song.py` → `song.png` |
| **Surfer** | Trình xem waveform | Đã cài (10/11) |
| **Synthesis (tổng hợp)** | Dịch RTL thành mạch cổng/LUT thật | Cần Vivado hoặc Yosys |
| **Yosys** | Công cụ tổng hợp mã nguồn mở, xem sơ đồ mạch, phát hiện latch | Chưa cài, để sau |
| **Latch** | Phần tử nhớ sinh ra **ngoài ý muốn** khi mạch tổ hợp thiếu nhánh gán | Lỗi hay gặp |
| **Vivado** | Bộ công cụ của AMD/Xilinx: tổng hợp, timing, nạp board | Không chạy được trên Mac M2 |
| **Timing** | Kiểm mạch có chạy kịp ở tần số xung nhịp mong muốn không | |
| **Bitstream** | File cấu hình nạp vào FPGA | |
| **Kria / Zynq-7000** | Các dòng board FPGA có CPU ARM. Kria bắt buộc Vivado; Zynq-7000 còn openXC7 | Board chưa chốt |
| **Prelab** | Giai đoạn chuẩn bị trước khi vào phòng lab, chỉ dùng máy cá nhân | Đang ở giai đoạn này |

<details>
<summary><b>reg / wire</b> — mở sau khi đã tự trả lời câu tự kiểm Bài 1</summary>

| Thuật ngữ | Giải thích |
|---|---|
| **reg** | Biến **giữ giá trị** tới khi được gán lại, gán trong khối `initial`/`always` |
| **wire** | Dây nối, giá trị do thứ đang **lái** nó quyết định (cổng ra của module, `assign`) |

</details>
