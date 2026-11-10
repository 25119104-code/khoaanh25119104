# Report Đợt 2 – Tuần 1 — Nhận diện chữ số viết tay (AI Project 1/3)

**Ngày:** 04/11/2026
**Giai đoạn:** Golden model C số nguyên (int8), cuối phần AI, làm chuẩn đối chiếu cho RTL

---

## 1. Kết quả / tiến độ

Sau đợt thi, em làm tiếp đúng thứ tự Thầy giao ngày 07/10: PTQ/QAT Python (xong tuần 6) → **C** → RTL.

| File | Vai trò |
|---|---|
| `golden_c/golden_int8.c` | **Golden model C int8.** Chỉ `stdio.h`, `stdlib.h`, `stdint.h`. Đường inference không có số thực nào. Không tính lại scale hay M0: đọc hết từ thư mục tham số, nên cùng một code chạy được cả tham số PTQ và QAT |
| `ptq_int8.py` | Thêm `--chi-tham-chieu`: sinh file tham chiếu 10.000 ảnh từ tham số đã xuất, **không calibrate lại** |
| `qat.py` | Thêm `--xuat`: xuất tham số số nguyên của checkpoint QAT ra `params_int8_qat/`, cùng định dạng `params_int8/` |
| `params_int8/golden_int8/fc_acc_test.txt` | 10 acc lớp fc của cả 10.000 ảnh test (Python tính), để C so từng bit |

Tham số và ảnh mẫu trong `params_int8/` **không đổi byte nào** (kiểm bằng md5 trước và sau), vẫn là bộ chạy trên Mac ngày 13/10.

### 1.1 Golden C int8 khớp Python từng bit

| Kiểm tra | PTQ (`params_int8/`) | QAT (`params_int8_qat/`) |
|---|---|---|
| Ảnh mẫu: 11 file từng lớp (input, acc, sau requantize, pool, fc) | **0 số khác** | **0 số khác** |
| 10.000 ảnh test: cả 10 acc lớp fc | **10.000/10.000 ảnh khớp từng bit** | **10.000/10.000** |
| Accuracy | 98,84% (bằng Python) | 98,91% (bằng Python) |
| Dự đoán khác float32 | 9 ảnh (float đúng → int8 sai: 3, ngược lại: 5) | 45 ảnh (17 / 26) |
| MAC conv mỗi ảnh | 395.136, bằng bản float | 395.136 |

Thêm hai kiểm tra:
- Biên dịch với `-fsanitize=undefined,address` rồi chạy đủ 10.000 ảnh: không có lỗi tràn số, dịch bit sai hay truy cập ngoài mảng.
- Biên dịch sạch với gcc 13 và clang 18 ở `-Wall -Wextra -pedantic`.

So **từng bit cả 10 acc** chặt hơn nhiều so với chỉ so dự đoán. Hai bản có thể đoán cùng số mà acc vẫn lệch, và lệch đó
sẽ thành lỗi ở RTL.

### 1.2 Thanh ghi accumulator 20 bit là đủ cho MỌI ảnh

Tuần 6 em mới đo |acc| lớn nhất trên tập test và ghi rõ là "chưa phải chặn trên lý thuyết". Tuần này golden C tính thêm
**chặn trên cho mọi ảnh có thể**: mỗi đầu vào `x = q − zp` nằm trong `[−128 − zp, 127 − zp]`, với từng trọng số chọn
đầu mút cho tích lớn nhất (hoặc nhỏ nhất) rồi cộng bias.

| Lớp | Đo trên 10.000 ảnh test | Chặn trên cho mọi ảnh |
|---|---|---|
| conv1 | 82.468 (18 bit) | 86.624 (18 bit) |
| conv2 | 149.729 (19 bit) | 357.383 (20 bit) |
| conv3 | 100.287 (18 bit) | 518.164 (20 bit) |
| fc | 63.673 (17 bit) | 435.342 (20 bit) |

Bộ tham số QAT cũng ra tối đa 20 bit. Kết luận: **RTL dùng thanh ghi acc 20 bit có dấu thì không bao giờ tràn**, không phải
32 bit, cũng không phải "chắc đủ" dựa trên tập test.

### 1.3 Đặt requantize SAU maxpool: giống hệt từng bit, ít hơn 4,08 lần

Golden C chạy song song thứ tự thứ hai: maxpool trên acc int32 trước, requantize sau.

| | Thứ tự gốc (như Python) | Requantize sau maxpool |
|---|---|---|
| Số lần requantize mỗi ảnh | 10.192 | **2.496** |
| Ảnh có pool1, pool2, pool3 hoặc fc khác thứ tự gốc | — | **0/10.000** (cả PTQ và QAT) |

Vì sao đúng: requantize là hàm không giảm (M0 > 0; cộng hằng, làm tròn xuống, kẹp đều không giảm), nên
`max(rq(a), rq(b)) = rq(max(a, b))`. Cùng lý do với việc đổi relu ↔ maxpool ở bản float (10.192 → 2.496, tuần 4).

Mỗi lần requantize có một phép nhân lớn (acc × M0), nên đổi thứ tự bớt **7.696 phép nhân lớn mỗi ảnh**. Cái giá: maxpool
phải so sánh trên acc 20 bit thay vì int8 8 bit. Bộ so sánh 20 bit vẫn rẻ hơn bộ nhân, nhưng đây là lựa chọn thiết kế RTL,
em sẽ quyết khi viết RTL.

### 1.4 Đường dữ liệu phần cứng, đọc ra từ golden C

| Chỗ | Phần cứng làm gì |
|---|---|
| Đầu vào | Chip nhận pixel `uint8`, `q = pixel − 128` = **đảo bit cao nhất**. Kiểm lại: khớp 7.840.000/7.840.000 pixel test |
| conv1 | `x = q − zp = pixel − 33`, nằm trong [−33, 222]: **bộ nhân 9 bit có dấu × 8 bit** |
| conv2, conv3, fc | zp = −128 nên `x = q + 128` ∈ [0, 255]: lại chỉ là **đảo bit cao nhất**, không tốn bộ cộng. Bộ nhân 8 bit không dấu × 8 bit có dấu |
| Accumulator | 20 bit có dấu, khởi tạo bằng bias |
| Requantize | `(acc · M0 + 2^38) >> 39`, cộng zp, kẹp về [−128, 127]. Kẹp chính là ReLU. Tích acc × M0 cần **51 bit** (20 + 31) |
| fc | Không requantize, argmax trên acc |

---

## 2. Vấn đề gặp phải

1. **README `params_int8/` ghi có `qat.py --xuat`, nhưng code chưa có.** Không xuất được tham số QAT thì không chạy được
   golden C với QAT.
2. **Muốn sinh file tham chiếu 10.000 ảnh mà không được đổi bộ tham số đã chốt.** Chạy lại `ptq_int8.py` sẽ calibrate lại
   bằng float của PyTorch, khác máy có thể lệch M0 (tuần 6 đã gặp: conv3 M0 trên Linux khác trên Mac).
3. **Dịch phải số âm trong C là implementation-defined.** Python `>>` làm tròn xuống, C `/` làm tròn về 0. Viết sai một chỗ
   là lệch 1 đơn vị ở các acc âm.
4. **Golden C chưa chạy trên Mac.** Em chạy trên máy ảo Linux arm64 (gcc). C chỉ có phép tính số nguyên nên kỳ vọng ra y hệt;
   sẽ chạy lại trên Mac trước khi commit.
5. **QAT int8 khác float ở 45 ảnh, PTQ chỉ 9.** QAT là một model đã train thêm, không phải bản lượng tử của cùng model.
   Mức hơn PTQ +0,07 điểm % (7 ảnh) vẫn chưa tách được là nhờ "biết lượng tử" hay chỉ nhờ train thêm (chưa có đối chứng).
6. **Vivado không cài được trên MacBook M2.** Xem mục 5.
7. **Hai việc tuần 6 chưa làm:** chạy QAT nhiều seed, và đối chứng QAT (fine-tune float cùng số epoch rồi PTQ).

---

## 3. Giải pháp & kiến thức mới

### Giải pháp

| Vấn đề | Cách xử lý |
|---|---|
| Thiếu `qat.py --xuat` | Viết thêm: nạp `models/small_cnn_qat_int8.pth`, đi đúng đường số nguyên của `ptq_int8.py`, ghi `params_int8_qat/`. Accuracy ra 98,91%, đúng bằng số QAT tuần 6 |
| Sinh tham chiếu không đổi tham số | `ptq_int8.py --chi-tham-chieu` dựng lại mô hình từ file `.txt` đã xuất. Tự kiểm: ảnh mẫu dựng lại khớp 10/10 file `golden_int8/` cũ, đầu vào khớp `pixel − 128` ở mọi pixel |
| Dịch phải số âm | Hàm `dich_phai_so_hoc()` tự tính `floor(x / 2^n)`, không dựa vào trình biên dịch |
| Đệm sai giá trị | Như tuần 6: trừ zp trước, rồi mới đệm 0 |
| acc × M0 tràn 32 bit | Tính tích bằng `int64_t`. acc vẫn để `int32_t`, như thanh ghi phần cứng |

### Kiến thức mới

**a) Vì sao bản số nguyên phải khớp từng bit, còn bản float chỉ cần lệch ≤ 1e-4.** Cộng số thực không có tính kết hợp:
cộng 144 số theo thứ tự khác là ra kết quả khác ở chữ số cuối, nên golden float chỉ so được trong dung sai. Cộng số nguyên
(khi không tràn) thì có tính kết hợp, thứ tự nào cũng ra đúng một số. Vì vậy với int8, **lệch 1 đơn vị là lỗi**, không phải sai số.

**b) Ba cách chia cho 2^n** (ví dụ −7 chia 2, đáp số thực −3,5):

| Cách | Kết quả | Ai dùng |
|---|---|---|
| Làm tròn về 0 | −3 | C: phép `/` trên số nguyên |
| Làm tròn xuống (floor) | −4 | Python `>>`, Verilog `>>>` trên tín hiệu signed, `dich_phai_so_hoc()` |
| Làm tròn nửa lên: cộng 2^(n−1) rồi floor | −3 | requantize: `(acc·M0 + 2^(shift−1)) >> shift` |

**c) Hàm không giảm giao hoán với max.** Mọi hàm f không giảm đều có `max(f(a), f(b)) = f(max(a, b))`. ReLU, requantize, kẹp
đều thuộc loại này, nên đặt sau maxpool được. AvgPool thì không (trung bình không giao hoán với hàm phi tuyến).

**d) Chặn trên accumulator.** Đo trên tập test chỉ cho biết các ảnh đã thấy. Chặn trên tính từ trọng số thật thì đúng cho mọi
ảnh. Ở đây hai số gần nhau ở conv1 (86.624 so với 82.468) nhưng cách xa ở conv3 (518.164 so với 100.287). Thiết kế phần
cứng phải theo chặn trên.

**e) zp = −128 biến phép trừ thành đảo bit.** `q − (−128) = q + 128`: lấy đúng mẫu bit của q, đảo bit cao nhất, rồi đọc
như số không dấu. Cùng lý do mà đầu vào `pixel − 128` cũng chỉ là đảo bit.

**f) Kiểm thử theo tầng.** Ảnh mẫu so từng lớp để chỉ ra **lớp nào** sai. 10.000 ảnh so lớp cuối để phủ nhiều trường hợp.
Sanitizer bắt hành vi không xác định mà kết quả đúng vẫn che đi được. Chạy thêm bộ tham số QAT để chứng minh code không
"khớp tình cờ" với một bộ số.

---

## 4. Kế hoạch tuần tới

1. Chạy lại golden C int8 trên Mac, rồi commit toàn bộ (gồm cả phần tuần 6 chưa commit).
2. **Bắt đầu RTL trên Mac** bằng Icarus Verilog + GTKWave:
   - Khối đầu tiên: **requantize**. Nhỏ, và kiểm được ngay bằng dữ liệu có sẵn: `01_conv1_acc.txt` (6.272 acc) vào,
     so với `02_relu1.txt` ra, từng bit.
   - Sau đó: MAC 3×3 của conv1, so với `01_conv1_acc.txt`.
3. Golden C: thêm chế độ xuất đầu ra từng lớp cho một ảnh bất kỳ, làm dữ liệu cho testbench RTL.
4. Chọn bề rộng M0 cho phần cứng. Bộ nhân trong DSP48 của Xilinx là 25 × 18 bit (7-series), 27 × 18 bit (UltraScale+).
   M0 31 bit không lọt cổng 18 bit nên mỗi bộ requantize phải ghép ít nhất 2 DSP. M0 15 bit thì acc 20 bit × M0 vừa 1 DSP.
   Thí nghiệm D tuần 6: M0 15 bit cho val bằng 31 bit. Nếu đổi, chỉ cần xuất lại tham số và chạy lại golden C, không phải sửa code C.
5. Vivado: làm theo phương án Thầy chốt (mục 5).

### Câu hỏi cần Thầy chốt

1. **Phương án Vivado** (mục 5): em đề xuất mô phỏng RTL ngay trên Mac, khi cần tổng hợp thì dùng Docker hoặc máy x86. Thầy đồng ý không?
2. **Board mục tiêu** là Zynq-7000 hay Kria? Kria bắt buộc phải có Vivado. Zynq-7000 còn có công cụ mã nguồn mở dự phòng.
3. **RTL dùng tham số PTQ hay QAT?** Em đề xuất PTQ: gần float nhất (khác 9 ảnh), còn mức hơn của QAT chưa chứng minh được.

---

## 5. Phương án Vivado trên MacBook (trình bày khi báo cáo)

**Máy em:** MacBook Air M2, SSD 256 GB (229 GiB dùng được), **còn trống 28 GiB**. RAM: ___ GB.
Vivado chỉ có bản Windows/Linux trên chip x86-64: không có bản macOS, không chạy trên chip ARM như M2.

### Vivado làm gì ngoài mô phỏng

| Chức năng | Để làm gì | Thay được trên Mac không |
|---|---|---|
| Mô phỏng (XSim) | Chạy testbench, xem dạng sóng | ✅ Icarus Verilog, Verilator, GTKWave |
| Synthesis + báo cáo tài nguyên | Ra số LUT/FF/DSP/BRAM | ⚠️ Yosys, chỉ dùng để so tương đối giữa các phương án |
| Implementation + timing | Đặt và đi dây, ra tần số tối đa | ⚠️ nextpnr-xilinx, chỉ cho chip 7-series |
| Bitstream + nạp board + ILA | Chạy trên chip thật, debug tín hiệu | ⚠️ openXC7 + openFPGALoader, không có ILA |
| Block Design | Ghép ARM (PS) với mạch của mình (PL) trên Zynq/Kria | ❌ |
| Vitis HLS | Chuyển C thành RTL | ❌ |

### Bốn phương án

| Phương án | Ưu | Nhược |
|---|---|---|
| **A. Docker + Rosetta** (giả lập x86 trong Docker) | Miễn phí, Vivado Linux đầy đủ | Không chính thức, hay crash dưới Rosetta. GUI qua VNC chậm. Không có USB trực tiếp: phải dùng Xilinx Virtual Cable. **Máy em chỉ còn 28 GiB**: gói cài đầy đủ 2025.1 là 110 GB, chỉ tải các họ chip Thầy chọn thì em ước lượng vẫn vài chục GB, chưa kể image Docker |
| **B. Windows 11 ARM** (Parallels/UTM) | GUI mượt nhất | AMD không hỗ trợ. Driver JTAG Digilent chưa có bản ARM, nên nạp board **tuỳ cáp**, không chắc được. Parallels mất phí |
| **C. Mô phỏng ngay trên Mac** | Nhanh, ổn định, miễn phí. Đủ cho viết và kiểm RTL mấy tuần tới | Không tổng hợp, không ra bitstream |
| **D. Máy x86** (phòng lab, máy bạn) | Được hỗ trợ chính thức, USB chạy thẳng | Phụ thuộc lịch dùng máy |

Về bảng so sánh Thầy gửi: ô "tự nhận bo FPGA qua USB: **Có**" ở cột Windows ARM là lạc quan. Ứng dụng x86 chạy được nhờ
giả lập, nhưng driver phần cứng phải là bản ARM, và Digilent nói chưa có lộ trình làm bản đó. Ô Docker "phải làm thủ công
ngoài host" thì đúng.

### Đề xuất

- **Bây giờ: C.** Bước tiếp theo là RTL và mô phỏng, chưa cần Vivado. Golden C int8 đã có dữ liệu từng lớp để testbench so.
- **Khi cần so phương án RTL:** thêm Yosys trên Mac để so tài nguyên tương đối.
- **Khi cần số chính thức (LUT, Fmax) hoặc nạp board:** D (máy lab), hoặc A nếu em dọn được ổ đĩa hay dùng ổ SSD ngoài.
  Synthesis và implementation chạy được bằng script TCL ở chế độ batch, không cần GUI, nên nhược điểm GUI chậm của A ít ảnh hưởng.
- **Board quyết định nhiều:** Kria thì bắt buộc Vivado. Zynq-7000 thì còn openXC7 làm dự phòng.

### Nói gọn với Thầy

> "Vivado không có bản cho Mac M2, máy em cũng chỉ còn 28 GB trống. Mấy tuần tới em làm RTL và mô phỏng bằng Icarus Verilog
> ngay trên Mac, vì giai đoạn này chưa cần Vivado. Tới lúc cần tổng hợp, đo timing hay nạp board thì em dùng máy lab, hoặc
> Docker nếu dọn được ổ. Windows ARM em không chọn vì driver JTAG chưa có bản ARM."

Nguồn: [vivado-on-silicon-mac](https://github.com/ichi4096/vivado-on-silicon-mac/blob/main/README.md),
[vivado-docker (lỗi Rosetta)](https://github.com/filmil/vivado-docker/),
[Digilent forum: driver cho Windows 11 ARM](https://forum.digilent.com/topic/22740-is-there-a-digilent-cable-driver-for-win11-on-arm/),
[AMD: tải Vivado 2025.1](https://www.amd.com/en/support/downloads/adaptive-socs-and-fpgas/development-tools/2025-1.html),
[openXC7](https://github.com/fpgawars/tools-openxc7).

---

**File liên quan:** `golden_c/golden_int8.c`, `golden_c/README.md`, `ptq_int8.py`, `qat.py`,
`params_int8/` (thêm `golden_int8/fc_acc_test.txt`), `params_int8_qat/`

**Chạy lại** (đứng ở thư mục gốc project, env `mnist`):
```
cc -O2 -std=c99 -Wall -Wextra -o golden_c/golden_int8 golden_c/golden_int8.c
./golden_c/golden_int8                          # PTQ: ảnh mẫu + 10.000 ảnh, vài giây
./golden_c/golden_int8 10000 params_int8_qat    # QAT
python ptq_int8.py --chi-tham-chieu             # sinh lại fc_acc_test.txt nếu thiếu (không calibrate lại)
python qat.py --bits 8 --xuat                   # xuất lại params_int8_qat/ từ checkpoint QAT
```
