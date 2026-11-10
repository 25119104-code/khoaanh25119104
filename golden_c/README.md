# golden_c/ — Golden model C của SmallCNN

Hai bản. Cùng đường inference 6 operator, cùng layout NCHW, conv luôn đủ 9 tap trên ảnh đã đệm.

| File | Số học | Đọc tham số từ | So với | Mức khớp |
|---|---|---|---|---|
| `golden_model.c` | float32 | `params/` | PyTorch (`golden/`) | lệch ≤ 3,05e-05, trùng dự đoán 10.000/10.000 |
| `golden_int8.c` | int8 / int32 / int64, không số thực | `params_int8/` hoặc `params_int8_qat/` | `ptq_int8.py`, `qat.py` (`<thư mục>/golden_int8/`) | **từng bit**: ảnh mẫu 11/11 file, 10 acc lớp fc của 10.000/10.000 ảnh |

File chạy được (`golden_model`, `golden_int8`) không commit — biên dịch lại trên máy.

---

## Golden C int8 (`golden_int8.c`, 03/11/2026)

Dịch từ đường số nguyên của `ptq_int8.py` (hàm `suy_luan_int`, cấu hình B: ReLU gộp vào bước kẹp, M0 31 bit).
Không tính lại scale hay M0: mọi hằng số đọc từ thư mục tham số, nên đổi bộ tham số không phải sửa code.

### Chạy (đứng ở thư mục gốc project)

```bash
cc -O2 -std=c99 -Wall -Wextra -o golden_c/golden_int8 golden_c/golden_int8.c
./golden_c/golden_int8                          # PTQ: ảnh mẫu + 10.000 ảnh test
./golden_c/golden_int8 100                      # chỉ 100 ảnh đầu
./golden_c/golden_int8 0                        # chỉ ảnh mẫu
./golden_c/golden_int8 10000 params_int8_qat    # bộ tham số QAT
```

Chương trình trả mã 0 khi mọi kiểm tra đạt, 1 khi có chỗ lệch.

Cần có:
- Thư mục tham số: `params_int8/` (do `python ptq_int8.py` sinh) hoặc `params_int8_qat/` (do `python qat.py --bits 8 --xuat` sinh).
- `<thư mục>/golden_int8/fc_acc_test.txt`: tham chiếu 10.000 ảnh. Thiếu thì chạy `python ptq_int8.py --chi-tham-chieu`
  (đọc tham số đã xuất, **không calibrate lại**). Không có file này thì chương trình vẫn chạy, chỉ bỏ phần so từng ảnh.
- `golden/logits_pytorch.txt`: chỉ để đếm số ảnh khác float32. Số thực chỉ xuất hiện ở phần kiểm tra này.
- `data/MNIST/raw/t10k-*` (file raw chưa nén).

### Kết quả (máy ảo Linux arm64 gcc, và Mac M2 clang 10/11: ra y hệt)

| Kiểm tra | PTQ (`params_int8/`) | QAT (`params_int8_qat/`) |
|---|---|---|
| Ảnh mẫu: 11 file từng lớp | 0 số khác | 0 số khác |
| 10.000 ảnh: cả 10 acc lớp fc | 10.000/10.000 khớp từng bit | 10.000/10.000 |
| Accuracy | 98,84% | 98,91% |
| Dự đoán khác float32 | 9 ảnh | 45 ảnh |
| MAC conv / ảnh | 395.136 | 395.136 |
| \|acc\| lớn nhất đo được (conv1, conv2, conv3, fc) | 18, 19, 18, 17 bit | 18, 19, 18, 18 bit |
| Chặn trên \|acc\| cho mọi ảnh | 18, 20, 20, 20 bit | 18, 20, 20, 20 bit |
| Requantize sau maxpool: số ảnh khác | 0/10.000 (10.192 → 2.496 lần requantize/ảnh) | 0/10.000 |

Thêm: chạy đủ 10.000 ảnh với `-fsanitize=undefined,address` không báo lỗi; biên dịch sạch với gcc 13 và clang 18
ở `-Wall -Wextra -pedantic`.

### Cấu trúc file

| Phần | Nội dung |
|---|---|
| [1] | Kích thước, `struct LopConv` (trọng số int8, bias int32, zp vào/ra, M0, shift) |
| [2] | Đọc file: kiểm dải từng số, kiểm thiếu/thừa số. `tham_so.txt` chỉ lấy zp, M0, shift (số thực bỏ qua bằng `%*f`) |
| [3] | Operator: `dich_phai_so_hoc`, `requantize`, `conv3x3_int` (trừ zp rồi mới đệm 0), `maxpool2_int8`, `maxpool2_int32`, `linear_int`, `argmax_int` |
| [4] | `infer_int` (giữ đầu ra từng lớp để so) và `infer_pool_truoc` (thí nghiệm [c]) |
| [5] | So với golden, `chan_tren` (chặn trên acc cho mọi ảnh), `so_bit` |
| [6] | `main`: [a] ảnh mẫu từng lớp → [b] 10.000 ảnh → [c] đổi thứ tự maxpool ↔ requantize |

### Bốn chỗ chỉ bản số nguyên mới có

1. **Đệm:** số thực 0 trong miền int8 là zp, không phải số nguyên 0. Trừ zp trước, rồi mới đệm 0.
2. **Dịch phải số âm:** chuẩn C để `>>` trên số âm cho trình biên dịch tự quyết. `dich_phai_so_hoc()` tự tính
   `floor(x / 2^n)`, giống Python `>>` và Verilog `>>>`.
3. **acc × M0 phải tính 64 bit:** acc cỡ 20 bit nhân M0 31 bit ra 51 bit.
4. **Đầu vào:** chip nhận pixel `uint8`, dùng thẳng `q = pixel − 128` (đảo bit cao nhất). `ptq_int8.py` đã kiểm phép
   này bằng lượng tử hoá đầy đủ ở 7.840.000/7.840.000 pixel test. Chương trình từ chối chạy nếu zp đầu vào khác −95.

---

## Golden C float32 (`golden_model.c`, Phase 2D)

Dịch dòng-đổi-dòng từ `inference_python.py` (bản conv **đệm 0**, luôn đủ 9 tap).

### Chạy (đứng ở thư mục gốc project)

```bash
cc -O2 -std=c99 -Wall -o golden_c/golden_model golden_c/golden_model.c
./golden_c/golden_model          # ảnh mẫu (so từng lớp) + 10.000 ảnh test
./golden_c/golden_model 100      # chỉ 100 ảnh đầu
./golden_c/golden_model 0        # chỉ ảnh mẫu
```

Cần: `params/weights.txt`, `params/biases.txt`, `golden/` (do `export_params.py` sinh) và
`data/MNIST/raw/t10k-*` (file raw chưa nén).

### Kết quả (Linux, gcc/clang)

| Kiểm tra | Kết quả |
|---|---|
| Thư viện | chỉ `stdio.h`, `stdlib.h` — không `math.h` |
| MAC conv / ảnh | 395.136 |
| Lệch PyTorch từng lớp (ảnh mẫu) | ≤ 3e-06 ở conv/pool, 1,9e-05 ở logit |
| 10.000 ảnh test | 98,82%, trùng dự đoán PyTorch 10.000/10.000, logit lệch ≤ 3,05e-05 |

### Cấu trúc file

| Phần | Nội dung |
|---|---|
| [1] | Hằng số kích thước + bảng offset (khớp `params/README.md`) |
| [2] | Đọc file `.txt` bằng `fscanf("%f")` |
| [3] | 6 operator: `conv2d` (đệm 0), `relu`, `maxpool2d`, flatten (không làm gì), `linear`, `argmax` (so sánh chặt) |
| [4] | `infer()` ghép 6 operator, giữ đầu ra từng lớp để so |
| [5] | So sánh với `golden/`, chuẩn hoá ảnh (làm trên PC) |
| [6] | `main`: ảnh mẫu từng lớp → 10.000 ảnh |
