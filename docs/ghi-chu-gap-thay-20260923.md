# Ghi chú gặp Thầy — 23/09/2026

> Ghi lại góp ý của Thầy và việc đã làm theo. Chỉ ghi phần Thầy nói, không thêm suy diễn.

## 1. Thầy hỏi — mình trả lời chưa tốt

| Câu Thầy hỏi | Trả lời đúng (ôn lại) | Ở đâu |
|---|---|---|
| Train và inference khác nhau ở đâu? | Train = forward + loss + **backward** + optimizer **cập nhật trọng số**. Inference = **chỉ forward**, trọng số **cố định**, ra argmax. FPGA chỉ làm inference | mục 3 bên dưới |
| Chỉ tính **1 điểm trên 1 kênh output** thì tính thế nào? | `bias[oc] + Σ_ic Σ_ky Σ_kx in·w`. conv1: 9 MAC, conv2: 72 MAC, conv3: 144 MAC. Mọi kênh vào **dồn vào 1 accumulator** → 1 số | `operators/conv2d.md` mục 3, 3 hình `figures/conv_*.png` |

## 2. Thầy dặn làm

| Thầy dặn | Đã làm |
|---|---|
| Viết **inference bằng Python** xem nó chạy thế nào | ✅ `inference_python.py` — vòng lặp thuần, không PyTorch, mảng 1 chiều như C. Khớp PyTorch mọi lớp (lệch ≤ 1,2e-5). **10.000 ảnh test: 98,82%, trùng dự đoán PyTorch 10.000/10.000**, logit lệch ≤ 2,5e-5 |
| **Vẽ ra** công thức, cách nhân chập, cộng | ✅ `make_conv_figures.py` → 3 hình, số thật, khớp PyTorch |
| Trích tham số ra `.txt`, **weight và bias 2 file riêng**, làm input cho code C | ✅ `export_params.py` → `params/weights.txt` (4.968) + `params/biases.txt` (50) + bảng offset |
| Tiếp theo: **golden model C** | ⬜ |
| Cứ nhờ AI làm, học dần hiểu dần | Nguyên tắc làm việc |

## 3. Train vs inference

| | Train | Inference |
|---|---|---|
| Trọng số | Thay đổi sau mỗi batch | **Cố định**, chỉ đọc |
| Chiều tính | Forward → loss → backward → cập nhật | **Chỉ forward** |
| Cần | Nhãn, loss, gradient, lr, optimizer | Ảnh + trọng số |
| Đầu ra | Loss giảm dần | 1 số 0–9 (argmax) |

## 4. Góp ý quan trọng nhất: `if` bỏ padding không có lợi trên FPGA

Mã giả conv có `if` để bỏ các tap rơi vào vùng đệm (ở góc bỏ 5/9 tap). Thầy chỉ ra: trên FPGA
việc này thành vấn đề — cần thêm mạch (LUT) để xét điều kiện rồi mới làm tiếp.

Hiểu lại cho rõ:
- Conv 3×3 trên FPGA = 9 bộ nhân **song song**, 1 chu kỳ. Bỏ bớt tap **không nhanh hơn**.
- `if` cần bộ so sánh + MUX cho từng tap → tốn LUT, luồng điều khiển không đều.
- Cách phần cứng làm: **đệm 0** vào dữ liệu, mọi vị trí luôn đủ 9 MAC.
- Cái giá: 395.136 thay vì 351.008 MAC conv/ảnh (+11,2%) — đo bằng `inference_python.py`.
- Hai cách ra kết quả **giống hệt** (cộng `0 × w` = không đổi).

→ Golden model C dùng **bản đệm 0**, vì nó giống phần cứng. Chi tiết: `operators/conv2d.md` mục 5.

**Bài học:** đếm số phép (thước đo phần mềm) ≠ chu kỳ, LUT/DSP, timing (thước đo phần cứng).

## 5. Phát hiện phụ khi làm

- **Giá trị đệm là 0 sau chuẩn hoá**, không phải màu nền. Nền MNIST sau chuẩn hoá = −0,4242.
  C phải đệm 0.0 như PyTorch.
- **Chuẩn hoá** `(x/255 − 0,1307)/0,3081` là phép affine `a·x + b` với `a, b` tính trước →
  không cần phép chia trên chip. Hiện golden model nhận ảnh **đã chuẩn hoá**.
- Python dùng float64, PyTorch float32 → lệch ~1e-5. Đó là độ chính xác, không phải sai thuật toán.
