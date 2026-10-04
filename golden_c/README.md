# golden_c/ — Golden model C (float32) của SmallCNN

Phase 2D. Dịch dòng-đổi-dòng từ `inference_python.py` (bản conv **đệm 0**, luôn đủ 9 tap).

## Chạy (đứng ở thư mục gốc project)

```bash
cc -O2 -std=c99 -Wall -o golden_c/golden_model golden_c/golden_model.c
./golden_c/golden_model          # ảnh mẫu (so từng lớp) + 10.000 ảnh test
./golden_c/golden_model 100      # chỉ 100 ảnh đầu
./golden_c/golden_model 0        # chỉ ảnh mẫu
```

Cần: `params/weights.txt`, `params/biases.txt`, `golden/` (do `export_params.py` sinh) và
`data/MNIST/raw/t10k-*` (file raw chưa nén).

## Kết quả (Linux, gcc/clang)

| Kiểm tra | Kết quả |
|---|---|
| Thư viện | chỉ `stdio.h`, `stdlib.h` — không `math.h` |
| MAC conv / ảnh | 395.136 |
| Lệch PyTorch từng lớp (ảnh mẫu) | ≤ 3e-06 ở conv/pool, 1,9e-05 ở logit |
| 10.000 ảnh test | 98,82%, trùng dự đoán PyTorch 10.000/10.000, logit lệch ≤ 3,05e-05 |

## Cấu trúc file

| Phần | Nội dung |
|---|---|
| [1] | Hằng số kích thước + bảng offset (khớp `params/README.md`) |
| [2] | Đọc file `.txt` bằng `fscanf("%f")` |
| [3] | 6 operator: `conv2d` (đệm 0), `relu`, `maxpool2d`, flatten (không làm gì), `linear`, `argmax` (so sánh chặt) |
| [4] | `infer()` ghép 6 operator, giữ đầu ra từng lớp để so |
| [5] | So sánh với `golden/`, chuẩn hoá ảnh (làm trên PC) |
| [6] | `main`: ảnh mẫu từng lớp → 10.000 ảnh |

File chạy được (`golden_c/golden_model`) không commit — biên dịch lại trên máy.
