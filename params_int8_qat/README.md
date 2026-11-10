# params_int8_qat/ — tham số số nguyên của SmallCNN (sinh tự động, không sửa tay)

Sinh bởi `ptq_int8.py` (PTQ → `params_int8/`) hoặc `qat.py --bits 8 --xuat` (QAT → `params_int8_qat/`).
Cấu hình: ReLU **gop**, M0 **31 bit**.

| File | Nội dung | Kiểu |
|---|---|---|
| `convN_w.txt` | trọng số [OC, IC, 3, 3] trải phẳng row-major | int8 đối xứng, zp = 0 |
| `convN_b.txt` | bias, scale = s_vào × s_w | int32 |
| `fc_w.txt`, `fc_b.txt` | [10, 144] và [10] | int8 / int32 |
| `tham_so.txt` | scale, zp, M0, shift từng lớp | |
| `golden_int8/` | ảnh test index 0, đầu ra số nguyên từng lớp | |
| `golden_int8/fc_acc_test.txt` | 10 acc lớp fc của cả 10.000 ảnh test (10.000 dòng × 10 số) | int32 |

Công thức mỗi lớp conv:

    acc  = bias + Σ (q_vào − zp_vào) · q_w          (int32, đệm 0 SAU khi trừ zp)
    q_ra = clamp(((acc · M0 + 2^(shift−1)) >> shift) + zp_ra, −128, 127)
    pool = max 2×2 trên q_ra

Lớp fc: `acc = bias + Σ (q − zp_vào) · q_w`, dự đoán = argmax(acc), so sánh chặt `>`.
Golden C int8 (`golden_c/golden_int8.c`) phải khớp **từng số** với `golden_int8/`.

| File golden | Shape | Số phần tử |
|---|---|---|
| `input.txt` | [1, 28, 28] | 784 |
| `01_conv1_acc.txt` | [8, 28, 28] | 6272 |
| `02_relu1.txt` | [8, 28, 28] | 6272 |
| `03_pool1.txt` | [8, 14, 14] | 1568 |
| `04_conv2_acc.txt` | [16, 14, 14] | 3136 |
| `05_relu2.txt` | [16, 14, 14] | 3136 |
| `06_pool2.txt` | [16, 7, 7] | 784 |
| `07_conv3_acc.txt` | [16, 7, 7] | 784 |
| `08_relu3.txt` | [16, 7, 7] | 784 |
| `09_pool3.txt` | [16, 3, 3] | 144 |
| `10_fc_acc.txt` | [10] | 10 |
