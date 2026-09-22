# Operator của `SmallCNN` — bộ tài liệu đầy đủ

> Bước 2 trong phương pháp Thực chiến của Thầy: **viết operator + mã giả** cho mọi phép toán
> trên đường inference, làm đầu vào cho golden model C (Phase 2D).

## Đường inference đầy đủ

Sau khi train xong, model chạy đúng 12 bước này và không có gì khác:

```
ảnh 1×28×28
   │
   ├─ conv2d  (1→8,  k3, p1)   →  8×28×28
   ├─ relu                     →  8×28×28
   ├─ maxpool (k2, s2)         →  8×14×14
   │
   ├─ conv2d  (8→16, k3, p1)   → 16×14×14
   ├─ relu                     → 16×14×14
   ├─ maxpool (k2, s2)         → 16× 7× 7
   │
   ├─ conv2d  (16→16, k3, p1)  → 16× 7× 7
   ├─ relu                     → 16× 7× 7
   ├─ maxpool (k2, s2)         → 16× 3× 3
   │
   ├─ flatten                  → 144
   ├─ linear  (144→10)         →  10 logit
   └─ argmax                   →  1 chữ số
```

**Chỉ có 6 operator khác nhau.** Không có softmax, không có loss function, không có optimizer —
những thứ đó chỉ tồn tại lúc train.

## Bảng tổng hợp

| Operator | File | Params | Phép tính / ảnh | FPGA |
|---|---|---|---|---|
| `Conv2d` | [conv2d.md](conv2d.md) | 3.568 | 395.136 MAC | ✅ Rất tốt |
| `ReLU` | [relu.md](relu.md) | 0 | 10.192 so sánh | ✅ Tốt nhất |
| `MaxPool2d` | [maxpool2d.md](maxpool2d.md) | 0 | 7.488 so sánh | ✅ Rất tốt |
| `Flatten` | [flatten.md](flatten.md) | 0 | 0 | ✅ Miễn phí |
| `Linear` | [linear.md](linear.md) | 1.450 | 1.440 MAC | ⚠️ Nghẽn băng thông |
| `argmax` | [argmax.md](argmax.md) | 0 | 9 so sánh | ✅ Gần như miễn phí |
| | **Tổng** | **5.018** | ~414.000 | |

Ba lớp conv chiếm **95%** khối lượng tính toán nhưng chỉ **71%** tham số.
Lớp `fc` ngược lại: 0,3% tính toán nhưng 29% tham số. Hai trục khác nhau, tối ưu khác nhau.

## Định dạng chung

Mỗi file có đúng 7 mục:

1. **Vai trò** — operator này tồn tại để làm gì
2. **Công thức** — kèm cách suy ra, không chỉ chép kết quả
3. **Mã giả** — viết được thành C ngay, không phụ thuộc thư viện
4. **Áp dụng vào `SmallCNN`** — số cụ thể của model này
5. **Mức FPGA-friendly** — phép toán nào tốn phần cứng
6. **Chú ý khi port sang C** — bẫy làm sai mà không báo lỗi
7. **Tự kiểm tra** — câu hỏi làm trên giấy

## Ba bẫy nguy hiểm nhất khi viết golden model C

Cả ba đều **không crash**, chỉ cho kết quả sai:

1. **Layout NCHW vs NHWC ở `flatten`** — vector 144 bị hoán vị, accuracy rơi về ~10%.
   Cách bắt: in vector 144 từ PyTorch và từ C cho cùng một ảnh, so từng phần tử.
   → [flatten.md](flatten.md) mục 6
2. **Thứ tự chiều trọng số** — PyTorch `[out][in]`, Keras `[in][out]`. Ngược nhau.
   → [conv2d.md](conv2d.md) và [linear.md](linear.md) mục 6
3. **Khởi tạo accumulator bằng 0 thay vì `bias`** — lệch đúng bằng bias, rất khó nhìn ra.
   → có ở cả `conv2d` và `linear`

## Hai chỗ bỏ bớt được phép toán cho FPGA

Đây là phần trả lời trực tiếp yêu cầu "thay/bỏ phép toán khó" trong phạm vi project:

| Bỏ gì | Tiết kiệm | Vì sao đúng |
|---|---|---|
| **Softmax** ở lớp cuối | 10 phép `exp()` + 1 phép chia | `exp` đơn điệu tăng nên không đổi thứ tự → `argmax` ra kết quả y hệt. [argmax.md](argmax.md) mục 2 |
| Đổi thứ tự **`relu` ↔ `maxpool`** | 10.192 → 2.496 phép ReLU (4,08 lần) | `max` và `relu` đều đơn điệu không giảm nên hoán vị được. Chỉ đúng với MaxPool, **sai** với AvgPool. [relu.md](relu.md) mục 5 |

Cả hai đều cho kết quả **giống hệt** bit-for-bit, không phải xấp xỉ.

## Chưa làm

- Lượng tử hoá từng operator (Phase 2E, làm **sau** golden model C float32).
  Kết quả khảo sát ban đầu ở `study/xem_quantization.py`.
- Operator cho phần train (`CrossEntropyLoss`, `Adam`) — **không nằm trên đường inference**,
  không cần cho chip.
