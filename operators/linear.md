# Operator: `Linear` (Fully Connected)

> Phase 1.5 nấc 2 → đầu vào cho Phase 2A (viết operator) và Phase 2D (golden model C).

## 1. Vai trò

Mỗi đầu ra nhìn **toàn bộ** đầu vào. Không có cửa sổ trượt, không dùng lại trọng số ở đâu cả: mỗi cặp (đầu vào, đầu ra) có một trọng số riêng.

Đó vừa là sức mạnh — nó tổng hợp được thông tin từ mọi vị trí trong ảnh — vừa là lý do nó đắt. `Conv2d` dùng 1 bộ kernel cho cả ảnh; `Linear` phải có một số riêng cho từng liên kết.

Trong ONNX nó hiện ra là node `Gemm` (General Matrix Multiply).

## 2. Công thức

```
y[o] = bias[o] + Σ (i = 0 .. IN-1)  W[o][i] × x[i]
```

```
params = in_features × out_features  +  out_features
                                        └── bias

MAC    = in_features × out_features
```

Chú ý: `params` và `MAC` gần như **bằng nhau**, chỉ lệch đúng phần bias. Điều này không đúng với conv, và mục 5 sẽ cho thấy vì sao nó quan trọng.

Công thức params **có** `in_features`, mà `in_features` đến từ shape sau `flatten` — tức là phụ thuộc kích thước ảnh. Đây chính là điểm khác căn bản với `Conv2d`, và là câu trả lời cho "vì sao `fc1` của `DigitCNN` chiếm 97% tham số".

## 3. Mã giả

```
linear(input[IN], weight[OUT][IN], bias[OUT], output[OUT]):

  for o = 0 .. OUT-1:

    acc = bias[o]                    # khởi tạo bằng bias, KHÔNG phải 0

    for i = 0 .. IN-1:
      acc += weight[o][i] * input[i]

    output[o] = acc
```

Hai vòng lặp. Đơn giản hơn `Conv2d` (6 vòng) rất nhiều — nhưng đơn giản không có nghĩa là rẻ trên phần cứng.

## 4. Áp dụng vào `SmallCNN`

| | Giá trị |
|---|---|
| `in_features` | 144 (= 16 × 3 × 3) |
| `out_features` | 10 (10 chữ số) |
| Params | 144 × 10 + 10 = **1.450** |
| MAC | 144 × 10 = **1.440** |
| Tỉ lệ trong tổng 5.018 params | **28,9%** |

So với `DigitCNN`:

| Model | Lớp FC | in → out | Params | % toàn mạng |
|---|---|---|---|---|
| `DigitCNN` | `fc1` | 1568 → 128 | 200.704 | **97,06%** |
| `SmallCNN` | `fc` | 144 → 10 | 1.450 | 28,9% |

Giảm 138 lần, và toàn bộ đến từ việc ép spatial xuống trước khi `flatten` cộng với bỏ hẳn lớp FC ẩn.

## 5. Mức FPGA-friendly

⚠️ **Phép toán thì tốt, kiểu truy cập bộ nhớ thì tệ.**

Bản thân phép tính vẫn là MAC, giống conv, không có gì khó. Vấn đề nằm ở **mỗi trọng số được dùng lại bao nhiêu lần**:

| Lớp | Params | MAC | Mỗi trọng số dùng lại |
|---|---|---|---|
| `conv1` | 72 | 56.448 | 784 lần |
| `conv2` | 1.152 | 225.792 | 196 lần |
| `conv3` | 2.304 | 112.896 | 49 lần |
| `fc` | 1.440 | 1.440 | **1 lần** |

Conv đọc một trọng số vào thanh ghi rồi dùng nó hàng trăm lần — chi phí đọc bộ nhớ được chia đều ra. FC đọc một trọng số, nhân **đúng một lần**, rồi vứt đi.

Hệ quả: FC bị giới hạn bởi **băng thông bộ nhớ**, không phải bởi số bộ nhân. Thêm DSP slice vào cũng không làm nó nhanh hơn, vì DSP phải ngồi chờ dữ liệu. Đây là lý do thật sự khiến `fc1` của `DigitCNN` (200.704 trọng số, mỗi cái dùng 1 lần) là thảm hoạ cho phần cứng — không chỉ vì nó to, mà vì nó ngốn băng thông mà không sinh ra phép tính tương xứng.

Đây cũng là lý do lượng tử hoá có lợi cho FC nhiều hơn cho conv: giảm 4 lần số byte phải đọc nghĩa là giảm thẳng 4 lần thời gian chạy của lớp bị nghẽn băng thông.

## 6. Chú ý khi port sang C

| Bẫy | Chi tiết |
|---|---|
| **Thứ tự chiều trọng số** | PyTorch lưu `weight[out_features][in_features]` → `W[o][i]`. Keras `Dense` lưu `[in][out]` — **chuyển vị**. Đọc file Keras bằng chỉ số PyTorch thì ra ma trận lật, không crash, kết quả sai hoàn toàn |
| **Khởi tạo `acc`** | Bằng `bias[o]`, không phải 0. Giống hệt bẫy ở `Conv2d` |
| **Tràn accumulator ở int8** | 144 tích của int8 × int8 có thể lên tới 144 × 127 × 127 = **2.322.576**. `int16` chỉ chứa tới 32.767 → **tràn**. Accumulator bắt buộc là `int32`. Đây là lỗi thật, không phải lý thuyết |
| **Bias ở fixed-point** | Bias nằm cùng thang với accumulator (tích của 2 số), không cùng thang với trọng số. Lượng tử hoá bias như trọng số là sai thang — thực tế phần cứng để bias ở int32 |
| **Không có softmax** | `fc` là lớp cuối, xuất logit thô. Đừng thêm softmax — xem `argmax.md` |

## 7. Tự kiểm tra

Làm trên giấy:

1. Muốn `fc` có **không quá 1.000** tham số (vẫn ra 10 lớp), `in_features` tối đa là bao nhiêu? Với 16 kênh thì spatial phải xuống còn bao nhiêu × bao nhiêu?
2. Với int8, tính giá trị tuyệt đối lớn nhất mà accumulator của `fc` có thể đạt. Cần tối thiểu bao nhiêu bit?
3. `fc1` của `DigitCNN` (1568 → 128): mỗi trọng số được dùng lại bao nhiêu lần? So với `conv2` của `SmallCNN`, chênh bao nhiêu lần?
4. Nếu thay `fc` bằng một `Conv2d(16 → 10, kernel 3, padding 0)` trên feature map 3×3, số params là bao nhiêu? Kết quả có khác `fc` không — vì sao?
