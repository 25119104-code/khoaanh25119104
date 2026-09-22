# Operator: `argmax`

> Phase 1.5 nấc 2 → đầu vào cho Phase 2A (viết operator) và Phase 2D (golden model C).
>
> Điểm kết thúc của inference. Sau operator này không còn gì nữa.

## 1. Vai trò

Nhận 10 logit, trả về **chỉ số** của số lớn nhất. Chỉ số đó chính là chữ số dự đoán: chỉ số 7 nghĩa là model đoán "7".

Không trả về giá trị, trả về **vị trí**. Đây là chỗ mạng chuyển từ "10 con số thực" sang "một câu trả lời".

## 2. Công thức

```
argmax(z) = chỉ số i sao cho z[i] ≥ z[j] với mọi j
```

### Vì sao không cần softmax

Công thức softmax:

```
softmax(z)[i] = exp(z[i]) / Σ exp(z[j])
```

`exp` là hàm **đơn điệu tăng nghiêm ngặt**, và mẫu số là cùng một hằng số dương cho cả 10 phần tử. Nên softmax **giữ nguyên thứ tự**: `z[a] > z[b]` thì `softmax(z)[a] > softmax(z)[b]`.

Suy ra:

```
argmax( softmax(z) )  =  argmax(z)
```

→ **Bỏ hẳn softmax trên chip.** Tiết kiệm 10 phép `exp()` và 1 phép chia — `exp` là phép đắt nhất trong toàn mạng nếu phải hiện thực (thường phải dùng bảng tra hoặc xấp xỉ đa thức trên FPGA). Đúng yêu cầu "thay/bỏ phép toán khó cho FPGA" của project.

### Vì sao model không có sẵn lớp softmax

`SmallCNN.forward()` kết thúc bằng `return self.fc(x)` — logit thô, không softmax. Đó không phải thiếu sót.

Lúc train dùng `nn.CrossEntropyLoss`, mà hàm này = `LogSoftmax` + `NLLLoss` gộp lại, **tự áp softmax bên trong**. Nếu thêm softmax vào `forward()` nữa thì bị áp hai lần, loss sai, model học kém đi.

Kết quả tiện lợi: model viết đúng chuẩn train thì đồng thời đã sẵn sàng cho phần cứng, không phải gỡ lớp nào ra.

## 3. Mã giả

```
argmax(logits[10]):

  best_idx = 0
  best_val = logits[0]

  for i = 1 .. 9:
    if logits[i] > best_val:        # ">" chứ không phải ">=" — xem mục 6
      best_val = logits[i]
      best_idx = i

  return best_idx
```

9 phép so sánh. Không nhân, không chia, không cộng.

## 4. Áp dụng vào `SmallCNN`

| | Giá trị |
|---|---|
| Đầu vào | 10 logit (float32, hoặc int32 nếu chạy fixed-point) |
| Đầu ra | 1 số nguyên trong khoảng 0–9 |
| Params | 0 |
| Phép so sánh | 9 |

Đặt cạnh 395.136 MAC của phần conv: argmax chiếm khoảng **0,002%** khối lượng tính toán.

## 5. Mức FPGA-friendly

✅ **Gần như miễn phí.** Một cây comparator: 10 đầu vào gộp đôi thành 5, rồi 3, rồi 2, rồi 1 — độ sâu 4 tầng. Không DSP, không bộ nhân, vài chục LUT.

So sánh thẳng trên **int32 accumulator** được, không cần dequantize về float trước. Vì phép dequantize là nhân với cùng một `scale` dương cho cả 10 logit, mà nhân với hằng số dương cũng giữ nguyên thứ tự — y hệt lý do bỏ được softmax. Bớt thêm 10 phép nhân nữa.

## 6. Chú ý khi port sang C

| Bẫy | Chi tiết |
|---|---|
| **`>` hay `>=`** | Dùng `>` thì khi hoà chọn chỉ số **nhỏ nhất**; dùng `>=` thì chọn **lớn nhất**. `torch.argmax` trả về chỉ số nhỏ nhất → phải viết `>`. Sai chỗ này thì golden model lệch PyTorch ở đúng những ảnh bị hoà |
| **Hoà không hiếm như tưởng** | Với float32 hầu như không bao giờ hoà. Nhưng sau khi lượng tử hoá, giá trị bị dồn về lưới — `xem_quantization.py` bảng 3 cho thấy 2.304 trọng số `conv3` chỉ còn 155 giá trị khác nhau ở int8. Logit cũng bị dồn tương tự, và khả năng hai logit trùng nhau tăng lên rõ rệt |
| **Đừng hiện thực softmax** | Xem mục 2. Nếu code C có `exp()` ở lớp cuối thì đang làm thừa |
| **Nếu cần độ tin cậy** | Lúc đó mới buộc phải có softmax, và `exp()` quay lại. Cách rẻ hơn: xuất **hiệu giữa logit lớn nhất và logit lớn nhì** làm thước đo "chắc chắn tới đâu" — chỉ tốn 1 phép trừ |
| **Khởi tạo** | `best_idx = 0`, `best_val = logits[0]`. Đừng khởi tạo `best_val` bằng 0 — logit có thể âm hết, khi đó hàm trả về sai |

## 7. Tự kiểm tra

Làm trên giấy:

1. Chứng minh `argmax(softmax(z)) = argmax(z)` bằng tính đơn điệu. Cần điều kiện gì về mẫu số?
2. Sau khi lượng tử hoá int8, khả năng hai logit bằng nhau tăng hay giảm? Giải thích dựa vào bảng 3 của `xem_quantization.py`.
3. Nếu đổi `>` thành `>=` trong mã giả, loại ảnh nào có thể đổi kết quả? Ước lượng trong 10.000 ảnh test thì có bao nhiêu ảnh như vậy ở float32, và ở int8?
4. Logit của một ảnh là `[-2.1, 0.4, 8.7, 8.7, -1.0, 3.2, 0.0, -5.5, 1.1, 2.9]`. `torch.argmax` trả về gì? Code C viết `>=` trả về gì?
