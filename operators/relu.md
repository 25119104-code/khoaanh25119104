# Operator: `ReLU`

> Phase 1.5 nấc 2 → đầu vào cho Phase 2A (viết operator) và Phase 2D (golden model C).

## 1. Vai trò

Cắt bỏ phần âm: số dương giữ nguyên, số âm ép về 0.

Đây là **nguồn phi tuyến duy nhất** của `SmallCNN`. Bỏ hết ReLU đi thì conv1 → conv2 → conv3 → fc chỉ còn là chuỗi phép biến đổi tuyến tính, mà tích của nhiều phép tuyến tính vẫn là **một** phép tuyến tính. Cả mạng 5.018 tham số sẽ sụp về đúng năng lực của một lớp `Linear(784, 10)` — vẫn chỉ vẽ được ranh giới thẳng. ReLU là thứ làm cho việc xếp chồng nhiều lớp có ý nghĩa.

## 2. Công thức

```
y = max(0, x)
```

Áp dụng **từng phần tử một** (element-wise). Shape vào = shape ra, không đổi chiều nào, không có tham số nào.

Đạo hàm (chỉ dùng khi train, inference không cần):

```
dy/dx = 1   nếu x > 0
        0   nếu x ≤ 0
```

Đạo hàm bằng đúng 1 ở nhánh dương là lý do ReLU không bị vanishing gradient như `sigmoid` hay `tanh` — gradient đi qua bao nhiêu lớp vẫn giữ nguyên độ lớn.

## 3. Mã giả

```
relu(data[N]):                      # N = tổng số phần tử, không quan tâm shape
  for i = 0 .. N-1:
    if data[i] < 0:
      data[i] = 0
```

Ba dòng. Không nhân, không chia, không cấp phát thêm bộ nhớ — sửa tại chỗ được.

## 4. Áp dụng vào `SmallCNN`

| Vị trí | Shape khi đó | Số phần tử phải xét |
|---|---|---|
| sau `conv1` | 8 × 28 × 28 | 6.272 |
| sau `conv2` | 16 × 14 × 14 | 3.136 |
| sau `conv3` | 16 × 7 × 7 | 784 |
| | **Tổng** | **10.192** |

Đặt cạnh tổng MAC của 3 lớp conv (**395.136**), ReLU chiếm khoảng **2,6%** số phép tính và **0** tham số. Rẻ tới mức không cần bàn khi tối ưu.

## 5. Mức FPGA-friendly

✅ **Tốt nhất trong mọi operator của mạng này.** Không nhân, không chia, không DSP slice. Chỉ là: đọc bit dấu, nếu là 1 thì xuất 0, ngược lại xuất nguyên. Một comparator + một mux, khoảng 1 LUT cho mỗi phần tử.

Với int8 đối xứng (`zero_point = 0`), ReLU rút gọn xuống mức **một cổng logic**: lấy bit dấu (MSB) đảo lại rồi AND với cả byte.

### Một tối ưu bỏ túi: đổi thứ tự `relu` và `pool`

`relu(maxpool(x))` cho kết quả **y hệt** `maxpool(relu(x))`, vì cả `max` lẫn `relu` đều là hàm đơn điệu không giảm:

- Nếu cả cửa sổ đều âm: vế trái = `relu(số âm)` = 0; vế phải = `max(0,0,0,0)` = 0.
- Nếu có ít nhất một số dương: cả hai vế đều trả về đúng số dương lớn nhất.

Mà `maxpool` giảm số phần tử, nên làm ReLU **sau** pool thì rẻ hơn:

| Thứ tự | conv1 | conv2 | conv3 | Tổng phép ReLU |
|---|---|---|---|---|
| `pool(relu(x))` — code hiện tại | 8·28·28 = 6.272 | 16·14·14 = 3.136 | 16·7·7 = 784 | 10.192 |
| `relu(pool(x))` | 8·14·14 = 1.568 | 16·7·7 = 784 | 16·3·3 = 144 | **2.496** |

Giảm **4,08 lần**, kết quả không đổi một bit.

> **Không phải đúng 4 lần — đây là chỗ dễ ghi sai.** Lấy 10.192 ÷ 4 = 2.548 là sai 52 phép.
> Hai pool đầu chia đúng 4 (28→14, 14→7), nhưng pool cuối 7→3 bị `floor` cắt hàng và cột cuối:
> 16·7·7 = 784 phần tử vào mà chỉ ra 16·3·3 = 144, tức chia 5,44 chứ không phải 4.
> Đúng cái hành vi `floor` đã nói ở `conv2d.md` — nó không chỉ đổi shape, nó đổi cả phép đếm. Đây đúng loại "thay/bỏ phép toán cho FPGA" mà project yêu cầu. **Lưu ý:** chỉ đúng với `MaxPool`. Với `AvgPool` thì sai, vì trung bình của các số đã cắt âm khác trung bình rồi mới cắt.

## 6. Chú ý khi port sang C

| Bẫy | Chi tiết |
|---|---|
| **Sửa tại chỗ** | `nn.ReLU(inplace=True)` ghi đè mảng đầu vào. Trong C làm vậy cũng được với inference, nhưng nếu còn cần giá trị gốc ở đâu đó thì mất |
| **`<` hay `<=`** | `max(0, x)` với `x = 0` ra 0 dù viết kiểu nào. Không có bẫy ở đây — khác với `argmax` |
| **Dead ReLU** | Nếu một kênh cho ra ≤ 0 với **mọi** ảnh thì kênh đó chết, trọng số conv sinh ra nó là vô dụng. Đáng kiểm khi muốn cắt tiếp params |
| **Số âm trong fixed-point** | Q1.7 dùng bù 2. Kiểm tra bit dấu, đừng so sánh như số không dấu |

## 7. Tự kiểm tra

Làm trên giấy:

1. Bỏ toàn bộ ReLU khỏi `SmallCNN`. Mạng còn lại tương đương với **một** lớp gì? Trong 5.018 tham số, bao nhiêu trở thành thừa?
2. Áp dụng tối ưu ở mục 5 cho riêng vị trí sau `conv1`: số phép ReLU ở đó giảm từ bao nhiêu xuống bao nhiêu?
3. Với int8 `zero_point = 0`, viết ReLU bằng **một** biểu thức bit (không dùng `if`).
