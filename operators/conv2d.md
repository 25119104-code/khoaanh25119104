# Operator: `Conv2d`

> Phase 1.5 nấc 2 → đầu vào cho Phase 2A (viết operator) và Phase 2D (golden model C).

## 1. Vai trò

Trượt một cửa sổ nhỏ (kernel) khắp ảnh, mỗi vị trí tính một tổng nhân-cộng. Cùng một bộ trọng số được **dùng lại** ở mọi vị trí — đó là điểm khác căn bản với `Linear`, và là lý do conv rẻ về bộ nhớ.

## 2. Công thức

### Kích thước đầu ra

```
H_out = floor( (H_in + 2×padding − kernel) / stride ) + 1
W_out = floor( (W_in + 2×padding − kernel) / stride ) + 1
```

Nguồn: PyTorch docs, mục *Shape* của `torch.nn.Conv2d`.

**Suy ra từ đâu:** đếm số vị trí đặt được cửa sổ. Mép trái cửa sổ nhảy theo bước `stride`, điều kiện còn nằm trong ảnh là `mép_trái + kernel ≤ H_in + 2×padding`. Số vị trí = số bước hợp lệ **+ 1** (vì vị trí 0 cũng tính).

**Hệ quả hay dùng:** `padding = (kernel − 1) / 2` thì `H_out = H_in` (same padding). Chỉ ra số nguyên khi kernel **lẻ** — một trong các lý do kernel luôn lẻ.

| kernel | padding giữ nguyên size |
|---|---|
| 3 | 1 |
| 5 | 2 |
| 7 | 3 |

### Số tham số

```
params = kernel_h × kernel_w × in_channels × out_channels  +  out_channels
                                                              └── bias
```

**Không có kích thước ảnh trong công thức.** Chạy trên 28×28 hay 7×7 cũng chừng đó trọng số. Đây là câu trả lời cho "vì sao conv rẻ mà fc đắt".

### Số phép tính (FLOPs)

```
MAC = kernel_h × kernel_w × in_ch × out_ch × H_out × W_out
```

Chỗ này **có** kích thước ảnh. Params quyết định bộ nhớ, MAC quyết định tốc độ/năng lượng — hai trục khác nhau, đừng lẫn.

## 3. Mã giả

Cho 1 ảnh (bỏ chiều batch). Zero-padding xử lý bằng kiểm tra biên, không cần cấp phát mảng đệm.

```
conv2d(input[IN_CH][IN_H][IN_W],
       weight[OUT_CH][IN_CH][K][K],
       bias[OUT_CH],
       output[OUT_CH][OUT_H][OUT_W]):

  for oc = 0 .. OUT_CH-1:                  # từng kênh đầu ra
    for oy = 0 .. OUT_H-1:
      for ox = 0 .. OUT_W-1:

        acc = bias[oc]                     # khởi tạo bằng bias, KHÔNG phải 0

        for ic = 0 .. IN_CH-1:             # cộng dồn qua mọi kênh đầu vào
          for ky = 0 .. K-1:
            for kx = 0 .. K-1:

              iy = oy * STRIDE - PAD + ky
              ix = ox * STRIDE - PAD + kx

              if 0 <= iy < IN_H and 0 <= ix < IN_W:      # ngoài biên = pixel 0
                acc += input[ic][iy][ix] * weight[oc][ic][ky][kx]

        output[oc][oy][ox] = acc
```

6 vòng lặp lồng nhau. Vòng ngoài duyệt **vị trí đầu ra**, vòng trong duyệt **cửa sổ kernel**.

Hai dòng quan trọng nhất là `iy` và `ix`: chúng ánh xạ toạ độ đầu ra về toạ độ đầu vào. Dấu `− PAD` là chỗ dễ sai nhất.

## 4. Áp dụng vào `SmallCNN`

| Lớp | in→out | H_in | H_out | Params | MAC |
|---|---|---|---|---|---|
| `conv1` | 1→8 | 28 | (28+2−3)/1+1 = **28** | 3·3·1·8+8 = **80** | 72 × 28² ≈ 56,000 |
| `conv2` | 8→16 | 14 | **14** | 3·3·8·16+16 = **1,168** | 1152 × 14² ≈ 226,000 |
| `conv3` | 16→16 | 7 | **7** | 3·3·16·16+16 = **2,320** | 2304 × 7² ≈ 113,000 |

Để ý `conv1` ít hơn `conv3` **29 lần** params nhưng chỉ ít hơn **2 lần** phép tính. Đúng minh hoạ params ≠ FLOPs.

## 5. Mức FPGA-friendly

✅ **Rất tốt.** Phép toán duy nhất là MAC (nhân + cộng tích luỹ) — đúng thứ mà mảng PE trên FPGA sinh ra để làm. Không có chia, không có hàm siêu việt.

Kernel 3×3 đặc biệt thân thiện: chỉ cần 9 bộ nhân và line-buffer 2 hàng. Kernel 7×7 cần 49 bộ nhân và buffer 6 hàng.

## 6. Chú ý khi port sang C

| Bẫy | Chi tiết |
|---|---|
| **Thứ tự chiều** | PyTorch: `weight[out_ch][in_ch][kH][kW]`. Keras: `[kH][kW][in_ch][out_ch]` — **ngược hẳn**. Lấy nhầm thứ tự thì không crash, chỉ ra kết quả sai |
| **Khởi tạo `acc`** | Bằng `bias[oc]`, không phải 0. Quên thì lệch đúng bằng bias, rất khó nhìn ra |
| **Dấu của PAD** | `iy = oy*STRIDE − PAD + ky`. Dấu trừ. Viết cộng thì ảnh bị dịch |
| **Biên** | Ngoài ảnh coi như 0, không phải lặp pixel mép |

## 7. Tự kiểm tra

Làm trên giấy, không chạy code:

1. Với `conv1` của SmallCNN (`IN_CH=1, OUT_CH=8, K=3, PAD=1, STRIDE=1`, ảnh 28×28): vòng lặp trong cùng chạy **tổng cộng bao nhiêu lần**? So với con số MAC ở mục 4.
2. Tính `output[0][0][0]` cần bao nhiêu phép nhân? Trong số đó bao nhiêu phép rơi vào vùng padding (bị bỏ qua)?
3. Nếu đổi `PAD` từ 1 xuống 0, `H_out` của `conv1` thành bao nhiêu? Shape trước `flatten` đổi thế nào? `fc` còn bao nhiêu params?
