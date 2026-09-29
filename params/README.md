# params/ — tham số SmallCNN cho golden model C

Sinh bởi `export_params.py` từ `models/small_cnn.pth`. **Không sửa tay.**

- `weights.txt`: 4968 số, `biases.txt`: 50 số, tổng **5018** params.
- 1 số float32 / dòng, định dạng `%.9g` (đọc lại ra đúng từng bit).
- Thứ tự trong mỗi lớp giữ nguyên PyTorch: conv `[out_ch][in_ch][ky][kx]`, fc `[out][in]`.

| Lớp | Shape weight | Offset weight | Số weight | Offset bias | Số bias |
|---|---|---|---|---|---|
| `conv1` | [8, 1, 3, 3] | 0 | 72 | 0 | 8 |
| `conv2` | [16, 8, 3, 3] | 72 | 1152 | 8 | 16 |
| `conv3` | [16, 16, 3, 3] | 1224 | 2304 | 24 | 16 |
| `fc` | [10, 144] | 3528 | 1440 | 40 | 10 |

Offset tính từ 0. Ví dụ phần tử `conv2.weight[oc][ic][ky][kx]` nằm ở dòng (tính từ 0):

```
72 + ((oc*8 + ic)*3 + ky)*3 + kx
```
