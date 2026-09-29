# golden/ — đầu ra chuẩn của PyTorch để đối chiếu

Ảnh mẫu: test set index **0**, nhãn đúng **7**. Sinh bởi `export_params.py`.

Mỗi file: 1 số / dòng, trải phẳng row-major theo shape (bỏ chiều batch).

| File | Shape | Số phần tử |
|---|---|---|
| `input.txt` | [1, 28, 28] | 784 |
| `01_conv1.txt` | [8, 28, 28] | 6272 |
| `02_relu1.txt` | [8, 28, 28] | 6272 |
| `03_pool1.txt` | [8, 14, 14] | 1568 |
| `04_conv2.txt` | [16, 14, 14] | 3136 |
| `05_relu2.txt` | [16, 14, 14] | 3136 |
| `06_pool2.txt` | [16, 7, 7] | 784 |
| `07_conv3.txt` | [16, 7, 7] | 784 |
| `08_relu3.txt` | [16, 7, 7] | 784 |
| `09_pool3.txt` | [16, 3, 3] | 144 |
| `10_flatten.txt` | [144] | 144 |
| `11_fc_logits.txt` | [10] | 10 |
| `logits_pytorch.txt` | [10000, 10] | 1 dòng / ảnh, 10 số cách nhau dấu cách |

Accuracy PyTorch trên 10000 ảnh test: **98.82%**.
