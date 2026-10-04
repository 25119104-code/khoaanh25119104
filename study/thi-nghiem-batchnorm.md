# Thí nghiệm: thêm BatchNorm vào SmallCNN

> Script: `study/thu_batchnorm.py` · Kết quả thô: `study/ket_qua_batchnorm.json` · Hình: `figures/thi_nghiem_batchnorm.png`
>
> Số liệu chạy trên **MacBook Air M2** (conda env `mnist`, CPU), 10,6 phút.
> Chạy lại: `conda activate mnist && python study/thu_batchnorm.py`

## 1. Câu hỏi

1. SmallCNN không có BatchNorm thì có thiệt gì không?
2. Nếu có BatchNorm, **gộp (fold) BN vào Conv** như Thầy nói trong video Step2 thì kết quả có giữ nguyên không?
3. Sau khi gộp, trọng số còn vừa int8 / Q1.7 không?

## 2. Cách làm

- **Model so sánh:** SmallCNN_BN giống hệt SmallCNN, chỉ chèn `BatchNorm2d` **ngay sau mỗi Conv, trước ReLU**. Conv đặt `bias=False` vì β của BN đã làm nhiệm vụ dịch.
- **Công bằng:** train **lại cả hai** trong cùng một lần chạy, cùng seed 42, cùng chia 50k/10k/10k, cùng Adam lr = 0,001, batch 64, 30 epoch. Dùng lại đúng hàm `train_model()` của `model_comparison.py`.
- **Không đụng model chính:** checkpoint riêng `models/study_*.pth`, không sửa `models.py`.
- **Kiểm tra tái lập:** bản "không BN" train lại ra **đúng y** model chính đã chốt (val 98,79% ở epoch 27, test 98,82%, max|w| conv2 = 0,9813). → Cùng máy, cùng seed thì kết quả tái lập được; so sánh với bản có BN là công bằng.

## 3. Kết quả

### 3.1 Accuracy và tốc độ hội tụ

| | Không BN | Có BN | BN đã gộp vào Conv |
|---|---|---|---|
| Tham số học được | 5.018 | 5.058 (+40 γ, +40 β) | **5.018** |
| Val acc tốt nhất | 98,79% | 98,80% | — |
| Epoch tốt nhất | 27/30 | 29/30 | — |
| Epoch đầu tiên val ≥ 98,0% | 5 | **2** | — |
| Epoch đầu tiên val ≥ 98,5% | 9 | **4** | — |
| Test acc | **98,82%** | 98,71% | 98,71% |

### 3.2 Gộp BN vào Conv

```
a        = γ / √(σ² + ε)            (tính 1 lần trên PC)
w_mới    = a · w
bias_mới = β − a · μ
```

- **0 / 10.000** ảnh đoán khác nhau giữa model có BN và model đã gộp.
- Lệch logit lớn nhất **9,87e-05**, chỉ do làm tròn float32 (thứ tự phép tính khác nhau).
- Sau khi gộp, model là **một SmallCNN bình thường**, đúng 5.018 tham số. `export_params.py`, `inference_python.py` và golden model C dùng lại được ngay, không phải sửa.

### 3.3 Trọng số sau gộp và int8 / Q1.7 (trần 0,9922)

| Lớp | Không BN | BN, trước gộp | BN, sau gộp | Hệ số `a` (min…max) |
|---|---|---|---|---|
| conv1 | 0,8697 | 0,5689 | **1,1246** (2 trọng số vượt trần) | 0,81 … 2,70 |
| conv2 | 0,9813 | 0,5668 | 0,4117 | 0,52 … 0,81 |
| conv3 | 0,7880 | 0,5848 | 0,9565 | 1,36 … 1,96 |
| fc | 0,9626 | 0,7924 | 0,7924 (không gộp) | — |

Bảng ghi max|w|. So **cột 2 với cột 3** để thấy tác dụng của phép gộp trên cùng một model: gộp nhân mỗi
kênh với hệ số `a`. Kênh có `a > 1` thì trọng số **to ra** (conv1: 0,57 → 1,12, gần gấp đôi; conv3: 0,58 → 0,96).
Kênh có `a < 1` thì **nhỏ đi** (conv2: 0,57 → 0,41). Cột 1 là model chính, train riêng, để so.

## 4. Nhận xét

1. **BN không làm model chính xác hơn.** Val gần như bằng nhau (98,80% vs 98,79%); test còn **thấp hơn 0,11%** (98,71% vs 98,82%). Với mạng nhỏ, nông như SmallCNN trên MNIST, BN không đem lại accuracy.
2. **BN giúp train nhanh hơn khoảng 2 lần.** Đạt 98,0% sau 2 epoch thay vì 5, đạt 98,5% sau 4 epoch thay vì 9. Loss cũng thấp hơn ở mọi epoch (xem hình). Đây là lợi ích thật của BN, nhưng chỉ có lợi **lúc train**.
3. **Lên chip thì không khác gì.** Gộp xong BN biến mất: không có `√`, không phép chia, không thêm MAC, số tham số vẫn 5.018. Đường inference vẫn đúng 6 operator.
4. **Với quantization, có BN còn bất lợi hơn.** Model chính (không BN) có **0** trọng số vượt trần Q1.7, dù conv2 sát trần (0,9813). Model BN sau khi gộp có conv1 lên **1,12 → 2 trọng số vượt trần**, conv3 lên 0,96 cũng sát trần. Nguyên nhân: gộp nhân trọng số với `a` tới 2,70. → Nếu sau này dùng model có BN thì phải chọn scale **riêng từng lớp / từng kênh**, không dùng chung Q1.7.

## 5. Quyết định

**Giữ SmallCNN không BN làm model chính.** Lý do:

- BN không tăng accuracy (test còn giảm 0,11%), mà Thầy dặn chưa cần tối ưu accuracy.
- Trên chip, có BN hay không đều như nhau sau khi gộp.
- Model không BN thuận lợi hơn cho quantization (không trọng số nào vượt trần Q1.7).
- Đổi model là phải sinh lại `params/`, `golden/` và mọi con số trong tài liệu, mà không được gì.

Giá trị của thí nghiệm là **đã tự làm và kiểm chứng được bước "gộp BatchNorm vào Conv"** trong Step2 của Thầy: kết quả giống hệt, 0 ảnh sai khác. Nếu sau này dùng model có BN (model lớn hơn, đề tài NCKH), quy trình đã sẵn sàng.

## 6. Câu nói ngắn để báo cáo Thầy

> Em thử thêm BatchNorm vào SmallCNN, train lại cùng điều kiện. Accuracy không tăng (val 98,80% so với
> 98,79%, test còn giảm 0,11%), nhưng hội tụ nhanh gấp đôi (98,5% sau 4 epoch thay vì 9).
> Em gộp BN vào Conv: 0/10.000 ảnh khác kết quả, số tham số về lại 5.018, trên chip không còn phép căn
> hay phép chia. Nhưng gộp làm trọng số conv1 to lên tới 1,12, vượt trần Q1.7. Vì vậy em giữ model không BN.
