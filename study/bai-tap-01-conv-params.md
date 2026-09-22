# Bài tập tự luyện 01 — Shape và Params

**Mục đích:** chứng minh mình *suy ra được* chứ không phải *học thuộc*. Đây là loại câu Thầy dùng để phân biệt hai thứ đó.

**Cách làm:**

- Làm trên giấy, **không mở tài liệu, không dùng máy tính**.
- Mỗi bài 2–3 phút. Quá 5 phút mà chưa ra → ghi "bí", chuyển bài tiếp.
- Viết cả **cách suy ra**, không chỉ đáp số. Ra đúng số mà không giải thích được thì coi như chưa làm.
- Làm xong hết 5 bài mới mở phần Phụ lục ở cuối.

---

## Bài 1 — Kiến trúc lạ, tính xuôi

```
Input 28×28×1

Conv2d(1  → 12, kernel 5, padding 2)  →  ReLU  →  MaxPool2d(2, 2)
Conv2d(12 → 24, kernel 3, padding 1)  →  ReLU  →  MaxPool2d(2, 2)
Flatten  →  Linear(? → 10)
```

**a)** Viết chuỗi shape đầy đủ, từ `1×28×28` tới `10`.

**b)** Params của từng lớp có trọng số.

**c)** Tổng params.

**Trả lời:**

| Bước | Phép tính | Shape ra | Params |
|---|---|---|---|
| input | — | 1 × 28 × 28 | 0 |
| conv1 |  |  |  |
| pool |  |  |  |
| conv2 |  |  |  |
| pool |  |  |  |
| flatten |  |  |  |
| fc |  |  |  |
| | | **Tổng** |  |

---

## Bài 2 — Tính ngược

Một lớp `Conv2d` kernel 3×3 có **4.640 tham số**. Biết `out_channels = 32`.

Tìm `in_channels`.

> Trình bày cách suy ra từ công thức, đừng thử số cho tới khi trúng.

**Trả lời:**

```


```

---

## Bài 3 — Bẫy floor

Ảnh vào **32×32**. Đi qua:

```
Conv2d(kernel=5, padding=0, stride=1)  →  MaxPool2d(kernel=3, stride=2)
```

**a)** Shape ra sau mỗi bước.

**b)** Ở bước pool, **bao nhiêu hàng và bao nhiêu cột bị bỏ hoàn toàn?** Chỉ rõ là hàng/cột nào.

**Trả lời:**

```


```

---

## Bài 4 — Thiết kế ngược

Bạn muốn lớp FC cuối (ra 10 lớp) có **không quá 2.000 tham số**.

**a)** Số phần tử sau `Flatten` tối đa là bao nhiêu?

**b)** Nếu feature map cuối có **32 kênh**, spatial tối đa là bao nhiêu × bao nhiêu?

> Gợi ý: spatial phải là số nguyên, và thường là hình vuông.

**Trả lời:**

```


```

---

## Bài 5 — Params vs FLOPs

Lớp `conv2` của SmallCNN: `Conv2d(8 → 16, kernel 3×3, padding 1)`, chạy trên feature map **14×14**.

**a)** Params của lớp này.

**b)** Số phép MAC (nhân–cộng) cho một ảnh.

**c)** Nếu feature map đầu vào là **28×28** thay vì 14×14:
- params đổi bao nhiêu lần?
- MAC đổi bao nhiêu lần?

**d)** Câu c nói lên điều gì về quan hệ giữa params và FLOPs?

**Trả lời:**

```


```

---

## Tự chấm

| Bài | Đã làm | Bí ở đâu |
|---|---|---|
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |
| 4 |  |  |
| 5 |  |  |

Bài **2** và **4** là loại khó nhất vì phải đi ngược công thức. Bí hai bài này là bình thường ở lần đầu — nhưng phải làm được trước khi gặp Thầy.

---

## Phụ lục — CHỈ MỞ SAU KHI ĐÃ LÀM XONG

<br><br><br><br><br><br><br><br><br><br>

Ba công thức cần dùng, không có gì khác:

```
Kích thước đầu ra
    H_out = floor( (H_in + 2·padding − kernel) / stride ) + 1

Params của Conv2d
    k · k · in_channels · out_channels  +  out_channels

Params của Linear
    in_features · out_features  +  out_features
```

Thêm một công thức cho bài 5:

```
MAC của một lớp conv (cho 1 ảnh)
    k · k · in_channels · out_channels · H_out · W_out
```

Không có đáp số ở đây — làm xong gửi bài cho Claude chấm.
